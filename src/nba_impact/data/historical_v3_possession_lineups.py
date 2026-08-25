"""Attach strict V3 ordinal lineups to separate historical V3 possessions.

This adapter joins only exact NBA Stats V3 ``actionId`` intervals.  It rebuilds
the frozen owner state before joining, so every owned action must reconcile to
one persisted possession candidate and one ten-player lineup state.  Outputs
are research candidates.  They never replace canonical CDN possessions.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .event_state import parse_clock_seconds
from .historical_v3_possessions import infer_v3_possession_owners
from .manifest import sha256_file, write_json_atomic
from .possessions import AWAY_LINEUP_COLUMNS, HOME_LINEUP_COLUMNS, LINEUP_COLUMNS


def _require(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    if missing := sorted(columns - set(frame.columns)):
        raise ValueError(f"{label} is missing required columns: {missing}")


def _elapsed_seconds(period: pd.Series, clock: pd.Series) -> pd.Series:
    remaining = parse_clock_seconds(clock)
    regulation = period.le(4)
    return pd.Series(
        np.where(
            regulation,
            (period - 1) * 720.0 + (720.0 - remaining),
            2880.0 + (period - 5) * 300.0 + (300.0 - remaining),
        ),
        index=period.index,
    )


def _v3_partition(root: Path, project_season: int, season_type: str) -> Path:
    direct = root / f"project_season={project_season}" / f"{season_type}.parquet"
    nested = root / "nbastatsv3" / f"project_season={project_season}" / f"{season_type}.parquet"
    if direct.exists():
        return direct
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Pinned V3 partition is absent: {direct} or {nested}")


def _owned_v3_actions(raw: pd.DataFrame) -> pd.DataFrame:
    """Reapply the frozen owner rule and number the exact owned action runs."""
    inferred, _ = infer_v3_possession_owners(raw)
    required = {
        "game_id", "event_order", "action_number", "period", "clock", "possession",
        "points_added", "scoring_team_id", "home_team_id", "away_team_id",
    }
    _require(inferred, required - {"home_team_id", "away_team_id"}, "inferred V3 actions")
    # Team identities are supplied by the persisted candidate.  The raw archive
    # does not carry them as immutable game dimensions.
    return inferred


def _candidate_action_frame(raw: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Map each owned action to one persisted candidate possession exactly once."""
    game_teams = candidates[["game_id", "home_team_id", "away_team_id"]].drop_duplicates()
    if game_teams.duplicated("game_id").any():
        raise ValueError("Historical possession candidates have conflicting game team identities.")
    actions = _owned_v3_actions(raw).merge(game_teams, on="game_id", how="inner", validate="many_to_one")
    valid = actions["possession"].eq(actions["home_team_id"]) | actions["possession"].eq(actions["away_team_id"])
    owned = actions.loc[valid].sort_values(["game_id", "event_order"], kind="stable").copy()
    new_possession = (
        owned["game_id"].ne(owned["game_id"].shift())
        | owned["period"].ne(owned["period"].shift())
        | owned["possession"].ne(owned["possession"].shift())
    )
    owned["possession_number"] = new_possession.groupby(owned["game_id"], sort=False).cumsum().astype(int)
    owned["possession_id"] = (
        owned["game_id"] + ":v3:" + owned["possession_number"].astype(str).str.zfill(3)
    )
    owned["home_points_added"] = np.where(
        owned["scoring_team_id"].eq(owned["home_team_id"]), owned["points_added"], 0
    )
    owned["away_points_added"] = np.where(
        owned["scoring_team_id"].eq(owned["away_team_id"]), owned["points_added"], 0
    )
    owned["seconds_elapsed_game"] = _elapsed_seconds(owned["period"], owned["clock"])
    if owned.duplicated(["game_id", "event_order"]).any():
        raise ValueError("An owned V3 action was duplicated before possession mapping.")
    return owned


def _candidate_reconciliation(owned: pd.DataFrame, candidates: pd.DataFrame) -> str | None:
    """Require the live owner reconstruction to exactly reproduce persisted rows."""
    aggregate = owned.groupby("possession_id", as_index=False, sort=False).agg(
        game_id=("game_id", "first"),
        possession_number=("possession_number", "first"),
        period=("period", "first"),
        start_order_number=("event_order", "first"),
        end_order_number=("event_order", "last"),
        start_action_number=("action_number", "first"),
        end_action_number=("action_number", "last"),
        offense_team_id=("possession", "first"),
        home_team_id=("home_team_id", "first"),
        away_team_id=("away_team_id", "first"),
        points=("points_added", "sum"),
        home_points=("home_points_added", "sum"),
        away_points=("away_points_added", "sum"),
        action_count=("event_order", "size"),
    )
    source = candidates.loc[candidates["possession_id"].isin(aggregate["possession_id"])].copy()
    if (
        len(candidates) != len(aggregate)
        or len(source) != len(aggregate)
        or source["possession_id"].duplicated().any()
    ):
        return "candidate_possession_id_mismatch"
    joined = aggregate.merge(source, on="possession_id", suffixes=("_rebuilt", "_candidate"), validate="one_to_one")
    columns = (
        "game_id", "possession_number", "period", "start_order_number", "end_order_number",
        "start_action_number", "end_action_number", "offense_team_id", "home_team_id",
        "away_team_id", "points", "home_points", "away_points", "action_count",
    )
    for column in columns:
        left = joined[f"{column}_rebuilt"]
        right = joined[f"{column}_candidate"]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left.astype(float), right.astype(float), equal_nan=True):
                return f"candidate_{column}_reconciliation_failed"
        elif not left.astype(str).eq(right.astype(str)).all():
            return f"candidate_{column}_reconciliation_failed"
    return None


def _attach_lineups(owned: pd.DataFrame, stints: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Attach one validated ten-player state to every owned V3 action."""
    ordered = stints.sort_values("start_action_id", kind="stable").reset_index(drop=True)
    required = {"stint_id", "start_action_id", "end_action_id_exclusive", *LINEUP_COLUMNS}
    _require(ordered, required, "historical V3 lineup stints")
    starts = pd.to_numeric(ordered["start_action_id"], errors="raise").to_numpy(dtype=np.int64)
    ends = pd.to_numeric(ordered["end_action_id_exclusive"], errors="raise").to_numpy(dtype=np.int64)
    action_ids = owned["event_order"].to_numpy(dtype=np.int64)
    if (
        len(ordered) == 0
        or starts[0] > action_ids.min()
        or ends[-1] <= action_ids.max()
        or (ends <= starts).any()
        or (starts[1:] != ends[:-1]).any()
    ):
        return owned.iloc[0:0].copy(), "lineup_action_interval_coverage_failed"
    players = ordered.loc[:, LINEUP_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if players.isna().any(axis=None) or (players.le(0).any(axis=None)):
        return owned.iloc[0:0].copy(), "lineup_player_id_missing"
    home = players.loc[:, HOME_LINEUP_COLUMNS]
    away = players.loc[:, AWAY_LINEUP_COLUMNS]
    if home.nunique(axis=1).ne(5).any() or away.nunique(axis=1).ne(5).any():
        return owned.iloc[0:0].copy(), "lineup_not_five_unique_players"
    if any(set(home.iloc[index]).intersection(set(away.iloc[index])) for index in range(len(players))):
        return owned.iloc[0:0].copy(), "lineup_home_away_overlap"
    indices = np.searchsorted(starts, action_ids, side="right") - 1
    if (indices < 0).any() or (action_ids >= ends[np.maximum(indices, 0)]).any():
        return owned.iloc[0:0].copy(), "owned_action_without_exact_lineup"
    attached = owned.copy()
    attached["ordinal_stint_id"] = ordered["stint_id"].to_numpy()[indices]
    for column in LINEUP_COLUMNS:
        attached[column] = players[column].to_numpy(dtype=np.int64)[indices]
    if attached.duplicated(["game_id", "event_order"]).any() or len(attached) != len(owned):
        return owned.iloc[0:0].copy(), "owned_action_mapping_not_one_to_one"
    return attached, None


def _segments(attached: pd.DataFrame) -> pd.DataFrame:
    work = attached.sort_values(["game_id", "possession_number", "event_order"], kind="stable").copy()
    changed = work["possession_id"].ne(work["possession_id"].shift()) | work["ordinal_stint_id"].ne(work["ordinal_stint_id"].shift())
    work["segment_number"] = changed.groupby(work["possession_id"], sort=False).cumsum().astype(int)
    aggregations: dict[str, tuple[str, str]] = {
        "period": ("period", "first"),
        "start_order_number": ("event_order", "first"),
        "end_order_number": ("event_order", "last"),
        "start_action_number": ("action_number", "first"),
        "end_action_number": ("action_number", "last"),
        "start_seconds_elapsed": ("seconds_elapsed_game", "first"),
        "end_seconds_elapsed": ("seconds_elapsed_game", "last"),
        "ordinal_stint_id": ("ordinal_stint_id", "first"),
        "offense_team_id": ("possession", "first"),
        "points": ("points_added", "sum"),
        "action_count": ("event_order", "size"),
    }
    for column in LINEUP_COLUMNS:
        aggregations[column] = (column, "first")
    output = work.groupby(
        ["possession_id", "game_id", "possession_number", "segment_number"],
        as_index=False,
        sort=False,
    ).agg(**aggregations)
    output["possession_segment_id"] = (
        output["possession_id"] + ":s" + output["segment_number"].astype(str).str.zfill(2)
    )
    return output


def _validate_outputs(possessions: pd.DataFrame, segments: pd.DataFrame, attached: pd.DataFrame) -> str | None:
    if possessions.empty or segments.empty or attached.empty:
        return "empty_output"
    if possessions.duplicated("possession_id").any() or segments.duplicated("possession_segment_id").any():
        return "duplicate_output_id"
    if len(attached) != attached[["game_id", "event_order"]].drop_duplicates().shape[0]:
        return "owned_action_mapping_not_one_to_one"
    reconcile = segments.groupby("possession_id", as_index=False).agg(
        points=("points", "sum"), action_count=("action_count", "sum")
    ).merge(possessions[["possession_id", "points", "action_count"]], on="possession_id", suffixes=("_segment", "_possession"), validate="one_to_one")
    if not np.allclose(reconcile["points_segment"], reconcile["points_possession"]):
        return "segment_points_not_conserved"
    if not reconcile["action_count_segment"].eq(reconcile["action_count_possession"]).all():
        return "segment_action_counts_not_conserved"
    players = segments.loc[:, LINEUP_COLUMNS]
    if players.isna().any(axis=None) or players.nunique(axis=1).ne(10).any():
        return "segment_lineup_not_ten_unique_players"
    return None


def build_historical_v3_possession_lineup_candidate(
    v3_root: str | Path,
    possession_candidates_path: str | Path,
    possession_quality_path: str | Path,
    lineup_stints_path: str | Path,
    lineup_quality_path: str | Path,
    possessions_destination: str | Path,
    segments_destination: str | Path,
    assigned_actions_destination: str | Path,
    quality_destination: str | Path,
    report_destination: str | Path,
    manifest_dir: str | Path,
    *,
    project_season: int,
    season_type: str = "regular",
) -> dict:
    """Build separate RAPM-shaped historical data only for double-passed games."""
    paths = [
        _v3_partition(Path(v3_root), project_season, season_type),
        Path(possession_candidates_path), Path(possession_quality_path),
        Path(lineup_stints_path), Path(lineup_quality_path),
    ]
    if missing := [str(path) for path in paths if not path.exists()]:
        raise FileNotFoundError(f"Required historical adapter input is absent: {missing}")
    candidates = pd.read_parquet(paths[1])
    _require(candidates, {"possession_id", "game_id", "season_end", "season_type", "home_team_id", "away_team_id", "lineup_ready"}, "historical possession candidates")
    candidates = candidates.loc[candidates["season_end"].eq(project_season) & candidates["season_type"].eq(season_type)].copy()
    candidates["game_id"] = candidates["game_id"].astype(str)
    possession_quality = pd.read_parquet(paths[2])
    lineup_quality = pd.read_parquet(paths[4])
    for frame, label in ((possession_quality, "historical possession quality"), (lineup_quality, "historical lineup quality")):
        _require(frame, {"game_id", "passed"}, label)
        frame["game_id"] = frame["game_id"].astype(str)
    if "project_season" in possession_quality:
        possession_quality = possession_quality.loc[possession_quality["project_season"].eq(project_season)]
    if "season_type" in possession_quality:
        possession_quality = possession_quality.loc[possession_quality["season_type"].eq(season_type)]
    if "season_type" in lineup_quality:
        lineup_quality = lineup_quality.loc[lineup_quality["season_type"].eq(season_type)]
    stints = pd.read_parquet(paths[3])
    _require(stints, {"game_id", "stint_id", "start_action_id", "end_action_id_exclusive", *LINEUP_COLUMNS}, "historical lineup stints")
    stints["game_id"] = stints["game_id"].astype(str)
    if "season_type" in stints:
        stints = stints.loc[stints["season_type"].eq(season_type)].copy()
    candidate_games = set(candidates["game_id"])
    possession_passed = set(possession_quality.loc[possession_quality["passed"].astype(bool), "game_id"])
    lineup_passed = set(lineup_quality.loc[lineup_quality["passed"].astype(bool), "game_id"])
    eligible = candidate_games & possession_passed & lineup_passed & set(stints["game_id"])
    raw = pd.read_parquet(paths[0])
    owned = _candidate_action_frame(raw, candidates.loc[candidates["game_id"].isin(eligible)])
    output_possessions: list[pd.DataFrame] = []
    output_segments: list[pd.DataFrame] = []
    output_actions: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    for game_id in sorted(eligible):
        game_candidates = candidates.loc[candidates["game_id"].eq(game_id)].copy()
        game_owned = owned.loc[owned["game_id"].eq(game_id)].copy()
        reasons: list[str] = []
        if game_owned.empty:
            reasons.append("missing_owned_actions")
        else:
            reconciliation = _candidate_reconciliation(game_owned, game_candidates)
            if reconciliation:
                reasons.append(reconciliation)
        if not reasons:
            attached, lineup_reason = _attach_lineups(game_owned, stints.loc[stints["game_id"].eq(game_id)])
            if lineup_reason:
                reasons.append(lineup_reason)
        else:
            attached = game_owned.iloc[0:0].copy()
        if not reasons:
            segments = _segments(attached)
            output = game_candidates.copy()
            output["lineup_ready"] = True
            output["lineup_segment_count"] = output["possession_id"].map(segments.groupby("possession_id")["segment_number"].max()).astype(int)
            output["lineup_source"] = "historical_v3_exact_action_interval"
            validation = _validate_outputs(output, segments, attached)
            if validation:
                reasons.append(validation)
        if reasons:
            quality_rows.append({"game_id": game_id, "passed": False, "failure_reasons": "|".join(sorted(set(reasons))), "owned_action_count": int(len(game_owned)), "mapped_action_count": 0, "possession_count": int(len(game_candidates)), "segment_count": 0})
            continue
        output_possessions.append(output)
        output_segments.append(segments)
        output_actions.append(attached)
        quality_rows.append({"game_id": game_id, "passed": True, "failure_reasons": "", "owned_action_count": int(len(game_owned)), "mapped_action_count": int(len(attached)), "possession_count": int(len(output)), "segment_count": int(len(segments))})
    possessions = pd.concat(output_possessions, ignore_index=True) if output_possessions else candidates.iloc[0:0].copy()
    segments = pd.concat(output_segments, ignore_index=True) if output_segments else pd.DataFrame(columns=["possession_segment_id", "possession_id", "game_id", "possession_number", "segment_number", "period", "start_order_number", "end_order_number", "start_action_number", "end_action_number", "start_seconds_elapsed", "end_seconds_elapsed", "ordinal_stint_id", "offense_team_id", "points", "action_count", *LINEUP_COLUMNS])
    assigned = pd.concat(output_actions, ignore_index=True) if output_actions else pd.DataFrame()
    quality = pd.DataFrame(quality_rows, columns=["game_id", "passed", "failure_reasons", "owned_action_count", "mapped_action_count", "possession_count", "segment_count"])
    destinations = {"possessions": Path(possessions_destination), "segments": Path(segments_destination), "assigned_actions": Path(assigned_actions_destination), "quality": Path(quality_destination), "report": Path(report_destination)}
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    possessions.to_parquet(destinations["possessions"], index=False)
    segments.to_parquet(destinations["segments"], index=False)
    assigned.to_parquet(destinations["assigned_actions"], index=False)
    quality.to_parquet(destinations["quality"], index=False)
    files = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths]
    passed_ids = quality.loc[quality["passed"], "game_id"].astype(str).tolist() if not quality.empty else []
    report = {
        "dataset": "historical_v3_possession_lineup_candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_status": "research_candidate_only",
        "project_season": project_season,
        "season_type": season_type,
        "scope": "Separate exact V3 actionId lineup attachment for games that pass both pre-existing historical possession and lineup QA.",
        "action_mapping_contract": "The frozen V3 owner reconstruction must exactly reproduce each persisted possession candidate. Every reconstructed owned action maps once to a contiguous, exact V3 actionId lineup interval.",
        "segment_contract": "Segment points and action counts must sum to the persisted possession outcome. Each segment has five unique players per side and ten unique players overall.",
        "candidate_game_count": len(candidate_games),
        "possession_quality_passed_game_count": len(candidate_games & possession_passed),
        "lineup_quality_passed_game_count": len(candidate_games & lineup_passed),
        "double_passed_input_game_count": len(eligible),
        "emitted_game_count": len(passed_ids),
        "rejected_after_attachment_count": len(eligible) - len(passed_ids),
        "emitted_possession_count": int(len(possessions)),
        "emitted_segment_count": int(len(segments)),
        "emitted_owned_action_count": int(len(assigned)),
        "passed_game_ids": passed_ids,
        "failure_reason_counts": quality.loc[~quality["passed"], "failure_reasons"].value_counts().to_dict() if not quality.empty else {},
        "source_files": files,
        "outputs": {key: str(path.resolve()) for key, path in destinations.items() if key != "report"},
        "forbidden_interpretation": "The output inherits a frozen V3 possession-owner model. It is a separate historical RAPM candidate, not canonical CDN data or proof of ground-truth possession boundaries.",
    }
    report["snapshot_id"] = "historical_v3_possession_lineups_" + hashlib.sha256(json.dumps([(item["path"], item["sha256"]) for item in files], sort_keys=True).encode("utf-8")).hexdigest()[:16]
    report["passed"] = bool(passed_ids) and not bool((~quality["passed"]).any())
    write_json_atomic(report, destinations["report"])
    write_json_atomic(report, Path(manifest_dir) / f"{report['snapshot_id']}.json")
    return report
