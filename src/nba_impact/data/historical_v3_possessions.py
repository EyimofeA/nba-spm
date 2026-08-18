"""Validated V3 possession-owner inference for the historical research spine.

NBA Stats V3 supplies complete ordered actions for 2017--2026, but it does not
publish the CDN ``possession`` owner field. This module uses a frozen forward
state machine. Its rules are checked against the CDN owner field in later
seasons before historical candidates can be built. Outputs stay separate from
canonical CDN possessions and contain no lineup claims.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .event_state import parse_clock_seconds
from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic


OWNER_RULE_VERSION = "v3_forward_owner_v1"
NBA_TEAM_ID_MIN = 1610612737
NBA_TEAM_ID_MAX = 1610612766
_FINAL_FREE_THROW = re.compile(r"(\d+)\s+of\s+(\d+)", re.IGNORECASE)
_RETAINED_FREE_THROW = ("technical", "flagrant", "clear path", "away from play")
_OFFENSIVE_FOUL = {"offensive", "offensive charge"}
_CORE_OWNER_TYPES = {"made shot", "missed shot", "rebound", "turnover", "free throw", "foul", "substitution", "timeout", "jump ball"}


def _actor_team(row: dict) -> int:
    for key in ("teamId", "personId"):
        value = row.get(key)
        if value is None or pd.isna(value):
            continue
        numeric = int(value)
        if NBA_TEAM_ID_MIN <= numeric <= NBA_TEAM_ID_MAX:
            return numeric
    return 0


def _is_miss(row: dict) -> bool:
    return str(row.get("description") or "").strip().upper().startswith("MISS")


def _is_final_free_throw(sub_type: str) -> bool:
    match = _FINAL_FREE_THROW.search(sub_type)
    return bool(match and match.group(1) == match.group(2))


def _is_retained_free_throw(sub_type: str) -> bool:
    return any(value in sub_type for value in _RETAINED_FREE_THROW)


def _description_points(row: dict) -> int:
    """Return points from the action itself, not sparse cumulative score states."""
    action_type = str(row.get("actionType") or "").strip().casefold()
    description = str(row.get("description") or "")
    if action_type == "made shot":
        return 3 if "3PT" in description.upper() else 2
    if action_type == "free throw" and not _is_miss(row):
        return 1
    return 0


def _prepare_v3(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required = {
        "gameId", "actionId", "actionNumber", "period", "clock", "teamId", "personId",
        "actionType", "subType", "description",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"V3 possession source is missing columns: {missing}")
    work = frame.copy()
    credit_rows = int(work["actionType"].isna().sum())
    work = work.loc[work["actionType"].notna()].copy()
    work["game_id"] = work["gameId"].map(canonical_game_id)
    work["event_order"] = pd.to_numeric(work["actionId"], errors="raise").astype("int64")
    work["action_number"] = pd.to_numeric(work["actionNumber"], errors="coerce").astype("Int64")
    work["period"] = pd.to_numeric(work["period"], errors="raise").astype("int64")
    if work.duplicated(["game_id", "event_order"], keep=False).any():
        raise ValueError("V3 event order is not unique within game.")
    return work.sort_values(["game_id", "event_order"], kind="stable"), credit_rows


def infer_v3_possession_owners(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Infer one offensive owner for each primary V3 action in physical order."""
    work, credit_rows = _prepare_v3(frame)
    outputs: list[pd.DataFrame] = []
    invalid_team_games = 0
    for game_id, game in work.groupby("game_id", sort=False):
        rows = list(game.to_dict("records"))
        team_ids = sorted({_actor_team(row) for row in rows} - {0})
        if len(team_ids) != 2:
            invalid_team_games += 1
            local = game.copy()
            local["possession"] = 0
            local["owner_rule"] = "invalid_game_teams"
            local["owner_confidence"] = "rejected"
            local["points_added"] = 0
            local["scoring_team_id"] = 0
            outputs.append(local)
            continue
        opponent = {team_ids[0]: team_ids[1], team_ids[1]: team_ids[0]}
        owners: list[int] = []
        rules: list[str] = []
        confidences: list[str] = []
        points: list[int] = []
        scoring_teams: list[int] = []
        current: int | None = None

        def next_anchor(index: int, *, same_clock: bool = False) -> tuple[str, int] | None:
            base = rows[index]
            for later in rows[index + 1 :]:
                if int(later["period"]) != int(base["period"]):
                    break
                if same_clock and later["clock"] != base["clock"]:
                    break
                action_type = str(later.get("actionType") or "").strip().casefold()
                actor = _actor_team(later)
                if action_type in {"made shot", "missed shot", "turnover", "rebound", "free throw"} and actor:
                    return action_type, actor
                sub_type = str(later.get("subType") or "").strip().casefold()
                if action_type == "foul" and sub_type in _OFFENSIVE_FOUL and actor:
                    return action_type, actor
            return None

        for index, row in enumerate(rows):
            action_type = str(row.get("actionType") or "").strip().casefold()
            sub_type = str(row.get("subType") or "").strip().casefold()
            actor = _actor_team(row)
            owner = current
            next_current = current
            rule = "inherit_current"
            confidence = "medium"

            if action_type == "period" and sub_type == "start":
                current = None
                owner = 0
                next_current = None
                rule, confidence = "period_start", "not_owned"
            elif action_type == "period" and sub_type == "end":
                owner = current or 0
                next_current = None
                rule, confidence = "period_end", "low"
            elif action_type in {"made shot", "missed shot"}:
                owner = actor or current
                next_current = owner
                rule, confidence = ("shot_actor", "high") if actor else ("shot_inherit", "low")
                if action_type == "made shot":
                    anchor = next_anchor(index, same_clock=True)
                    and_one = anchor is not None and anchor == ("free throw", owner)
                    if not and_one and owner:
                        next_current = opponent[owner]
            elif action_type == "rebound":
                owner = actor or current
                next_current = owner
                rule, confidence = ("rebound_actor", "high") if actor else ("rebound_inherit", "low")
            elif action_type == "turnover":
                if actor:
                    owner = actor
                    rule, confidence = "turnover_actor", "high"
                elif current:
                    owner = current
                    rule, confidence = "turnover_inherit", "low"
                else:
                    anchor = next_anchor(index)
                    owner = opponent[anchor[1]] if anchor is not None else None
                    rule, confidence = "turnover_reverse_next_anchor", "low"
                next_current = opponent[owner] if owner else None
            elif action_type == "free throw":
                if "technical" in sub_type:
                    owner = current or actor
                    next_current = current or actor
                    rule, confidence = "technical_ft_inherit", "medium"
                else:
                    owner = actor
                    next_current = actor
                    rule, confidence = "free_throw_actor", "high"
                    if (
                        _is_final_free_throw(sub_type)
                        and not _is_miss(row)
                        and not _is_retained_free_throw(sub_type)
                    ):
                        next_current = opponent[actor]
            elif action_type == "foul":
                if sub_type in _OFFENSIVE_FOUL:
                    owner = actor
                    next_current = opponent[actor]
                    rule, confidence = "offensive_foul_actor", "high"
                else:
                    owner = current or (opponent.get(actor) if actor else None)
                    next_current = current or owner
                    rule, confidence = "defensive_foul_inherit", "medium"
            elif action_type == "jump ball":
                anchor = next_anchor(index)
                owner = anchor[1] if anchor is not None else current
                next_current = owner
                rule, confidence = "jump_ball_next_anchor", "medium"
            elif action_type == "violation":
                owner = current
                next_current = current
                rule, confidence = "violation_inherit", "low"
            elif current is None:
                anchor = next_anchor(index)
                owner = anchor[1] if anchor is not None else 0
                next_current = owner or None
                rule, confidence = "next_anchor", "low"

            action_points = _description_points(row)
            owners.append(int(owner or 0))
            rules.append(rule)
            confidences.append(confidence)
            points.append(action_points)
            scoring_teams.append(actor if action_points else 0)
            current = next_current
        local = game.copy()
        local["possession"] = owners
        local["owner_rule"] = rules
        local["owner_confidence"] = confidences
        local["points_added"] = points
        local["scoring_team_id"] = scoring_teams
        outputs.append(local)
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    return result, {
        "secondary_credit_rows_removed": credit_rows,
        "invalid_two_team_games": invalid_team_games,
    }


def _owner_runs(frame: pd.DataFrame, column: str) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for period, part in frame.sort_values("event_order", kind="stable").groupby("period", sort=True):
        last = None
        for raw in part[column]:
            owner = int(raw) if pd.notna(raw) else 0
            if owner and owner != last:
                runs.append((int(period), owner))
                last = owner
    return runs


def validate_v3_owners_against_cdn(v3: pd.DataFrame, cdn: pd.DataFrame) -> dict:
    """Measure the frozen state machine against independent CDN owner labels."""
    inferred, source_issues = infer_v3_possession_owners(v3)
    required = {"gameId", "actionNumber", "possession"}
    if missing := sorted(required - set(cdn.columns)):
        raise ValueError(f"CDN validation source is missing columns: {missing}")
    truth = cdn.copy()
    truth["game_id"] = truth["gameId"].map(canonical_game_id)
    truth["action_number"] = pd.to_numeric(truth["actionNumber"], errors="coerce").astype("Int64")
    truth = truth[["game_id", "action_number", "possession"]]
    if truth.duplicated(["game_id", "action_number"], keep=False).any():
        raise ValueError("CDN validation action numbers are not unique.")
    joined = inferred.merge(truth, on=["game_id", "action_number"], how="left", suffixes=("_inferred", "_cdn"), validate="one_to_one")
    mapped = joined["possession_cdn"].notna()
    owned = mapped & joined["possession_cdn"].ne(0)
    action_type = joined["actionType"].astype(str).str.strip().str.casefold()
    core = owned & action_type.isin(_CORE_OWNER_TYPES)
    all_match = joined.loc[owned, "possession_inferred"].eq(joined.loc[owned, "possession_cdn"].astype(int))
    core_match = joined.loc[core, "possession_inferred"].eq(joined.loc[core, "possession_cdn"].astype(int))
    run_rows: list[dict] = []
    for game_id, game in joined.loc[mapped].groupby("game_id", sort=False):
        inferred_runs = _owner_runs(game, "possession_inferred")
        truth_runs = _owner_runs(game, "possession_cdn")
        run_rows.append(
            {
                "game_id": game_id,
                "inferred_count": len(inferred_runs),
                "truth_count": len(truth_runs),
                "exact_sequence": inferred_runs == truth_runs,
            }
        )
    runs = pd.DataFrame(run_rows)
    runs["count_difference"] = runs["inferred_count"] - runs["truth_count"]
    metrics = {
        **source_issues,
        "v3_action_rows": int(len(inferred)),
        "mapped_action_rows": int(mapped.sum()),
        "mapping_rate": float(mapped.mean()),
        "owned_action_rows": int(owned.sum()),
        "all_owned_action_agreement": float(all_match.mean()),
        "core_action_agreement": float(core_match.mean()),
        "game_count": int(len(runs)),
        "exact_owner_sequence_game_rate": float(runs["exact_sequence"].mean()),
        "possession_count_within_two_game_rate": float(runs["count_difference"].abs().le(2).mean()),
        "mean_possession_count_difference": float(runs["count_difference"].mean()),
        "mean_absolute_possession_count_difference": float(runs["count_difference"].abs().mean()),
    }
    metrics["passed"] = bool(
        metrics["mapping_rate"] >= 0.99
        and metrics["core_action_agreement"] >= 0.998
        and metrics["exact_owner_sequence_game_rate"] >= 0.90
        and metrics["possession_count_within_two_game_rate"] >= 0.99
        and abs(metrics["mean_possession_count_difference"]) <= 0.25
    )
    return metrics


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


def _collapse_candidate_actions(actions: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = actions["possession"].eq(actions["home_team_id"]) | actions["possession"].eq(actions["away_team_id"])
    owned = actions.loc[valid].sort_values(["game_id", "event_order"], kind="stable").copy()
    new_possession = (
        owned["game_id"].ne(owned["game_id"].shift())
        | owned["period"].ne(owned["period"].shift())
        | owned["possession"].ne(owned["possession"].shift())
    )
    owned["possession_number"] = new_possession.groupby(owned["game_id"]).cumsum().astype(int)
    owned["possession_id"] = owned["game_id"] + ":v3:" + owned["possession_number"].astype(str).str.zfill(3)
    owned["home_points_added"] = np.where(owned["scoring_team_id"].eq(owned["home_team_id"]), owned["points_added"], 0)
    owned["away_points_added"] = np.where(owned["scoring_team_id"].eq(owned["away_team_id"]), owned["points_added"], 0)
    owned["seconds_elapsed_game"] = _elapsed_seconds(owned["period"], owned["clock"])
    possessions = owned.groupby(["possession_id", "game_id", "possession_number"], as_index=False, sort=False).agg(
        season_start=("season_start", "first"), season_end=("season_end", "first"),
        season_label=("season_label", "first"), season_type=("season_type", "first"),
        game_date=("game_date", "first"), period=("period", "first"),
        start_order_number=("event_order", "first"), end_order_number=("event_order", "last"),
        start_action_number=("action_number", "first"), end_action_number=("action_number", "last"),
        start_seconds_elapsed=("seconds_elapsed_game", "first"), end_seconds_elapsed=("seconds_elapsed_game", "last"),
        offense_team_id=("possession", "first"), home_team_id=("home_team_id", "first"),
        away_team_id=("away_team_id", "first"), points=("points_added", "sum"),
        home_points=("home_points_added", "sum"), away_points=("away_points_added", "sum"),
        action_count=("event_order", "size"), low_confidence_actions=("owner_confidence", lambda values: int(pd.Series(values).isin(["low"]).sum())),
    )
    possessions["defense_team_id"] = np.where(
        possessions["offense_team_id"].eq(possessions["home_team_id"]),
        possessions["away_team_id"], possessions["home_team_id"],
    ).astype("int64")
    possessions["offense_is_home"] = possessions["offense_team_id"].eq(possessions["home_team_id"])
    possessions["lineup_ready"] = False
    possessions["owner_source"] = OWNER_RULE_VERSION
    score = possessions.groupby("game_id", as_index=False).agg(
        possession_count=("possession_id", "size"), home_points=("home_points", "sum"),
        away_points=("away_points", "sum"), home_possessions=("offense_is_home", "sum"),
    ).merge(games[["game_id", "home_score", "away_score"]], on="game_id", validate="one_to_one")
    score["away_possessions"] = score["possession_count"] - score["home_possessions"]
    score["score_conserved"] = score["home_points"].eq(score["home_score"]) & score["away_points"].eq(score["away_score"])
    score["plausible_count"] = score["possession_count"].between(150, 300)
    score["balanced_sides"] = (score["home_possessions"] - score["away_possessions"]).abs().le(5)
    score["passed"] = score[["score_conserved", "plausible_count", "balanced_sides"]].all(axis=1)
    return possessions, score


def build_historical_v3_possession_candidates(
    v3_root: str | Path,
    official_scores_path: str | Path,
    output_root: str | Path,
    quality_destination: str | Path,
    manifest_dir: str | Path,
    *,
    seasons: tuple[int, ...] = tuple(range(2017, 2024)),
    season_types: tuple[str, ...] = ("regular", "playoffs"),
) -> dict:
    """Build separate score-conserved V3 possession candidates by partition."""
    if not seasons:
        raise ValueError("At least one season is required.")
    official_path = Path(official_scores_path)
    official = pd.read_parquet(official_path)
    official["game_id"] = official["game_id"].map(canonical_game_id)
    official["game_date"] = pd.to_datetime(official["game_date"], errors="raise")
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_frames: list[pd.DataFrame] = []
    source_files: list[dict] = [{"path": str(official_path.resolve()), "bytes": official_path.stat().st_size, "sha256": sha256_file(official_path)}]
    output_files: list[dict] = []
    partition_metrics: list[dict] = []
    for season in seasons:
        for season_type in season_types:
            source = Path(v3_root) / f"project_season={season}" / f"{season_type}.parquet"
            if not source.exists():
                partition_metrics.append({"project_season": season, "season_type": season_type, "passed": False, "issue": "missing_v3_partition"})
                continue
            raw = pd.read_parquet(source)
            inferred, inference_issues = infer_v3_possession_owners(raw)
            games = official.loc[
                official["project_season"].eq(season) & official["season_type"].eq(season_type)
            ].copy()
            games["season_start"] = season - 1
            games["season_end"] = season
            games["season_label"] = f"{season - 1}-{str(season)[-2:]}"
            game_ids = set(games["game_id"])
            inferred = inferred.loc[inferred["game_id"].isin(game_ids)].merge(
                games[["game_id", "season_start", "season_end", "season_label", "season_type", "game_date", "home_team_id", "away_team_id"]],
                on="game_id", validate="many_to_one",
            )
            possessions, quality = _collapse_candidate_actions(inferred, games)
            passed_games = set(quality.loc[quality["passed"], "game_id"])
            accepted = possessions.loc[possessions["game_id"].isin(passed_games)].copy()
            destination = output_dir / f"project_season={season}" / f"{season_type}.parquet"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".parquet.partial")
            accepted.to_parquet(temporary, index=False)
            temporary.replace(destination)
            quality["project_season"] = season
            quality["season_type"] = season_type
            quality_frames.append(quality)
            source_record = {"path": str(source.resolve()), "bytes": source.stat().st_size, "sha256": sha256_file(source)}
            output_record = {"path": str(destination.resolve()), "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}
            source_files.append(source_record)
            output_files.append(output_record)
            partition_metrics.append(
                {
                    "project_season": season, "season_type": season_type,
                    "source_games": int(inferred["game_id"].nunique()),
                    "passed_games": int(len(passed_games)), "rejected_games": int(len(quality) - len(passed_games)),
                    "possession_rows": int(len(accepted)), "passed": bool(quality["passed"].all()),
                    **inference_issues,
                }
            )
    quality_rows = pd.concat(quality_frames, ignore_index=True) if quality_frames else pd.DataFrame()
    quality_output = Path(quality_destination)
    quality_output.parent.mkdir(parents=True, exist_ok=True)
    quality_rows.to_parquet(quality_output, index=False)
    identity = hashlib.sha256(
        json.dumps([(item["path"], item["sha256"]) for item in source_files]).encode("utf-8")
    ).hexdigest()[:16]
    passed = bool(partition_metrics) and all(item.get("passed", False) for item in partition_metrics)
    snapshot = {
        "snapshot_id": f"historical_v3_possessions_{identity}",
        "dataset": "historical_v3_possession_candidates",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "scope": "separate V3 state-machine candidate; no lineup claims and not canonical CDN data",
        "owner_rule_version": OWNER_RULE_VERSION,
        "partition_metrics": partition_metrics,
        "game_count": int(quality_rows.loc[quality_rows["passed"], "game_id"].nunique()) if not quality_rows.empty else 0,
        "rejected_game_count": int((~quality_rows["passed"]).sum()) if not quality_rows.empty else 0,
        "possession_row_count": int(sum(item.get("possession_rows", 0) for item in partition_metrics)),
        "quality_path": str(quality_output.resolve()),
        "source_files": source_files,
        "output_files": output_files,
        "forbidden_interpretation": "This candidate does not prove exact historical possession boundaries and is not RAPM-ready until ordinal lineups pass.",
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
