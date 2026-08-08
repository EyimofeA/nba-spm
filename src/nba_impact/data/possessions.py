"""Canonical CDN possessions and ordinal lineup segments.

CDN ``possession`` is the offensive team for an action, while ``orderNumber``
is the physical event order.  ``actionNumber`` is used only as a guarded join
key to the independently validated V3 score states; it is never used to order
events.
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
from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic


HOME_LINEUP_COLUMNS = tuple(f"home_player_{index}" for index in range(1, 6))
AWAY_LINEUP_COLUMNS = tuple(f"away_player_{index}" for index in range(1, 6))
LINEUP_COLUMNS = HOME_LINEUP_COLUMNS + AWAY_LINEUP_COLUMNS


def _load_cdn_actions(root: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((root / "cdnnba").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No CDN NBA event partitions found under {root / 'cdnnba'}")
    columns = [
        "gameId", "orderNumber", "actionNumber", "period", "clock", "possession",
        "scoreHome", "scoreAway", "actionType", "description", "personId", "teamId",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_parquet(path, columns=columns)
        frame["game_id"] = frame["gameId"].map(canonical_game_id)
        frames.append(frame.drop(columns="gameId"))
    actions = pd.concat(frames, ignore_index=True)
    duplicate_order = actions.duplicated(["game_id", "orderNumber"], keep=False)
    duplicate_action = actions.duplicated(["game_id", "actionNumber"], keep=False)
    if duplicate_order.any() or duplicate_action.any():
        raise ValueError(
            "CDN event keys are not unique: "
            f"order={int(duplicate_order.sum())}, action={int(duplicate_action.sum())}."
        )
    return actions.sort_values(["game_id", "orderNumber"], kind="stable"), paths


def _add_elapsed_seconds(actions: pd.DataFrame) -> pd.DataFrame:
    output = actions.copy()
    remaining = parse_clock_seconds(output["clock"])
    period = pd.to_numeric(output["period"], errors="raise")
    regulation = period <= 4
    output["seconds_elapsed_game"] = np.where(
        regulation,
        (period - 1) * 720.0 + (720.0 - remaining),
        4 * 720.0 + (period - 5) * 300.0 + (300.0 - remaining),
    )
    return output


def reconcile_action_points(
    actions: pd.DataFrame, event_states: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Use ordered CDN scores, repairing only proven terminal gaps from V3."""
    ordered = actions.sort_values(["game_id", "orderNumber"], kind="stable").copy()
    for side in ("home", "away"):
        score_column = f"score{side.title()}"
        score = pd.to_numeric(ordered[score_column], errors="raise")
        previous = score.groupby(ordered["game_id"], sort=False).shift(fill_value=0)
        ordered[f"cdn_{side}_points"] = score - previous
    if ordered[["cdn_home_points", "cdn_away_points"]].lt(0).any(axis=None):
        raise ValueError("CDN contains a negative score correction; explicit handling is required.")

    v3 = event_states.groupby(["game_id", "actionNumber"], as_index=False).agg(
        v3_period=("period", "first"),
        v3_clock=("clock", "first"),
        v3_home_points=("home_points_added", "sum"),
        v3_away_points=("away_points_added", "sum"),
    )
    ordered = ordered.merge(v3, on=["game_id", "actionNumber"], how="left", validate="one_to_one")
    aligned = ordered["v3_period"].eq(ordered["period"]) & ordered["v3_clock"].eq(ordered["clock"])
    corrected = pd.Series(False, index=ordered.index)
    unresolved_games: set[str] = set()
    terminal = ordered.groupby("game_id", sort=False).tail(1).set_index("game_id")
    for side in ("home", "away"):
        cdn_points = ordered[f"cdn_{side}_points"]
        v3_points = ordered[f"v3_{side}_points"]
        candidate_delta = (v3_points - cdn_points).where(aligned, 0.0)
        positive_candidates = candidate_delta.where(candidate_delta.gt(0), 0.0)
        candidate_total = positive_candidates.groupby(ordered["game_id"], sort=False).sum()
        terminal_gap = (
            pd.to_numeric(terminal[side + "_score"], errors="raise")
            - pd.to_numeric(terminal["score" + side.title()], errors="raise")
        )
        approved_games = {
            str(game_id)
            for game_id, gap in terminal_gap.items()
            if gap > 0 and candidate_total.get(game_id, 0.0) == gap
        }
        unresolved_games.update(
            str(game_id)
            for game_id, gap in terminal_gap.items()
            if gap != 0 and str(game_id) not in approved_games
        )
        repair = aligned & positive_candidates.gt(0) & ordered["game_id"].isin(approved_games)
        ordered[f"{side}_points_added"] = cdn_points.where(~repair, v3_points)
        corrected |= repair
    ordered["points_added"] = ordered["home_points_added"] + ordered["away_points_added"]
    stats: dict[str, int | float] = {
        "cdn_rows": int(len(ordered)),
        "v3_aligned_rows": int(aligned.sum()),
        "v3_alignment_rate": float(aligned.mean()),
        "score_rows_corrected_by_v3": int(corrected.sum()),
        "unresolved_terminal_score_gap_games": int(len(unresolved_games)),
    }
    return ordered, stats


def build_ordinal_lineup_stints(
    actions: pd.DataFrame, player_games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay exact CDN substitutions and retain event-order boundaries."""
    starters: dict[tuple[str, int], set[int]] = {}
    for (game_id, team_id), group in player_games.loc[player_games["starter"]].groupby(
        ["game_id", "team_id"]
    ):
        starters[(str(game_id), int(team_id))] = set(group["player_id"].astype(int))

    output: list[dict] = []
    quality: list[dict] = []
    for game_id, game in actions.groupby("game_id", sort=False):
        game = game.sort_values("orderNumber", kind="stable")
        home_id = int(game["home_team_id"].iloc[0])
        away_id = int(game["away_team_id"].iloc[0])
        lineups = {
            home_id: set(starters.get((str(game_id), home_id), set())),
            away_id: set(starters.get((str(game_id), away_id), set())),
        }
        pending_out: dict[int, set[int]] = defaultdict(set)
        pending_in: dict[int, set[int]] = defaultdict(set)
        start_order = int(game["orderNumber"].min())
        max_order = int(game["orderNumber"].max())
        local: list[dict] = []
        errors = 0

        def append_stint(end_order: int) -> None:
            if end_order <= start_order:
                return
            home = sorted(lineups[home_id])
            away = sorted(lineups[away_id])
            if len(home) != 5 or len(away) != 5 or set(home).intersection(away):
                return
            local.append(
                {
                    "game_id": str(game_id),
                    "start_order_number": start_order,
                    "end_order_number": end_order,
                    **{f"home_player_{index + 1}": player for index, player in enumerate(home)},
                    **{f"away_player_{index + 1}": player for index, player in enumerate(away)},
                }
            )

        substitutions = game.loc[game["actionType"].astype(str).str.casefold().eq("substitution")]
        for row in substitutions.itertuples(index=False):
            team_id = int(row.teamId) if pd.notna(row.teamId) else 0
            player_id = int(row.personId) if pd.notna(row.personId) else 0
            description = str(row.description).casefold()
            if team_id not in lineups or player_id <= 0:
                errors += 1
                continue
            if description.startswith("sub out:"):
                pending_out[team_id].add(player_id)
            elif description.startswith("sub in:"):
                pending_in[team_id].add(player_id)
            else:
                errors += 1
                continue
            if pending_out[team_id] and len(pending_out[team_id]) == len(pending_in[team_id]):
                outs = pending_out[team_id]
                ins = pending_in[team_id]
                if not outs.issubset(lineups[team_id]) or ins.intersection(lineups[team_id] - outs):
                    errors += 1
                    pending_out[team_id] = set()
                    pending_in[team_id] = set()
                    continue
                append_stint(int(row.orderNumber))
                lineups[team_id] = (lineups[team_id] - outs) | ins
                start_order = int(row.orderNumber)
                pending_out[team_id] = set()
                pending_in[team_id] = set()
        if any(pending_out.values()) or any(pending_in.values()):
            errors += 1
        append_stint(max_order + 1)
        starter_valid = len(starters.get((str(game_id), home_id), set())) == 5 and len(
            starters.get((str(game_id), away_id), set())
        ) == 5
        passed = starter_valid and errors == 0 and bool(local)
        quality.append(
            {
                "game_id": str(game_id),
                "starter_valid": starter_valid,
                "ordinal_transition_errors": errors,
                "ordinal_stint_count": len(local),
                "passed": passed,
            }
        )
        if passed:
            for number, stint in enumerate(local, start=1):
                output.append(
                    {
                        "ordinal_stint_id": f"{game_id}_o{number:03d}",
                        "ordinal_stint_number": number,
                        **stint,
                    }
                )
    return pd.DataFrame(output), pd.DataFrame(quality)


def attach_ordinal_lineups(actions: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Attach the exact ordinal lineup state to every owned action."""
    outputs: list[pd.DataFrame] = []
    stint_groups = {str(game_id): group for game_id, group in stints.groupby("game_id", sort=False)}
    for game_id, group in actions.groupby("game_id", sort=False):
        game_stints = stint_groups.get(str(game_id))
        if game_stints is None or game_stints.empty:
            raise ValueError(f"No ordinal lineup stints for game {game_id}.")
        game_stints = game_stints.sort_values("start_order_number", kind="stable")
        starts = game_stints["start_order_number"].to_numpy(dtype=np.int64)
        ends = game_stints["end_order_number"].to_numpy(dtype=np.int64)
        orders = group["orderNumber"].to_numpy(dtype=np.int64)
        indices = np.searchsorted(starts, orders, side="right") - 1
        if (indices < 0).any() or (orders >= ends[np.maximum(indices, 0)]).any():
            raise ValueError(f"Action falls outside an ordinal lineup stint for game {game_id}.")
        local = group.copy()
        for column in LINEUP_COLUMNS:
            local[column] = game_stints[column].to_numpy()[indices]
        local["ordinal_stint_id"] = game_stints["ordinal_stint_id"].to_numpy()[indices]
        outputs.append(local)
    return pd.concat(outputs, ignore_index=True)


def collapse_cdn_possessions(actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create possession outcomes plus lossless ordinal lineup segments."""
    owned = actions.copy()
    valid = owned["possession"].eq(owned["home_team_id"]) | owned["possession"].eq(
        owned["away_team_id"]
    )
    local_scoring_owner = ~valid & owned["home_points_added"].gt(0)
    owned.loc[local_scoring_owner, "possession"] = owned.loc[local_scoring_owner, "home_team_id"]
    local_scoring_owner = ~valid & owned["away_points_added"].gt(0)
    owned.loc[local_scoring_owner, "possession"] = owned.loc[local_scoring_owner, "away_team_id"]
    valid = owned["possession"].eq(owned["home_team_id"]) | owned["possession"].eq(
        owned["away_team_id"]
    )
    owned = owned.loc[valid].sort_values(["game_id", "orderNumber"], kind="stable").copy()
    new_possession = (
        owned["game_id"].ne(owned["game_id"].shift())
        | owned["period"].ne(owned["period"].shift())
        | owned["possession"].ne(owned["possession"].shift())
    )
    owned["possession_number"] = new_possession.groupby(owned["game_id"]).cumsum().astype(int)
    owned["possession_id"] = (
        owned["game_id"] + ":" + owned["possession_number"].astype(str).str.zfill(3)
    )
    new_segment = new_possession | owned["ordinal_stint_id"].ne(owned["ordinal_stint_id"].shift())
    owned["segment_number"] = new_segment.groupby(owned["possession_id"]).cumsum().astype(int)

    segment_aggregations: dict[str, tuple[str, str]] = {
        "period": ("period", "first"),
        "start_order_number": ("orderNumber", "first"),
        "end_order_number": ("orderNumber", "last"),
        "start_action_number": ("actionNumber", "first"),
        "end_action_number": ("actionNumber", "last"),
        "start_seconds_elapsed": ("seconds_elapsed_game", "first"),
        "end_seconds_elapsed": ("seconds_elapsed_game", "last"),
        "ordinal_stint_id": ("ordinal_stint_id", "first"),
        "offense_team_id": ("possession", "first"),
        "points": ("points_added", "sum"),
        "action_count": ("orderNumber", "size"),
    }
    for column in LINEUP_COLUMNS:
        segment_aggregations[column] = (column, "first")
    segments = owned.groupby(
        ["possession_id", "game_id", "possession_number", "segment_number"],
        as_index=False,
        sort=False,
    ).agg(**segment_aggregations)
    segments["possession_segment_id"] = (
        segments["possession_id"] + ":s" + segments["segment_number"].astype(str).str.zfill(2)
    )

    possessions = owned.groupby(["possession_id", "game_id", "possession_number"], as_index=False, sort=False).agg(
        season_start=("season_start", "first"), season_end=("season_end", "first"),
        season_label=("season_label", "first"), season_type=("season_type", "first"),
        game_date=("game_date", "first"), period=("period", "first"),
        start_order_number=("orderNumber", "first"), end_order_number=("orderNumber", "last"),
        start_action_number=("actionNumber", "first"), end_action_number=("actionNumber", "last"),
        start_seconds_elapsed=("seconds_elapsed_game", "first"),
        end_seconds_elapsed=("seconds_elapsed_game", "last"),
        offense_team_id=("possession", "first"), home_team_id=("home_team_id", "first"),
        away_team_id=("away_team_id", "first"), points=("points_added", "sum"),
        home_points=("home_points_added", "sum"), away_points=("away_points_added", "sum"),
        action_count=("orderNumber", "size"), lineup_segment_count=("segment_number", "max"),
    )
    possessions["offense_team_id"] = possessions["offense_team_id"].astype("int64")
    possessions["defense_team_id"] = np.where(
        possessions["offense_team_id"].eq(possessions["home_team_id"]),
        possessions["away_team_id"], possessions["home_team_id"],
    ).astype("int64")
    possessions["offense_is_home"] = possessions["offense_team_id"].eq(possessions["home_team_id"])
    possessions["lineup_ready"] = True
    return possessions, segments


def build_possessions(
    event_root: str | Path,
    event_states_path: str | Path,
    game_dim_path: str | Path,
    player_games_path: str | Path,
    lineup_quality_path: str | Path,
    destination: str | Path,
    segments_destination: str | Path,
    manifest_dir: str | Path,
) -> dict:
    """Build validated possession outcomes and ordinal lineup segments."""
    root = Path(event_root)
    source_paths = [
        Path(event_states_path), Path(game_dim_path), Path(player_games_path), Path(lineup_quality_path)
    ]
    actions, cdn_paths = _load_cdn_actions(root)
    games = pd.read_parquet(game_dim_path)
    event_states = pd.read_parquet(
        event_states_path,
        columns=["game_id", "actionNumber", "period", "clock", "home_points_added", "away_points_added"],
    )
    players = pd.read_parquet(player_games_path)
    lineup_quality = pd.read_parquet(lineup_quality_path)
    cdn_games = set(actions["game_id"])
    passed_lineup_games = set(lineup_quality.loc[lineup_quality["passed"], "game_id"].astype(str))
    candidate_games = set(games["game_id"].astype(str)) & cdn_games & passed_lineup_games
    dimension_columns = [
        "game_id", "season_start", "season_end", "season_label", "season_type", "game_date",
        "home_team_id", "away_team_id", "home_score", "away_score",
    ]
    actions = actions.loc[actions["game_id"].isin(candidate_games)].merge(
        games[dimension_columns], on="game_id", validate="many_to_one"
    )
    actions = _add_elapsed_seconds(actions)
    actions, point_stats = reconcile_action_points(actions, event_states)
    ordinal_stints, ordinal_quality = build_ordinal_lineup_stints(actions, players)
    ordinal_passed_games = set(ordinal_quality.loc[ordinal_quality["passed"], "game_id"])
    eligible_games = candidate_games & ordinal_passed_games
    actions = actions.loc[actions["game_id"].isin(eligible_games)].copy()
    actions = attach_ordinal_lineups(
        actions, ordinal_stints.loc[ordinal_stints["game_id"].isin(eligible_games)]
    )
    possessions, segments = collapse_cdn_possessions(actions)

    score_check = possessions.groupby("game_id", as_index=False).agg(
        possession_count=("possession_id", "size"), home_points=("home_points", "sum"),
        away_points=("away_points", "sum"), home_possessions=("offense_is_home", "sum"),
    ).merge(games[["game_id", "home_score", "away_score"]], on="game_id", validate="one_to_one")
    score_check["away_possessions"] = score_check["possession_count"] - score_check["home_possessions"]
    side_imbalance = (score_check["home_possessions"] - score_check["away_possessions"]).abs()
    issues = {
        "ordinal_lineup_failed_games": int(len(candidate_games - ordinal_passed_games)),
        "score_mismatch_games": int(
            (score_check["home_points"].ne(score_check["home_score"]) |
             score_check["away_points"].ne(score_check["away_score"])).sum()
        ),
        "duplicate_possession_ids": int(possessions.duplicated("possession_id", keep=False).sum()),
        "duplicate_segment_ids": int(segments.duplicated("possession_segment_id", keep=False).sum()),
        "negative_point_rows": int(possessions["points"].lt(0).sum()),
        "implausible_point_rows": int(possessions["points"].gt(7).sum()),
        "games_outside_possession_bounds": int((~score_check["possession_count"].between(150, 300)).sum()),
        "games_with_side_imbalance_over_five": int(side_imbalance.gt(5).sum()),
        "segment_point_mismatch_possessions": int(
            possessions[["possession_id", "points"]].merge(
                segments.groupby("possession_id", as_index=False)["points"].sum(),
                on="possession_id", suffixes=("_possession", "_segments"), validate="one_to_one",
            ).eval("points_possession != points_segments").sum()
        ),
    }
    warnings = {
        "games_with_side_imbalance_four_or_five": int(side_imbalance.between(4, 5).sum()),
        "possessions_with_multiple_lineup_segments": int(possessions["lineup_segment_count"].gt(1).sum()),
    }
    passed = not any(issues.values()) and set(possessions["game_id"]) == eligible_games

    possession_output = Path(destination)
    segment_output = Path(segments_destination)
    possession_output.parent.mkdir(parents=True, exist_ok=True)
    segment_output.parent.mkdir(parents=True, exist_ok=True)
    possession_tmp = possession_output.with_suffix(possession_output.suffix + ".partial")
    segment_tmp = segment_output.with_suffix(segment_output.suffix + ".partial")
    possessions.to_parquet(possession_tmp, index=False)
    segments.to_parquet(segment_tmp, index=False)
    possession_tmp.replace(possession_output)
    segment_tmp.replace(segment_output)

    all_sources = cdn_paths + source_paths
    source_records = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in all_sources
    ]
    identity = hashlib.sha256(
        json.dumps([(item["path"], item["sha256"]) for item in source_records]).encode("utf-8")
    ).hexdigest()[:16]
    snapshot = {
        "snapshot_id": f"possessions_{identity}", "dataset": "possessions",
        "grain": "one maximal same-offense CDN order run; lineup changes retained in a segment child table",
        "created_at": datetime.now(timezone.utc).isoformat(), "passed": passed,
        "row_count": int(len(possessions)), "segment_row_count": int(len(segments)),
        "source_game_count": int(len(games)), "cdn_game_count": int(len(cdn_games)),
        "candidate_game_count": int(len(candidate_games)), "game_count": int(len(eligible_games)),
        "coverage_rate": float(len(eligible_games) / len(games)),
        "excluded_no_cdn_games": int(len(set(games["game_id"]) - cdn_games)),
        "excluded_lineup_quality_games": int(len(set(games["game_id"]) - passed_lineup_games)),
        "point_reconciliation": point_stats, "issues": issues, "warnings": warnings,
        "ordering_policy": "CDN orderNumber only; actionNumber is never an ordering field.",
        "lineup_policy": "Substitutions are replayed ordinally; every owned action maps to one ten-player segment.",
        "builder_code_sha256": sha256_file(Path(__file__)),
        "path": str(possession_output.resolve()), "segments_path": str(segment_output.resolve()),
        "source_files": source_records,
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
