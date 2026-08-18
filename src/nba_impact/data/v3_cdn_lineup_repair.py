"""Strict research candidate for repairing CDN-present lineup games.

The CDN feed remains the possession and event-order authority.  NBA Stats V3
contributes only a substitution pair.  Each V3 substitution must resolve to a
single CDN *incoming* substitution action using game, period, clock, team, and
player identity.  A clock match by itself is never sufficient.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .game_dim import canonical_game_id
from .lineups import _elapsed_seconds, _normalize_name, _player_aliases
from .manifest import sha256_file, write_json_atomic
from .possessions import attach_ordinal_lineups


REPAIR_GAME_IDS = (
    "0022301210",
    "0022300339",
    "0022400061",
    "0022500264",
)
_V3_SUB = re.compile(r"^SUB:\s*(.+?)\s+FOR\s+(.+?)$", re.IGNORECASE)
_LINEUP_COLUMNS = tuple(
    [f"home_player_{index}" for index in range(1, 6)]
    + [f"away_player_{index}" for index in range(1, 6)]
)


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def parse_v3_substitutions(v3: pd.DataFrame, player_games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return V3 pairs with IDs, plus failures.  `personId` is the outgoing ID."""
    _require_columns(
        v3,
        {
            "game_id", "actionId", "actionNumber", "period", "clock", "actionType", "description",
            "personId", "teamId",
        },
        "V3 events",
    )
    source = v3.loc[v3["actionType"].astype(str).str.casefold().eq("substitution")].copy()
    aliases = _player_aliases(player_games, set(source["game_id"].astype(str)))
    rows: list[dict] = []
    failures: list[dict] = []
    for row in source.itertuples(index=False):
        match = _V3_SUB.fullmatch(str(row.description).strip())
        if not match:
            failures.append({"game_id": str(row.game_id), "action_id": int(row.actionId), "reason": "parse"})
            continue
        incoming_name, _ = match.groups()
        key = (str(row.game_id), int(row.teamId), _normalize_name(incoming_name))
        incoming_ids = aliases.get(key, set())
        if len(incoming_ids) != 1:
            failures.append(
                {
                    "game_id": str(row.game_id),
                    "action_id": int(row.actionId),
                    "reason": f"incoming_alias_matches_{len(incoming_ids)}",
                }
            )
            continue
        rows.append(
            {
                "game_id": str(row.game_id),
                "v3_action_id": int(row.actionId),
                "v3_action_number": int(row.actionNumber) if pd.notna(row.actionNumber) else None,
                "period": int(row.period),
                "clock": str(row.clock),
                "team_id": int(row.teamId),
                "out_player_id": int(row.personId),
                "in_player_id": next(iter(incoming_ids)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(failures)


def infer_v3_period_starts(v3: pd.DataFrame, v3_pairs: pd.DataFrame) -> tuple[dict[tuple[str, int, int], set[int]], pd.DataFrame]:
    """Infer five players at a period start from the V3 game record.

    This is only used to reset a valid prior-period state at an explicit CDN
    period-start ordinal.  It is not a time-based substitution join.
    """
    valid = v3.loc[
        v3["teamId"].notna() & v3["personId"].notna() & v3["teamId"].gt(0) & v3["personId"].gt(0)
    ].copy()
    output: dict[tuple[str, int, int], set[int]] = {}
    failures: list[dict] = []
    for (game_id, team_id, period), group in valid.groupby(["game_id", "teamId", "period"], sort=False):
        if int(period) == 1:
            continue
        substitutions = v3_pairs.loc[
            v3_pairs["game_id"].eq(str(game_id))
            & v3_pairs["team_id"].eq(int(team_id))
            & v3_pairs["period"].eq(int(period))
        ].sort_values("v3_action_id", kind="stable")
        first_direction: dict[int, str] = {}
        for row in substitutions.itertuples(index=False):
            first_direction.setdefault(int(row.out_player_id), "out")
            first_direction.setdefault(int(row.in_player_id), "in")
        first_out = {player_id for player_id, direction in first_direction.items() if direction == "out"}
        all_sub_players = set(first_direction)
        active_non_sub = set(
            group.loc[
                ~group["actionType"].astype(str).str.casefold().eq("substitution"), "personId"
            ].astype(int)
        ) - all_sub_players
        # A player first substituted out must have started the period.  A player
        # who appears before any substitution and is never substituted is also a
        # starter.  Restrict the latter to first-period activity before the first
        # mapped substitution to avoid collecting a later entrant.
        candidates = first_out | active_non_sub
        if len(candidates) == 5:
            output[(str(game_id), int(team_id), int(period))] = candidates
        else:
            failures.append(
                {
                    "game_id": str(game_id),
                    "team_id": int(team_id),
                    "period": int(period),
                    "reason": f"period_starter_candidates_{len(candidates)}",
                }
            )
    return output, pd.DataFrame(failures)


def align_v3_substitutions_to_cdn(v3_pairs: pd.DataFrame, cdn: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map each V3 actionId to one CDN ordinal using a full identity key.

    The V3 event names the outgoing player.  The canonical CDN row that marks
    the same change is its incoming player row.  The join key includes the game,
    period, clock, team, incoming player, and the fact that the CDN row is a
    substitution-in row.  This deliberately rejects a clock-only match.
    """
    _require_columns(
        cdn,
        {"game_id", "orderNumber", "period", "clock", "actionType", "subType", "personId", "teamId"},
        "CDN events",
    )
    if v3_pairs.empty:
        return pd.DataFrame(), pd.DataFrame(columns=["game_id", "v3_action_id", "reason"])
    incoming = cdn.loc[
        cdn["actionType"].astype(str).str.casefold().eq("substitution")
        & cdn["subType"].astype(str).str.casefold().eq("in")
    ].copy()
    incoming["team_id"] = pd.to_numeric(incoming["teamId"], errors="coerce").astype("Int64")
    incoming["in_player_id"] = pd.to_numeric(incoming["personId"], errors="coerce").astype("Int64")
    incoming = incoming.dropna(subset=["team_id", "in_player_id"])
    incoming["team_id"] = incoming["team_id"].astype(int)
    incoming["in_player_id"] = incoming["in_player_id"].astype(int)
    key = ["game_id", "period", "clock", "team_id", "in_player_id"]
    candidate_counts = incoming.groupby(key, dropna=False)["orderNumber"].size().rename("candidate_count")
    candidates = v3_pairs.merge(candidate_counts, left_on=key, right_index=True, how="left")
    candidates["candidate_count"] = candidates["candidate_count"].fillna(0).astype(int)
    failures = candidates.loc[candidates["candidate_count"].ne(1), ["game_id", "v3_action_id", "candidate_count"]].copy()
    failures["reason"] = failures["candidate_count"].map(lambda count: f"cdn_incoming_candidates_{count}")
    aligned = candidates.loc[candidates["candidate_count"].eq(1)].merge(
        incoming[key + ["orderNumber", "actionNumber"]], on=key, how="left", validate="one_to_one"
    )
    aligned = aligned.rename(columns={"orderNumber": "cdn_order_number", "actionNumber": "cdn_action_number"})
    aligned["alignment_key"] = "game_id|period|clock|team_id|in_player_id|substitution_in"
    duplicate_order = aligned.duplicated(["game_id", "cdn_order_number"], keep=False)
    if duplicate_order.any():
        duplicate = aligned.loc[duplicate_order, ["game_id", "v3_action_id"]].copy()
        duplicate["reason"] = "duplicate_cdn_order_number"
        failures = pd.concat([failures[["game_id", "v3_action_id", "reason"]], duplicate], ignore_index=True)
        aligned = aligned.loc[~duplicate_order].copy()
    else:
        failures = failures[["game_id", "v3_action_id", "reason"]]
    monotonic_failures: list[dict] = []
    for game_id, group in aligned.groupby("game_id", sort=False):
        ordered = group.sort_values("v3_action_id", kind="stable")
        if not ordered["cdn_order_number"].is_monotonic_increasing:
            monotonic_failures.append({"game_id": str(game_id), "v3_action_id": -1, "reason": "nonmonotonic_ordinal_alignment"})
    if monotonic_failures:
        failures = pd.concat([failures, pd.DataFrame(monotonic_failures)], ignore_index=True)
    return aligned.sort_values(["game_id", "cdn_order_number"], kind="stable"), failures


def replay_aligned_lineups(
    actions: pd.DataFrame,
    player_games: pd.DataFrame,
    aligned: pd.DataFrame,
    period_starts: dict[tuple[str, int, int], set[int]],
    period_start_failures: pd.DataFrame,
    game_dim: pd.DataFrame,
    game_ids: tuple[str, ...] = REPAIR_GAME_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay V3 pairs at mapped CDN ordinals and audit every lineup state."""
    starters: dict[tuple[str, int], set[int]] = {}
    for (game_id, team_id), group in player_games.loc[player_games["starter"]].groupby(["game_id", "team_id"]):
        starters[(str(game_id), int(team_id))] = set(group["player_id"].astype(int))
    expected: dict[str, dict[int, float]] = defaultdict(dict)
    for row in player_games.itertuples(index=False):
        expected[str(row.game_id)][int(row.player_id)] = float(row.minutes_seconds)
    games = game_dim.set_index("game_id", drop=False)
    outputs: list[dict] = []
    quality: list[dict] = []
    for game_id in game_ids:
        game_actions = actions.loc[actions["game_id"].eq(game_id)].sort_values("orderNumber", kind="stable")
        repair = aligned.loc[aligned["game_id"].eq(game_id)].sort_values("cdn_order_number", kind="stable")
        game = games.loc[game_id] if game_id in games.index else None
        errors: list[str] = []
        if game is None or game_actions.empty:
            errors.append("missing_game_inputs")
            quality.append({"game_id": game_id, "passed": False, "failure_reasons": errors})
            continue
        home_id, away_id = int(game.home_team_id), int(game.away_team_id)
        lineups = {
            home_id: set(starters.get((game_id, home_id), set())),
            away_id: set(starters.get((game_id, away_id), set())),
        }
        starter_valid = all(len(lineups[team_id]) == 5 for team_id in (home_id, away_id))
        if not starter_valid:
            errors.append("invalid_canonical_starters")
        if repair.empty:
            errors.append("no_aligned_v3_substitutions")
        start_order = int(game_actions["orderNumber"].min())
        max_order = int(game_actions["orderNumber"].max())
        end_seconds = 2880.0 + max(int(game.max_period) - 4, 0) * 300.0
        local: list[dict] = []

        def append(end_order: int) -> None:
            nonlocal start_order
            if end_order <= start_order:
                return
            home, away = sorted(lineups[home_id]), sorted(lineups[away_id])
            if len(home) != 5 or len(away) != 5 or set(home).intersection(away):
                errors.append("invalid_ten_player_state")
                return
            local.append(
                {
                    "game_id": game_id,
                    "start_order_number": start_order,
                    "end_order_number": end_order,
                    **{f"home_player_{index + 1}": player for index, player in enumerate(home)},
                    **{f"away_player_{index + 1}": player for index, player in enumerate(away)},
                }
            )

        instructions: list[tuple[int, int, object]] = []
        for period in range(2, int(game.max_period) + 1):
            period_actions = game_actions.loc[game_actions["period"].eq(period)]
            if period_actions.empty:
                errors.append("missing_cdn_period_start")
                continue
            instructions.append((int(period_actions["orderNumber"].min()), 0, period))
        for row in repair.itertuples(index=False):
            instructions.append((int(row.cdn_order_number), 1, row))
        for order, kind, payload in sorted(instructions, key=lambda item: (item[0], item[1])):
            append(order)
            if kind == 0:
                period = int(payload)
                for team_id in (home_id, away_id):
                    inferred = period_starts.get((game_id, team_id, period))
                    if inferred is None:
                        errors.append("missing_v3_period_start")
                    else:
                        lineups[team_id] = set(inferred)
                start_order = order
                continue
            row = payload
            team_id, outgoing, incoming = int(row.team_id), int(row.out_player_id), int(row.in_player_id)
            if team_id not in lineups or outgoing not in lineups[team_id] or incoming in lineups[team_id]:
                errors.append("invalid_substitution_transition")
                continue
            lineups[team_id] = (lineups[team_id] - {outgoing}) | {incoming}
            start_order = order
        append(max_order + 1)

        order_clock = game_actions.set_index("orderNumber")["clock"].to_dict()
        actual: dict[int, float] = defaultdict(float)
        for stint in local:
            start_seconds = _elapsed_seconds(int(game_actions.loc[game_actions["orderNumber"].eq(stint["start_order_number"]), "period"].iloc[0]), order_clock[stint["start_order_number"]])
            if stint["end_order_number"] > max_order:
                end = end_seconds
            else:
                end_period = int(game_actions.loc[game_actions["orderNumber"].eq(stint["end_order_number"]), "period"].iloc[0])
                end = _elapsed_seconds(end_period, order_clock[stint["end_order_number"]])
            duration = end - start_seconds
            if duration < 0:
                errors.append("negative_elapsed_stint")
                continue
            for column in _LINEUP_COLUMNS:
                actual[int(stint[column])] += duration
        minute_errors = [abs(actual.get(player_id, 0.0) - seconds) for player_id, seconds in expected[game_id].items()]
        max_minute_error = max(minute_errors, default=float("inf"))
        action_scores = game_actions[["scoreHome", "scoreAway"]].apply(pd.to_numeric, errors="coerce")
        score_conserved = bool(
            action_scores.notna().all(axis=None)
            and action_scores.diff().fillna(0).ge(0).all(axis=None)
            and int(action_scores.iloc[-1, 0]) == int(game.home_score)
            and int(action_scores.iloc[-1, 1]) == int(game.away_score)
        )
        if not score_conserved:
            errors.append("cdn_score_not_conserved")
        if max_minute_error > 5.0:
            errors.append("official_minute_error_over_5_seconds")
        passed = not errors and bool(local)
        quality.append(
            {
                "game_id": game_id,
                "aligned_v3_substitution_count": int(len(repair)),
                "ordinal_stint_count": int(len(local)),
                "starter_valid": starter_valid,
                "max_player_minute_error": max_minute_error,
                "score_conserved": score_conserved,
                "passed": passed,
                "failure_reasons": "|".join(sorted(set(errors))),
            }
        )
        if passed:
            for number, stint in enumerate(local, start=1):
                outputs.append({"ordinal_stint_id": f"{game_id}_v3r{number:03d}", "ordinal_stint_number": number, **stint})
    return pd.DataFrame(outputs), pd.DataFrame(quality)


def _load_events(root: Path, source: str, game_ids: tuple[str, ...], columns: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((root / source).rglob("*.parquet"))
    if not paths:
        raise ValueError(f"No {source} partitions found under {root / source}")
    raw_ids = [int(game_id) for game_id in game_ids]
    frames = [pd.read_parquet(path, columns=columns, filters=[("gameId", "in", raw_ids)]) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["game_id"] = frame["gameId"].map(canonical_game_id)
    return frame.loc[frame["game_id"].isin(game_ids)].copy(), paths


def build_v3_cdn_lineup_repair_candidate(
    event_root: str | Path,
    v3_event_root: str | Path,
    player_games_path: str | Path,
    game_dim_path: str | Path,
    alignment_destination: str | Path,
    stints_destination: str | Path,
    assigned_actions_destination: str | Path,
    quality_destination: str | Path,
    report_destination: str | Path,
    manifest_dir: str | Path,
) -> dict:
    """Write a separate, non-production candidate for four predeclared games."""
    root = Path(event_root)
    v3_root = Path(v3_event_root)
    player_path, game_path = Path(player_games_path), Path(game_dim_path)
    players, games = pd.read_parquet(player_path), pd.read_parquet(game_path)
    players = players.loc[players["game_id"].isin(REPAIR_GAME_IDS)].copy()
    games = games.loc[games["game_id"].isin(REPAIR_GAME_IDS)].copy()
    v3, v3_paths = _load_events(
        v3_root, "nbastatsv3", REPAIR_GAME_IDS,
        ["gameId", "actionId", "actionNumber", "period", "clock", "actionType", "description", "personId", "teamId"],
    )
    cdn, cdn_paths = _load_events(
        root, "cdnnba", REPAIR_GAME_IDS,
        ["gameId", "orderNumber", "actionNumber", "period", "clock", "actionType", "subType", "personId", "teamId", "possession", "scoreHome", "scoreAway"],
    )
    v3_pairs, parse_failures = parse_v3_substitutions(v3, players)
    period_starts, period_start_failures = infer_v3_period_starts(v3, v3_pairs)
    alignment, alignment_failures = align_v3_substitutions_to_cdn(v3_pairs, cdn)
    alignment_failure_games = set(alignment_failures["game_id"].astype(str)) if not alignment_failures.empty else set()
    stints, quality = replay_aligned_lineups(
        cdn, players, alignment, period_starts, period_start_failures, games
    )
    if not quality.empty:
        quality.loc[quality["game_id"].isin(alignment_failure_games), "passed"] = False
        quality.loc[quality["game_id"].isin(alignment_failure_games), "failure_reasons"] = quality.loc[
            quality["game_id"].isin(alignment_failure_games), "failure_reasons"
        ].fillna("").map(lambda text: "|".join(filter(None, [text, "strict_alignment_failed"])))
    valid_games = set(quality.loc[quality["passed"], "game_id"].astype(str))
    attached_frames: list[pd.DataFrame] = []
    for game_id in sorted(valid_games):
        try:
            attached_frames.append(
                attach_ordinal_lineups(
                    cdn.loc[cdn["game_id"].eq(game_id)].sort_values("orderNumber", kind="stable"),
                    stints.loc[stints["game_id"].eq(game_id)],
                )
            )
        except ValueError:
            quality.loc[quality["game_id"].eq(game_id), "passed"] = False
            quality.loc[quality["game_id"].eq(game_id), "failure_reasons"] = quality.loc[
                quality["game_id"].eq(game_id), "failure_reasons"
            ].map(lambda text: "|".join(filter(None, [text, "ordinal_coverage_failed"])))
    valid_games = set(quality.loc[quality["passed"], "game_id"].astype(str))
    if stints.empty:
        stints = pd.DataFrame(columns=["ordinal_stint_id", "ordinal_stint_number", "game_id", *_LINEUP_COLUMNS])
    else:
        stints = stints.loc[stints["game_id"].isin(valid_games)].copy()
    assigned_actions = (
        pd.concat(attached_frames, ignore_index=True)
        if attached_frames
        else pd.DataFrame(columns=[*cdn.columns, *_LINEUP_COLUMNS, "ordinal_stint_id"])
    )
    assigned_actions = assigned_actions.loc[assigned_actions["game_id"].isin(valid_games)].copy()
    paths = {
        "alignment": Path(alignment_destination),
        "stints": Path(stints_destination),
        "assigned_actions": Path(assigned_actions_destination),
        "quality": Path(quality_destination),
        "report": Path(report_destination),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    alignment.to_parquet(paths["alignment"], index=False)
    stints.to_parquet(paths["stints"], index=False)
    assigned_actions.to_parquet(paths["assigned_actions"], index=False)
    quality.to_parquet(paths["quality"], index=False)
    source_paths = sorted(set(cdn_paths + v3_paths + [player_path, game_path]))
    outcome_columns = ["game_id", "orderNumber", "possession", "scoreHome", "scoreAway"]
    canonical_outcomes = cdn.loc[cdn["game_id"].isin(valid_games), outcome_columns].sort_values(
        ["game_id", "orderNumber"], kind="stable"
    )
    outcome_hash = hashlib.sha256(
        canonical_outcomes.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    assigned_outcomes = assigned_actions[outcome_columns].sort_values(["game_id", "orderNumber"], kind="stable")
    assigned_outcome_hash = hashlib.sha256(assigned_outcomes.to_csv(index=False).encode("utf-8")).hexdigest()
    report = {
        "dataset": "v3_cdn_lineup_repair_candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_status": "research_candidate_only",
        "repair_game_ids": list(REPAIR_GAME_IDS),
        "passed_game_ids": sorted(valid_games),
        "quarantined_game_ids": sorted(set(REPAIR_GAME_IDS) - valid_games),
        "alignment_contract": (
            "Each V3 actionId maps to one CDN orderNumber only when game, period, clock, team, "
            "incoming player ID, and substitution-in event type all match. Clock-only joins are forbidden."
        ),
        "canonical_possession_contract": "CDN orderNumber, possession owner, and scores are retained; V3 only supplies substitution pairs.",
        "cdn_possession_outcomes_hash_before": outcome_hash,
        "cdn_possession_outcomes_hash_after": assigned_outcome_hash,
        "cdn_possession_outcomes_unchanged": outcome_hash == assigned_outcome_hash,
        "assigned_action_row_count": int(len(assigned_actions)),
        "minute_tolerance_seconds": 5.0,
        "v3_parse_failures": parse_failures.to_dict("records"),
        "period_start_failures": period_start_failures.to_dict("records"),
        "alignment_failures": alignment_failures.to_dict("records"),
        "quality": quality.to_dict("records"),
        "outputs": {key: str(path.resolve()) for key, path in paths.items() if key != "report"},
        "source_files": [
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
        ],
    }
    report["snapshot_id"] = "v3_cdn_lineup_repair_" + hashlib.sha256(
        json.dumps(report["source_files"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    report["passed"] = not report["quarantined_game_ids"] and report["cdn_possession_outcomes_unchanged"]
    write_json_atomic(report, paths["report"])
    write_json_atomic(report, Path(manifest_dir) / f"{report['snapshot_id']}.json")
    return report
