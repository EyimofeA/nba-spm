"""Strict, targeted repairs for games quarantined by canonical lineup QA.

Gabriel's event files are used only to repair named games that still have a
canonical CDN action stream.  The CDN stream supplies possession ownership and
physical order.  The fallback supplies observed on-court player sets.  This
module never invents a player: a source action must show ten players, or a
missing non-outcome boundary must be bracketed by the same observed lineup.
"""
from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .event_state import parse_clock_seconds
from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic
from .possessions import (
    AWAY_LINEUP_COLUMNS,
    HOME_LINEUP_COLUMNS,
    _add_elapsed_seconds,
    collapse_cdn_possessions,
    reconcile_action_points,
)


_RAW_COLUMNS = [
    "gameId", "orderNumber", "actionNumber", "period", "clock", "possession",
    "scoreHome", "scoreAway", "actionType", "description", "personId", "teamId",
]
_GABRIEL_COLUMNS = [
    "game_id", "actionNumber", "actionType", "players_on", "scoreHome", "scoreAway",
]
_NON_OUTCOME_BOUNDARIES = {"period", "game", "jumpball"}


def _load_plan(manifest_path: str | Path, fallback_root: Path) -> list[dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    target_games = manifest.get("target_games") or {}
    tasks: list[dict] = []
    for task in manifest["tasks"]:
        names = target_games.get(task["name"])
        if not names:
            raise ValueError(f"{task['name']}: missing target_games entry")
        source = fallback_root / task["destination"]
        if not source.exists():
            raise FileNotFoundError(f"{task['name']}: missing fallback file {source}")
        for game_id in names:
            tasks.append({**task, "game_id": canonical_game_id(game_id), "source": source})
    return tasks


def _parse_players(value: object) -> tuple[int, ...] | None:
    if value is None or pd.isna(value):
        return None
    values = tuple(int(item) for item in str(value).split("|") if item.strip())
    return values if len(values) == 10 and len(set(values)) == 10 else None


def _lineup_from_players(
    player_ids: tuple[int, ...], player_teams: dict[int, int], home_team_id: int, away_team_id: int
) -> tuple[int, ...] | None:
    home = sorted(player_id for player_id in player_ids if player_teams.get(player_id) == home_team_id)
    away = sorted(player_id for player_id in player_ids if player_teams.get(player_id) == away_team_id)
    if len(home) != 5 or len(away) != 5 or len(set(home) | set(away)) != 10:
        return None
    return tuple(home + away)


def _source_lineups(
    source: pd.DataFrame, player_teams: dict[int, int], home_team_id: int, away_team_id: int
) -> dict[int, tuple[int, ...]]:
    if source.duplicated("actionNumber", keep=False).any():
        raise ValueError("fallback source has duplicate actionNumber values")
    output: dict[int, tuple[int, ...]] = {}
    for row in source.itertuples(index=False):
        players = _parse_players(row.players_on)
        if players is None:
            continue
        lineup = _lineup_from_players(players, player_teams, home_team_id, away_team_id)
        if lineup is not None:
            output[int(row.actionNumber)] = lineup
    return output


def _bridge_lineup(
    action_number: int,
    action_type: str,
    source_actions: list[int],
    source_lineups: dict[int, tuple[int, ...]],
) -> tuple[tuple[int, ...] | None, str]:
    """Use only exact continuity, never a guessed player state."""
    if action_number in source_lineups:
        return source_lineups[action_number], "observed"
    position = bisect_left(source_actions, action_number)
    previous = source_actions[position - 1] if position else None
    following = source_actions[position] if position < len(source_actions) else None
    previous_lineup = source_lineups.get(previous) if previous is not None else None
    following_lineup = source_lineups.get(following) if following is not None else None
    if str(action_type).casefold() in _NON_OUTCOME_BOUNDARIES:
        return following_lineup or previous_lineup, "boundary_continuity"
    if previous_lineup is not None and previous_lineup == following_lineup:
        return previous_lineup, "two_sided_continuity"
    return None, "unmapped"


def _elapsed_seconds(period: int, clock: object) -> float:
    remaining = float(parse_clock_seconds(pd.Series([clock])).iloc[0])
    if period <= 4:
        return (period - 1) * 720.0 + 720.0 - remaining
    return 2880.0 + (period - 5) * 300.0 + 300.0 - remaining


def _ordinal_stint_ids(game_id: str, lineups: list[tuple[int, ...]]) -> list[str]:
    """Number contiguous observed lineup states; returning lineups get new IDs."""
    lineup_token = pd.Series(["|".join(str(value) for value in lineup) for lineup in lineups])
    stint_number = lineup_token.ne(lineup_token.shift()).cumsum().astype(int)
    return [f"{game_id}:g{number:03d}" for number in stint_number]


def _minute_error(actions: pd.DataFrame, player_games: pd.DataFrame, max_period: int) -> float:
    """Reconcile observed lineup-state durations to official player minutes."""
    actions = actions.sort_values("orderNumber", kind="stable")
    elapsed = [_elapsed_seconds(int(row.period), row.clock) for row in actions.itertuples(index=False)]
    states = [
        tuple(int(getattr(row, column)) for column in HOME_LINEUP_COLUMNS + AWAY_LINEUP_COLUMNS)
        for row in actions.itertuples(index=False)
    ]
    observed: dict[int, float] = defaultdict(float)
    prior_time = 0.0
    prior_state = states[0]
    for current_time, state in zip(elapsed, states, strict=True):
        duration = current_time - prior_time
        if duration < -1e-6:
            raise ValueError("event order has decreasing game time")
        for player_id in prior_state:
            observed[player_id] += max(duration, 0.0)
        prior_time = max(prior_time, current_time)
        prior_state = state
    game_seconds = 2880.0 + max(int(max_period) - 4, 0) * 300.0
    for player_id in prior_state:
        observed[player_id] += max(game_seconds - prior_time, 0.0)
    expected = player_games.set_index("player_id")["minutes_seconds"].to_dict()
    player_ids = set(expected) | set(observed)
    return max((abs(float(expected.get(player_id, 0.0)) - observed.get(player_id, 0.0)) for player_id in player_ids), default=float("inf"))


def _repair_game(
    raw_actions: pd.DataFrame,
    fallback: pd.DataFrame,
    game: pd.Series,
    player_games: pd.DataFrame,
    event_states: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    game_id = str(game.game_id)
    home_team_id, away_team_id = int(game.home_team_id), int(game.away_team_id)
    player_teams = {
        int(row.player_id): int(row.team_id)
        for row in player_games.itertuples(index=False)
    }
    source_lineups = _source_lineups(fallback, player_teams, home_team_id, away_team_id)
    source_actions = sorted(source_lineups)
    prepared = raw_actions.sort_values("orderNumber", kind="stable").copy()
    lineups: list[tuple[int, ...]] = []
    methods: list[str] = []
    raw_rows = list(prepared.itertuples(index=False))
    for index, row in enumerate(raw_rows):
        action_number = int(row.actionNumber)
        action_type = str(row.actionType).casefold()
        lineup, method = _bridge_lineup(action_number, action_type, source_actions, source_lineups)
        if lineup is None and action_type == "substitution":
            source_position = bisect_right(source_actions, action_number)
            following = source_actions[source_position] if source_position < len(source_actions) else None
            following_lineup = source_lineups.get(following) if following is not None else None
            if following is not None and following_lineup is not None:
                intervening = [
                    candidate for candidate in raw_rows[index + 1:]
                    if int(candidate.actionNumber) < following
                ]
                same_clock_batch = all(
                    str(candidate.actionType).casefold() == "substitution"
                    and int(candidate.period) == int(row.period)
                    and str(candidate.clock) == str(row.clock)
                    for candidate in intervening
                )
                if same_clock_batch:
                    lineup, method = following_lineup, "post_substitution_observed"
        if lineup is None:
            return None, None, {
                "game_id": game_id,
                "passed": False,
                "reason": f"unmapped_action:{int(row.actionNumber)}:{row.actionType}",
            }
        lineups.append(lineup)
        methods.append(method)
    for index, column in enumerate(HOME_LINEUP_COLUMNS + AWAY_LINEUP_COLUMNS):
        prepared[column] = [lineup[index] for lineup in lineups]
    prepared["fallback_lineup_method"] = methods
    prepared["game_id"] = game_id
    prepared = prepared.assign(
        season_start=int(game.season_start),
        season_end=int(game.season_end),
        season_label=str(game.season_label),
        season_type=str(game.season_type),
        game_date=game.game_date,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_score=int(game.home_score),
        away_score=int(game.away_score),
    )
    prepared = _add_elapsed_seconds(prepared)
    prepared, point_stats = reconcile_action_points(prepared, event_states)
    # A lineup may recur later in a game.  Preserve that return as a distinct
    # ordinal stint, otherwise a possession that spans the return would falsely
    # look like it has one unbroken lineup segment.
    prepared["ordinal_stint_id"] = _ordinal_stint_ids(game_id, lineups)
    minute_error = _minute_error(prepared, player_games, int(game.max_period))
    possessions, segments = collapse_cdn_possessions(prepared)
    home_points, away_points = possessions[["home_points", "away_points"]].sum()
    issues = {
        "score_mismatch": int(home_points != int(game.home_score) or away_points != int(game.away_score)),
        "duplicate_possessions": int(possessions.duplicated("possession_id", keep=False).sum()),
        "duplicate_segments": int(segments.duplicated("possession_segment_id", keep=False).sum()),
        "invalid_segment_lineups": int(
            segments.loc[:, HOME_LINEUP_COLUMNS + AWAY_LINEUP_COLUMNS].nunique(axis=1).ne(10).sum()
        ),
        "minute_error_over_five_seconds": int(minute_error > 5.0),
    }
    passed = not any(issues.values())
    summary = {
        "game_id": game_id,
        "passed": passed,
        "candidate_possessions": int(len(possessions)),
        "candidate_segments": int(len(segments)),
        "observed_action_lineups": int(sum(method == "observed" for method in methods)),
        "post_substitution_observed_actions": int(
            sum(method == "post_substitution_observed" for method in methods)
        ),
        "two_sided_continuity_actions": int(sum(method == "two_sided_continuity" for method in methods)),
        "boundary_continuity_actions": int(sum(method == "boundary_continuity" for method in methods)),
        "max_player_minute_error_seconds": float(minute_error),
        "point_reconciliation": point_stats,
        "issues": issues,
    }
    return (possessions if passed else None), (segments if passed else None), summary


def build_gabriel_fallback_repairs(
    manifest_path: str | Path,
    fallback_root: str | Path,
    event_root: str | Path,
    event_states_path: str | Path,
    game_dim_path: str | Path,
    player_games_path: str | Path,
    possessions_path: str | Path,
    segments_path: str | Path,
    output_path: str | Path,
    segments_output_path: str | Path,
    report_path: str | Path,
) -> dict:
    """Create a merged RAPM input only from individually passing repaired games."""
    fallback_root = Path(fallback_root)
    event_root = Path(event_root)
    plan = _load_plan(manifest_path, fallback_root)
    games = pd.read_parquet(game_dim_path)
    players = pd.read_parquet(player_games_path)
    states = pd.read_parquet(
        event_states_path,
        columns=["game_id", "actionNumber", "period", "clock", "home_points_added", "away_points_added"],
    )
    base_possessions = pd.read_parquet(possessions_path)
    base_segments = pd.read_parquet(segments_path)
    summaries: list[dict] = []
    repaired_possessions: list[pd.DataFrame] = []
    repaired_segments: list[pd.DataFrame] = []
    for item in plan:
        game_id = item["game_id"]
        game_rows = games.loc[games["game_id"].astype(str).eq(game_id)]
        if len(game_rows) != 1:
            summaries.append({"game_id": game_id, "passed": False, "reason": "canonical_game_missing"})
            continue
        game = game_rows.iloc[0]
        raw_path = event_root / "cdnnba" / f"season={int(game.season_start)}" / f"{game.season_type}.parquet"
        if not raw_path.exists():
            summaries.append({"game_id": game_id, "passed": False, "reason": "canonical_cdn_events_missing"})
            continue
        raw = pd.read_parquet(raw_path, columns=_RAW_COLUMNS)
        raw = raw.loc[raw["gameId"].map(canonical_game_id).eq(game_id)].copy()
        fallback = pd.read_csv(item["source"], usecols=lambda name: name in _GABRIEL_COLUMNS, low_memory=False)
        fallback = fallback.loc[fallback["game_id"].map(canonical_game_id).eq(game_id)].copy()
        player_rows = players.loc[players["game_id"].astype(str).eq(game_id)].copy()
        state_rows = states.loc[states["game_id"].astype(str).eq(game_id)].copy()
        if raw.empty or fallback.empty or player_rows.empty or state_rows.empty:
            summaries.append({"game_id": game_id, "passed": False, "reason": "required_game_source_missing"})
            continue
        candidate_possessions, candidate_segments, summary = _repair_game(
            raw, fallback, game, player_rows, state_rows
        )
        summaries.append(summary)
        if candidate_possessions is not None and candidate_segments is not None:
            repaired_possessions.append(candidate_possessions)
            repaired_segments.append(candidate_segments)

    passing_ids = {item["game_id"] for item in summaries if item.get("passed")}
    merged_possessions = pd.concat(
        [base_possessions.loc[~base_possessions["game_id"].astype(str).isin(passing_ids)], *repaired_possessions],
        ignore_index=True,
    ).sort_values(["game_id", "possession_number"], kind="stable")
    merged_segments = pd.concat(
        [base_segments.loc[~base_segments["game_id"].astype(str).isin(passing_ids)], *repaired_segments],
        ignore_index=True,
    ).sort_values(["game_id", "possession_number", "segment_number"], kind="stable")
    output_path, segments_output_path, report_path = map(Path, (output_path, segments_output_path, report_path))
    for path in (output_path, segments_output_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    merged_possessions.to_parquet(output_path, index=False)
    merged_segments.to_parquet(segments_output_path, index=False)
    report = {
        "dataset": "gabriel_targeted_lineup_repairs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "CDN possession ownership and order plus observed fallback on-court states; no unmatched outcome action is admitted.",
        "requested_games": [item["game_id"] for item in plan],
        "repaired_games": sorted(passing_ids),
        "blocked_games": [item for item in summaries if not item.get("passed")],
        "game_summaries": summaries,
        "merged_possession_rows": int(len(merged_possessions)),
        "merged_segment_rows": int(len(merged_segments)),
        "output_path": str(output_path.resolve()),
        "segments_output_path": str(segments_output_path.resolve()),
        "passed": len(passing_ids) == len(plan),
    }
    write_json_atomic(report, report_path)
    return report
