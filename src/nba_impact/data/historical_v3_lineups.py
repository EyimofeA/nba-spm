"""Strict, separate historical lineup candidates from NBA Stats V3 event order.

NBA Stats V3 ``actionId`` is the only event order used here.  The builder does
not infer possession ownership and does not write canonical lineup tables.
Starter and minute evidence comes from a separate player-game candidate.
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
from .lineups import _elapsed_seconds, _normalize_name
from .manifest import sha256_file, write_json_atomic
from .v3_cdn_lineup_repair import infer_v3_period_starts


_V3_SUB = re.compile(r"^SUB:\s*(.+?)\s+FOR\s+(.+?)$", re.IGNORECASE)
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
_LINEUP_COLUMNS = tuple(
    [f"home_player_{index}" for index in range(1, 6)]
    + [f"away_player_{index}" for index in range(1, 6)]
)


def _require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    if missing := sorted(columns - set(frame.columns)):
        raise ValueError(f"{source} is missing required columns: {missing}")


def _name_aliases(name: object) -> set[str]:
    """Return conservative aliases for a player-game roster name."""
    text = str(name or "").strip()
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    if not tokens:
        return set()
    aliases = {_normalize_name(text), _normalize_name(tokens[-1])}
    if len(tokens) >= 2:
        aliases.add(_normalize_name(f"{tokens[0][0]} {tokens[-1]}"))
        aliases.add(_normalize_name(f"{tokens[0]} {tokens[-1]}"))
    if len(tokens) >= 2 and tokens[-1].casefold().strip(".") in _SUFFIXES:
        aliases.add(_normalize_name(" ".join(tokens[-2:])))
        aliases.add(_normalize_name(tokens[-2]))
        if len(tokens) >= 3:
            aliases.add(_normalize_name(f"{tokens[0][0]} {' '.join(tokens[-2:])}"))
    return {alias for alias in aliases if alias}


def _historical_aliases(player_games: pd.DataFrame) -> dict[tuple[str, int, str], set[int]]:
    """Resolve only against the named game and team roster."""
    aliases: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    for row in player_games.itertuples(index=False):
        for alias in _name_aliases(row.player_name):
            aliases[(str(row.game_id), int(row.team_id), alias)].add(int(row.player_id))
    return aliases


def parse_historical_v3_substitutions(
    v3: pd.DataFrame, player_games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse V3 pairs; incoming names must map to one same-game team roster ID."""
    _require_columns(
        v3,
        {"game_id", "actionId", "actionNumber", "period", "clock", "actionType", "description", "personId", "teamId"},
        "V3 events",
    )
    aliases = _historical_aliases(player_games)
    source = v3.loc[v3["actionType"].astype(str).str.casefold().eq("substitution")].copy()
    rows: list[dict] = []
    failures: list[dict] = []
    for row in source.itertuples(index=False):
        match = _V3_SUB.fullmatch(str(row.description).strip())
        if not match:
            failures.append({"game_id": str(row.game_id), "action_id": int(row.actionId), "reason": "parse"})
            continue
        incoming_name, _ = match.groups()
        candidates = aliases.get(
            (str(row.game_id), int(row.teamId), _normalize_name(incoming_name)), set()
        )
        if len(candidates) != 1:
            failures.append(
                {
                    "game_id": str(row.game_id),
                    "action_id": int(row.actionId),
                    "reason": f"incoming_alias_matches_{len(candidates)}",
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
                "in_player_id": next(iter(candidates)),
            }
        )
    pair_columns = [
        "game_id", "v3_action_id", "v3_action_number", "period", "clock", "team_id",
        "out_player_id", "in_player_id",
    ]
    failure_columns = ["game_id", "action_id", "reason"]
    return pd.DataFrame(rows, columns=pair_columns), pd.DataFrame(failures, columns=failure_columns)


def _official_scores(
    path: Path, project_season: int, season_type: str
) -> pd.DataFrame:
    scores = pd.read_parquet(path)
    _require_columns(
        scores,
        {"project_season", "season_type", "game_id", "home_team_id", "away_team_id", "home_score", "away_score"},
        "official game scores",
    )
    scores["game_id"] = scores["game_id"].map(canonical_game_id)
    output = scores.loc[
        scores["project_season"].eq(project_season) & scores["season_type"].eq(season_type)
    ].copy()
    if output["game_id"].duplicated().any():
        raise ValueError("Official score source has duplicate game IDs")
    return output


def _v3_score_conserved(actions: pd.DataFrame, score: pd.Series) -> bool:
    values = actions[["scoreHome", "scoreAway"]].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if values.empty:
        return False
    if not values.diff().fillna(0).ge(0).all(axis=None):
        return False
    last = values.iloc[-1]
    return bool(int(last["scoreHome"]) == int(score.home_score) and int(last["scoreAway"]) == int(score.away_score))


def _lineup_state_valid(lineups: dict[int, set[int]], home_id: int, away_id: int) -> bool:
    home, away = lineups[home_id], lineups[away_id]
    return len(home) == 5 and len(away) == 5 and not home.intersection(away)


def _build_game_stints(
    actions: pd.DataFrame,
    players: pd.DataFrame,
    score: pd.Series,
    pairs: pd.DataFrame,
    period_starts: dict[tuple[str, int, int], set[int]],
    parse_failure_count: int,
    period_failure_count: int,
) -> tuple[list[dict], dict]:
    """Replay one game at exact V3 action IDs and return stints plus quality."""
    game_id = str(score.game_id)
    home_id, away_id = int(score.home_team_id), int(score.away_team_id)
    actions = actions.sort_values("actionId", kind="stable").copy()
    errors: set[str] = set()
    if actions.empty:
        return [], {"game_id": game_id, "passed": False, "failure_reasons": "missing_v3_actions"}
    if actions["actionId"].duplicated().any():
        errors.add("duplicate_v3_action_id")
    max_period = int(actions["period"].max())
    roster = {
        home_id: set(players.loc[players["team_id"].eq(home_id), "player_id"].astype(int)),
        away_id: set(players.loc[players["team_id"].eq(away_id), "player_id"].astype(int)),
    }
    starters = {
        home_id: set(players.loc[players["team_id"].eq(home_id) & players["starter"], "player_id"].astype(int)),
        away_id: set(players.loc[players["team_id"].eq(away_id) & players["starter"], "player_id"].astype(int)),
    }
    lineups = {home_id: set(starters[home_id]), away_id: set(starters[away_id])}
    starter_valid = _lineup_state_valid(lineups, home_id, away_id)
    if not starter_valid:
        errors.add("invalid_player_game_starters")
    if parse_failure_count:
        errors.add("v3_substitution_name_resolution_failed")
    if period_failure_count:
        errors.add("v3_period_start_inference_failed")

    action_times = {
        int(row.actionId): _elapsed_seconds(int(row.period), row.clock)
        for row in actions.itertuples(index=False)
    }
    first_by_period = actions.groupby("period", sort=False)["actionId"].min().to_dict()
    transitions: dict[int, list[tuple[str, object]]] = defaultdict(list)
    for period in range(2, max_period + 1):
        if period not in first_by_period:
            errors.add("missing_v3_period_start_action")
            continue
        transitions[int(first_by_period[period])].append(("period", period))
    for pair in pairs.sort_values("v3_action_id", kind="stable").itertuples(index=False):
        transitions[int(pair.v3_action_id)].append(("substitution", pair))

    local: list[dict] = []
    first_action = int(actions["actionId"].iloc[0])
    start_action, start_time = first_action, action_times[first_action]

    def append(end_action: int, end_time: float) -> None:
        nonlocal start_action, start_time
        if end_time < start_time:
            errors.add("negative_elapsed_stint")
            return
        if end_action <= start_action or end_time == start_time:
            return
        if not _lineup_state_valid(lineups, home_id, away_id):
            errors.add("invalid_ten_player_state")
            return
        home, away = sorted(lineups[home_id]), sorted(lineups[away_id])
        local.append(
            {
                "game_id": game_id,
                "start_action_id": start_action,
                "end_action_id_exclusive": end_action,
                "start_seconds_elapsed": start_time,
                "end_seconds_elapsed": end_time,
                "duration_seconds": end_time - start_time,
                **{f"home_player_{index + 1}": value for index, value in enumerate(home)},
                **{f"away_player_{index + 1}": value for index, value in enumerate(away)},
            }
        )

    for action_id in sorted(transitions):
        transition_time = action_times.get(action_id)
        if transition_time is None:
            errors.add("substitution_action_missing_from_v3")
            continue
        append(action_id, transition_time)
        for kind, payload in transitions[action_id]:
            if kind == "period":
                period = int(payload)
                for team_id in (home_id, away_id):
                    inferred = period_starts.get((game_id, team_id, period))
                    if inferred is None or len(inferred) != 5 or not inferred.issubset(roster[team_id]):
                        errors.add("invalid_v3_period_start")
                    else:
                        lineups[team_id] = set(inferred)
            else:
                pair = payload
                team_id, outgoing, incoming = int(pair.team_id), int(pair.out_player_id), int(pair.in_player_id)
                if (
                    team_id not in lineups
                    or outgoing not in lineups[team_id]
                    or incoming in lineups[team_id]
                    or incoming not in roster[team_id]
                ):
                    errors.add("invalid_substitution_transition")
                else:
                    lineups[team_id] = (lineups[team_id] - {outgoing}) | {incoming}
        start_action, start_time = action_id, transition_time

    end_seconds = 2880.0 + max(max_period - 4, 0) * 300.0
    append(int(actions["actionId"].max()) + 1, end_seconds)

    # Event coverage is a state audit, not a duration-stint count. Multiple
    # substitutions can occur at the same clock, so a valid action-level state
    # can have zero elapsed duration and must not be emitted as a zero-length
    # stint. Apply each action's transition first, then audit its lineup.
    coverage_lineups = {home_id: set(starters[home_id]), away_id: set(starters[away_id])}
    covered = 0
    for action in actions.itertuples(index=False):
        action_id = int(action.actionId)
        for kind, payload in transitions.get(action_id, []):
            if kind == "period":
                period = int(payload)
                for team_id in (home_id, away_id):
                    inferred = period_starts.get((game_id, team_id, period))
                    if inferred is None or len(inferred) != 5 or not inferred.issubset(roster[team_id]):
                        errors.add("invalid_v3_period_start")
                    else:
                        coverage_lineups[team_id] = set(inferred)
            else:
                pair = payload
                team_id, outgoing, incoming = int(pair.team_id), int(pair.out_player_id), int(pair.in_player_id)
                if (
                    team_id not in coverage_lineups
                    or outgoing not in coverage_lineups[team_id]
                    or incoming in coverage_lineups[team_id]
                    or incoming not in roster[team_id]
                ):
                    errors.add("invalid_substitution_transition")
                else:
                    coverage_lineups[team_id] = (coverage_lineups[team_id] - {outgoing}) | {incoming}
        if _lineup_state_valid(coverage_lineups, home_id, away_id):
            covered += 1
    if covered != len(actions):
        errors.add("v3_event_lineup_coverage_incomplete")
    actual: dict[int, float] = defaultdict(float)
    for stint in local:
        for column in _LINEUP_COLUMNS:
            actual[int(stint[column])] += float(stint["duration_seconds"])
    expected = {
        int(row.player_id): float(row.minutes_seconds)
        for row in players.itertuples(index=False)
    }
    minute_errors = [abs(actual.get(player, 0.0) - expected.get(player, 0.0)) for player in set(actual) | set(expected)]
    max_minute_error = max(minute_errors, default=float("inf"))
    if max_minute_error > 5.0:
        errors.add("official_player_minute_error_over_5_seconds")
    score_conserved = _v3_score_conserved(actions, score)
    if not score_conserved:
        errors.add("v3_score_not_conserved")
    passed = not errors and bool(local)
    quality = {
        "game_id": game_id,
        "season_label": str(players["season_label"].iloc[0]),
        "season_type": str(players["season_type"].iloc[0]),
        "game_date": players["game_date"].iloc[0],
        "home_team_id": home_id,
        "away_team_id": away_id,
        "max_period": max_period,
        "v3_action_count": len(actions),
        "v3_substitution_count": len(pairs),
        "v3_event_coverage_count": covered,
        "v3_event_coverage_complete": covered == len(actions),
        "starter_valid": starter_valid,
        "max_player_minute_error": max_minute_error,
        "score_conserved": score_conserved,
        "stint_count": len(local),
        "passed": passed,
        "failure_reasons": "|".join(sorted(errors)),
    }
    return local if passed else [], quality


def build_historical_v3_lineup_candidate(
    v3_root: str | Path,
    player_games_path: str | Path,
    official_scores_path: str | Path,
    stints_destination: str | Path,
    quality_destination: str | Path,
    report_destination: str | Path,
    manifest_dir: str | Path,
    *,
    project_season: int,
    season_type: str = "regular",
) -> dict:
    """Build a strict non-production historical V3 ordinal-lineup candidate."""
    root = Path(v3_root)
    player_path, scores_path = Path(player_games_path), Path(official_scores_path)
    v3_path = root / "nbastatsv3" / f"project_season={project_season}" / f"{season_type}.parquet"
    if not v3_path.exists():
        raise FileNotFoundError(f"Pinned V3 partition is absent: {v3_path}")
    players = pd.read_parquet(player_path)
    _require_columns(
        players,
        {"game_id", "season_end", "season_label", "season_type", "game_date", "team_id", "player_id", "player_name", "starter", "minutes_seconds"},
        "historical player games",
    )
    players["game_id"] = players["game_id"].map(canonical_game_id)
    players = players.loc[
        players["season_end"].eq(project_season) & players["season_type"].eq(season_type)
    ].copy()
    players["starter"] = players["starter"].astype(bool)
    players["team_id"] = pd.to_numeric(players["team_id"], errors="raise").astype(int)
    players["player_id"] = pd.to_numeric(players["player_id"], errors="raise").astype(int)
    players["minutes_seconds"] = pd.to_numeric(players["minutes_seconds"], errors="raise")
    scores = _official_scores(scores_path, project_season, season_type)
    v3 = pd.read_parquet(
        v3_path,
        columns=["gameId", "actionId", "actionNumber", "period", "clock", "actionType", "description", "personId", "teamId", "scoreHome", "scoreAway"],
    )
    v3["game_id"] = v3["gameId"].map(canonical_game_id)
    target_ids = set(scores["game_id"])
    v3 = v3.loc[v3["game_id"].isin(target_ids)].copy()
    pairs, parse_failures = parse_historical_v3_substitutions(v3, players)
    period_starts, period_failures = infer_v3_period_starts(v3, pairs)
    parse_by_game = parse_failures.groupby("game_id").size().to_dict() if not parse_failures.empty else {}
    period_by_game = period_failures.groupby("game_id").size().to_dict() if not period_failures.empty else {}
    player_games = {game_id: frame.copy() for game_id, frame in players.groupby("game_id", sort=False)}
    event_games = {game_id: frame.copy() for game_id, frame in v3.groupby("game_id", sort=False)}
    pairs_by_game = {game_id: frame.copy() for game_id, frame in pairs.groupby("game_id", sort=False)} if not pairs.empty else {}
    all_stints: list[dict] = []
    quality_rows: list[dict] = []
    for score in scores.sort_values("game_id", kind="stable").itertuples(index=False):
        game_id = str(score.game_id)
        if game_id not in player_games:
            quality_rows.append({"game_id": game_id, "passed": False, "failure_reasons": "missing_player_game_rows"})
            continue
        if game_id not in event_games:
            quality_rows.append({"game_id": game_id, "passed": False, "failure_reasons": "missing_v3_actions"})
            continue
        local, quality = _build_game_stints(
            event_games[game_id], player_games[game_id], score,
            pairs_by_game.get(game_id, pairs.iloc[0:0]), period_starts,
            int(parse_by_game.get(game_id, 0)), int(period_by_game.get(game_id, 0)),
        )
        quality_rows.append(quality)
        if quality["passed"]:
            for number, stint in enumerate(local, start=1):
                all_stints.append(
                    {
                        "stint_id": f"{game_id}_v3_{number:03d}",
                        "stint_number": number,
                        "season_label": quality["season_label"],
                        "season_type": quality["season_type"],
                        "game_date": quality["game_date"],
                        "home_team_id": quality["home_team_id"],
                        "away_team_id": quality["away_team_id"],
                        "substitution_source": "nbastatsv3_action_id_name_resolved",
                        **stint,
                    }
                )
    quality = pd.DataFrame(quality_rows)
    stints = pd.DataFrame(all_stints)
    if stints.empty:
        stints = pd.DataFrame(columns=["stint_id", "stint_number", "season_label", "season_type", "game_date", "home_team_id", "away_team_id", "substitution_source", "game_id", "start_action_id", "end_action_id_exclusive", "start_seconds_elapsed", "end_seconds_elapsed", "duration_seconds", *_LINEUP_COLUMNS])
    output_paths = {"stints": Path(stints_destination), "quality": Path(quality_destination), "report": Path(report_destination)}
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    stints.to_parquet(output_paths["stints"], index=False)
    quality.to_parquet(output_paths["quality"], index=False)
    source_files = [v3_path, player_path, scores_path]
    records = [{"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in source_files]
    passed_ids = quality.loc[quality["passed"], "game_id"].astype(str).tolist() if not quality.empty else []
    report = {
        "dataset": "historical_v3_ordinal_lineup_candidate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "production_status": "research_candidate_only",
        "project_season": project_season,
        "season_type": season_type,
        "physical_order_contract": "NBA Stats V3 actionId is the only event order; no clock-only join is used.",
        "lineup_contract": "Starters and official minutes come from separate player_games. Each substitution is replayed at its V3 actionId. Period reset requires exactly five V3-inferred players per team.",
        "event_coverage_contract": "V3 has no possession-owner field. Every V3 action must have exactly one ordinal lineup state; possession ownership is not claimed.",
        "minute_tolerance_seconds": 5.0,
        "source_game_count": int(len(scores)),
        "passed_game_count": len(passed_ids),
        "quarantined_game_count": int(len(scores) - len(passed_ids)),
        "passed_game_ids": passed_ids,
        "failure_reason_counts": quality.loc[~quality["passed"], "failure_reasons"].value_counts().to_dict() if not quality.empty else {},
        "stint_row_count": int(len(stints)),
        "v3_parse_failure_count": int(len(parse_failures)),
        "v3_period_start_failure_count": int(len(period_failures)),
        "source_files": records,
        "outputs": {key: str(path.resolve()) for key, path in output_paths.items() if key != "report"},
    }
    report["snapshot_id"] = "historical_v3_lineups_" + hashlib.sha256(
        json.dumps(records, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    report["passed"] = report["quarantined_game_count"] == 0
    write_json_atomic(report, output_paths["report"])
    write_json_atomic(report, Path(manifest_dir) / f"{report['snapshot_id']}.json")
    return report
