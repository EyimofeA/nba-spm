"""Reconstruct validated five-player lineup stints from starter seeds and substitutions."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic


_CLOCK = re.compile(r"^PT(?:(\d+)M)?(?:([0-9.]+)S)?$")
_V3_SUB = re.compile(r"^SUB:\s*(.+?)\s+FOR\s+(.+?)$", re.IGNORECASE)


def _clock_seconds(value: object) -> float:
    match = _CLOCK.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"Unsupported game clock: {value!r}")
    minutes, seconds = match.groups()
    return float(minutes or 0) * 60.0 + float(seconds or 0)


def _elapsed_seconds(period: int, clock: object) -> float:
    period_length = 720.0 if period <= 4 else 300.0
    prior = min(period - 1, 4) * 720.0 + max(period - 5, 0) * 300.0
    return prior + period_length - _clock_seconds(clock)


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def _load_cdn_substitutions(root: Path) -> tuple[pd.DataFrame, set[str], list[Path]]:
    frames: list[pd.DataFrame] = []
    game_ids: set[str] = set()
    paths = sorted((root / "cdnnba").rglob("*.parquet"))
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=["gameId", "period", "clock", "actionType", "description", "personId", "teamId"],
        )
        frame["game_id"] = frame["gameId"].map(canonical_game_id)
        game_ids.update(frame["game_id"].unique().tolist())
        frame = frame.loc[frame["actionType"].astype(str).str.casefold().eq("substitution")].copy()
        frame["direction"] = frame["description"].astype(str).str.extract(r"^SUB\s+(in|out):", expand=False)
        frame["player_id"] = pd.to_numeric(frame["personId"], errors="coerce").astype("Int64")
        frame["team_id"] = pd.to_numeric(frame["teamId"], errors="coerce").astype("Int64")
        frame["substitution_source"] = "cdnnba_exact"
        frames.append(
            frame[["game_id", "period", "clock", "team_id", "player_id", "direction", "substitution_source"]]
        )
    if not paths:
        raise ValueError(f"No CDN NBA event partitions found under {root / 'cdnnba'}")
    return pd.concat(frames, ignore_index=True), game_ids, paths


def _player_aliases(player_games: pd.DataFrame, game_ids: set[str]) -> dict[tuple[str, int, str], set[int]]:
    aliases: dict[tuple[str, int, str], set[int]] = defaultdict(set)
    subset = player_games.loc[player_games["game_id"].isin(game_ids)]
    for row in subset.itertuples(index=False):
        first = str(row.first_name or "")
        family = str(row.family_name or "")
        values = (family, row.player_name, f"{first[:1]} {family}", f"{first[:3]} {family}")
        for value in values:
            aliases[(row.game_id, int(row.team_id), _normalize_name(value))].add(int(row.player_id))
    return aliases


def _load_v3_fallback_substitutions(
    root: Path, game_ids: set[str], player_games: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, int, int], set[int]], list[Path], list[dict]]:
    frames: list[pd.DataFrame] = []
    paths = sorted((root / "nbastatsv3").rglob("*.parquet"))
    for path in paths:
        frame = pd.read_parquet(
            path,
            columns=[
                "gameId",
                "actionId",
                "period",
                "clock",
                "actionType",
                "description",
                "personId",
                "teamId",
            ],
        )
        frame["game_id"] = frame["gameId"].map(canonical_game_id)
        frame = frame.loc[frame["game_id"].isin(game_ids)].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(), {}, paths, []
    event_source = pd.concat(frames, ignore_index=True)
    source = event_source.loc[
        event_source["actionType"].astype(str).str.casefold().eq("substitution")
    ].copy()
    aliases = _player_aliases(player_games, game_ids)
    output: list[dict] = []
    failures: list[dict] = []
    for row in source.itertuples(index=False):
        match = _V3_SUB.fullmatch(str(row.description).strip())
        if not match:
            failures.append({"game_id": row.game_id, "description": str(row.description), "reason": "parse"})
            continue
        incoming_name, _ = match.groups()
        key = (row.game_id, int(row.teamId), _normalize_name(incoming_name))
        incoming_ids = aliases.get(key, set())
        if len(incoming_ids) != 1:
            failures.append(
                {
                    "game_id": row.game_id,
                    "description": str(row.description),
                    "reason": f"incoming_alias_matches_{len(incoming_ids)}",
                }
            )
            continue
        common = {
            "game_id": row.game_id,
            "period": int(row.period),
            "clock": row.clock,
            "team_id": int(row.teamId),
            "substitution_source": "nbastatsv3_name_resolved",
        }
        output.append({**common, "player_id": int(row.personId), "direction": "out"})
        output.append({**common, "player_id": next(iter(incoming_ids)), "direction": "in"})
    output_frame = pd.DataFrame(output)
    period_starts: dict[tuple[str, int, int], set[int]] = {}
    if not output_frame.empty:
        output_frame["elapsed_seconds"] = [
            _elapsed_seconds(int(period), clock)
            for period, clock in zip(output_frame["period"], output_frame["clock"], strict=True)
        ]
    valid_events = event_source.loc[
        event_source["teamId"].notna()
        & event_source["personId"].notna()
        & event_source["teamId"].gt(0)
        & event_source["personId"].gt(0)
    ]
    for (game_id, team_id, period), event_group in valid_events.groupby(
        ["game_id", "teamId", "period"]
    ):
        group_subs = output_frame.loc[
            output_frame["game_id"].eq(game_id)
            & output_frame["team_id"].eq(int(team_id))
            & output_frame["period"].eq(int(period))
        ]
        first_direction: dict[int, str] = {}
        for row in group_subs.sort_values(
            ["elapsed_seconds", "direction"], ascending=[True, False]
        ).itertuples(index=False):
            first_direction.setdefault(int(row.player_id), str(row.direction))
        first_out = {player_id for player_id, direction in first_direction.items() if direction == "out"}
        no_sub_activity = set(
            event_group.loc[
                ~event_group["actionType"].astype(str).str.casefold().eq("substitution"), "personId"
            ].astype(int)
        ) - set(first_direction)
        candidates = first_out | no_sub_activity
        if len(candidates) == 5:
            period_starts[(str(game_id), int(team_id), int(period))] = candidates
        elif int(period) > 1:
            failures.append(
                {
                    "game_id": str(game_id),
                    "team_id": int(team_id),
                    "period": int(period),
                    "reason": f"period_starter_candidates_{len(candidates)}",
                }
            )
    output_frame = output_frame.drop(columns="elapsed_seconds", errors="ignore")
    return output_frame, period_starts, paths, failures


def _substitution_groups(substitutions: pd.DataFrame) -> dict[str, dict[float, list[tuple[int, set[int], set[int]]]]]:
    substitutions = substitutions.copy()
    substitutions["elapsed_seconds"] = [
        _elapsed_seconds(int(period), clock)
        for period, clock in zip(substitutions["period"], substitutions["clock"], strict=True)
    ]
    games: dict[str, dict[float, list[tuple[int, set[int], set[int]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (game_id, elapsed, team_id), group in substitutions.groupby(
        ["game_id", "elapsed_seconds", "team_id"], sort=True
    ):
        outs = set(group.loc[group["direction"].eq("out"), "player_id"].dropna().astype(int))
        ins = set(group.loc[group["direction"].eq("in"), "player_id"].dropna().astype(int))
        games[str(game_id)][float(elapsed)].append((int(team_id), outs, ins))
    return games


def build_lineup_stints(
    event_root: str | Path,
    player_games_path: str | Path,
    game_dim_path: str | Path,
    destination: str | Path,
    quality_destination: str | Path,
    manifest_dir: str | Path,
    *,
    minute_tolerance_seconds: float = 5.0,
    max_quarantine_fraction: float = 0.005,
) -> dict:
    """Build stints and quarantine any game whose lineup minutes do not reconcile."""
    root = Path(event_root)
    player_source = Path(player_games_path)
    game_source = Path(game_dim_path)
    players = pd.read_parquet(player_source)
    games = pd.read_parquet(game_source)
    cdn_subs, cdn_games, cdn_paths = _load_cdn_substitutions(root)
    target_games = set(games["game_id"])
    fallback_game_ids = target_games - cdn_games
    fallback_subs, fallback_period_starts, v3_paths, fallback_failures = _load_v3_fallback_substitutions(
        root, fallback_game_ids, players
    )
    substitutions = pd.concat([cdn_subs, fallback_subs], ignore_index=True)
    substitutions = substitutions.loc[substitutions["game_id"].isin(target_games)].copy()
    invalid_direction_rows = int((~substitutions["direction"].isin(["in", "out"])).sum())
    substitution_groups = _substitution_groups(substitutions)

    starters: dict[tuple[str, int], set[int]] = {}
    for (game_id, team_id), group in players.loc[players["starter"]].groupby(["game_id", "team_id"]):
        starters[(str(game_id), int(team_id))] = set(group["player_id"].astype(int))
    expected_by_game: dict[str, dict[int, float]] = defaultdict(dict)
    for row in players.itertuples(index=False):
        expected_by_game[str(row.game_id)][int(row.player_id)] = float(row.minutes_seconds)

    all_stints: list[dict] = []
    game_quality: list[dict] = []
    fallback_failure_games = {item["game_id"] for item in fallback_failures}
    for game in games.sort_values(["game_date", "game_id"]).itertuples(index=False):
        game_id = str(game.game_id)
        home_id = int(game.home_team_id)
        away_id = int(game.away_team_id)
        lineups = {
            home_id: set(starters.get((game_id, home_id), set())),
            away_id: set(starters.get((game_id, away_id), set())),
        }
        end_seconds = 2880.0 + max(int(game.max_period) - 4, 0) * 300.0
        transitions_invalid = 0
        local_stints: list[dict] = []
        previous = 0.0

        def append_stint(end: float) -> None:
            nonlocal previous
            if end <= previous:
                return
            home = sorted(lineups[home_id])
            away = sorted(lineups[away_id])
            if len(home) != 5 or len(away) != 5:
                return
            local_stints.append(
                {
                    "game_id": game_id,
                    "start_seconds_elapsed": previous,
                    "end_seconds_elapsed": end,
                    "duration_seconds": end - previous,
                    **{f"home_player_{index + 1}": player for index, player in enumerate(home)},
                    **{f"away_player_{index + 1}": player for index, player in enumerate(away)},
                }
            )
            previous = end

        game_changes = substitution_groups.get(game_id, {})
        period_boundaries: dict[float, int] = {}
        if game_id in fallback_game_ids:
            for period in range(2, int(game.max_period) + 1):
                boundary = min(period - 1, 4) * 720.0 + max(period - 5, 0) * 300.0
                period_boundaries[boundary] = period
        for elapsed in sorted(set(game_changes) | set(period_boundaries)):
            if elapsed < previous or elapsed > end_seconds:
                transitions_invalid += 1
                continue
            append_stint(elapsed)
            if elapsed in period_boundaries:
                period = period_boundaries[elapsed]
                for team_id in (home_id, away_id):
                    inferred = fallback_period_starts.get((game_id, team_id, period))
                    if inferred is None:
                        transitions_invalid += 1
                    else:
                        lineups[team_id] = set(inferred)
            for team_id, outs, ins in game_changes.get(elapsed, []):
                if team_id not in lineups:
                    transitions_invalid += 1
                    continue
                out_only = outs - ins
                in_only = ins - outs
                if not out_only.issubset(lineups[team_id]) or in_only.intersection(lineups[team_id]):
                    transitions_invalid += 1
                lineups[team_id] = (lineups[team_id] - out_only) | in_only
                if len(lineups[team_id]) != 5:
                    transitions_invalid += 1
        append_stint(end_seconds)

        actual = defaultdict(float)
        for stint in local_stints:
            for side in ("home", "away"):
                for index in range(1, 6):
                    actual[(game_id, int(stint[f"{side}_player_{index}"]))] += float(
                        stint["duration_seconds"]
                    )
        expected_minutes = expected_by_game.get(game_id, {})
        observed_minutes = {
            player_id: seconds for (observed_game, player_id), seconds in actual.items() if observed_game == game_id
        }
        minute_errors = [
            abs(observed_minutes.get(player_id, 0.0) - expected_minutes.get(player_id, 0.0))
            for player_id in set(expected_minutes) | set(observed_minutes)
        ]
        max_minute_error = max(minute_errors, default=float("inf"))
        duration_covered = sum(row["duration_seconds"] for row in local_stints)
        starter_valid = len(starters.get((game_id, home_id), set())) == 5 and len(
            starters.get((game_id, away_id), set())
        ) == 5
        source = "cdnnba_exact" if game_id in cdn_games else "nbastatsv3_name_resolved"
        passed = bool(
            starter_valid
            and game_id not in fallback_failure_games
            and transitions_invalid == 0
            and abs(duration_covered - end_seconds) < 1e-6
            and max_minute_error <= minute_tolerance_seconds
        )
        game_quality.append(
            {
                "game_id": game_id,
                "season_label": game.season_label,
                "season_type": game.season_type,
                "game_date": game.game_date,
                "substitution_source": source,
                "starter_valid": starter_valid,
                "transition_errors": transitions_invalid,
                "fallback_resolution_errors": sum(
                    item["game_id"] == game_id for item in fallback_failures
                ),
                "stint_count": len(local_stints),
                "covered_seconds": duration_covered,
                "game_seconds": end_seconds,
                "max_player_minute_error": max_minute_error,
                "passed": passed,
            }
        )
        if passed:
            for number, stint in enumerate(local_stints, start=1):
                all_stints.append(
                    {
                        "stint_id": f"{game_id}_{number:03d}",
                        "stint_number": number,
                        "season_label": game.season_label,
                        "season_type": game.season_type,
                        "game_date": game.game_date,
                        "home_team_id": home_id,
                        "away_team_id": away_id,
                        "substitution_source": source,
                        **stint,
                    }
                )

    quality = pd.DataFrame(game_quality)
    stints = pd.DataFrame(all_stints)
    output = Path(destination)
    quality_output = Path(quality_destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    quality_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    quality_temporary = quality_output.with_suffix(quality_output.suffix + ".partial")
    stints.to_parquet(temporary, index=False)
    quality.to_parquet(quality_temporary, index=False)
    temporary.replace(output)
    quality_temporary.replace(quality_output)

    home_columns = [f"home_player_{index}" for index in range(1, 6)]
    away_columns = [f"away_player_{index}" for index in range(1, 6)]
    invalid_lineup_rows = 0
    if not stints.empty:
        invalid_lineup_rows = int(
            sum(
                len(set(row[home_columns])) != 5
                or len(set(row[away_columns])) != 5
                or bool(set(row[home_columns]).intersection(row[away_columns]))
                for _, row in stints.iterrows()
            )
        )
    issue_counts = {
        "invalid_substitution_direction_rows": invalid_direction_rows,
        "fallback_resolution_errors": len(fallback_failures),
        "failed_games": int((~quality["passed"]).sum()),
        "starter_failures": int((~quality["starter_valid"]).sum()),
        "games_with_transition_errors": int(quality["transition_errors"].gt(0).sum()),
        "games_with_minute_mismatch": int(
            quality["max_player_minute_error"].gt(minute_tolerance_seconds).sum()
        ),
        "duplicate_stint_ids": int(stints.duplicated("stint_id", keep=False).sum()),
        "nonpositive_stint_durations": int(stints["duration_seconds"].le(0).sum()),
        "invalid_output_lineup_rows": invalid_lineup_rows,
    }
    quarantine_fraction = float((~quality["passed"]).mean())
    passed = bool(
        len(quality) == len(games)
        and quarantine_fraction <= max_quarantine_fraction
        and issue_counts["invalid_substitution_direction_rows"] == 0
        and issue_counts["duplicate_stint_ids"] == 0
        and issue_counts["nonpositive_stint_durations"] == 0
        and issue_counts["invalid_output_lineup_rows"] == 0
    )
    source_paths = cdn_paths + v3_paths + [player_source, game_source]
    source_records = [
        {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in source_paths
    ]
    identity = hashlib.sha256(
        json.dumps([(item["path"], item["sha256"]) for item in source_records]).encode("utf-8")
    ).hexdigest()[:16]
    snapshot = {
        "snapshot_id": f"lineup_stints_{identity}",
        "dataset": "lineup_stints",
        "grain": "constant ten-player on-court interval",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "row_count": int(len(stints)),
        "source_game_count": int(len(games)),
        "passed_game_count": int(quality["passed"].sum()),
        "quarantined_game_count": int((~quality["passed"]).sum()),
        "quarantine_fraction": quarantine_fraction,
        "max_quarantine_fraction": max_quarantine_fraction,
        "source_counts": {str(key): int(value) for key, value in quality["substitution_source"].value_counts().items()},
        "minute_tolerance_seconds": minute_tolerance_seconds,
        "validation_policy": (
            "Only fully reconciled games are emitted. The snapshot passes when every source game is audited, "
            "all emitted stints have valid ten-player lineups, and the quarantined share is at most the configured limit."
        ),
        "issues": issue_counts,
        "fallback_failures_sample": fallback_failures[:20],
        "path": str(output.resolve()),
        "quality_path": str(quality_output.resolve()),
        "source_files": source_records,
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
