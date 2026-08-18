"""Resumable NBA Stats V3 matchup ingestion and strict raw-row materialization."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import boxscorematchupsv3
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic
from .matchup_defense_features import RAW_COLUMNS


_STATISTICS = {
    "matchupMinutes": "matchup_minutes",
    "matchupMinutesSort": "matchup_minutes_sort",
    "partialPossessions": "partial_possessions",
    "percentageDefenderTotalTime": "percentage_defender_total_time",
    "percentageOffensiveTotalTime": "percentage_offensive_total_time",
    "percentageTotalTimeBothOn": "percentage_total_time_both_on",
    "switchesOn": "switches_on",
    "playerPoints": "player_points",
    "teamPoints": "team_points",
    "matchupAssists": "matchup_assists",
    "matchupPotentialAssists": "matchup_potential_assists",
    "matchupTurnovers": "matchup_turnovers",
    "matchupBlocks": "matchup_blocks",
    "matchupFieldGoalsMade": "matchup_field_goals_made",
    "matchupFieldGoalsAttempted": "matchup_field_goals_attempted",
    "matchupFieldGoalsPercentage": "matchup_field_goals_percentage",
    "matchupThreePointersMade": "matchup_three_pointers_made",
    "matchupThreePointersAttempted": "matchup_three_pointers_attempted",
    "matchupThreePointersPercentage": "matchup_three_pointers_percentage",
    "helpBlocks": "help_blocks",
    "helpFieldGoalsMade": "help_field_goals_made",
    "helpFieldGoalsAttempted": "help_field_goals_attempted",
    "helpFieldGoalsPercentage": "help_field_goals_percentage",
    "matchupFreeThrowsMade": "matchup_free_throws_made",
    "matchupFreeThrowsAttempted": "matchup_free_throws_attempted",
    "shootingFouls": "shooting_fouls",
}


def _validated_payload(payload: dict, game_id: str) -> dict:
    box = payload.get("boxScoreMatchups")
    if not isinstance(box, dict) or canonical_game_id(box.get("gameId")) != game_id:
        raise ValueError(f"{game_id}: response does not contain the requested matchup boxscore")
    for side in ("homeTeam", "awayTeam"):
        team = box.get(side)
        if not isinstance(team, dict) or not team.get("teamId") or not isinstance(team.get("players"), list):
            raise ValueError(f"{game_id}: incomplete {side} matchup boxscore")
    return box


def flatten_official_matchups(payload: dict, game_id: str) -> pd.DataFrame:
    """Convert one NBA V3 payload to the existing matchup-feature raw contract."""
    box = _validated_payload(payload, game_id)
    rows: list[dict] = []
    for side in ("homeTeam", "awayTeam"):
        team = box[side]
        for player in team["players"]:
            person_id = player.get("personId")
            if person_id is None:
                raise ValueError(f"{game_id}: player row has no personId")
            for matchup in player.get("matchups") or []:
                defender_id = matchup.get("personId")
                statistics = matchup.get("statistics")
                if defender_id is None or not isinstance(statistics, dict):
                    raise ValueError(f"{game_id}: malformed matchup row")
                row = {
                    "game_id": game_id,
                    "away_team_id": int(box["awayTeamId"]),
                    "home_team_id": int(box["homeTeamId"]),
                    "team_id": int(team["teamId"]),
                    "person_id": int(person_id),
                    "matchups_person_id": int(defender_id),
                }
                for source, destination in _STATISTICS.items():
                    row[destination] = statistics.get(source)
                rows.append(row)
    frame = pd.DataFrame(rows)
    missing = set(RAW_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"{game_id}: flattened matchup rows missing {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{game_id}: official matchup payload has no matchup rows")
    if frame.duplicated(["game_id", "person_id", "matchups_person_id"]).any():
        raise ValueError(f"{game_id}: duplicate player-defender matchup rows")
    return frame


def _fetch_game(game_id: str, destination: Path, *, max_attempts: int) -> dict:
    status = "verified_existing"
    if destination.exists():
        payload = json.loads(destination.read_text())
        _validated_payload(payload, game_id)
    else:
        retrying = Retrying(
            retry=retry_if_exception_type((requests.RequestException, json.JSONDecodeError, ValueError)),
            wait=wait_exponential_jitter(initial=2, max=300),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )
        payload = None
        for attempt in retrying:
            with attempt:
                candidate = boxscorematchupsv3.BoxScoreMatchupsV3(game_id=game_id, timeout=300).get_dict()
                _validated_payload(candidate, game_id)
                flatten_official_matchups(candidate, game_id)
                payload = candidate
        if payload is None:
            raise RuntimeError(f"{game_id}: retry loop ended without a response")
        write_json_atomic(payload, destination)
        status = "downloaded"
    return {"game_id": game_id, "status": status, "path": str(destination.resolve()), "bytes": destination.stat().st_size, "sha256": sha256_file(destination)}


def ingest_official_matchups(
    game_ids: list[str] | tuple[str, ...], raw_root: str | Path, output_path: str | Path,
    manifest_dir: str | Path, *, season: int, season_type: str, max_attempts: int = 20,
    minimum_delay_seconds: float = 0.2, max_workers: int = 2,
) -> dict:
    """Fetch one season of official matchup data with atomic, resumable raw JSON."""
    if max_workers < 1:
        raise ValueError("max_workers must be at least one")
    normalized = sorted({canonical_game_id(value) for value in game_ids})
    if not normalized:
        raise ValueError("At least one game ID is required")
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    def fetch_one(game_id: str) -> dict:
        return _fetch_game(game_id, raw_root / f"game_id={game_id}.json", max_attempts=max_attempts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, game_id): game_id for game_id in normalized}
        for future in as_completed(futures):
            game_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                results.append({"game_id": game_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                print(f"{'failed':>17} {game_id} {type(exc).__name__}: {str(exc)[:160]}")
            else:
                results.append(result)
                print(f"{result['status']:>17} {game_id} {result['bytes'] / 1000:7.1f} KB")
            if minimum_delay_seconds > 0:
                time.sleep(minimum_delay_seconds)

    results.sort(key=lambda item: item["game_id"])
    failed = [item for item in results if item["status"] == "failed"]
    output = Path(output_path)
    output_hash = None
    output_rows = 0
    if not failed:
        frames = []
        for item in results:
            payload = json.loads(Path(item["path"]).read_text())
            frames.append(flatten_official_matchups(payload, item["game_id"]))
        panel = pd.concat(frames, ignore_index=True)
        panel["_season"] = int(season)
        panel["_season_type"] = str(season_type)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".partial")
        panel.to_parquet(temporary, index=False)
        temporary.replace(output)
        output_hash = sha256_file(output)
        output_rows = int(len(panel))
    identity = hashlib.sha256(json.dumps([(item["game_id"], item["status"], item.get("sha256")) for item in results]).encode()).hexdigest()[:16]
    snapshot = {
        "snapshot_id": f"official_matchups_{identity}", "dataset": "official_nba_stats_v3_matchups",
        "grain": "offensive-player by defender within game", "created_at": datetime.now(timezone.utc).isoformat(),
        "season": int(season), "season_type": str(season_type), "source": "NBA Stats BoxScoreMatchupsV3",
        "requested_game_count": len(normalized), "downloaded_game_count": sum(item["status"] == "downloaded" for item in results),
        "verified_existing_game_count": sum(item["status"] == "verified_existing" for item in results),
        "failed_game_count": len(failed), "output_path": str(output.resolve()), "output_sha256": output_hash,
        "output_rows": output_rows, "max_attempts": max_attempts, "max_workers": max_workers,
        "passed": not failed and output_rows > 0, "results": results,
    }
    if snapshot["passed"]:
        write_json_atomic(snapshot, Path(f"{output}.manifest.json"))
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    if failed:
        raise RuntimeError(f"{len(failed)} official matchup downloads failed; rerun resumes completed games")
    return snapshot
