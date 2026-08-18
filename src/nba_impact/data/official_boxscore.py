"""Resumable per-game NBA Stats boxscore repairs for quarantined lineups."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import boxscoretraditionalv3
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .game_dim import canonical_game_id
from .manifest import sha256_file, write_json_atomic


def _validated_box(payload: dict, game_id: str) -> dict:
    box = payload.get("boxScoreTraditional")
    if not isinstance(box, dict) or canonical_game_id(box.get("gameId")) != game_id:
        raise ValueError(f"{game_id}: response does not contain the requested boxscore")
    for side in ("homeTeam", "awayTeam"):
        team = box.get(side)
        if not isinstance(team, dict) or not team.get("teamId") or not team.get("players"):
            raise ValueError(f"{game_id}: incomplete {side} boxscore")
    return box


def load_official_boxscore_rows(root: str | Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load cached repair JSON into the canonical NBA-box input columns."""
    records: list[dict] = []
    paths = sorted(Path(root).glob("game_id=*.json"))
    for path in paths:
        game_id = canonical_game_id(path.stem.split("=", maxsplit=1)[1])
        box = _validated_box(json.loads(path.read_text()), game_id)
        for side in ("homeTeam", "awayTeam"):
            team = box[side]
            for player in team["players"]:
                statistics = player.get("statistics") or {}
                minutes = statistics.get("minutes") or ""
                records.append(
                    {
                        "gameId": game_id,
                        "game_id": game_id,
                        "team_id": int(team["teamId"]),
                        "player_id": int(player["personId"]),
                        "first_name": str(player.get("firstName") or ""),
                        "family_name": str(player.get("familyName") or ""),
                        "starter_position": str(player.get("position") or "").strip(),
                        "comment": str(player.get("comment") or ""),
                        "minutes": minutes,
                        "player_game_source": "nba_stats_boxscore_v3_repair",
                    }
                )
    return pd.DataFrame(records), paths


def ingest_official_boxscores(
    game_ids: list[str] | tuple[str, ...],
    root: str | Path,
    manifest_dir: str | Path,
    *,
    max_attempts: int = 20,
    minimum_delay_seconds: float = 0.6,
) -> dict:
    """Cache official boxscores atomically, skipping already-valid games.

    A failed game is recorded and does not stop the rest of a large resumable
    batch. The returned snapshot fails until every requested game is valid.
    """
    destination_root = Path(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    normalized_ids = sorted({canonical_game_id(game_id) for game_id in game_ids})
    for game_id in normalized_ids:
        destination = destination_root / f"game_id={game_id}.json"
        try:
            status = "verified_existing"
            if destination.exists():
                payload = json.loads(destination.read_text())
                _validated_box(payload, game_id)
            else:
                retrying = Retrying(
                    retry=retry_if_exception_type(
                        (requests.RequestException, json.JSONDecodeError, ValueError)
                    ),
                    wait=wait_exponential_jitter(initial=2, max=300),
                    stop=stop_after_attempt(max_attempts),
                    reraise=True,
                )
                payload = None
                for attempt in retrying:
                    with attempt:
                        response = boxscoretraditionalv3.BoxScoreTraditionalV3(
                            game_id=game_id, timeout=300
                        ).get_dict()
                        _validated_box(response, game_id)
                        payload = response
                if payload is None:
                    raise RuntimeError(f"{game_id}: retry loop ended without a response")
                write_json_atomic(payload, destination)
                status = "downloaded"
                time.sleep(minimum_delay_seconds)
            results.append(
                {
                    "game_id": game_id,
                    "status": status,
                    "path": str(destination.resolve()),
                    "bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
            print(f"{status:>17} {game_id} {destination.stat().st_size / 1000:7.1f} KB")
        except (requests.RequestException, json.JSONDecodeError, ValueError) as exc:
            results.append(
                {
                    "game_id": game_id,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
            )
            print(f"{'failed':>17} {game_id} {type(exc).__name__}: {str(exc)[:160]}")

    identity = hashlib.sha256(
        json.dumps(
            [(item["game_id"], item["status"], item.get("sha256")) for item in results]
        ).encode("utf-8")
    ).hexdigest()[:16]
    failed_count = sum(item["status"] == "failed" for item in results)
    snapshot = {
        "snapshot_id": f"official_boxscores_{identity}",
        "dataset": "official_nba_boxscore_repairs",
        "grain": "one official NBA Stats V3 game boxscore JSON",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": len(results) == len(normalized_ids) and failed_count == 0,
        "requested_game_count": len(normalized_ids),
        "downloaded_game_count": sum(item["status"] == "downloaded" for item in results),
        "verified_existing_game_count": sum(
            item["status"] == "verified_existing" for item in results
        ),
        "failed_game_count": failed_count,
        "max_attempts": max_attempts,
        "minimum_delay_seconds": minimum_delay_seconds,
        "source": "NBA Stats BoxScoreTraditionalV3",
        "results": results,
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
