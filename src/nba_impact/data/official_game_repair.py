"""Download official NBA play-by-play and box scores for a finite repair queue."""

from __future__ import annotations

import hashlib
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .manifest import write_json_atomic


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json,text/plain,*/*",
}
BASE = "https://stats.nba.com/stats"


def _url(endpoint: str, game_id: str) -> str:
    params = {"GameID": game_id, "StartPeriod": 0, "EndPeriod": 14}
    if endpoint == "boxscoretraditionalv3":
        params.update({"StartRange": 0, "EndRange": 0, "RangeType": 0})
    return f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"


def _request_json(url: str, *, retries: int, timeout: float) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(60.0, (2.0**attempt) + random.random()))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last_error}")


def _validate(endpoint: str, payload: dict, game_id: str) -> int:
    if endpoint == "playbyplayv3":
        actions = payload.get("game", {}).get("actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"{game_id}: V3 play-by-play has no actions")
        periods = {int(row.get("period", 0) or 0) for row in actions}
        if max(periods, default=0) < 4:
            raise ValueError(f"{game_id}: V3 play-by-play ends before period four")
        return len(actions)
    box = payload.get("boxScoreTraditional")
    if not isinstance(box, dict):
        raise ValueError(f"{game_id}: V3 traditional box score is missing")
    teams = [box.get("homeTeam"), box.get("awayTeam")]
    if any(not isinstance(team, dict) for team in teams):
        raise ValueError(f"{game_id}: V3 traditional box score lacks two teams")
    return sum(len(team.get("players") or []) for team in teams)


def _write_payload(path: Path, payload: dict) -> tuple[int, str]:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _valid_existing(path: Path, endpoint: str, game_id: str) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        rows = _validate(endpoint, payload, game_id)
        raw = path.read_bytes()
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "rows": rows}
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _fetch_game(
    game_id: str,
    season: int,
    output_root: Path,
    *,
    retries: int,
    timeout: float,
) -> dict:
    files: dict[str, dict] = {}
    for endpoint, label in (("playbyplayv3", "play_by_play"), ("boxscoretraditionalv3", "box_score")):
        path = output_root / f"season={season}" / label / f"{game_id}.json"
        existing = _valid_existing(path, endpoint, game_id)
        if existing is not None:
            files[label] = {**existing, "relative_path": str(path.relative_to(output_root)), "resumed": True}
            continue
        payload = _request_json(_url(endpoint, game_id), retries=retries, timeout=timeout)
        rows = _validate(endpoint, payload, game_id)
        size, digest = _write_payload(path, payload)
        files[label] = {
            "bytes": size,
            "sha256": digest,
            "rows": rows,
            "relative_path": str(path.relative_to(output_root)),
            "resumed": False,
        }
    return {
        "game_id": game_id,
        "season": season,
        "status": "ok",
        "files": files,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def download_repair_queue(
    repair_queue_path: str | Path,
    output_root: str | Path,
    *,
    workers: int = 4,
    retries: int = 6,
    timeout: float = 45.0,
) -> dict:
    """Download and verify every game in a canonical repair queue."""
    queue_path = Path(repair_queue_path)
    output = Path(output_root)
    queue = pd.read_parquet(queue_path, columns=["season", "game_id"])
    queue["game_id"] = queue["game_id"].astype(str).str.zfill(10)
    queue = queue.drop_duplicates(["season", "game_id"]).sort_values(["season", "game_id"])
    results: list[dict] = []
    failures: list[dict] = []
    output.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _fetch_game,
                row.game_id,
                int(row.season),
                output,
                retries=retries,
                timeout=timeout,
            ): (int(row.season), row.game_id)
            for row in queue.itertuples(index=False)
        }
        for number, future in enumerate(as_completed(futures), start=1):
            season, game_id = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # each failed game remains visible and resumable
                failures.append({"season": season, "game_id": game_id, "error": str(error)})
            if number % 10 == 0 or number == len(futures):
                print(f"official repair progress {number}/{len(futures)} failures={len(failures)}", flush=True)

    manifest = {
        "schema_version": "official_game_repair_v1",
        "source_queue": str(queue_path.resolve()),
        "requested_games": int(len(queue)),
        "completed_games": len(results),
        "failed_games": len(failures),
        "games": sorted(results, key=lambda row: (row["season"], row["game_id"])),
        "failures": sorted(failures, key=lambda row: (row["season"], row["game_id"])),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(manifest, output / "manifest.json")
    if failures:
        raise RuntimeError(f"Official repair download left {len(failures)} failed games; rerun to resume")
    return manifest
