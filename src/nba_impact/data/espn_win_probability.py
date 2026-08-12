"""Resumable ingestion of ESPN play-level NBA win probabilities."""
from __future__ import annotations

import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from nba_impact.data.manifest import sha256_file, write_json_atomic


SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
TEAM_ALIASES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def canonical_team_abbreviation(value: object) -> str:
    abbreviation = str(value or "").strip().upper()
    return TEAM_ALIASES.get(abbreviation, abbreviation)


def parse_espn_clock(value: object) -> float:
    text = str(value).strip()
    if ":" not in text:
        return float(text)
    minutes, seconds = text.split(":", 1)
    return 60.0 * float(minutes) + float(seconds)


def _request_json(url: str, params: dict[str, str], *, attempts: int = 4) -> dict:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=(10, 45))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("ESPN response is not a JSON object")
            return payload
        except (requests.RequestException, ValueError) as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5 * 2**attempt)
    raise RuntimeError(f"ESPN request failed after {attempts} attempts: {error}") from error


def _write_gzip_json_atomic(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with gzip.open(partial, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    os.replace(partial, destination)


def read_gzip_json(path: str | Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _competition_teams(event: dict) -> tuple[str, str, int | None, int | None]:
    competition = event["competitions"][0]
    sides: dict[str, tuple[str, int | None]] = {}
    for competitor in competition["competitors"]:
        score = competitor.get("score")
        sides[str(competitor["homeAway"])] = (
            canonical_team_abbreviation(competitor["team"].get("abbreviation")),
            int(score) if score not in (None, "") else None,
        )
    home, home_score = sides["home"]
    away, away_score = sides["away"]
    return home, away, home_score, away_score


def match_scoreboard_games(games: pd.DataFrame, scoreboard: dict) -> list[dict]:
    """Match one local schedule date to ESPN events by teams, then verify final scores."""
    events = scoreboard.get("events", [])
    rows: list[dict] = []
    for game in games.itertuples(index=False):
        matches = []
        for event in events:
            home, away, home_score, away_score = _competition_teams(event)
            if home == canonical_team_abbreviation(game.home_team_tricode) and away == canonical_team_abbreviation(
                game.away_team_tricode
            ):
                matches.append((event, home_score, away_score))
        status = "matched"
        error = None
        espn_event_id = None
        if len(matches) != 1:
            status = "unmatched" if not matches else "ambiguous"
            error = f"expected one team match; found {len(matches)}"
        else:
            event, home_score, away_score = matches[0]
            espn_event_id = str(event["id"])
            if home_score != int(game.home_score) or away_score != int(game.away_score):
                status = "score_mismatch"
                error = f"ESPN {home_score}-{away_score}; local {game.home_score}-{game.away_score}"
        rows.append(
            {
                "game_id": str(game.game_id),
                "season_label": str(game.season_label),
                "game_date": pd.Timestamp(game.game_date),
                "home_team_tricode": str(game.home_team_tricode),
                "away_team_tricode": str(game.away_team_tricode),
                "espn_event_id": espn_event_id,
                "status": status,
                "error": error,
            }
        )
    return rows


def extract_espn_win_probability(payload: dict, *, game_id: str, season_label: str) -> pd.DataFrame:
    """Extract one row per ESPN play carrying a home-win probability."""
    probabilities = {
        str(row["playId"]): float(row["homeWinPercentage"])
        for row in payload.get("winprobability", [])
        if row.get("playId") is not None and row.get("homeWinPercentage") is not None
    }
    rows: list[dict[str, Any]] = []
    for play in payload.get("plays", []):
        play_id = str(play.get("id"))
        if play_id not in probabilities:
            continue
        rows.append(
            {
                "game_id": str(game_id),
                "season_label": str(season_label),
                "espn_play_id": play_id,
                "sequence_number": int(play.get("sequenceNumber", 0)),
                "period": int(play["period"]["number"]),
                "seconds_remaining_period_espn": parse_espn_clock(play["clock"]["displayValue"]),
                "home_score_after": int(play.get("homeScore", 0)),
                "away_score_after": int(play.get("awayScore", 0)),
                "espn_home_win_probability": probabilities[play_id],
                "description": play.get("text"),
            }
        )
    return pd.DataFrame(rows)


def ingest_espn_win_probability(
    game_dim_path: str | Path,
    *,
    season_labels: tuple[str, ...],
    raw_root: str | Path,
    index_output: str | Path,
    manifest_dir: str | Path,
    max_workers: int = 4,
) -> dict:
    """Discover ESPN game IDs by schedule date and cache raw summary payloads."""
    games = pd.read_parquet(game_dim_path)
    games = games.loc[games["season_label"].isin(season_labels)].copy()
    if games.empty:
        raise ValueError("No games found for the selected season labels.")
    if max_workers < 1 or max_workers > 16:
        raise ValueError("max_workers must be between 1 and 16")
    raw_root = Path(raw_root)
    scoreboard_root = raw_root / "scoreboards"
    summary_root = raw_root / "summaries"

    def scoreboard_for_date(date: pd.Timestamp) -> tuple[pd.Timestamp, dict]:
        stamp = pd.Timestamp(date).strftime("%Y%m%d")
        path = scoreboard_root / f"{stamp}.json.gz"
        if path.exists():
            return pd.Timestamp(date), read_gzip_json(path)
        payload = _request_json(SCOREBOARD_URL, {"dates": stamp, "limit": "100"})
        _write_gzip_json_atomic(payload, path)
        return pd.Timestamp(date), payload

    scoreboards: dict[pd.Timestamp, dict] = {}
    dates = sorted(pd.to_datetime(games["game_date"].drop_duplicates()))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(scoreboard_for_date, date) for date in dates]
        for future in as_completed(futures):
            date, payload = future.result()
            scoreboards[date] = payload

    index_rows: list[dict] = []
    for date, date_games in games.groupby("game_date", sort=True):
        index_rows.extend(match_scoreboard_games(date_games, scoreboards[pd.Timestamp(date)]))
    index = pd.DataFrame(index_rows)

    def fetch_summary(row: Any) -> tuple[str, str | None, str | None, int]:
        if row.status != "matched" or row.espn_event_id is None:
            return str(row.game_id), None, row.error, 0
        path = summary_root / f"season={row.season_label}" / f"game_id={row.game_id}.json.gz"
        try:
            payload = read_gzip_json(path) if path.exists() else _request_json(
                SUMMARY_URL, {"event": str(row.espn_event_id)}
            )
            if not payload.get("plays") or not payload.get("winprobability"):
                raise ValueError("summary is missing plays or winprobability")
            if not path.exists():
                _write_gzip_json_atomic(payload, path)
            count = len(extract_espn_win_probability(
                payload, game_id=str(row.game_id), season_label=str(row.season_label)
            ))
            return str(row.game_id), str(path.resolve()), None, count
        except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
            return str(row.game_id), None, str(exc), 0

    summaries: dict[str, tuple[str | None, str | None, int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch_summary, row) for row in index.itertuples(index=False)]
        for future in as_completed(futures):
            game_id, path, error, count = future.result()
            summaries[game_id] = (path, error, count)
    index["summary_path"] = index["game_id"].map(lambda game_id: summaries[str(game_id)][0])
    index["summary_error"] = index["game_id"].map(lambda game_id: summaries[str(game_id)][1])
    index["wp_play_rows"] = index["game_id"].map(lambda game_id: summaries[str(game_id)][2]).astype(int)
    summary_failed = index["status"].eq("matched") & index["summary_error"].notna()
    index.loc[summary_failed, "status"] = "summary_error"
    index_output = Path(index_output)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    index.to_parquet(index_output, index=False)

    counts = {str(key): int(value) for key, value in index["status"].value_counts().items()}
    snapshot = {
        "snapshot_id": f"espn_wp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "dataset": "espn_play_win_probability",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "season_labels": list(season_labels),
        "games_expected": int(len(games)),
        "status_counts": counts,
        "games_ready": int((index["status"] == "matched").sum()),
        "wp_play_rows": int(index["wp_play_rows"].sum()),
        "passed": bool((index["status"] == "matched").all()),
        "game_dim_sha256": sha256_file(game_dim_path),
        "index_path": str(index_output.resolve()),
        "index_sha256": sha256_file(index_output),
        "source": {"scoreboard_url": SCOREBOARD_URL, "summary_url": SUMMARY_URL},
    }
    destination = Path(manifest_dir) / f"{snapshot['snapshot_id']}.json"
    write_json_atomic(snapshot, destination)
    snapshot["manifest_path"] = str(destination.resolve())
    return snapshot
