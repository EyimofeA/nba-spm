"""Resumable annual NBA Stats close-defender dashboard ingestion."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import leaguedashptdefend
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from .manifest import sha256_file, write_json_atomic


def _season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def _fetch_dashboard(season: int, category: str, destination: Path, *, max_attempts: int) -> tuple[pd.DataFrame, dict]:
    if destination.exists():
        payload = json.loads(destination.read_text())
        status = "verified_existing"
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
                candidate = leaguedashptdefend.LeagueDashPtDefend(
                    defense_category=category, season=_season_label(season), timeout=300
                ).get_dict()
                result_sets = candidate.get("resultSets") or []
                if len(result_sets) != 1 or not (result_sets[0].get("data") or result_sets[0].get("rowSet")):
                    raise ValueError(f"{season}/{category}: empty or malformed NBA Stats response")
                payload = candidate
        if payload is None:
            raise RuntimeError(f"{season}/{category}: retry loop ended without a response")
        write_json_atomic(payload, destination)
        status = "downloaded"
    result = payload["resultSets"][0]
    frame = pd.DataFrame(result.get("data") or result.get("rowSet"), columns=result["headers"])
    if frame.empty:
        raise ValueError(f"{season}/{category}: cached NBA Stats response is empty")
    return frame, {"season": season, "category": category, "status": status, "path": str(destination.resolve()), "rows": int(len(frame)), "sha256": sha256_file(destination)}


def _to_source(frame: pd.DataFrame, season: int, *, rim: bool) -> pd.DataFrame:
    if rim:
        required = {"CLOSE_DEF_PERSON_ID", "PLAYER_NAME", "FGM_LT_06", "FGA_LT_06", "LT_06_PCT", "NS_LT_06_PCT", "PLUSMINUS"}
        if missing := required - set(frame.columns):
            raise ValueError(f"{season}/rim: missing {sorted(missing)}")
        return pd.DataFrame({
            "PLAYER": frame["PLAYER_NAME"], "PLAYER_ID": frame["CLOSE_DEF_PERSON_ID"],
            "DFGM": frame["FGM_LT_06"], "DFGA": frame["FGA_LT_06"],
            "DFG%": 100.0 * pd.to_numeric(frame["LT_06_PCT"], errors="coerce"),
            "FG%": 100.0 * pd.to_numeric(frame["NS_LT_06_PCT"], errors="coerce"),
            "DIFF%": 100.0 * pd.to_numeric(frame["PLUSMINUS"], errors="coerce"), "year": season,
        })
    required = {"CLOSE_DEF_PERSON_ID", "PLAYER_NAME", "D_FGM", "D_FGA", "D_FG_PCT", "NORMAL_FG_PCT", "PCT_PLUSMINUS"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{season}/overall: missing {sorted(missing)}")
    return pd.DataFrame({
        "PLAYER": frame["PLAYER_NAME"], "PLAYER_ID": frame["CLOSE_DEF_PERSON_ID"],
        "DFGM": frame["D_FGM"], "DFGA": frame["D_FGA"],
        "DFG%": 100.0 * pd.to_numeric(frame["D_FG_PCT"], errors="coerce"),
        "FG%": 100.0 * pd.to_numeric(frame["NORMAL_FG_PCT"], errors="coerce"),
        "DIFF%": 100.0 * pd.to_numeric(frame["PCT_PLUSMINUS"], errors="coerce"), "year": season,
    })


def ingest_official_defense_dashboards(
    seasons: tuple[int, ...], raw_root: str | Path, output_dir: str | Path, manifest_dir: str | Path,
    *, max_attempts: int = 20, minimum_delay_seconds: float = 0.2,
) -> dict:
    """Fetch official overall and rim close-defender tables for every requested season."""
    raw_root = Path(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    results = []
    overall_rows, rim_rows = [], []
    for season in seasons:
        for category, is_rim, rows in (("Overall", False, overall_rows), ("Less Than 6Ft", True, rim_rows)):
            destination = raw_root / f"season={season}" / f"{category.lower().replace(' ', '_')}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame, record = _fetch_dashboard(season, category, destination, max_attempts=max_attempts)
            rows.append(_to_source(frame, season, rim=is_rim))
            results.append(record)
            print(f"{record['status']:>17} {season} {category:<14} {record['rows']:4d} rows")
            if minimum_delay_seconds:
                time.sleep(minimum_delay_seconds)
    dfg, rim = pd.concat(overall_rows, ignore_index=True), pd.concat(rim_rows, ignore_index=True)
    for label, frame in (("dfg", dfg), ("rim_dfg", rim)):
        # NBA returns one final-team row per player and, for a small set of
        # trades, additional team rows.  The downstream feature builder
        # explicitly aggregates them after ID resolution.
        if frame[["DFGM", "DFGA", "DIFF%"]].isna().any().any():
            raise ValueError(f"{label}: invalid official dashboard rows")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dfg_path, rim_path = output_dir / "dfg.csv", output_dir / "rim_dfg.csv"
    dfg.to_csv(dfg_path, index=False)
    rim.to_csv(rim_path, index=False)
    identity = hashlib.sha256(json.dumps([(x["season"], x["category"], x["sha256"]) for x in results]).encode()).hexdigest()[:16]
    snapshot = {
        "snapshot_id": f"official_defense_dashboards_{identity}", "dataset": "official_nba_stats_close_defender_dashboards",
        "created_at": datetime.now(timezone.utc).isoformat(), "seasons": list(seasons),
        "source": "NBA Stats LeagueDashPtDefend", "passed": True, "results": results,
        "dfg_path": str(dfg_path.resolve()), "rim_dfg_path": str(rim_path.resolve()),
        "row_counts": {"dfg": int(len(dfg)), "rim_dfg": int(len(rim))},
        "sha256": {"dfg": sha256_file(dfg_path), "rim_dfg": sha256_file(rim_path)},
    }
    write_json_atomic(snapshot, output_dir / "run.json")
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
