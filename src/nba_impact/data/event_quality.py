"""Source-aware audits and cross-source reconciliation for event Parquets."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .manifest import sha256_file


SOURCE_CONTRACTS = {
    "cdnnba": {
        "game": "gameId",
        "key": ("gameId", "orderNumber"),
        "required": (
            "gameId",
            "orderNumber",
            "actionNumber",
            "period",
            "clock",
            "possession",
            "_season",
            "_season_type",
        ),
    },
    "nbastatsv3": {
        "game": "gameId",
        "key": ("gameId", "actionId"),
        "required": ("gameId", "actionId", "actionNumber", "period", "clock", "_season", "_season_type"),
    },
    "pbpstats": {
        "game": "GAMEID",
        "key": (),
        "required": ("GAMEID", "PERIOD", "STARTTIME", "ENDTIME", "EVENTS", "_season", "_season_type"),
    },
    "shotdetail": {
        "game": "GAME_ID",
        "key": ("GAME_ID", "GAME_EVENT_ID"),
        "required": ("GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID", "PERIOD", "_season", "_season_type"),
    },
    "matchups": {
        "game": "game_id",
        "key": ("game_id", "person_id", "matchups_person_id"),
        "required": (
            "game_id",
            "away_team_id",
            "home_team_id",
            "team_id",
            "person_id",
            "matchups_person_id",
            "_season",
            "_season_type",
        ),
    },
}


def _partition_identity(path: Path) -> tuple[str, int, str]:
    source = path.parents[1].name
    season = int(path.parent.name.split("=", maxsplit=1)[1])
    season_type = path.stem
    return source, season, season_type


def audit_event_file(path: str | Path) -> tuple[dict, set[int]]:
    source_path = Path(path)
    source, season, season_type = _partition_identity(source_path)
    contract = SOURCE_CONTRACTS.get(source)
    report = {
        "source": source,
        "season": season,
        "season_type": season_type,
        "path": str(source_path.resolve()),
        "bytes": source_path.stat().st_size,
        "sha256": sha256_file(source_path),
        "row_count": 0,
        "game_count": 0,
        "issues": [],
    }
    if contract is None:
        report["issues"].append({"severity": "high", "code": "unknown_source_contract", "count": 1})
        report["passed"] = False
        return report, set()

    schema_columns = set(pq.ParquetFile(source_path).schema_arrow.names)
    missing = sorted(set(contract["required"]) - schema_columns)
    if missing:
        report["issues"].append(
            {"severity": "critical", "code": "missing_required_columns", "count": len(missing), "columns": missing}
        )
        report["passed"] = False
        return report, set()

    selected = list(dict.fromkeys((*contract["required"], *contract["key"])))
    frame = pd.read_parquet(source_path, columns=selected)
    report["row_count"] = int(len(frame))
    game_column = contract["game"]
    games = set(pd.to_numeric(frame[game_column], errors="coerce").dropna().astype(int).tolist())
    report["game_count"] = len(games)
    if frame.empty:
        report["issues"].append({"severity": "critical", "code": "empty_partition", "count": 1})

    null_key_columns = (game_column, *contract["key"])
    null_rows = int(frame.loc[:, list(dict.fromkeys(null_key_columns))].isna().any(axis=1).sum())
    if null_rows:
        report["issues"].append({"severity": "critical", "code": "null_identity_rows", "count": null_rows})
    if contract["key"]:
        duplicate_rows = int(frame.duplicated(list(contract["key"]), keep=False).sum())
        if duplicate_rows:
            report["issues"].append(
                {"severity": "critical", "code": "duplicate_source_keys", "count": duplicate_rows}
            )

    season_mismatch = int((pd.to_numeric(frame["_season"], errors="coerce") != season).sum())
    if season_mismatch:
        report["issues"].append({"severity": "critical", "code": "season_mismatch", "count": season_mismatch})
    type_values = frame["_season_type"].dropna().astype(str).str.lower().unique().tolist()
    expected_tokens_by_partition = {
        "regular": {"regular", "regular season", "rg"},
        "playoffs": {"playoffs", "postseason", "po"},
        "play_in": {"play_in", "play-in", "pi"},
    }
    expected_tokens = expected_tokens_by_partition.get(season_type.lower(), {season_type.lower()})
    unexpected_types = [value for value in type_values if value.lower() not in expected_tokens]
    if unexpected_types:
        report["issues"].append(
            {"severity": "high", "code": "season_type_mismatch", "count": len(unexpected_types), "values": unexpected_types}
        )
    report["passed"] = not any(issue["severity"] in {"critical", "high"} for issue in report["issues"])
    return report, games


def build_event_snapshot(root: str | Path) -> dict:
    root_path = Path(root)
    reports: list[dict] = []
    game_sets: dict[tuple[int, str, str], set[int]] = {}
    for path in sorted(root_path.rglob("*.parquet")):
        report, games = audit_event_file(path)
        reports.append(report)
        game_sets[(report["season"], report["season_type"], report["source"])] = games

    reconciliation: list[dict] = []
    partitions = sorted({(season, season_type) for season, season_type, _ in game_sets})
    for season, season_type in partitions:
        available = {
            source: games
            for (item_season, item_type, source), games in game_sets.items()
            if item_season == season and item_type == season_type
        }
        if "nbastatsv3" not in available:
            continue
        reference = available["nbastatsv3"]
        for source, games in sorted(available.items()):
            if source == "nbastatsv3":
                continue
            missing = sorted(reference - games)
            extra = sorted(games - reference)
            reconciliation.append(
                {
                    "season": season,
                    "season_type": season_type,
                    "reference_source": "nbastatsv3",
                    "source": source,
                    "missing_games": len(missing),
                    "extra_games": len(extra),
                    "missing_game_ids": missing[:20],
                    "extra_game_ids": extra[:20],
                    "passed": not missing and not extra,
                }
            )

    identity = hashlib.sha256(
        json.dumps([(item["path"], item["sha256"], item["row_count"]) for item in reports]).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "snapshot_id": f"nba_events_{identity}",
        "dataset": "nba_event_sources",
        "grain": "source-native event or matchup row",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(reports)
        and all(report["passed"] for report in reports)
        and all(item["passed"] for item in reconciliation),
        "files": reports,
        "reconciliation": reconciliation,
    }
