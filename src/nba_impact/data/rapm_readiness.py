"""Audit whether local event data is safe to use as a RAPM input.

The report is deliberately stricter than a source-file inventory. A game is
RAPM-ready only when canonical possessions and ordinal ten-player lineup
segments agree. A V3 event file or a score-only table does not pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


_V3_REQUIRED = {"gameId", "actionId", "actionNumber", "period", "clock", "teamId", "personId"}
_CDN_REQUIRED = {
    "gameId", "orderNumber", "actionNumber", "period", "clock", "possession",
    "actionType", "personId", "teamId",
}
_LINEUP_COLUMNS = tuple(
    f"{side}_player_{number}" for side in ("home", "away") for number in range(1, 6)
)


def _partition(root: Path, source: str, season_start: int, season_type: str) -> Path:
    return root / source / f"season={season_start}" / f"{season_type}.parquet"


def _source_summary(path: Path, required: set[str], order_column: str) -> dict:
    if not path.exists():
        return {"present": False, "rows": 0, "games": 0, "issues": ["missing_partition"]}
    frame = pd.read_parquet(path)
    missing = sorted(required - set(frame.columns))
    issues = [f"missing_column:{name}" for name in missing]
    game_column = "gameId" if "gameId" in frame.columns else None
    if game_column and order_column in frame.columns:
        duplicate = int(frame.duplicated([game_column, order_column], keep=False).sum())
        if duplicate:
            issues.append(f"duplicate_{order_column}:{duplicate}")
    return {
        "present": True,
        "rows": int(len(frame)),
        "games": int(frame[game_column].nunique()) if game_column else 0,
        "issues": issues,
    }


def _season_games(frame: pd.DataFrame, season_start: int, season_type: str) -> set[str]:
    if frame.empty or not {"game_id", "season_start", "season_type"}.issubset(frame.columns):
        return set()
    return set(
        frame.loc[
            frame["season_start"].eq(season_start) & frame["season_type"].eq(season_type),
            "game_id",
        ].astype(str)
    )


def audit_rapm_inputs(
    event_root: str | Path,
    silver_root: str | Path,
    project_seasons: Iterable[int] = range(2017, 2027),
) -> dict:
    """Return RAPM-input coverage for project seasons, labelled by season end."""
    event_root = Path(event_root)
    silver_root = Path(silver_root)
    table_columns = {
        "game_dim": ["game_id", "season_start", "season_type"],
        "player_games": ["game_id", "season_start", "season_type"],
        "possessions": ["game_id", "season_start", "season_type"],
        "possession_lineup_segments": ["game_id", "possession_segment_id", *_LINEUP_COLUMNS],
    }
    tables = {}
    for name, columns in table_columns.items():
        file_path = silver_root / f"{name}.parquet"
        tables[name] = pd.read_parquet(file_path, columns=columns) if file_path.exists() else pd.DataFrame()

    segments = tables["possession_lineup_segments"]
    valid_segment_games: set[str] = set()
    invalid_segment_games: set[str] = set()
    if not segments.empty:
        if not set(_LINEUP_COLUMNS).issubset(segments.columns) or "possession_segment_id" not in segments:
            invalid_segment_games = set(segments.get("game_id", pd.Series(dtype=str)).astype(str))
        else:
            duplicate = segments.duplicated("possession_segment_id", keep=False)
            valid_lineup = segments.loc[:, _LINEUP_COLUMNS].notna().all(axis=1) & (
                segments.loc[:, _LINEUP_COLUMNS].nunique(axis=1).eq(10)
            )
            invalid_segment_games = set(segments.loc[duplicate | ~valid_lineup, "game_id"].astype(str))
            valid_segment_games = set(segments["game_id"].astype(str)) - invalid_segment_games

    rows: list[dict] = []
    for project_season in project_seasons:
        source_season = int(project_season) - 1
        for season_type in ("regular", "playoffs"):
            v3 = _source_summary(
                _partition(event_root, "nbastatsv3", source_season, season_type),
                _V3_REQUIRED,
                "actionId",
            )
            cdn = _source_summary(
                _partition(event_root, "cdnnba", source_season, season_type),
                _CDN_REQUIRED,
                "orderNumber",
            )
            game_ids = _season_games(tables["game_dim"], source_season, season_type)
            player_games = _season_games(tables["player_games"], source_season, season_type)
            possession_games = _season_games(tables["possessions"], source_season, season_type)
            ready_games = possession_games & valid_segment_games
            issues: list[str] = []
            if not v3["present"]:
                issues.append("missing_v3_events")
            if not cdn["present"]:
                issues.append("missing_cdn_ordinal_events")
            if game_ids and not player_games.issuperset(game_ids):
                issues.append(f"player_game_coverage_gap:{len(game_ids - player_games)}")
            if game_ids and not ready_games.issuperset(game_ids):
                issues.append(f"rapm_ready_game_gap:{len(game_ids - ready_games)}")
            if invalid_segment_games & possession_games:
                issues.append(f"invalid_lineup_segments:{len(invalid_segment_games & possession_games)}")
            rows.append(
                {
                    "project_season": int(project_season),
                    "source_season_start": source_season,
                    "season_label": f"{source_season}-{str(source_season + 1)[-2:]}",
                    "season_type": season_type,
                    "v3_events": v3,
                    "cdn_ordinal_events": cdn,
                    "game_dim_games": len(game_ids),
                    "player_game_games": len(player_games),
                    "possession_games": len(possession_games),
                    "rapm_ready_games": len(ready_games),
                    "issues": issues,
                    "passed": not issues,
                }
            )
    return {
        "dataset": "rapm_input_readiness",
        "grain": "one project season and season type",
        "contract": (
            "RAPM-ready requires canonical possession rows and ordinal ten-player lineup segments. "
            "V3-only and score-only data do not pass."
        ),
        "event_root": str(event_root.resolve()),
        "silver_root": str(silver_root.resolve()),
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
    }
