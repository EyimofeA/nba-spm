"""Normalize legacy possession records without heuristic game classification."""
from __future__ import annotations

import pandas as pd


GAME_TYPE_BY_PREFIX = {
    "002": "regular",
    "004": "playoffs",
    "005": "play_in",
    "006": "other",
}


def classify_game_type(game_id: object) -> str:
    value = str(game_id).strip()
    return GAME_TYPE_BY_PREFIX.get(value[:3], "other")


def normalize_legacy_possessions(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["gameid"] = normalized["gameid"].astype(str).str.strip()
    normalized["game_type"] = normalized["gameid"].map(classify_game_type)
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise")
    normalized = normalized.sort_values(
        ["season", "date", "gameid", "period", "num"], kind="stable"
    ).reset_index(drop=True)
    return normalized
