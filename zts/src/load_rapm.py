"""
RAPM data loaders for the zTS project.

Data sources (from JE_rapm/):
  - 3 Year RAPM.csv   — 3-year rolling RAPM per player (target for box model training)
  - 1 Year RAPM.csv   — single-season RAPM (raw, used in prior blend)
  - RAPM_with_prior_all_seasons.csv — already-computed prior+raw final RAPM (2017–2024)

Season key convention
---------------------
All DataFrames returned here use an integer `Season` column that represents the
**ending** year of the RAPM window:
  "2022-24"  →  Season = 2024
  1 Year     →  Season already integer (e.g. 2024)

Player names are normalised via _norm_name for fuzzy joining to box score data.
"""

from __future__ import annotations

import pathlib
import unicodedata

import pandas as pd
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent.parent
_JE   = _HERE.parent / "JE_rapm"

# Columns we always emit
RAPM_3Y_COLS = ["_name_key", "Name", "Season", "ORAPM_3y", "DRAPM_3y", "RAPM_3y"]
RAPM_1Y_COLS = ["_name_key", "Name", "Season", "ORAPM_1y", "DRAPM_1y", "RAPM_1y"]
RAPM_PR_COLS = ["_name_key", "Name", "Season",
                "OPrior", "DPrior",   # what the prior model predicted
                "ORaw",   "DRaw",     # 1-year raw RAPM (offensive / defensive)
                "OFinal", "DFinal",   # blended final
                "RAPM_prior"]         # total final


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm(s: pd.Series) -> pd.Series:
    """Lowercase + strip accents — matches features.py._norm_name logic."""
    def _strip(text: str) -> str:
        nfkd = unicodedata.normalize("NFKD", str(text))
        return "".join(c for c in nfkd if not unicodedata.combining(c))
    return s.fillna("").astype(str).str.lower().str.strip().apply(_strip)


def _end_year(season_str: str) -> int:
    """Parse '2022-24' → 2024 or '1999-01' → 2001."""
    parts = season_str.split("-")
    last = parts[-1]
    if len(last) == 2:
        century = int(parts[0][:2])
        year = int(last)
        # Handle century rollover: "1999-01" → 2001 not 1901
        if year < int(parts[0][2:]):
            century += 1
        return century * 100 + year
    return int(last)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_rapm_3year(
    path: pathlib.Path | None = None,
    min_season: int = 2016,
    max_season: int = 2025,
) -> pd.DataFrame:
    """Load 3-year rolling RAPM.

    Season column = end year of the 3-year window.
    Only keeps regular-season windows (named NNN-NN format).

    Returns columns: _name_key, Name, Season, ORAPM_3y, DRAPM_3y, RAPM_3y
    """
    path = path or (_JE / "3 Year RAPM.csv")
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={"Off": "ORAPM_3y", "Def": "DRAPM_3y", "Rapm": "RAPM_3y"})

    # Parse "2022-24" → integer end year
    df["Season"] = df["Season"].astype(str).apply(_end_year)
    df = df[(df["Season"] >= min_season) & (df["Season"] <= max_season)]

    df["ORAPM_3y"] = pd.to_numeric(df["ORAPM_3y"], errors="coerce")
    df["DRAPM_3y"] = pd.to_numeric(df["DRAPM_3y"], errors="coerce")
    df["RAPM_3y"]  = pd.to_numeric(df["RAPM_3y"],  errors="coerce")
    df["_name_key"] = _norm(df["Name"])

    return df[RAPM_3Y_COLS].dropna(subset=["RAPM_3y"]).reset_index(drop=True)


def load_rapm_1year(
    path: pathlib.Path | None = None,
    min_season: int = 2014,
    max_season: int = 2025,
) -> pd.DataFrame:
    """Load single-season RAPM (raw, unregularised with prior).

    Returns columns: _name_key, Name, Season, ORAPM_1y, DRAPM_1y, RAPM_1y
    """
    path = path or (_JE / "1 Year RAPM.csv")
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={"Off": "ORAPM_1y", "Def": "DRAPM_1y", "Rapm": "RAPM_1y"})

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
    df = df[(df["Season"] >= min_season) & (df["Season"] <= max_season)]

    for col in ["ORAPM_1y", "DRAPM_1y", "RAPM_1y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_name_key"] = _norm(df["Name"])

    return df[RAPM_1Y_COLS].dropna(subset=["RAPM_1y"]).reset_index(drop=True)


def load_rapm_with_prior(
    path: pathlib.Path | None = None,
    min_season: int = 2017,
    max_season: int = 2025,
) -> pd.DataFrame:
    """Load the pre-computed prior-blended RAPM (2017–2024).

    The 'prior' in this file was computed from a previous box-score model
    (not the zTS features).  Use this for benchmarking against your new prior.

    Returns columns: _name_key, Name, Season,
                     OPrior, DPrior, ORaw, DRaw, OFinal, DFinal, RAPM_prior
    """
    path = path or (_JE / "RAPM_with_prior_all_seasons.csv")
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={
        "Offensive Prior RAPM":    "OPrior",
        "Defensive Prior RAPM":    "DPrior",
        "Offensive Raw RAPM":      "ORaw",
        "Defensive Raw RAPM":      "DRaw",
        "Offensive Final RAPM":    "OFinal",
        "Defensive Final RAPM":    "DFinal",
        "RAPM":                    "RAPM_prior",
    })

    df["Season"] = pd.to_numeric(df["Season"], errors="coerce").astype(int)
    df = df[(df["Season"] >= min_season) & (df["Season"] <= max_season)]

    for col in ["OPrior","DPrior","ORaw","DRaw","OFinal","DFinal","RAPM_prior"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_name_key"] = _norm(df["Name"])

    return df[RAPM_PR_COLS].dropna(subset=["RAPM_prior"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Name-based join helper
# ---------------------------------------------------------------------------

def join_rapm_to_box(
    rapm_df: pd.DataFrame,
    box_df: pd.DataFrame,
    rapm_col: str = "RAPM_3y",
) -> pd.DataFrame:
    """Left-join RAPM onto box_df by (normalised player name, Season).

    box_df must have a 'Player' column and a 'Season' column.
    Returns box_df with RAPM columns appended (NaN where no match found).

    Duplicate matches are resolved by keeping the row with the largest
    |RAPM| value (most active player with that name).
    """
    box = box_df.copy()
    box["_name_key"] = _norm(box["Player"])

    rapm_cols = ["_name_key", "Season"] + [c for c in rapm_df.columns
                                            if c not in ["_name_key", "Name", "Season"]]
    rapm_dedup = (
        rapm_df[rapm_cols]
        .assign(_abs=lambda d: d[rapm_col].abs())
        .sort_values("_abs", ascending=False)
        .drop_duplicates(subset=["_name_key", "Season"])
        .drop(columns=["_abs"])
    )

    merged = box.merge(rapm_dedup, on=["_name_key", "Season"], how="left")
    return merged.drop(columns=["_name_key"])
