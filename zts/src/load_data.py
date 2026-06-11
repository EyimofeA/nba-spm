"""
Data loading functions for the zTS project.

Two data sources:
  1. Box scores  — ../playersheets/year_totals/{YEAR}.csv  (local)
  2. Synergy playtypes — GitHub CSV (remote, cached locally on first fetch)
"""

from __future__ import annotations

import os
import pathlib
import urllib.request

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent.parent          # zts/
_REPO_ROOT = _HERE.parent                                       # New SPM/
_YEAR_TOTALS_DIR = _REPO_ROOT / "playersheets" / "year_totals"
_RAW_DIR = _HERE / "data" / "raw"
_CACHE_FILE = _RAW_DIR / "playtype_raw.csv"

SYNERGY_URL = (
    "https://raw.githubusercontent.com/gabriel1200/site_Data/master/playtype.csv"
)

# Synergy years available via the GitHub source
SYNERGY_YEAR_RANGE = (2014, 2026)

# Minimum thresholds
MIN_SYNERGY_POSS = 20      # for league-average computation per playtype
MIN_PLAYER_MINUTES = 250   # filter from final output

# ---------------------------------------------------------------------------
# Box score loader
# ---------------------------------------------------------------------------

_BOX_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "year",
    # team (available in 2025+ mega-sheets)
    "TEAM_ABBREVIATION",
    # zTS core
    "PTS", "FGA", "FTA", "Minutes", "AGE",
    # games played / wins (for contextual features + team wins derivation)
    "GP", "W",
    # creation / load
    "AST", "TOV", "FG3A", "FG3_PCT", "FG2A", "FG2M", "OffPoss", "DefPoss",
    # defensive counting
    "STL", "BLK", "OREB", "DREB",
    # rim
    "AtRimFGA", "AtRimFGM", "AtRimAssists",
    # rebounding splits
    "OREB_CONTEST", "OREB_UNCONTEST", "OREB_CHANCE_DEFER",
    "DREB_CONTEST", "DREB_UNCONTEST", "DREB_CHANCE_DEFER",
    # shot difficulty
    "tight_FG3A", "very_tight_FG3A", "tight_FG3M", "very_tight_FG3M",
    # assisted field goals
    "PtsAssisted2s", "PtsAssisted3s",
    # misc offense — two spellings depending on file vintage
    "Offensive Fouls Drawn",   # pre-2025 files
    "Offensive_Fouls_Drawn",   # 2025+ mega-sheets
]

# Columns that are rates (must be recomputed after summing team rows, not averaged)
_BOX_RATE_COLS = {"FG3_PCT"}

# All counting columns that should be summed across team rows for traded players
_BOX_SUM_COLS = [
    "PTS", "FGA", "FTA", "Minutes", "AST", "TOV",
    "FG3A", "FG2A", "FG2M", "OffPoss", "DefPoss",
    "GP", "W",
    "STL", "BLK", "OREB", "DREB",
    "AtRimFGA", "AtRimFGM", "AtRimAssists",
    "OREB_CONTEST", "OREB_UNCONTEST", "OREB_CHANCE_DEFER",
    "DREB_CONTEST", "DREB_UNCONTEST", "DREB_CHANCE_DEFER",
    "tight_FG3A", "very_tight_FG3A", "tight_FG3M", "very_tight_FG3M",
    "PtsAssisted2s", "PtsAssisted3s",
    "Offensive Fouls Drawn",
]


def load_box_scores(
    min_year: int = SYNERGY_YEAR_RANGE[0],
    max_year: int = SYNERGY_YEAR_RANGE[1],
) -> pd.DataFrame:
    """Load regular-season player box score totals for a range of seasons.

    Only columns that exist in every yearly CSV are loaded; missing optional
    columns are silently skipped and will appear as NaN in the output.

    Returns a DataFrame with PLAYER_ID, Player, Season, and all available
    box score columns (see _BOX_COLS for the full list).
    """
    frames: list[pd.DataFrame] = []

    for year in range(min_year, max_year + 1):
        path = _YEAR_TOTALS_DIR / f"{year}.csv"
        if not path.exists():
            continue
        # Only load columns that actually exist in this year's file
        available = pd.read_csv(path, nrows=0).columns.tolist()
        cols_to_load = [c for c in _BOX_COLS if c in available]
        df = pd.read_csv(path, usecols=cols_to_load, low_memory=False)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No year_totals CSVs found under {_YEAR_TOTALS_DIR} "
            f"for years {min_year}–{max_year}."
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"PLAYER_NAME": "Player", "year": "Season"})

    # Unify the two spellings of offensive fouls drawn (2025+ uses underscore)
    if "Offensive_Fouls_Drawn" in out.columns:
        if "Offensive Fouls Drawn" not in out.columns:
            out = out.rename(columns={"Offensive_Fouls_Drawn": "Offensive Fouls Drawn"})
        else:
            out["Offensive Fouls Drawn"] = out["Offensive Fouls Drawn"].fillna(
                out["Offensive_Fouls_Drawn"]
            )
            out = out.drop(columns=["Offensive_Fouls_Drawn"])

    # Standardise team abbreviation column name
    if "TEAM_ABBREVIATION" in out.columns:
        out = out.rename(columns={"TEAM_ABBREVIATION": "Team"})

    # Coerce all numeric columns (exclude string identity columns)
    _str_cols = {"Player", "Team"}
    id_cols = {"PLAYER_ID", "Season"}
    for col in out.columns:
        if col not in _str_cols | id_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["PLAYER_ID", "Season", "PTS", "FGA", "FTA"])
    out["PLAYER_ID"] = out["PLAYER_ID"].astype(int)
    out["Season"] = out["Season"].astype(int)

    # Deduplicate traded players: sum counting stats, reconstruct rates.
    # Pre-derive FG3M from FG3A × FG3_PCT so we can recompute after summing.
    out["_FG3M"] = out["FG3A"].fillna(0) * out["FG3_PCT"].fillna(0)

    sum_cols = [c for c in _BOX_SUM_COLS if c in out.columns] + ["_FG3M"]
    agg_dict: dict = {
        "Player": ("Player", "first"),
        "AGE":    ("AGE", "first"),
    }
    if "Team" in out.columns:
        # For traded players use "TOT" (multiple teams); single-team players keep their abbrev
        team_count = out.groupby(["PLAYER_ID", "Season"])["Team"].transform("nunique")
        out["Team"] = out["Team"].where(team_count == 1, "TOT")
        agg_dict["Team"] = ("Team", "first")
    for col in sum_cols:
        agg_dict[col] = (col, "sum")

    out = out.groupby(["PLAYER_ID", "Season"], as_index=False).agg(**agg_dict)

    # Reconstruct FG3_PCT from aggregated makes/attempts
    out["FG3_PCT"] = np.where(out["FG3A"] > 0, out["_FG3M"] / out["FG3A"], 0.0)
    out = out.drop(columns=["_FG3M"])

    return out


# ---------------------------------------------------------------------------
# Synergy playtype loader
# ---------------------------------------------------------------------------

_SYNERGY_COLS = ["PLAYER_ID", "Player", "year", "playtype", "Poss", "Points", "FGA", "FGM", "FTFreq%"]

_PLAYTYPE_KEYS = {
    "iso", "pr_ball", "post",           # Creation
    "off_screen", "hand_off", "spot",   # Spacing
    "pr_roll", "oreb", "misc", "cut",   # Finishing
    "tran",                             # Transition
}


def load_synergy_playtypes(
    min_year: int = SYNERGY_YEAR_RANGE[0],
    max_year: int = SYNERGY_YEAR_RANGE[1],
    force_download: bool = False,
) -> pd.DataFrame:
    """Load player-level Synergy playtype data.

    The file is downloaded once and cached to data/raw/playtype_raw.csv.
    Pass force_download=True to refresh the cache.

    Returns a DataFrame with columns:
        PLAYER_ID, Player, Season, playtype, Poss, Points, FGA, FGM, FTFreq_pct
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    if force_download or not _CACHE_FILE.exists():
        print(f"Downloading Synergy playtype data from GitHub...")
        urllib.request.urlretrieve(SYNERGY_URL, _CACHE_FILE)
        print(f"Saved to {_CACHE_FILE}")
    else:
        print(f"Using cached file: {_CACHE_FILE}")

    df = pd.read_csv(_CACHE_FILE, low_memory=False)

    # Normalise column names — the CSV has some inconsistencies
    df.columns = [c.strip() for c in df.columns]

    # The raw CSV contains two merged formats with different FT frequency columns:
    #   Old rows (2014–2023): 'FTFreq%'  stored as decimal  (0.143 = 14.3%)
    #   New rows (2024+):     '%FT'      stored as percentage (14.7 = 14.7%)
    # We unify both into a single decimal column 'FTFreq_pct' (0–1 scale).
    for old_col, is_pct in [("FTFreq%", False), ("%FT", True)]:
        if old_col in df.columns:
            src = pd.to_numeric(df[old_col], errors="coerce")
            if is_pct:
                src = src / 100.0
            if "FTFreq_pct" not in df.columns:
                df["FTFreq_pct"] = src
            else:
                df["FTFreq_pct"] = df["FTFreq_pct"].fillna(src)

    if "FTFreq_pct" not in df.columns:
        raise KeyError(
            "Cannot find FT frequency column ('FTFreq%' or '%FT') in Synergy data. "
            f"Available columns: {list(df.columns)}"
        )

    # Standardise year column
    year_col = "year" if "year" in df.columns else "SEASON_ID"
    df = df.rename(columns={year_col: "Season"})

    # Keep only needed columns
    keep = ["PLAYER_ID", "Player", "Season", "playtype", "Poss", "Points", "FGA", "FGM", "FTFreq_pct"]
    available = [c for c in keep if c in df.columns]
    df = df[available].copy()

    # Coerce types
    for col in ["PLAYER_ID", "Season", "Poss", "Points", "FGA", "FGM", "FTFreq_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["PLAYER_ID", "Season", "playtype", "Poss", "Points"])
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
    df["Season"] = df["Season"].astype(int)

    # Filter to known playtypes and year range
    df = df[df["playtype"].isin(_PLAYTYPE_KEYS)]
    df = df[(df["Season"] >= min_year) & (df["Season"] <= max_year)]
    # Synergy data ends at 2025 for now even when loading up to 2026
    df = df[df["Season"] <= 2026]

    return df.reset_index(drop=True)
