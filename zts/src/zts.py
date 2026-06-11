"""
Core zTS computation functions.

All functions are pure transformations — they take DataFrames and return DataFrames.
No I/O or side effects here.

Pipeline:
    1. compute_player_ts(box_df)         → adds TS_pct per player-season
    2. compute_league_ts(box_df)         → league avg TS% per season
    3. compute_playtype_ts(synergy_df)   → TS% per player per playtype (for league agg)
    4. compute_league_ts_by_playtype(synergy_df)  → league avg TS% per (season, playtype)
    5. compute_playtype_shares(synergy_df)         → share of possessions per playtype
    6. compute_expected_ts(shares_df, league_pt_ts_df) → expected TS% from playtype mix
    7. compute_zts(box_df, synergy_df)   → full zTS table (main entry point)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps Synergy playtype keys → bucket label (informational only)
PLAYTYPE_BUCKET = {
    "iso":        "Creation",
    "pr_ball":    "Creation",
    "post":       "Creation",
    "off_screen": "Spacing",
    "hand_off":   "Spacing",
    "spot":       "Spacing",
    "pr_roll":    "Finishing",
    "oreb":       "Finishing",
    "misc":       "Finishing",
    "cut":        "Finishing",
    "tran":       "Transition",
}

FT_FACTOR = 0.44   # standard FTA weight in TS% denominator
MIN_POSS_FOR_LEAGUE_AVG = 20   # min Synergy poss to count toward league playtype TS


# ---------------------------------------------------------------------------
# Step 1 — Player TS%
# ---------------------------------------------------------------------------

def compute_player_ts(box_df: pd.DataFrame) -> pd.DataFrame:
    """Add TS_pct column to box score DataFrame.

    Requires columns: PTS, FGA, FTA
    Returns input df with new column TS_pct (0–100 scale, e.g., 55.3).
    Rows where FGA + FTA == 0 receive NaN.
    """
    df = box_df.copy()
    denom = 2 * (df["FGA"] + FT_FACTOR * df["FTA"])
    df["TS_pct"] = np.where(denom > 0, 100 * df["PTS"] / denom, np.nan)
    return df


# ---------------------------------------------------------------------------
# Step 2 — League average TS% per season
# ---------------------------------------------------------------------------

def compute_league_ts(box_df: pd.DataFrame, min_minutes: float = 0) -> pd.DataFrame:
    """Compute league average TS% per season from box scores.

    Uses counting-stat aggregation (sum PTS / sum attempts) rather than
    averaging individual player rates, matching how league TS% is published.

    Returns DataFrame with columns: Season, League_TS
    """
    df = box_df if min_minutes == 0 else box_df[box_df["Minutes"] >= min_minutes]
    agg = df.groupby("Season").agg(
        total_pts=("PTS", "sum"),
        total_fga=("FGA", "sum"),
        total_fta=("FTA", "sum"),
    )
    agg["League_TS"] = 100 * agg["total_pts"] / (
        2 * (agg["total_fga"] + FT_FACTOR * agg["total_fta"])
    )
    return agg[["League_TS"]].reset_index()


# ---------------------------------------------------------------------------
# Step 3 — TS% per row in Synergy data (player × playtype)
# ---------------------------------------------------------------------------

def _estimate_fta(synergy_df: pd.DataFrame) -> pd.Series:
    """Estimate FTA from Synergy FTFreq_pct and Poss.

    FTFreq_pct is the fraction of possessions that result in free throws.
    Approximating 2 FTs per FT trip gives FTA ≈ FTFreq_pct × Poss × 2.
    """
    ft = synergy_df["FTFreq_pct"].fillna(0)
    return ft * synergy_df["Poss"] * 2


def compute_playtype_ts(synergy_df: pd.DataFrame) -> pd.DataFrame:
    """Add TS_pct column to the Synergy DataFrame (per player-playtype row).

    Uses estimated FTA from FTFreq_pct.
    Returns copy of input with new column: TS_pct (0–100 scale).
    """
    df = synergy_df.copy()
    fta_est = _estimate_fta(df)
    fga = df["FGA"].fillna(0)
    denom = 2 * (fga + FT_FACTOR * fta_est)
    df["TS_pct"] = np.where(denom > 0, 100 * df["Points"] / denom, np.nan)
    return df


# ---------------------------------------------------------------------------
# Step 4 — League average TS% per (season, playtype)
# ---------------------------------------------------------------------------

def compute_league_ts_by_playtype(synergy_df: pd.DataFrame) -> pd.DataFrame:
    """Compute counting-based league TS% for each (Season, playtype).

    Only rows with Poss >= MIN_POSS_FOR_LEAGUE_AVG are included so that
    players with trivial usage don't distort the average.

    Returns DataFrame with columns: Season, playtype, League_PT_TS
    """
    df = synergy_df[synergy_df["Poss"] >= MIN_POSS_FOR_LEAGUE_AVG].copy()
    fta_est = _estimate_fta(df)
    df["fta_est"] = fta_est
    df["fga_fill"] = df["FGA"].fillna(0)

    agg = df.groupby(["Season", "playtype"]).agg(
        total_pts=("Points", "sum"),
        total_fga=("fga_fill", "sum"),
        total_fta=("fta_est", "sum"),
    )
    agg["League_PT_TS"] = 100 * agg["total_pts"] / (
        2 * (agg["total_fga"] + FT_FACTOR * agg["total_fta"])
    )
    return agg[["League_PT_TS"]].reset_index()


# ---------------------------------------------------------------------------
# Step 5 — Playtype possession shares per player-season
# ---------------------------------------------------------------------------

def compute_playtype_shares(synergy_df: pd.DataFrame) -> pd.DataFrame:
    """Compute each player's fraction of total Synergy possessions per playtype.

    Returns a wide DataFrame indexed by (PLAYER_ID, Season) with one column
    per playtype key (e.g., iso_share, pr_ball_share, …) plus SynPoss (total).
    """
    df = synergy_df[["PLAYER_ID", "Season", "playtype", "Poss"]].copy()

    # Total Synergy possessions per player-season
    total_poss = (
        df.groupby(["PLAYER_ID", "Season"])["Poss"]
        .sum()
        .rename("SynPoss")
        .reset_index()
    )

    # Merge total back and compute share
    df = df.merge(total_poss, on=["PLAYER_ID", "Season"])
    df["share"] = np.where(df["SynPoss"] > 0, df["Poss"] / df["SynPoss"], 0)

    # Pivot to wide format
    wide = df.pivot_table(
        index=["PLAYER_ID", "Season"],
        columns="playtype",
        values="share",
        aggfunc="sum",
        fill_value=0,
    )
    wide.columns = [f"{pt}_share" for pt in wide.columns]
    wide = wide.reset_index()

    wide = wide.merge(total_poss, on=["PLAYER_ID", "Season"])
    return wide


# ---------------------------------------------------------------------------
# Step 6 — Expected TS from playtype mix
# ---------------------------------------------------------------------------

def compute_expected_ts(
    shares_df: pd.DataFrame,
    league_pt_ts_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute ExpectedTS for each player-season.

    ExpectedTS = Σ (playtype_share × League_PT_TS for that playtype and season)

    Args:
        shares_df:       Output of compute_playtype_shares().
        league_pt_ts_df: Output of compute_league_ts_by_playtype().

    Returns DataFrame with columns: PLAYER_ID, Season, ExpectedTS
    """
    # Pivot league playtype TS to wide: one column per playtype
    league_wide = league_pt_ts_df.pivot_table(
        index="Season", columns="playtype", values="League_PT_TS"
    )
    league_wide.columns = [f"{pt}_league_ts" for pt in league_wide.columns]
    league_wide = league_wide.reset_index()

    df = shares_df.merge(league_wide, on="Season", how="left")

    all_playtypes = list(
        {col.replace("_share", "") for col in df.columns if col.endswith("_share")}
    )

    expected = pd.Series(np.zeros(len(df)), index=df.index)
    for pt in all_playtypes:
        share_col = f"{pt}_share"
        league_col = f"{pt}_league_ts"
        if share_col in df.columns and league_col in df.columns:
            share = df[share_col].fillna(0)
            lts = df[league_col].fillna(np.nan)
            expected += share * lts

    df["ExpectedTS"] = expected
    return df[["PLAYER_ID", "Season", "ExpectedTS"]].copy()


# ---------------------------------------------------------------------------
# Step 7 — Full zTS pipeline (main entry point)
# ---------------------------------------------------------------------------

def compute_zts(
    box_df: pd.DataFrame,
    synergy_df: pd.DataFrame,
    min_minutes: float = 250,
    min_syn_poss: float = 50,
) -> pd.DataFrame:
    """Compute zTS for all player-seasons where both data sources overlap.

    Args:
        box_df:      Output of load_box_scores().
        synergy_df:  Output of load_synergy_playtypes().
        min_minutes: Drop players below this minute threshold.
        min_syn_poss: Drop players with fewer Synergy possessions than this.

    Returns DataFrame with final zTS output columns:
        PLAYER_ID, Player, Season, Minutes, TS_pct, League_TS,
        rTS, ExpectedTS, Difficulty, zTS, SynPoss
    """
    # --- player TS% ---
    box = compute_player_ts(box_df)
    box = box[box["Minutes"] >= min_minutes].copy()

    # --- league TS% ---
    league_ts = compute_league_ts(box_df)   # use full population for league avg

    # --- playtype-level league TS% ---
    league_pt_ts = compute_league_ts_by_playtype(synergy_df)

    # --- playtype shares ---
    shares = compute_playtype_shares(synergy_df)
    shares = shares[shares["SynPoss"] >= min_syn_poss]

    # --- expected TS ---
    expected = compute_expected_ts(shares, league_pt_ts)

    # --- merge everything ---
    result = box.merge(expected, on=["PLAYER_ID", "Season"], how="inner")
    result = result.merge(shares[["PLAYER_ID", "Season", "SynPoss"]], on=["PLAYER_ID", "Season"], how="left")
    result = result.merge(league_ts, on="Season", how="left")

    # --- final metrics ---
    result["rTS"] = result["TS_pct"] - result["League_TS"]
    result["Difficulty"] = result["League_TS"] - result["ExpectedTS"]
    result["zTS"] = result["rTS"] + result["Difficulty"]   # = TS_pct - ExpectedTS

    # Round for readability
    for col in ["TS_pct", "League_TS", "rTS", "ExpectedTS", "Difficulty", "zTS"]:
        result[col] = result[col].round(2)

    out_cols = [
        "PLAYER_ID", "Player", "Season", "Minutes",
        "TS_pct", "League_TS", "rTS",
        "ExpectedTS", "Difficulty", "zTS",
        "SynPoss",
    ]
    result = result[out_cols].sort_values(["Season", "zTS"], ascending=[True, False])
    return result.reset_index(drop=True)
