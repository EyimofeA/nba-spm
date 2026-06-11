"""
Feature engineering functions for the zTS project.

All functions are pure transformations with no I/O side effects.
Data sources:
  - box_df:  output of load_data.load_box_scores()
  - site_Data/{year}/hustle/hustle.csv
  - site_Data/{year}/defense/rimdfg.csv
  - site_Data/{year}/player_tracking/close_6.csv
  - site_Data/{year}/player_tracking/passing.csv
  - site_Data/{year}/player_tracking/drives.csv  (kept but excluded from main output)

NOTE on RAPM-adjacent stats:
  Do NOT include on/off ratings, net rating, or RAPM-derived stats as features
  in a box-score model used as a RAPM prior — that is circular. These features
  are safe for general player profiling and non-RAPM downstream tasks.

Public API
----------
compute_box_creation(box_df)          → Creation, Load, ThreePtP, cTOV_pct
compute_passer_rating(box_df, ...)    → PasserRating
compute_rim_points_saved(box_df)      → RimPointsSaved (rim defense)
compute_close6_defense(box_df)        → OppFGM, OppFGA, OppDiff perimeter defense
compute_hustle_stats(box_df)          → Deflections, ChargesDrawn, ContestShots
compute_ftov(box_df, hustle_df)       → fTOV_per100 (forced turnovers / stops)
compute_advanced_box(box_df, ...)     → rebounding splits, tight 3s, assisted FGs,
                                        rim assists, offensive fouls drawn
compute_passing_load(box_df)          → PotentialAst, SecondaryAst per 100
compute_drive_efficiency(box_df)      → Drives stats (secondary; not in main output)
compute_all_features(box_df)          → all features joined (drives excluded)
"""

from __future__ import annotations

import math
import pathlib
import unicodedata

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent.parent          # zts/
_SITE_DATA = _HERE.parent / "site_Data"
_SYNERGY_YEAR_RANGE = (2014, 2026)

# Position → estimated height in inches (used when player height unavailable)
_POS_HEIGHT: dict[str, float] = {
    "G":   76.0,   # PG / SG
    "G-F": 78.0,
    "F-G": 78.0,
    "F":   80.0,   # SF / SF-PF
    "F-C": 82.0,
    "C-F": 82.0,
    "C":   84.0,   # Centre
}
_LEAGUE_AVG_HEIGHT = 79.0   # fallback when position also unknown

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _norm_name(s: pd.Series) -> pd.Series:
    """Lowercase + strip whitespace + strip accents.

    Resolves mismatches like 'Luka Dončić' ↔ 'Luka Doncic'.
    """
    def _strip(text: str) -> str:
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    return s.fillna("").astype(str).str.lower().str.strip().apply(_strip)


def _load_tracking_files(
    rel_path: str,
    min_year: int = _SYNERGY_YEAR_RANGE[0],
    max_year: int = _SYNERGY_YEAR_RANGE[1],
) -> pd.DataFrame:
    """Concatenate site_Data/{year}/{rel_path} for a range of seasons.

    Adds a 'Season' column and a '_player_key' normalised name column.
    """
    frames: list[pd.DataFrame] = []
    for year in range(min_year, max_year + 1):
        path = _SITE_DATA / str(year) / rel_path
        if not path.exists():
            continue
        df = pd.read_csv(path, low_memory=False)
        df["Season"] = year
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Coalesce all player-name variants into a single canonical column
    # then build the normalised join key.
    _NAME_ALIASES = ["Player", "PLAYER", "PLAYER_NAME", "player"]
    name_cols_present = [c for c in _NAME_ALIASES if c in out.columns]
    if name_cols_present:
        # Build a unified series by coalescing in order
        name_series = out[name_cols_present[0]].copy()
        for nc in name_cols_present[1:]:
            name_series = name_series.fillna(out[nc])
        out["_player_key"] = _norm_name(name_series)
    return out


def _merge_by_name(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    """Join on (_player_key, Season), coalescing PLAYER_ID from both sides."""
    merged = left.merge(right, on=["_player_key", "Season"], how=how, suffixes=("", "_r"))
    if "PLAYER_ID_r" in merged.columns:
        if "PLAYER_ID" not in merged.columns:
            merged["PLAYER_ID"] = np.nan
        merged["PLAYER_ID"] = merged["PLAYER_ID"].fillna(merged["PLAYER_ID_r"])
        merged = merged.drop(columns=["PLAYER_ID_r"])
    return merged


def _box_key(box_df: pd.DataFrame, poss_col: str) -> pd.DataFrame:
    """Return a join-ready slice of box_df with normalised player key."""
    key = box_df[["PLAYER_ID", "Season", "Player", poss_col]].copy()
    key["_player_key"] = _norm_name(key["Player"])
    return key


# ---------------------------------------------------------------------------
# 1. Box Creation, Load, ThreePtP, cTOV
# ---------------------------------------------------------------------------

def compute_box_creation(box_df: pd.DataFrame) -> pd.DataFrame:
    """Compute Box Creation, Load, 3-pt Proficiency, and cTOV% from box scores.

    All rate inputs are per-100 offensive possessions.

    Box Creation (BPM-style):
        ThreePtP = (2/(1+exp(−FG3A_per100)) − 1) × FG3_PCT
        Creation = 0.1843·AST + 0.0969·(PTS+TOV) − 2.3021·ThreePtP
                 + 0.0582·(AST·(PTS+TOV)·ThreePtP) − 1.1942

    Load (total offensive burden per 100):
        Load = (AST − 0.38·Creation·0.75) + FGA + 0.44·FTA + Creation + TOV

    cTOV% (controlled turnover rate — turnovers as fraction of load):
        cTOV_pct = TOV_per100 / Load

    Returns DataFrame: PLAYER_ID, Season, ThreePtP, Creation, Load, cTOV_pct
    """
    df = box_df.copy()
    op = df["OffPoss"].replace(0, np.nan)

    pts_100  = df["PTS"]  / op * 100
    ast_100  = df["AST"]  / op * 100
    tov_100  = df["TOV"]  / op * 100
    fga_100  = df["FGA"]  / op * 100
    fta_100  = df["FTA"]  / op * 100
    fg3a_100 = df["FG3A"] / op * 100
    fg3_pct  = pd.to_numeric(df["FG3_PCT"], errors="coerce").fillna(0)

    three_pt_p = (2 / (1 + np.exp(-fg3a_100)) - 1) * fg3_pct

    creation = (
        0.1843 * ast_100
        + 0.0969 * (pts_100 + tov_100)
        - 2.3021 * three_pt_p
        + 0.0582 * (ast_100 * (pts_100 + tov_100) * three_pt_p)
        - 1.1942
    )

    load = (
        (ast_100 - 0.38 * creation * 0.75)
        + fga_100
        + fta_100 * 0.44
        + creation
        + tov_100
    )

    ctov = np.where(load > 0, tov_100 / load, np.nan)

    out = df[["PLAYER_ID", "Season"]].copy()
    out["ThreePtP"]  = three_pt_p.round(4)
    out["Creation"]  = creation.round(3)
    out["Load"]      = load.round(3)
    out["cTOV_pct"]  = pd.Series(ctov, index=df.index).round(4)
    return out


# ---------------------------------------------------------------------------
# 2. Passer Rating
# ---------------------------------------------------------------------------

def compute_passer_rating(
    box_df: pd.DataFrame,
    creation_df: pd.DataFrame,
    position_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute a logistic Passer Rating for each player-season.

    Formula:
        inner = 4.79
              + 6.48 × Layup%
              + 0.19 × Load
              − 76.71 × (AST/Load)
              − 0.07 × Height
              − 0.08 × TOV_per100²
              − 2.71 × (NonThreeCreation / AST_per100)
              + 1.19 × (Height × AST/Load)

        PasserRating = sigmoid(−ln(2.5) × (inner − 5.6)) × 11.3
                     = 1 / (1 + exp(−ln(2.5) × (inner − 5.6))) × 11.3

    Where:
        Layup%          = AtRimFGA / FGA
        AST/Load        = AST_per100 / Load
        Height          = positional estimate (see _POS_HEIGHT); no height
                          column exists in the box score data
        TOV_per100²     = (TOV / OffPoss × 100)²
        NonThreeCreation = Creation + 2.3021·ThreePtP
                           − 0.0582·(AST·(PTS+TOV)·ThreePtP)
                           (creation with 3-pt adjustment removed)

    Args:
        box_df:       load_box_scores() output.
        creation_df:  compute_box_creation() output (needs Creation, Load,
                      ThreePtP already computed).
        position_df:  optional DataFrame with columns PLAYER_ID, Season,
                      Position (string). If None, uses _LEAGUE_AVG_HEIGHT.

    Returns DataFrame: PLAYER_ID, Season, PasserRating
    """
    _LN25 = math.log(2.5)   # ≈ 0.916

    df = box_df.merge(creation_df, on=["PLAYER_ID", "Season"], how="left")

    op = df["OffPoss"].replace(0, np.nan)
    ast_100 = df["AST"]  / op * 100
    tov_100 = df["TOV"]  / op * 100
    pts_100 = df["PTS"]  / op * 100
    fg3a_100 = df["FG3A"] / op * 100
    fg3_pct  = pd.to_numeric(df["FG3_PCT"], errors="coerce").fillna(0)

    layup_pct = (df["AtRimFGA"].fillna(0) / df["FGA"].replace(0, np.nan)).fillna(0)

    load    = df["Load"]
    creation = df["Creation"]
    three_pt_p = df["ThreePtP"]

    ast_load = np.where(load > 0, ast_100 / load, 0.0)

    # Non-3pt creation: strip out the 3PtP adjustment terms
    non3_creation = creation + 2.3021 * three_pt_p - 0.0582 * (ast_100 * (pts_100 + tov_100) * three_pt_p)
    non3_creation_ast = np.where(ast_100 > 0, non3_creation / ast_100, 0.0)

    # Height: use positional estimates if position_df provided, else league avg
    if position_df is not None:
        df = df.merge(position_df[["PLAYER_ID", "Season", "Position"]], on=["PLAYER_ID", "Season"], how="left")
        df["_height"] = df["Position"].map(_POS_HEIGHT).fillna(_LEAGUE_AVG_HEIGHT)
    else:
        df["_height"] = _LEAGUE_AVG_HEIGHT

    height = df["_height"]

    inner = (
        4.79
        + 6.48 * layup_pct
        + 0.19 * load
        - 76.71 * ast_load
        - 0.07 * height
        - 0.08 * tov_100 ** 2
        - 2.71 * non3_creation_ast
        + 1.19 * (height * ast_load)
    )

    passer_rating = (1 / (1 + np.exp(-_LN25 * (inner - 5.6)))) * 11.3

    out = df[["PLAYER_ID", "Season"]].copy()
    out["PasserRating"] = pd.Series(passer_rating.values, index=df.index).round(3)
    return out


# ---------------------------------------------------------------------------
# 3. Rim Points Saved
# ---------------------------------------------------------------------------

def compute_rim_points_saved(
    box_df: pd.DataFrame,
    min_year: int = _SYNERGY_YEAR_RANGE[0],
    max_year: int = _SYNERGY_YEAR_RANGE[1],
) -> pd.DataFrame:
    """Rim Points Saved per player-season from defense/rimdfg.csv.

    Formula:
        rDiff%         = League_avg_Diff% − Player_Diff%  (positive = better)
        DFGA_per100    = DFGA / DefPoss × 100
        RimPointsSaved = DFGA_per100 × rDiff% × 2 / 100

    Diff% = player_rim_FG% − league_rim_FG%. Negative is good.
    """
    rim = _load_tracking_files("defense/rimdfg.csv", min_year, max_year)
    if rim.empty:
        return pd.DataFrame(columns=["PLAYER_ID", "Season", "RimPointsSaved"])

    # Coalesce inconsistent casing: 'Diff%' (2022-23) and 'DIFF%' (2024)
    diff_cols = [c for c in rim.columns if c.upper() == "DIFF%"]
    if not diff_cols:
        return pd.DataFrame(columns=["PLAYER_ID", "Season", "RimPointsSaved"])

    rim["Diff_pct"] = pd.to_numeric(rim[diff_cols[0]], errors="coerce")
    for extra in diff_cols[1:]:
        rim["Diff_pct"] = rim["Diff_pct"].fillna(pd.to_numeric(rim[extra], errors="coerce"))
    rim = rim.drop(columns=diff_cols)

    for col in ["DFGA", "DFGM"]:
        rim[col] = pd.to_numeric(rim[col], errors="coerce")

    league_avg = (
        rim.groupby("Season")["Diff_pct"].mean()
        .rename("League_Diff_pct").reset_index()
    )
    rim = rim.merge(league_avg, on="Season", how="left")
    rim["rDiff_pct"] = rim["League_Diff_pct"] - rim["Diff_pct"]

    bk = _box_key(box_df, "DefPoss")
    merged = _merge_by_name(rim, bk, how="left")

    dp = merged["DefPoss"].replace(0, np.nan)
    merged["DFGA_per100"]    = merged["DFGA"] / dp * 100
    merged["RimPointsSaved"] = (merged["DFGA_per100"] * merged["rDiff_pct"] * 2 / 100).round(3)

    result = merged[["PLAYER_ID", "Season", "DFGA", "DFGA_per100", "Diff_pct", "rDiff_pct", "RimPointsSaved"]].dropna(subset=["PLAYER_ID"])
    result["PLAYER_ID"] = result["PLAYER_ID"].astype(int)

    return (
        result.groupby(["PLAYER_ID", "Season"], as_index=False)
        .agg(
            DFGA=("DFGA", "sum"),
            DFGA_per100=("DFGA_per100", "mean"),
            Diff_pct=("Diff_pct", "mean"),
            rDiff_pct=("rDiff_pct", "mean"),
            RimPointsSaved=("RimPointsSaved", "mean"),
        )
    )


# ---------------------------------------------------------------------------
# 4. Close-Out Defense (opponents' FGM/FGA within 6 ft)
# ---------------------------------------------------------------------------

def compute_close6_defense(
    box_df: pd.DataFrame,
    min_year: int = _SYNERGY_YEAR_RANGE[0],
    max_year: int = _SYNERGY_YEAR_RANGE[1],
) -> pd.DataFrame:
    """Opponent shooting stats when the player is the primary closeout defender.

    Source: site_Data/{year}/player_tracking/close_6.csv
    Columns: PLAYER, DFGM, DFGA, DFG%, FG%, Diff%

    Returns per-100 defended FGA and the player's FG% differential when
    contesting perimeter/mid-range shots within 6 feet.
    """
    c6 = _load_tracking_files("player_tracking/close_6.csv", min_year, max_year)
    if c6.empty:
        return pd.DataFrame(columns=["PLAYER_ID", "Season", "C6_DFGA_per100"])

    diff_cols = [c for c in c6.columns if c.upper() == "DIFF%"]
    if diff_cols:
        c6["C6_Diff_pct"] = pd.to_numeric(c6[diff_cols[0]], errors="coerce")
        for extra in diff_cols[1:]:
            c6["C6_Diff_pct"] = c6["C6_Diff_pct"].fillna(pd.to_numeric(c6[extra], errors="coerce"))

    for col in ["DFGM", "DFGA"]:
        c6[col] = pd.to_numeric(c6.get(col, pd.Series()), errors="coerce")

    bk = _box_key(box_df, "DefPoss")
    merged = _merge_by_name(c6, bk, how="left")

    dp = merged["DefPoss"].replace(0, np.nan)
    merged["C6_DFGA_per100"] = (merged["DFGA"] / dp * 100).round(2)

    cols = ["PLAYER_ID", "Season", "C6_DFGA_per100"]
    if "C6_Diff_pct" in merged.columns:
        merged["C6_Diff_pct"] = merged["C6_Diff_pct"].round(2)
        cols.append("C6_Diff_pct")

    result = merged[cols].dropna(subset=["PLAYER_ID"])
    result["PLAYER_ID"] = result["PLAYER_ID"].astype(int)

    return (
        result.groupby(["PLAYER_ID", "Season"], as_index=False)
        .agg({c: "mean" for c in cols if c not in ["PLAYER_ID", "Season"]})
    )


# ---------------------------------------------------------------------------
# 5. Hustle Stats
# ---------------------------------------------------------------------------

def compute_hustle_stats(
    box_df: pd.DataFrame,
    min_year: int = _SYNERGY_YEAR_RANGE[0],
    max_year: int = _SYNERGY_YEAR_RANGE[1],
) -> pd.DataFrame:
    """Deflections, ChargesDrawn, contested shots from hustle/hustle.csv.

    Hustle data begins in 2018; seasons before that return NaN.

    Returns DataFrame:
        PLAYER_ID, Season, Position,
        Deflections_per100, ChargesDrawn_per100,
        Contested2PT_per100, Contested3PT_per100
    """
    hustle = _load_tracking_files("hustle/hustle.csv", min_year, max_year)
    if hustle.empty:
        return pd.DataFrame(columns=["PLAYER_ID", "Season", "Deflections_per100"])

    # Normalise non-breaking-space in column names (common in these files)
    hustle.columns = [c.replace("\xa0", " ").strip() for c in hustle.columns]

    # 2025+ hustle files use UPPERCASE_UNDERSCORE names; map to canonical names.
    # After concat, both old-format (e.g., "Deflections") and new-format
    # (e.g., "DEFLECTIONS") may coexist — coalesce them into one canonical column.
    _HUSTLE_ALIASES: dict[str, list[str]] = {
        "Deflections":        ["DEFLECTIONS"],
        "ChargesDrawn":       ["CHARGES_DRAWN"],
        "Contested2PT Shots": ["CONTESTED_SHOTS_2PT"],
        "Contested3PT Shots": ["CONTESTED_SHOTS_3PT"],
        "ScreenAssists":      ["SCREEN_ASSISTS"],
        "DistMilesOff":       ["DIST_MILES_OFF"],
    }
    for canonical, aliases in _HUSTLE_ALIASES.items():
        for alias in aliases:
            if alias in hustle.columns:
                if canonical in hustle.columns:
                    hustle[canonical] = hustle[canonical].fillna(hustle[alias])
                else:
                    hustle = hustle.rename(columns={alias: canonical})

    for col in ["Deflections", "ChargesDrawn", "Contested2PT Shots", "Contested3PT Shots"]:
        if col in hustle.columns:
            hustle[col] = pd.to_numeric(hustle[col], errors="coerce")

    # Position column for passer rating height lookup
    pos_col = next((c for c in hustle.columns if c.upper() in ("POSITION", "POS")), None)
    if pos_col:
        hustle = hustle.rename(columns={pos_col: "Position"})

    bk = _box_key(box_df, "DefPoss")
    merged = _merge_by_name(hustle, bk, how="left")

    dp = merged["DefPoss"].replace(0, np.nan)
    op = merged["OffPoss"].replace(0, np.nan) if "OffPoss" in merged.columns else dp

    out = merged[["PLAYER_ID", "Season"]].copy()
    for stat, poss in [
        ("Deflections",        dp),
        ("ChargesDrawn",       dp),
        ("Contested2PT Shots", dp),
        ("Contested3PT Shots", dp),
        ("ScreenAssists",      op),   # screen assists → per 100 off poss
    ]:
        if stat in merged.columns:
            out[stat.replace(" ", "") + "_per100"] = (merged[stat] / poss * 100).round(3)

    # Distance traveled on offense (raw miles per game — not per-100, already a rate)
    if "DistMilesOff" in merged.columns:
        out["DistMilesOff"] = pd.to_numeric(merged["DistMilesOff"], errors="coerce").round(3)

    if "Position" in merged.columns:
        out["Position"] = merged["Position"]

    result = out.dropna(subset=["PLAYER_ID"])
    result["PLAYER_ID"] = result["PLAYER_ID"].astype(int)

    agg_dict: dict = {}
    for col in result.columns:
        if col in ("PLAYER_ID", "Season"):
            continue
        if col == "Position":
            agg_dict[col] = (col, "first")
        else:
            agg_dict[col] = (col, "mean")

    return result.groupby(["PLAYER_ID", "Season"], as_index=False).agg(**agg_dict)


# ---------------------------------------------------------------------------
# 6. fTOV — Forced Turnovers / Stops Rate
# ---------------------------------------------------------------------------

def compute_ftov(
    box_df: pd.DataFrame,
    hustle_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forced turnover / stop rate per 100 defensive possessions.

    fTOV_per100 = (STL + Deflections + ChargesDrawn) / DefPoss × 100

    Deflections and ChargesDrawn come from hustle_df if provided;
    otherwise only steals contribute.

    Returns DataFrame: PLAYER_ID, Season, fTOV_per100
    """
    df = box_df[["PLAYER_ID", "Season", "STL", "DefPoss"]].copy()
    df["STL"] = pd.to_numeric(df["STL"], errors="coerce").fillna(0)
    df["DefPoss"] = pd.to_numeric(df["DefPoss"], errors="coerce")
    df["_stops"] = df["STL"]

    if hustle_df is not None and not hustle_df.empty:
        hustle_slim = hustle_df[["PLAYER_ID", "Season"] + [c for c in hustle_df.columns if c in ("Deflections_per100", "ChargesDrawn_per100")]].copy()
        df = df.merge(hustle_slim, on=["PLAYER_ID", "Season"], how="left")
        dp = df["DefPoss"].replace(0, np.nan)
        # Convert per-100 back to raw counts, then add to stops
        if "Deflections_per100" in df.columns:
            df["_stops"] += (df["Deflections_per100"].fillna(0) / 100 * dp).fillna(0)
        if "ChargesDrawn_per100" in df.columns:
            df["_stops"] += (df["ChargesDrawn_per100"].fillna(0) / 100 * dp).fillna(0)

    dp = df["DefPoss"].replace(0, np.nan)
    df["fTOV_per100"] = (df["_stops"] / dp * 100).round(3)

    return df[["PLAYER_ID", "Season", "fTOV_per100"]]


# ---------------------------------------------------------------------------
# 7. Advanced Box Features
# ---------------------------------------------------------------------------

def compute_advanced_box(box_df: pd.DataFrame) -> pd.DataFrame:
    """Compute miscellaneous box-score features per player-season.

    All rate stats normalised per-100 offensive or defensive possessions.

    Features:
        Rebounding (per 100 def poss):
            OREB_Contest_per100, OREB_Uncontest_per100, OREB_Defer_per100
            DREB_Contest_per100, DREB_Uncontest_per100, DREB_Defer_per100

        Shot difficulty (per 100 off poss):
            TightThree_per100   = (tight_FG3A + very_tight_FG3A) / OffPoss × 100
                                  — measures 3-point shot creation under pressure

        Rim creation (per 100 off poss):
            RimAssists_per100   = AtRimAssists / OffPoss × 100

        Assisted field goals (raw percentages, no normalisation needed):
            Assisted2s_pct      = PtsAssisted2s / (FG2M × 2)
            Assisted3s_pct      = PtsAssisted3s / (FG3M × 3)

        Offensive fouls drawn (per 100 off poss):
            OffFoulsDrawn_per100

    Returns DataFrame with PLAYER_ID, Season, and all above columns.
    """
    df = box_df.copy()
    op = df["OffPoss"].replace(0, np.nan)
    dp = df["DefPoss"].replace(0, np.nan)

    out = df[["PLAYER_ID", "Season"]].copy()

    # Rebounding splits
    for stat, poss_ser in [
        ("OREB_CONTEST",      op), ("OREB_UNCONTEST", op), ("OREB_CHANCE_DEFER", op),
        ("DREB_CONTEST",      dp), ("DREB_UNCONTEST", dp), ("DREB_CHANCE_DEFER", dp),
    ]:
        col = stat.replace("CHANCE_", "")  # shorten OREB_CHANCE_DEFER → OREB_DEFER
        col = col.replace("_CONTEST", "_Contest").replace("_UNCONTEST", "_Uncontest").replace("_DEFER", "_Defer")
        if stat in df.columns:
            out[f"{col}_per100"] = (pd.to_numeric(df[stat], errors="coerce") / poss_ser * 100).round(2)

    # Tight 3s under pressure
    t3a = pd.to_numeric(df.get("tight_FG3A", 0), errors="coerce").fillna(0)
    vt3a = pd.to_numeric(df.get("very_tight_FG3A", 0), errors="coerce").fillna(0)
    out["TightThree_per100"] = ((t3a + vt3a) / op * 100).round(2)

    # Rim assists
    if "AtRimAssists" in df.columns:
        out["RimAssists_per100"] = (pd.to_numeric(df["AtRimAssists"], errors="coerce") / op * 100).round(2)

    # Assisted FG rates
    fg2m = pd.to_numeric(df.get("FG2M", pd.Series(np.nan, index=df.index)), errors="coerce")
    # FG3M is derived: FG3A × FG3_PCT (FG3M is not stored directly in box scores)
    fg3a = pd.to_numeric(df.get("FG3A", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    fg3p = pd.to_numeric(df.get("FG3_PCT", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0)
    fg3m = fg3a * fg3p
    pts_ast2 = pd.to_numeric(df.get("PtsAssisted2s", pd.Series(np.nan, index=df.index)), errors="coerce")
    pts_ast3 = pd.to_numeric(df.get("PtsAssisted3s", pd.Series(np.nan, index=df.index)), errors="coerce")
    out["Assisted2s_pct"] = (pts_ast2 / (fg2m * 2).where((fg2m * 2) > 0)).round(3)
    out["Assisted3s_pct"] = (pts_ast3 / (fg3m * 3).where((fg3m * 3) > 0)).round(3)

    # Offensive fouls drawn
    if "Offensive Fouls Drawn" in df.columns:
        out["OffFoulsDrawn_per100"] = (
            pd.to_numeric(df["Offensive Fouls Drawn"], errors="coerce") / op * 100
        ).round(2)

    # Blocks per 100 defensive possessions
    if "BLK" in df.columns:
        out["BLK_per100"] = (pd.to_numeric(df["BLK"], errors="coerce") / dp * 100).round(2)

    return out


# ---------------------------------------------------------------------------
# 8. Passing Load
# ---------------------------------------------------------------------------

def compute_passing_load(
    box_df: pd.DataFrame,
    min_year: int = _SYNERGY_YEAR_RANGE[0],
    max_year: int = _SYNERGY_YEAR_RANGE[1],
) -> pd.DataFrame:
    """Potential assists and secondary assists per 100 off possessions.

    PotentialAST counts passes that *would* be assists if the shot went in —
    a playmaking signal independent of teammate shooting quality.
    SecondaryAST counts the assist on the assist.
    """
    pas = _load_tracking_files("player_tracking/passing.csv", min_year, max_year)
    if pas.empty:
        return pd.DataFrame(columns=["PLAYER_ID", "Season", "PotentialAst_per100"])

    for col in ["PotentialAST", "SecondaryAST"]:
        pas[col] = pd.to_numeric(pas[col], errors="coerce")

    bk = _box_key(box_df, "OffPoss")
    merged = _merge_by_name(pas, bk, how="left")

    op = merged["OffPoss"].replace(0, np.nan)
    merged["PotentialAst_per100"] = (merged["PotentialAST"] / op * 100).round(2)
    merged["SecondaryAst_per100"] = (merged["SecondaryAST"] / op * 100).round(2)

    result = merged[["PLAYER_ID", "Season", "PotentialAst_per100", "SecondaryAst_per100"]].dropna(subset=["PLAYER_ID"])
    result["PLAYER_ID"] = result["PLAYER_ID"].astype(int)

    return (
        result.groupby(["PLAYER_ID", "Season"], as_index=False)
        .agg(PotentialAst_per100=("PotentialAst_per100", "mean"),
             SecondaryAst_per100=("SecondaryAst_per100", "mean"))
    )


# ---------------------------------------------------------------------------
# 9. Drive Efficiency  (secondary — excluded from main output by default)
# ---------------------------------------------------------------------------

def compute_drive_efficiency(
    box_df: pd.DataFrame,
    min_year: int = _SYNERGY_YEAR_RANGE[0],
    max_year: int = _SYNERGY_YEAR_RANGE[1],
) -> pd.DataFrame:
    """Drive-based efficiency metrics (excluded from compute_all_features).

    Available via direct call if needed; not included in the main feature table
    per project design decision.
    """
    drv = _load_tracking_files("player_tracking/drives.csv", min_year, max_year)
    if drv.empty:
        return pd.DataFrame(columns=["PLAYER_ID", "Season", "Drives_per100"])

    for col in ["DRIVES", "FGM", "FGA", "PTS", "AST"]:
        drv[col] = pd.to_numeric(drv[col], errors="coerce")

    bk = _box_key(box_df, "OffPoss")
    merged = _merge_by_name(drv, bk, how="left")

    op = merged["OffPoss"].replace(0, np.nan)
    merged["Drives_per100"]   = (merged["DRIVES"] / op * 100).round(2)
    merged["DrivePts_per100"] = (merged["PTS"] / op * 100).round(2)
    merged["DriveFG_pct"]     = (merged["FGM"] / merged["FGA"].replace(0, np.nan)).round(3)
    merged["DriveAst_pct"]    = (merged["AST"] / merged["DRIVES"].replace(0, np.nan)).round(3)

    result = merged[["PLAYER_ID", "Season", "Drives_per100", "DrivePts_per100", "DriveFG_pct", "DriveAst_pct"]].dropna(subset=["PLAYER_ID"])
    result["PLAYER_ID"] = result["PLAYER_ID"].astype(int)

    return (
        result.groupby(["PLAYER_ID", "Season"], as_index=False)
        .agg(Drives_per100=("Drives_per100", "mean"),
             DrivePts_per100=("DrivePts_per100", "mean"),
             DriveFG_pct=("DriveFG_pct", "mean"),
             DriveAst_pct=("DriveAst_pct", "mean"))
    )


# ---------------------------------------------------------------------------
# 10. Master feature assembler
# ---------------------------------------------------------------------------

def compute_all_features(
    box_df: pd.DataFrame,
    min_year: int = _SYNERGY_YEAR_RANGE[0],
    max_year: int = _SYNERGY_YEAR_RANGE[1],
) -> pd.DataFrame:
    """Compute all engineered features and join into a single player-season table.

    Drive efficiency is intentionally excluded from this output (call
    compute_drive_efficiency() directly if needed).

    Args:
        box_df:   load_box_scores() output.
        min_year: First season to include.
        max_year: Last season to include.

    Returns DataFrame indexed by (PLAYER_ID, Season) with all feature columns.
    """
    box = box_df[(box_df["Season"] >= min_year) & (box_df["Season"] <= max_year)].copy()

    # --- Compute all feature groups ---
    creation  = compute_box_creation(box)
    hustle    = compute_hustle_stats(box, min_year, max_year)

    # Build position table from hustle (for passer rating height)
    pos_df: pd.DataFrame | None = None
    if not hustle.empty and "Position" in hustle.columns:
        pos_df = hustle[["PLAYER_ID", "Season", "Position"]].dropna(subset=["Position"])

    passer    = compute_passer_rating(box, creation, pos_df)
    rim       = compute_rim_points_saved(box, min_year, max_year)
    close6    = compute_close6_defense(box, min_year, max_year)
    ftov      = compute_ftov(box, hustle)
    adv_box   = compute_advanced_box(box)
    passing   = compute_passing_load(box, min_year, max_year)

    # --- Join everything onto the box skeleton ---
    base = box[["PLAYER_ID", "Player", "Season", "Minutes", "AGE"]].copy()

    def _left(df: pd.DataFrame, keys: list[str] = ["PLAYER_ID", "Season"]) -> pd.DataFrame:
        cols = [c for c in df.columns if c not in ["Player", "Minutes", "AGE"]]
        return base.merge(df[cols], on=keys, how="left")

    result = _left(creation)
    result = result.merge(passer[["PLAYER_ID", "Season", "PasserRating"]], on=["PLAYER_ID", "Season"], how="left")
    result = result.merge(rim,    on=["PLAYER_ID", "Season"], how="left")
    result = result.merge(close6, on=["PLAYER_ID", "Season"], how="left")
    result = result.merge(hustle[[c for c in hustle.columns if c not in ["Position"]]], on=["PLAYER_ID", "Season"], how="left")
    result = result.merge(ftov,   on=["PLAYER_ID", "Season"], how="left")
    result = result.merge(adv_box, on=["PLAYER_ID", "Season"], how="left")
    result = result.merge(passing, on=["PLAYER_ID", "Season"], how="left")

    return result.sort_values(["Season", "PLAYER_ID"]).reset_index(drop=True)
