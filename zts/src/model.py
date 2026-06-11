"""
Box-score SPM model + retrodiction testing.

Pipeline
--------
1. Assemble box + zTS + engineered features + contextual features.
2. Join 3-year RAPM as the target (by normalised player name + season).
3. 3-year trailing rolling average of all feature columns per player.
4. Train separate Ridge + XGBoost models predicting offensive RAPM and
   defensive RAPM (and total = off + def).
5. Export box-model predictions as prior.csv (PLAYER_ID, SPM_O, SPM_D
   in per-possession units) — the format expected by JE_rapm/rapm_with_prior.py.
   Feed this into a ridge-with-prior RAPM run to get the blended final metric.
6. Retrodiction test (Pearson r², Krishna Narsu LEBRON methodology):
     - Predict player value for year Y
     - Multiply by actual year-Y+1 minutes
     - Sum by team → Pearson r² vs actual year-Y+1 wins (scraped from
       basketball-reference)
   Run for BOTH the box model alone AND the ridge-with-prior blended metric.

Season convention
-----------------
All DataFrames use an integer Season = ending calendar year of the NBA
season (e.g. 2024 = 2023-24 season).

RAPM prior methodology (JE approach)
-------------------------------------
The correct way to use a box-model as a prior for RAPM is NOT a post-hoc
Bayesian blend.  Instead, the prior vector μ is subtracted from the target
before the ridge solve, then added back:

    offset   = X @ μ          (prior's contribution to each possession)
    y_adj    = y - offset      (residual the ridge needs to explain)
    raw_coef = Ridge(X, y_adj, α)    (ridge toward 0 on the residual)
    final    = raw_coef + μ    (total = residual + prior)

This is equivalent to ridge regularization toward μ instead of toward 0:
    min ||y - Xθ||² + α||θ - μ||²
This formulation is available in rapm_with_prior.py (requires MySQL matchup DB).
write_prior_csv() exports μ in the expected format so you can run it there.
"""

from __future__ import annotations

import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

_HERE      = pathlib.Path(__file__).resolve().parent.parent
_SITE_DATA = _HERE.parent / "site_Data"
_CACHE_DIR = _HERE / "data" / "processed"

# --- NBA team name → abbreviation (basketball-reference naming) ---
_TEAM_ABV: dict[str, str] = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS",
    # Historical names
    "New Jersey Nets": "NJN", "New Orleans Hornets": "NOH", "Charlotte Bobcats": "CHA",
    "Seattle SuperSonics": "SEA", "Vancouver Grizzlies": "MEM",
    "New Orleans/Oklahoma City Hornets": "NOK",
}

SEASONS         = list(range(2014, 2027))
MAX_GAMES       = 82
REPL_MIN_THRESH = 200    # minutes — players below this get replacement value
ROOKIE_BONUS    = 0.5    # RAPM points above replacement for rookies
POSS_PER_MIN    = 2.05   # rough NBA average possessions per minute

# Feature columns used in the model (after rolling average)
# NOTE: rTS (raw True Shooting vs league avg) is intentionally excluded.
# zTS (playtype-adjusted TS) subsumes it and is strictly more informative.
# Including both creates collinearity that distorts Ridge coefficients.
MODEL_FEATURES = [
    # Scoring / efficiency
    "PTS_per100", "ScoringValue", "zTS", "ThreePtP",
    # Creation / load
    "Creation", "Load", "cTOV_pct", "PasserRating",
    # Rim offense
    "TightThree_per100", "RimAssists_per100",
    # Assisted shots
    "Assisted2s_pct", "Assisted3s_pct",
    # Playmaking (tracking)
    "PotentialAst_per100", "SecondaryAst_per100",
    # Screen creation
    "ScreenAssists_per100",
    # Defense — rim
    "RimPointsSaved", "BLK_per100",
    # Defense — perimeter
    "C6_Diff_pct",
    # Hustle / forced stops
    "fTOV_per100", "Deflections_per100",
    # Rebounding
    "DREB_Uncontest_per100", "OREB_Contest_per100",
    # Movement
    "DistMilesOff",
    # Context
    "GP_pct", "team_ortg_ctx", "team_drtg_ctx",
]

# Feature split for separate O-RAPM / D-RAPM models
OFFENSE_FEATURES = [
    "PTS_per100", "ScoringValue", "zTS", "ThreePtP", "Creation", "Load", "cTOV_pct",
    "PasserRating", "TightThree_per100", "RimAssists_per100",
    "Assisted2s_pct", "Assisted3s_pct", "PotentialAst_per100", "SecondaryAst_per100",
    "ScreenAssists_per100", "DistMilesOff",
    "GP_pct", "team_ortg_ctx",
]
DEFENSE_FEATURES = [
    "RimPointsSaved", "BLK_per100", "C6_Diff_pct", "fTOV_per100", "Deflections_per100",
    "DREB_Uncontest_per100", "OREB_Contest_per100",
    "GP_pct", "team_drtg_ctx",
]


# ===========================================================================
# Team wins + team ratings — scrape from basketball-reference
# ===========================================================================

def _parse_team_ratings_table(t: pd.DataFrame, season: int) -> list[dict]:
    """Parse a basketball-reference misc/team-ratings table into rows.

    The table has the real column names in row 0 (pandas reads the multi-level
    header as Unnamed: N).  Row 0 values: Rk, Team, Age, W, L, ..., ORtg, DRtg.
    """
    header = t.iloc[0].tolist()
    data   = t.iloc[1:].copy()
    data.columns = header
    data = data[data["Rk"].notna() & (data["Rk"] != "Rk")]  # drop sub-headers

    rows = []
    for _, row in data.iterrows():
        raw_name = str(row.get("Team", ""))
        if not raw_name or raw_name in ("nan", "League Average"):
            continue
        clean = raw_name.split("(")[0].replace("*", "").replace("†", "").strip()
        abv = _TEAM_ABV.get(clean)
        if abv is None:
            for full, a in _TEAM_ABV.items():
                if clean.lower() in full.lower() or full.split()[-1].lower() in clean.lower():
                    abv = a
                    break
        if abv is None:
            continue
        ortg = pd.to_numeric(row.get("ORtg", ""), errors="coerce")
        drtg = pd.to_numeric(row.get("DRtg", ""), errors="coerce")
        wins = pd.to_numeric(row.get("W",    ""), errors="coerce")
        if pd.isna(ortg):
            continue
        rows.append({
            "Team": abv, "Season": season,
            "team_ortg": float(ortg),
            "team_drtg": float(drtg) if not pd.isna(drtg) else np.nan,
            "team_wins": int(wins) if not pd.isna(wins) else np.nan,
        })
    return rows


def scrape_team_ratings(
    seasons: list[int] | None = None,
    cache_path: pathlib.Path | None = None,
    sleep_s: float = 1.5,
) -> pd.DataFrame:
    """Scrape team ORtg, DRtg, and wins from basketball-reference.

    Source: https://www.basketball-reference.com/leagues/NBA_{season}.html
    Table 10 on that page has the misc team ratings (ORtg, DRtg, Pace, …).

    Cached to data/processed/team_ratings.csv.  Re-uses cache if all
    requested seasons are already present.

    Returns: Team (abbrev), Season (int), team_ortg, team_drtg, team_wins
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = cache_path or (_CACHE_DIR / "team_ratings.csv")

    if seasons is None:
        seasons = SEASONS

    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached["Season"] = cached["Season"].astype(int)
        missing = [s for s in seasons if s not in set(cached["Season"])]
    else:
        cached = pd.DataFrame()
        missing = list(seasons)

    if not missing:
        return cached[cached["Season"].isin(seasons)].reset_index(drop=True)

    print(f"  Scraping team ratings for seasons: {missing}")
    new_rows: list[dict] = []
    for season in missing:
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
        try:
            tables = pd.read_html(url, header=0)
            # Table index 10 is the Misc / Team Ratings table (ORtg, DRtg, Pace…)
            # Find it by looking for a table whose first-row values contain 'ORtg'
            for tbl in tables:
                first_row = tbl.iloc[0].astype(str).tolist()
                if "ORtg" in first_row:
                    new_rows.extend(_parse_team_ratings_table(tbl, season))
                    break
        except Exception as e:
            print(f"    Warning: could not scrape ratings for {season}: {e}")
        time.sleep(sleep_s)

    if new_rows:
        new_df   = pd.DataFrame(new_rows).drop_duplicates(["Team", "Season"])
        combined = pd.concat([cached, new_df], ignore_index=True).drop_duplicates(["Team","Season"])
        combined.to_csv(cache_path, index=False)
        print(f"  Saved team_ratings.csv ({len(combined)} rows)")
    else:
        combined = cached

    return combined[combined["Season"].isin(seasons)].reset_index(drop=True)


def scrape_team_wins(
    seasons: list[int] | None = None,
    cache_path: pathlib.Path | None = None,
    sleep_s: float = 1.5,
) -> pd.DataFrame:
    """Thin wrapper: return just Team/Season/team_wins from scrape_team_ratings.

    Kept for backward compatibility with existing notebook code.
    """
    ratings = scrape_team_ratings(seasons=seasons, sleep_s=sleep_s)
    return ratings[["Team", "Season", "team_wins"]].dropna(subset=["team_wins"]).copy()


# ===========================================================================
# Contextual features
# ===========================================================================

def compute_contextual_features(
    box_df: pd.DataFrame,
    team_ratings_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute player-season contextual features.

    Features:
        PTS_per100        Points per 100 offensive possessions
        GP_pct            Games appeared / 82
        team_ortg         Actual team ORtg (from basketball-reference)
        team_drtg         Actual team DRtg (from basketball-reference)
        team_ortg_ctx     team_ortg × player share of team minutes
        team_drtg_ctx     team_drtg × player share of team minutes

    NOTE: team ORtg/DRtg are scraped directly from basketball-reference
    (scrape_team_ratings).  Computing them from player box data is wrong
    because summing player OffPoss double/triple-counts shared possessions
    among teammates.

    `team_ratings_df` should be the output of scrape_team_ratings().
    If absent, team context features will be NaN.
    """
    df = box_df.copy()
    has_team = "Team" in df.columns and df["Team"].notna().any()

    op = df["OffPoss"].replace(0, np.nan)
    df["PTS_per100"] = (df["PTS"] / op * 100).round(2)
    df["GP_pct"]     = (pd.to_numeric(df["GP"], errors="coerce") / MAX_GAMES).clip(0, 1).round(3)

    if has_team and team_ratings_df is not None and not team_ratings_df.empty:
        # Sum minutes per team-season from player data (this aggregation IS correct)
        team_mins = (
            df.groupby(["Team", "Season"])["Minutes"]
            .sum()
            .rename("team_min")
            .reset_index()
        )
        # Join actual ORtg/DRtg from scraped data
        ratings = team_ratings_df[["Team","Season","team_ortg","team_drtg"]].copy()
        team_info = team_mins.merge(ratings, on=["Team","Season"], how="left")

        df = df.merge(team_info, on=["Team","Season"], how="left")
        df["player_min_share"] = (df["Minutes"] / df["team_min"].replace(0, np.nan)).clip(0, 1)
        df["team_ortg_ctx"]    = (df["team_ortg"] * df["player_min_share"]).round(3)
        df["team_drtg_ctx"]    = (df["team_drtg"] * df["player_min_share"]).round(3)
    else:
        for col in ["team_ortg","team_drtg","team_ortg_ctx","team_drtg_ctx"]:
            df[col] = np.nan

    keep = ["PLAYER_ID","Season","PTS_per100","GP_pct","team_ortg","team_drtg","team_ortg_ctx","team_drtg_ctx"]
    if has_team:
        keep = ["Team"] + keep
    return df[[c for c in keep if c in df.columns]].copy()


# ===========================================================================
# Feature assembly + rolling average
# ===========================================================================

def assemble_model_data(
    box_df: pd.DataFrame,
    zts_df: pd.DataFrame,
    features_df: pd.DataFrame,
    contextual_df: pd.DataFrame,
    rapm3_df: pd.DataFrame,    # 3-year RAPM with _name_key
    rapm1_df: pd.DataFrame,    # 1-year RAPM
    lebron_df: pd.DataFrame | None = None,  # for Team column if needed
) -> pd.DataFrame:
    """Join all sources into one wide player-season table.

    Returns DataFrame with PLAYER_ID, Season, Player, Minutes, Team (if
    available), all feature columns, and RAPM target columns.
    """
    from load_rapm import join_rapm_to_box

    # Keep Team from box_df if already present (2025+ mega-sheets have TEAM_ABBREVIATION)
    box_base_cols = ["PLAYER_ID", "Season", "Player", "Minutes", "GP"]
    if "Team" in box_df.columns:
        box_base_cols.append("Team")
    base = box_df[box_base_cols].copy()

    # Attach Team from LEBRON data if provided and Team not already in base
    if lebron_df is not None and "Team" in lebron_df.columns:
        if "Team" not in base.columns:
            base = base.merge(
                lebron_df[["PLAYER_ID", "Season", "Team"]],
                on=["PLAYER_ID", "Season"], how="left",
            )
        else:
            # Fill missing teams for older seasons from LEBRON
            team_patch = lebron_df[["PLAYER_ID", "Season", "Team"]].rename(
                columns={"Team": "_Team_lb"}
            )
            base = base.merge(team_patch, on=["PLAYER_ID", "Season"], how="left")
            base["Team"] = base["Team"].fillna(base["_Team_lb"])
            base = base.drop(columns=["_Team_lb"])

    # zTS columns
    zts_cols = ["PLAYER_ID","Season"] + [c for c in ["rTS","zTS","Difficulty","ExpectedTS","TS_pct"]
                                          if c in zts_df.columns]
    base = base.merge(zts_df[zts_cols], on=["PLAYER_ID","Season"], how="left")

    # Engineered features
    feat_cols = ["PLAYER_ID","Season"] + [c for c in features_df.columns
                                           if c not in base.columns and c not in ["PLAYER_ID","Season","Player","Minutes","Team","GP"]]
    base = base.merge(features_df[feat_cols], on=["PLAYER_ID","Season"], how="left")

    # Contextual
    ctx_join = [c for c in contextual_df.columns if c not in ["Player","Minutes","GP"]]
    base = base.merge(contextual_df[ctx_join], on=["PLAYER_ID","Season"], how="left")

    # 3-year RAPM target (name-based join)
    base = join_rapm_to_box(rapm3_df, base, rapm_col="RAPM_3y")

    # 1-year RAPM (for prior blend)
    import unicodedata
    def _norm(s):
        def strip(t):
            nfkd = unicodedata.normalize("NFKD", str(t))
            return "".join(c for c in nfkd if not unicodedata.combining(c))
        return s.fillna("").astype(str).str.lower().str.strip().apply(strip)

    rapm1_slim = rapm1_df[["_name_key","Season","ORAPM_1y","DRAPM_1y","RAPM_1y"]].copy()
    base["_name_key"] = _norm(base["Player"])
    base = base.merge(rapm1_slim, on=["_name_key","Season"], how="left")
    base = base.drop(columns=["_name_key"], errors="ignore")

    # Coalesce any Team_x / Team_y collision from intermediate merges
    if "Team" not in base.columns:
        for suffix in ("_x", "_y"):
            cand = f"Team{suffix}"
            if cand in base.columns:
                base = base.rename(columns={cand: "Team"})
                break
    # Drop any leftover Team_x / Team_y after coalescing
    for col in [c for c in base.columns if c.startswith("Team_")]:
        base = base.drop(columns=[col])

    # ScoringValue = 2 × TSA_per100 × (zTS / 100)
    # Combines volume (how many scoring attempts per 100) with playtype-adjusted
    # efficiency above baseline.  zTS is in percentage-point units, so /100 gives decimal.
    if "zTS" in base.columns:
        fga = pd.to_numeric(box_df.set_index(["PLAYER_ID","Season"]).reindex(
            pd.MultiIndex.from_frame(base[["PLAYER_ID","Season"]])
        )["FGA"].values, errors="coerce")
        fta = pd.to_numeric(box_df.set_index(["PLAYER_ID","Season"]).reindex(
            pd.MultiIndex.from_frame(base[["PLAYER_ID","Season"]])
        )["FTA"].values, errors="coerce")
        op  = pd.to_numeric(box_df.set_index(["PLAYER_ID","Season"]).reindex(
            pd.MultiIndex.from_frame(base[["PLAYER_ID","Season"]])
        )["OffPoss"].values, errors="coerce")
        tsa_per100 = (fga + 0.44 * fta) / np.where(op > 0, op, np.nan) * 100
        base["ScoringValue"] = (2 * tsa_per100 * (base["zTS"].values / 100)).round(3)

    return base.sort_values(["Season","PLAYER_ID"]).reset_index(drop=True)


def compute_rolling_features(
    df: pd.DataFrame,
    window: int = 3,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Trailing N-year rolling average per player for all feature columns."""
    if feature_cols is None:
        feature_cols = [c for c in MODEL_FEATURES if c in df.columns]

    rolled_parts = []
    for pid, grp in df.sort_values("Season").groupby("PLAYER_ID"):
        g = grp.set_index("Season")[feature_cols].sort_index()
        g_rolled = g.rolling(window=window, min_periods=1).mean()
        g_rolled.index.name = "Season"
        g_rolled = g_rolled.reset_index()
        g_rolled["PLAYER_ID"] = pid
        rolled_parts.append(g_rolled)

    rolled = pd.concat(rolled_parts, ignore_index=True)
    meta_cols = [c for c in df.columns if c not in feature_cols]
    out = df[meta_cols].merge(rolled, on=["PLAYER_ID","Season"], how="left")
    return out


# ===========================================================================
# Model training
# ===========================================================================

def _impute(X: pd.DataFrame, medians: dict | None = None) -> tuple[np.ndarray, dict]:
    """Median-impute a feature DataFrame. Returns (array, medians_dict)."""
    if medians is None:
        medians = {col: float(X[col].median()) for col in X.columns}
    X2 = X.copy()
    for col in X2.columns:
        fv = medians.get(col, 0.0)
        X2[col] = X2[col].fillna(0.0 if pd.isna(fv) else fv)
    return X2.values, medians


def build_train_matrix(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    min_minutes: int = 200,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    """Build imputed X and y for a single target column.

    Uses the 3-year rolling-averaged features so that the training input
    and target (3-year RAPM) operate at the same temporal resolution.

    Returns (X_df, X_array, y_array, global_medians).
    """
    avail = [c for c in feature_cols if c in model_df.columns]
    df = model_df[(model_df["Minutes"] >= min_minutes)].dropna(subset=[target]).copy()
    X, medians = _impute(df[avail])
    y = df[target].values
    return df[avail], X, y, medians


def build_inference_matrix(
    raw_model_df: pd.DataFrame,
    feature_cols: list[str],
    season: int,
    medians: dict,
    min_minutes: int = 500,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build imputed feature matrix for a SINGLE season using raw (non-rolled) values.

    The training pipeline uses 3-year rolling features to match the 3-year RAPM
    target.  For out-of-sample seasons (no RAPM available), it is better to use
    the *current season's actual values* rather than a rolling mean that mixes in
    stale prior seasons — especially when a player's role or team has changed.

    Args:
        raw_model_df: assemble_model_data() output (before compute_rolling_features).
        feature_cols: list of feature column names.
        season:       integer season year (e.g. 2026).
        medians:      imputation medians from build_train_matrix (training set).
        min_minutes:  minimum playing time filter.

    Returns (sub_df, X_array).
    """
    avail = [c for c in feature_cols if c in raw_model_df.columns]
    sub = raw_model_df[
        (raw_model_df["Season"] == season) & (raw_model_df["Minutes"] >= min_minutes)
    ].copy()
    X_df = sub[avail].copy()
    for col in avail:
        fv = medians.get(col, 0.0)
        X_df[col] = X_df[col].fillna(0.0 if pd.isna(fv) else fv)
    return sub, X_df.values


def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple:
    """Fit Ridge regression; returns (model, scaler)."""
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m  = Ridge(alpha=alpha)
    m.fit(Xs, y)
    return m, sc


def train_xgboost(X: np.ndarray, y: np.ndarray) -> object:
    """Fit XGBoostRegressor."""
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost required: pip install xgboost")
    m = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=1.5, random_state=42, verbosity=0,
    )
    m.fit(X, y)
    return m


def train_mlp(X: np.ndarray, y: np.ndarray) -> tuple:
    """Fit a small MLP regressor; returns (model, scaler).

    Architecture: 64 → 32 → 16, ReLU, early stopping on 10% validation set.
    Requires StandardScaler pre-processing (same as Ridge).
    """
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    m  = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        learning_rate_init=0.001,
    )
    m.fit(Xs, y)
    return m, sc


def train_stacked_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> dict:
    """Stacked ensemble: XGB + Ridge + MLP base learners, Ridge meta-learner.

    Uses K-fold OOF predictions to train the meta-learner without leakage.
    Returns a dict with final base models (trained on all data) + meta Ridge.

    Prediction:
        preds = stack_predict(ensemble, X_new)
    """
    kf  = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros((len(y), 3))   # columns: XGB, Ridge, MLP

    for tr_idx, va_idx in kf.split(X):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr = y[tr_idx]
        # XGB
        xm = train_xgboost(Xtr, ytr)
        oof[va_idx, 0] = xm.predict(Xva)
        # Ridge
        rm, sc_r = train_ridge(Xtr, ytr, alpha=100)
        oof[va_idx, 1] = rm.predict(sc_r.transform(Xva))
        # MLP
        mm, sc_m = train_mlp(Xtr, ytr)
        oof[va_idx, 2] = mm.predict(sc_m.transform(Xva))

    meta = Ridge(alpha=1.0)
    meta.fit(oof, y)

    # Final base models trained on full training set
    xgb_f          = train_xgboost(X, y)
    ridge_f, sc_rf = train_ridge(X, y, alpha=100)
    mlp_f,   sc_mf = train_mlp(X, y)

    return {
        "xgb":      xgb_f,
        "ridge":    ridge_f,  "sc_ridge": sc_rf,
        "mlp":      mlp_f,    "sc_mlp":   sc_mf,
        "meta":     meta,
        "meta_weights": meta.coef_,    # XGB / Ridge / MLP blend weights
    }


def stack_predict(ensemble: dict, X: np.ndarray) -> np.ndarray:
    """Run stacked ensemble inference."""
    base_preds = np.column_stack([
        ensemble["xgb"].predict(X),
        ensemble["ridge"].predict(ensemble["sc_ridge"].transform(X)),
        ensemble["mlp"].predict(ensemble["sc_mlp"].transform(X)),
    ])
    return ensemble["meta"].predict(base_preds)


def predict(model, X_arr: np.ndarray, scaler=None) -> np.ndarray:
    Xa = scaler.transform(X_arr) if scaler else X_arr
    return model.predict(Xa)


# ===========================================================================
# Replacement / rookie values
# ===========================================================================

def compute_replacement_value(rapm3_df: pd.DataFrame, box_df: pd.DataFrame) -> float:
    """Median 3-year RAPM for players below the minute threshold."""
    import unicodedata
    def _norm(s):
        def strip(t):
            nfkd = unicodedata.normalize("NFKD", str(t))
            return "".join(c for c in nfkd if not unicodedata.combining(c))
        return s.fillna("").astype(str).str.lower().str.strip().apply(strip)

    box_slim = box_df[["PLAYER_ID","Season","Player","Minutes"]].copy()
    box_slim["_name_key"] = _norm(box_slim["Player"])
    r3 = rapm3_df[["_name_key","Season","RAPM_3y"]].copy()
    merged = box_slim.merge(r3, on=["_name_key","Season"], how="left")
    low = merged[merged["Minutes"] < REPL_MIN_THRESH]["RAPM_3y"]
    return float(low.median()) if len(low) > 0 else -2.5


# ===========================================================================
# Prior blend + prior CSV export
# ===========================================================================

def compute_prior_blend(
    model_df: pd.DataFrame,
    pred_col: str,
    raw1y_col: str,
    alpha: float = 4000,
    poss_per_min: float = POSS_PER_MIN,
    out_col: str = "RAPM_blended",
) -> pd.DataFrame:
    """Approximate Bayesian shrinkage of raw 1-year RAPM toward box prior.

    This approximates the JE ridge-with-prior formula under the assumption
    that the RAPM design matrix X'X ≈ N·I (players are roughly uncorrelated
    across possessions).  The exact formulation requires the matchup X matrix
    (see JE_rapm/rapm_with_prior.py and write_prior_csv() below).

    Formula:
        N     = player_minutes × poss_per_min   (proxy for single-season possessions)
        Final = (Prior × α + Raw × N) / (α + N)

    α controls shrinkage strength:
        α = 2000 → less shrinkage, raw 1yr RAPM dominates sooner
        α = 4000 → JE default
        α = 6000 → heavier shrinkage toward box prior

    Players without 1-year RAPM fall back to the box prior.
    """
    df  = model_df.copy()
    N   = df["Minutes"].fillna(0) * poss_per_min
    mu  = df[pred_col].fillna(0)
    raw = df[raw1y_col]
    blended = np.where(
        raw.notna(),
        (mu * alpha + raw * N) / (alpha + N),
        mu,
    )
    df[out_col] = blended
    return df


def write_prior_csv(
    model_df: pd.DataFrame,
    off_pred_col: str,
    def_pred_col: str,
    season: int,
    out_path: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Export box model predictions as prior.csv for rapm_with_prior.py.

    The JE RAPM prior format is:
        Season, PLAYER_ID, SPM_O, SPM_D
    where SPM_O / SPM_D are in per-possession units (divide per-100 by 100).

    Note on SPM_D sign convention: in the JE code, SPM_D represents the
    defensive *cost* per possession with this player on court, so a good
    defender has a *negative* SPM_D (fewer opponent points).  If your
    DRAPM_3y is positive = good defense (fewer pts allowed), negate when
    writing SPM_D.

    Usage:
        prior_df = write_prior_csv(rolled, "box_ORAPM", "box_DRAPM", 2024)
        # Then from JE_rapm/: python rapm_with_prior.py  (uses this prior.csv)
    """
    out_path = out_path or (_CACHE_DIR / "prior_export.csv")
    df = model_df[model_df["Season"] == season].copy()

    prior = pd.DataFrame({
        "Season":    df["Season"].values,
        "PLAYER_ID": df["PLAYER_ID"].values,
        "SPM_O":     (df[off_pred_col].fillna(0) / 100).round(6),
        # JE convention: SPM_D > 0 = good defender.
        # rapm_with_prior.py negates SPM_D when building the prior vector
        # so the ridge coefficient for good defenders ends up negative
        # (negative coef = reduces opponent scoring = good).
        "SPM_D":     (df[def_pred_col].fillna(0) / 100).round(6),
    }).drop_duplicates("PLAYER_ID")

    prior.to_csv(out_path, index=False)
    print(f"  Wrote {len(prior)} player priors for season {season} → {out_path}")
    return prior


# ===========================================================================
# Retrodiction test
# ===========================================================================

def retrodiction_test(
    model,
    model_df: pd.DataFrame,
    team_wins_df: pd.DataFrame,
    feature_cols: list[str],
    score_col: str | None = None,   # if provided, use pre-computed score column
    scaler=None,
    replacement_value: float = -2.5,
    test_seasons: list[int] | None = None,
    min_minutes_qualifier: int = 200,
    label: str = "Model",
    global_medians: dict | None = None,
) -> pd.DataFrame:
    """Retrodiction test using Pearson r².

    For each test season Y (within model_df):
      1. Predict All-in-One scores for year Y  (or use pre-computed score_col)
      2. Look up year-Y+1 roster & actual minutes
      3. Score assignment:
           returning player with score       → predicted score
           player with < min_minutes in Y+1  → replacement_value
           no Y score (rookie/new)           → replacement_value + ROOKIE_BONUS
      4. contribution = score × Y+1_minutes
      5. team_score = Σ contributions per team
      6. R² = Pearson r² between team_score and actual team_wins[Y+1]

    Returns DataFrame: Season_predicted, Season_test, R2, Pearson_r, Label
    """
    avail_feats = [c for c in feature_cols if c in model_df.columns]

    if global_medians is None and model is not None:
        global_medians = {c: float(model_df[c].median()) for c in avail_feats}

    if test_seasons is None:
        max_s = model_df["Season"].max()
        test_seasons = [s for s in sorted(model_df["Season"].unique()) if s < max_s]

    results = []
    for y in sorted(test_seasons):
        y_next = y + 1
        df_y    = model_df[model_df["Season"] == y].copy()
        df_next = model_df[model_df["Season"] == y_next].copy()
        if df_y.empty or df_next.empty:
            continue

        # ── Compute or retrieve scores for year Y ────────────────────
        if score_col and score_col in df_y.columns:
            df_y["_score"] = df_y[score_col]
        else:
            Xy = df_y[avail_feats].copy()
            for col in Xy.columns:
                fv = global_medians.get(col, 0.0)
                Xy[col] = Xy[col].fillna(0.0 if pd.isna(fv) else fv)
            Xa = scaler.transform(Xy.values) if scaler else Xy.values
            df_y["_score"] = model.predict(Xa)

        score_map = dict(zip(df_y["PLAYER_ID"], df_y["_score"]))

        # ── Year Y+1 roster ──────────────────────────────────────────
        if "Team" not in df_next.columns or df_next["Team"].isna().all():
            continue
        df_next = df_next.dropna(subset=["Team"])
        df_next["raw_score"] = df_next["PLAYER_ID"].map(score_map)

        repl = replacement_value
        rookie = replacement_value + ROOKIE_BONUS
        df_next["adj_score"] = np.where(
            df_next["Minutes"] < min_minutes_qualifier,
            repl,
            df_next["raw_score"].fillna(rookie),
        )

        df_next["contribution"] = df_next["adj_score"] * df_next["Minutes"]
        team_scores = df_next.groupby("Team")["contribution"].sum().rename("team_score").reset_index()

        wins_next = team_wins_df[team_wins_df["Season"] == y_next][["Team","team_wins"]]
        merged = team_scores.merge(wins_next, on="Team", how="inner")
        if len(merged) < 10:
            continue

        r = float(merged[["team_wins","team_score"]].corr().iloc[0, 1])
        results.append({
            "Season_predicted": y,
            "Season_test":      y_next,
            "R2":               round(r ** 2, 4),
            "Pearson_r":        round(r, 4),
            "Label":            label,
        })

    return pd.DataFrame(results)


# ===========================================================================
# Feature importance helpers
# ===========================================================================

def ridge_importance(model, scaler, feature_cols: list[str]) -> pd.DataFrame:
    coefs = model.coef_
    avail = feature_cols[:len(coefs)]
    return (
        pd.DataFrame({"Feature": avail, "Coefficient": coefs})
        .assign(AbsCoef=lambda d: d["Coefficient"].abs())
        .sort_values("AbsCoef", ascending=False)
        .reset_index(drop=True)
    )


def xgb_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    try:
        gain = model.get_booster().get_score(importance_type="gain")
        rows = []
        for k, v in gain.items():
            idx = int(k[1:]) if k.startswith("f") and k[1:].isdigit() else None
            fname = feature_cols[idx] if idx is not None and idx < len(feature_cols) else k
            rows.append({"Feature": fname, "Gain": v})
        return pd.DataFrame(rows).sort_values("Gain", ascending=False).reset_index(drop=True)
    except Exception:
        imp = model.feature_importances_
        return (
            pd.DataFrame({"Feature": feature_cols[:len(imp)], "Importance": imp})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
