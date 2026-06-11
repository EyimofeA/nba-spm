"""rapm_core.py — block-ridge RAPM with CV-chosen lambdas, optional time decay, and tier shrinkage.

Possession-level design matrix, two columns per player (off, def), plus meta
columns for home-court advantage, rubberband (signed score margin at possession
start), playoff, and optional back-to-back. Each block carries its own lambda.

Two lambda-search modes:
  --search-mode ridgecv  Fast path: sklearn RidgeCV with LOO-GCV. Meta-block
                         lambdas scale with the chosen player alpha via
                         configurable ratios.
  --search-mode grid     Canonical path: exhaustive CV grid over (λ_off, λ_def)
                         only. Meta columns (home, rubber-band, playoff, b2b,
                         coach) are single indicator variables, so they get
                         fixed scalar lambdas from --lam-{home,rb,playoff,…}.

Replacement-level shrinkage bins players into MPG buckets (default 5-minute
buckets 0-5, 5-10, …, 35-40, 40+) and pulls each bucket's coefficients toward
the bottom-quartile mean of that bucket.

Playoff classification uses per-season date overrides for lockout (1999, 2012)
and COVID-affected (2020, 2021) postseasons; all other seasons use
April 12 – June 30.

Standard errors follow Jacobs (squared2020.com, RAPM Part IV):
σ² · (XᵀWX + diag(λ))⁻¹ with off-def covariance captured so Var(Net RAPM) is
exact. See `jacobs_standard_errors`.

Time decay (within-season exp decay, cross-season discrete multiplier) is
available but OFF by default — opt in via `--within-season-decay` and/or
`--cross-season-decay`.

See the RAPM charter for conventions: rapm/AGENTS.md.
"""
from __future__ import annotations

# =============================================================================
# Imports
# =============================================================================
import argparse
import csv
import json
import math
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import cg, splu, svds
from sklearn.linear_model import Ridge, RidgeCV

from paths import (
    ALL_NAMES_CSV,
    COMBINED_CORE,
    DIAGNOSTICS_DIR,
    PLAYERSHEETS_YEAR_TOTALS,
    RAPM_CORE_DUMP,
    RAPM_CORE_RESULTS,
    ensure_dirs,
)

ensure_dirs()


# =============================================================================
# Config dataclasses
# =============================================================================
@dataclass
class MetaLambdas:
    home: float = 100.0
    rb: float = 1000.0
    playoff: float = 500.0
    b2b: float = 500.0
    coach: float = 2000.0


@dataclass
class TierEdges:
    """Per-tier MPG upper bounds (ascending). Default: 5-minute buckets 0-5, 5-10, ..., 40+.

    We bucket this finely (not just low/mid/high) because the script is run
    once per window and the replacement levels are then frozen.
    """
    edges: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)


@dataclass
class RunConfig:
    seasons: list[int]
    window_label: str
    suffix: str
    # Meta toggles
    use_home: bool = True
    use_rubberband: bool = True
    use_playoff: bool = True
    use_b2b: bool = False
    use_coach: bool = False
    # Time decay (both default OFF; opt in per run)
    within_season_gamma: float | None = None     # e.g., 0.01 per game
    cross_season_rate: float | None = None       # e.g., 0.7 per season back
    # Lambda search mode
    search_mode: str = "ridgecv"                 # "ridgecv" (fast LOO-GCV) or "grid" (per-block exhaustive CV)
    # Fast path — sklearn RidgeCV with column scaling so a single α yields per-block effective λ
    cv_folds: int = 5                            # only used if cv_method='kfold'
    cv_method: str = "gcv"                       # "gcv" (efficient LOO) or "kfold"
    alpha_grid: tuple[float, ...] = (
        100, 250, 500, 1000, 1500, 2000, 3000, 4500, 6000, 8000,
    )
    ratio_home: float = 0.05
    ratio_rb: float = 0.50
    ratio_playoff: float = 0.25
    ratio_b2b: float = 0.25
    ratio_coach: float = 1.00
    # Per-block exhaustive grid — grid ONLY over (off, def). Meta blocks are
    # single indicator columns (home, rb, playoff, b2b, coach), so a single
    # fixed lambda is sufficient and makes the grid ~O(|off|·|def|) instead of
    # O(|off|·|def|·|home|·|rb|·|playoff|).
    grid_off: tuple[float, ...] = (500, 1500, 3000, 6000)
    grid_def: tuple[float, ...] = (500, 1500, 3000, 6000)
    grid_cv_folds: int = 3
    # Fixed meta-block lambdas (used in grid mode; used as absolute values
    # rather than ratios).
    lam_home: float = 200.0
    lam_rb: float = 1500.0
    lam_playoff: float = 500.0
    lam_b2b: float = 500.0
    lam_coach: float = 2000.0
    # Retained for backward compat; unused by grid mode
    meta_lambdas: MetaLambdas = field(default_factory=MetaLambdas)
    # Replacement-level shrinkage
    replacement_mode: str = "tier"               # "off", "tier", "uniform"
    tier_edges: TierEdges = field(default_factory=TierEdges)
    # Diagnostics
    run_diagnostics: bool = True
    compute_std_errors: bool = True
    # Playoff-only mode
    playoff_only: bool = False
    # Output
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


# =============================================================================
# Helpers
# =============================================================================
def format_season_list(seasons: list[int]) -> str:
    start_year = str(seasons[0])
    end_years = "-".join(str(y)[-2:] for y in seasons[1:])
    return f"{start_year}-{end_years}" if end_years else start_year


def _ensure_mysql_driver():
    """Return a MySQLdb-compatible module. Auto-pip-install pymysql if missing.

    Order of preference:
      1. mysqlclient (native C; faster)
      2. pymysql already installed (pure-Python)
      3. pip-install pymysql into the current interpreter and retry
    """
    try:
        import MySQLdb  # type: ignore
        return MySQLdb
    except ImportError:
        pass
    try:
        import pymysql  # type: ignore
        pymysql.install_as_MySQLdb()
        import MySQLdb  # type: ignore
        return MySQLdb
    except ImportError:
        print("[setup] No MySQL driver found — pip installing pymysql into this interpreter…", flush=True)
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pymysql"])
        import pymysql  # type: ignore
        pymysql.install_as_MySQLdb()
        import MySQLdb  # type: ignore
        return MySQLdb


def connect_db():
    # Password is hardcoded in the other scripts in this folder; keep the same
    # shape so the socket/auth is consistent. Move to env var before committing.
    MySQLdb = _ensure_mysql_driver()
    return MySQLdb.connect(
        host="localhost",
        user="root",
        password=os.environ.get("NBA_DB_PASSWORD", ""),
        db="nba_api",
        unix_socket="/tmp/mysql.sock",
    )


# =============================================================================
# Fetch possessions
# =============================================================================
def fetch_possessions(seasons: Iterable[int], playoff_only: bool = False):
    """Pull matchups rows for the given seasons from MySQL.

    Returns a list of tuples:
      (home_poss, pts, a1..a5, h1..h5, season, date, period, num, gameid)

    `playoff_only` fetches only playoff possessions. Uses per-season date
    overrides (lockout / COVID) from PLAYOFF_DATE_OVERRIDES rather than a
    single hardcoded April-12/June-30 window.  When False (default), all games
    are returned and the caller derives a playoff indicator from `date`.
    """
    seasons = list(seasons)
    placeholders = ",".join(["%s"] * len(seasons))

    if playoff_only:
        # Build a per-season OR clause using the canonical playoff windows so
        # lockout (1999, 2012) and COVID (2020, 2021) seasons are handled correctly.
        _default_lo = "04-12"
        _default_hi = "06-30"
        _overrides = {
            1999: ("05-08", "06-25"),
            2012: ("04-28", "06-21"),
            2020: ("08-17", "10-11"),
            2021: ("05-22", "07-20"),
        }
        season_clauses = []
        for s in seasons:
            lo, hi = _overrides.get(s, (_default_lo, _default_hi))
            season_clauses.append(
                f"(season = {s} AND date BETWEEN '{s}-{lo}' AND '{s}-{hi}')"
            )
        date_clause = "AND (" + " OR ".join(season_clauses) + ")"
    else:
        date_clause = ""

    query = f"""
        SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5,
               season, date, period, num, gameid
        FROM matchups
        WHERE season IN ({placeholders})
          {date_clause}
          AND pts IS NOT NULL
        ORDER BY season, date, gameid, period, num
    """
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(query, seasons)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# =============================================================================
# Playoff / B2B / time ordering
# =============================================================================
def _as_date(d) -> _date | None:
    if d is None:
        return None
    if isinstance(d, _date):
        return d
    # pandas / MySQLdb usually returns datetime.date already; handle strings too
    try:
        return pd.to_datetime(d).date()
    except Exception:
        return None


# Per-season playoff date overrides. `season` here is the calendar year the
# playoffs began in (e.g. season=2012 → 2011-12 NBA season, lockout playoffs).
# Default for all other seasons: (April 12, June 30) of that year.
PLAYOFF_DATE_OVERRIDES: dict[int, tuple[_date, _date]] = {
    # 1998-99 lockout: playoffs ran May 8 – June 25, 1999
    1999: (_date(1999, 5, 8), _date(1999, 6, 25)),
    # 2011-12 lockout: playoffs April 28 – June 21, 2012
    2012: (_date(2012, 4, 28), _date(2012, 6, 21)),
    # 2019-20 COVID bubble: regular season resumed July 30, playoffs Aug 17 – Oct 11, 2020
    2020: (_date(2020, 8, 17), _date(2020, 10, 11)),
    # 2020-21 COVID delay: May 22 – July 20, 2021
    2021: (_date(2021, 5, 22), _date(2021, 7, 20)),
}


def playoff_window(season: int) -> tuple[_date, _date]:
    return PLAYOFF_DATE_OVERRIDES.get(season, (_date(season, 4, 12), _date(season, 6, 30)))


def is_playoff_game(season: int, game_date) -> bool:
    d = _as_date(game_date)
    if d is None:
        return False
    lo, hi = playoff_window(season)
    return lo <= d <= hi


def build_team_game_order(rows) -> tuple[dict, dict]:
    """Build two dicts:

    - `team_games[(team_key, season)] -> list[(date, gameid)]` for each team
    - `b2b_flag[gameid][team_key] -> bool` (True if this team played the previous day)

    `team_key` is a stable id per side of the possession; since matchups only
    carries player ids we use the hash of the 5 home / away PLAYER_IDs. That's
    robust enough for a b2b flag within a season (teams rarely field identical
    lineups across different days anyway, but we use the gameid as the actual
    grouping key).

    For the purposes of B2B we actually need team identity per game. We derive
    it as the sorted tuple of (h1..h5) for the home team on the first possession
    of each gameid, and the sorted tuple of (a1..a5) for the away team. That's
    not perfect if the starting lineup varies, so we fall back to using the
    gameid itself for ordering and use date gaps per team-id as best-effort.

    Returns (team_games_by_season, b2b_by_game_by_side) where side is 'home'/'away'.
    """
    # Step 1: first-possession home/away player sets per gameid
    home_set_per_game: dict = {}
    away_set_per_game: dict = {}
    game_meta: dict = {}
    for r in rows:
        gameid = r[16]
        if gameid in home_set_per_game:
            continue
        home_set_per_game[gameid] = tuple(sorted(r[7:12]))
        away_set_per_game[gameid] = tuple(sorted(r[2:7]))
        game_meta[gameid] = (r[12], _as_date(r[13]))  # (season, date)

    # Step 2: cluster games into team threads by majority-overlap of starting lineups
    # Simpler approximation: treat each distinct starter-5 tuple as a team identity.
    # For B2B we just need "did this team play yesterday?" which is consistent as
    # long as starting lineups are stable within a season, which is roughly true.
    team_dates: dict[tuple, list[tuple]] = defaultdict(list)
    for gameid, (season, gdate) in game_meta.items():
        if gdate is None:
            continue
        team_dates[("home_key", home_set_per_game[gameid], season)].append((gdate, gameid))
        team_dates[("away_key", away_set_per_game[gameid], season)].append((gdate, gameid))

    for key, lst in team_dates.items():
        lst.sort()

    # Step 3: for each game, compute whether the team played on the immediately
    # preceding calendar day.
    b2b_home: dict = {}
    b2b_away: dict = {}
    for key, lst in team_dates.items():
        side = key[0]
        sink = b2b_home if side == "home_key" else b2b_away
        prev_date = None
        for gdate, gameid in lst:
            is_b2b = prev_date is not None and (gdate - prev_date).days == 1
            sink[gameid] = is_b2b
            prev_date = gdate
    return game_meta, (b2b_home, b2b_away)


# =============================================================================
# Time-decay weights (opt-in; default off)
# =============================================================================
def compute_time_decay_weights(
    rows,
    gamma: float | None,
    cross_season_rate: float | None,
    game_meta: dict,
) -> np.ndarray:
    """Return a per-possession weight vector.

    Within-season weight:   w = exp(-gamma · games_between_this_game_and_last_in_window)
    Cross-season multiplier: rate^(seasons_back_from_latest)

    When both args are None, returns ones. Time decay is intentionally OFF by
    default — callers opt in via `--within-season-decay GAMMA` and/or
    `--cross-season-decay RATE`.
    """
    n = len(rows)
    if gamma is None and cross_season_rate is None:
        return np.ones(n, dtype=np.float64)

    seasons = sorted({season for season, _ in game_meta.values()})
    latest_season = max(seasons)

    games_by_season: dict[int, list[tuple]] = defaultdict(list)
    for gameid, (season, gdate) in game_meta.items():
        games_by_season[season].append((gdate or _date(season, 1, 1), gameid))
    game_index: dict[str, int] = {}
    last_index: dict[int, int] = {}
    for season, lst in games_by_season.items():
        lst.sort()
        for i, (_, gid) in enumerate(lst):
            game_index[gid] = i
        last_index[season] = len(lst) - 1

    weights = np.empty(n, dtype=np.float64)
    for i, r in enumerate(rows):
        season = r[12]
        gameid = r[16]
        w = 1.0
        if gamma is not None:
            dist = last_index[season] - game_index.get(gameid, 0)
            w *= math.exp(-gamma * dist)
        if cross_season_rate is not None:
            seasons_back = latest_season - season
            w *= cross_season_rate ** seasons_back
        weights[i] = w
    return weights


# =============================================================================
# Design matrix
# =============================================================================
@dataclass
class DesignMatrix:
    X: csr_matrix
    y: np.ndarray
    base_weight: np.ndarray            # pre-timedecay weights (ones currently)
    col_to_key: dict[int, str]
    block_of_col: np.ndarray           # block tag per column (int code)
    gameids: np.ndarray                # per-row gameid for bootstrap / CV ordering
    row_order: np.ndarray              # pre-sorted time order indices


# Block codes
BLOCK_OFF = 0
BLOCK_DEF = 1
BLOCK_HOME = 2
BLOCK_RB = 3
BLOCK_PLAYOFF = 4
BLOCK_B2B = 5
BLOCK_COACH = 6


def build_design_matrix(rows, cfg: RunConfig, game_meta: dict, b2b_pair: tuple[dict, dict]) -> DesignMatrix:
    """Construct the sparse design matrix.

    Uses COO-style triplet accumulation then converts to CSR once.
    """
    b2b_home, b2b_away = b2b_pair

    # Enumerate players encountered on court
    all_players: set[int] = set()
    for r in rows:
        for i in range(2, 12):
            all_players.add(r[i])

    # Build column index
    col_to_key: dict[int, str] = {}
    key_to_col: dict[str, int] = {}

    def add_col(key: str):
        idx = len(col_to_key)
        col_to_key[idx] = key
        key_to_col[key] = idx
        return idx

    for pid in sorted(all_players):
        add_col(f"{pid}_off")
        add_col(f"{pid}_def")

    meta_cols: list[tuple[str, int]] = []   # (key, block_code)
    if cfg.use_home:
        meta_cols.append(("META_home", BLOCK_HOME))
    if cfg.use_rubberband:
        meta_cols.append(("META_rb_margin", BLOCK_RB))
    if cfg.use_playoff:
        meta_cols.append(("META_playoff", BLOCK_PLAYOFF))
    if cfg.use_b2b:
        meta_cols.append(("META_b2b_off", BLOCK_B2B))
        meta_cols.append(("META_b2b_def", BLOCK_B2B))
    # coach left as stub (no columns) in v1

    for key, _ in meta_cols:
        add_col(key)

    n_cols = len(col_to_key)
    block_of_col = np.empty(n_cols, dtype=np.int32)
    for idx, key in col_to_key.items():
        if key.endswith("_off"):
            block_of_col[idx] = BLOCK_OFF
        elif key.endswith("_def"):
            block_of_col[idx] = BLOCK_DEF
        else:
            block_of_col[idx] = dict(meta_cols)[key]

    # Walk possessions, accumulate triplets + running score
    n = len(rows)
    rows_idx: list[int] = []
    cols_idx: list[int] = []
    vals: list[float] = []
    y = np.empty(n, dtype=np.float64)
    gameids = np.empty(n, dtype=object)
    game_score: dict = defaultdict(lambda: [0, 0])  # home_pts, away_pts

    for i, r in enumerate(rows):
        home_poss, pts = r[0], r[1]
        a = [r[j] for j in range(2, 7)]
        h = [r[j] for j in range(7, 12)]
        season = r[12]
        gdate = r[13]
        gameid = r[16]

        # Running score BEFORE this possession
        hp, ap = game_score[gameid]
        margin_from_off = (hp - ap) if home_poss else (ap - hp)

        # Off/def role
        if home_poss:
            off_list, def_list = h, a
        else:
            off_list, def_list = a, h

        for p in off_list:
            rows_idx.append(i)
            cols_idx.append(key_to_col[f"{p}_off"])
            vals.append(1.0)
        for p in def_list:
            rows_idx.append(i)
            cols_idx.append(key_to_col[f"{p}_def"])
            vals.append(1.0)

        if cfg.use_home:
            # +1 if home team is on offense, -1 if away team is on offense
            rows_idx.append(i)
            cols_idx.append(key_to_col["META_home"])
            vals.append(1.0 if home_poss else -1.0)

        if cfg.use_rubberband:
            rows_idx.append(i)
            cols_idx.append(key_to_col["META_rb_margin"])
            vals.append(float(margin_from_off))

        if cfg.use_playoff and is_playoff_game(int(season), gdate):
            rows_idx.append(i)
            cols_idx.append(key_to_col["META_playoff"])
            vals.append(1.0)

        if cfg.use_b2b:
            off_b2b = (b2b_home.get(gameid, False) if home_poss else b2b_away.get(gameid, False))
            def_b2b = (b2b_away.get(gameid, False) if home_poss else b2b_home.get(gameid, False))
            if off_b2b:
                rows_idx.append(i)
                cols_idx.append(key_to_col["META_b2b_off"])
                vals.append(1.0)
            if def_b2b:
                rows_idx.append(i)
                cols_idx.append(key_to_col["META_b2b_def"])
                vals.append(1.0)

        y[i] = pts
        gameids[i] = gameid

        # Update running score AFTER building features
        if home_poss:
            game_score[gameid][0] += pts
        else:
            game_score[gameid][1] += pts

    X = csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows_idx, dtype=np.int64), np.array(cols_idx, dtype=np.int64))),
        shape=(n, n_cols),
    )

    # Rows are already time-ordered by the SQL ORDER BY; record that order.
    row_order = np.arange(n, dtype=np.int64)

    return DesignMatrix(
        X=X,
        y=y,
        base_weight=np.ones(n, dtype=np.float64),
        col_to_key=col_to_key,
        block_of_col=block_of_col,
        gameids=gameids,
        row_order=row_order,
    )


# =============================================================================
# Block ridge solver
# =============================================================================
def _lambda_vec(block_of_col: np.ndarray, lam_off: float, lam_def: float, meta: MetaLambdas) -> np.ndarray:
    lv = np.empty(block_of_col.shape, dtype=np.float64)
    lv[block_of_col == BLOCK_OFF] = lam_off
    lv[block_of_col == BLOCK_DEF] = lam_def
    lv[block_of_col == BLOCK_HOME] = meta.home
    lv[block_of_col == BLOCK_RB] = meta.rb
    lv[block_of_col == BLOCK_PLAYOFF] = meta.playoff
    lv[block_of_col == BLOCK_B2B] = meta.b2b
    lv[block_of_col == BLOCK_COACH] = meta.coach
    return lv


def fit_block_ridge(
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    lambda_vec: np.ndarray,
    beta_target: np.ndarray | None = None,
) -> np.ndarray:
    """Solve (X' W X + diag(lambda_vec)) beta = X' W (y - X beta_target) + diag(lambda_vec) beta_target.

    When beta_target is None it's treated as zero (standard ridge).
    Uses CG on the sparse normal equations; falls back to direct SPLU if CG
    fails to converge.
    """
    n_cols = X.shape[1]
    if beta_target is None:
        beta_target = np.zeros(n_cols, dtype=np.float64)

    # Center y
    y_av = float(np.average(y, weights=weights)) if weights.sum() > 0 else float(np.mean(y))
    y_c = y - y_av

    # Normal equations
    W_sqrt = np.sqrt(weights)
    Xw = X.multiply(W_sqrt[:, None]).tocsr()
    XtWX = (Xw.T @ Xw).tocsc()
    # Add diag
    XtWX = XtWX + diags(lambda_vec, format="csc")

    # RHS: X' W (y_c - X beta_target) + diag(lambda) beta_target
    resid = y_c - X @ beta_target
    rhs = X.T @ (weights * resid) + lambda_vec * beta_target

    # Solve
    try:
        sol, info = cg(XtWX.tocsr(), rhs, rtol=1e-6, maxiter=5000)
        if info != 0:
            raise RuntimeError(f"CG did not converge cleanly (info={info}); falling back to SPLU")
    except Exception:
        solver = splu(XtWX.tocsc())
        sol = solver.solve(rhs)

    return sol + beta_target


# =============================================================================
# Column scaling → effective per-block lambdas via a single alpha
# =============================================================================
def _ratios_per_col(block_of_col: np.ndarray, cfg: RunConfig) -> np.ndarray:
    """Return per-column penalty ratios (lambda_j = alpha * ratio_j).

    Player columns (off, def) have ratio=1 so the chosen `alpha` IS the player
    lambda. Meta blocks use user-configurable ratios so their regularization
    scales with the player alpha.
    """
    r = np.ones(block_of_col.shape, dtype=np.float64)
    r[block_of_col == BLOCK_HOME] = cfg.ratio_home
    r[block_of_col == BLOCK_RB] = cfg.ratio_rb
    r[block_of_col == BLOCK_PLAYOFF] = cfg.ratio_playoff
    r[block_of_col == BLOCK_B2B] = cfg.ratio_b2b
    r[block_of_col == BLOCK_COACH] = cfg.ratio_coach
    # Clip to avoid div-by-zero when a ratio is accidentally set to 0
    r = np.maximum(r, 1e-8)
    return r


def _scale_columns(X: csr_matrix, ratios: np.ndarray) -> csr_matrix:
    """Return X' = X * diag(1 / sqrt(ratios)) so that fitting Ridge(alpha) on X'
    is equivalent to fitting ridge on X with per-column penalty alpha*ratio_j.
    """
    inv_s = 1.0 / np.sqrt(ratios)
    return X.multiply(inv_s[None, :]).tocsr()


# =============================================================================
# Alpha selection via sklearn RidgeCV (built-in efficient LOO-GCV)
# =============================================================================
def ridge_cv_select_alpha(
    dm: DesignMatrix,
    cfg: RunConfig,
) -> tuple[float, np.ndarray, float, pd.DataFrame]:
    """Use sklearn's RidgeCV to pick the player alpha automatically.

    We apply column scaling so that a *single* alpha from RidgeCV maps to
    per-block lambdas with the ratios defined in `cfg`. This is the equivalent
    of block-ridge lambda search but uses sklearn's efficient LOO-GCV (or
    k-fold) instead of our handcrafted time-series CV.

    Returns:
      alpha_chosen        chosen player-alpha (== lambda for off/def cols)
      beta                recovered coefficients on the *original* X scale
      intercept           fitted intercept
      scores_df           dataframe of per-alpha CV scores (mean_rmse)
    """
    ratios = _ratios_per_col(dm.block_of_col, cfg)
    X_scaled = _scale_columns(dm.X, ratios)

    alphas = list(cfg.alpha_grid)
    print(f"  grid: {alphas}")
    print(f"  cv_method: {cfg.cv_method}")

    if cfg.cv_method == "gcv":
        # cv=None triggers efficient LOO-GCV.
        # sklearn >=1.5 renamed `store_cv_values` → `store_cv_results`.
        import inspect
        ridge_kwargs = dict(alphas=alphas, cv=None, fit_intercept=True)
        sig = inspect.signature(RidgeCV.__init__).parameters
        if "store_cv_results" in sig:
            ridge_kwargs["store_cv_results"] = True
            cv_attr = "cv_results_"
        else:
            ridge_kwargs["store_cv_values"] = True
            cv_attr = "cv_values_"
        ridge = RidgeCV(**ridge_kwargs)
        ridge.fit(X_scaled, dm.y)
        cv_vals = getattr(ridge, cv_attr)
        mean_mse = cv_vals.mean(axis=0)
    else:
        ridge = RidgeCV(alphas=alphas, cv=cfg.cv_folds, fit_intercept=True)
        ridge.fit(X_scaled, dm.y)
        # With cv != None, cv_values_ is not stored; rerun per-alpha with Ridge to get scores
        from sklearn.model_selection import cross_val_score
        mean_mse = []
        for a in alphas:
            scores = -cross_val_score(
                Ridge(alpha=a, fit_intercept=True),
                X_scaled, dm.y, cv=cfg.cv_folds, scoring="neg_mean_squared_error",
            )
            mean_mse.append(float(scores.mean()))
        mean_mse = np.asarray(mean_mse)

    alpha_chosen = float(ridge.alpha_)
    beta_scaled = ridge.coef_
    intercept = float(ridge.intercept_)
    # Recover β on the original X scale
    beta = beta_scaled / np.sqrt(ratios)

    scores_df = pd.DataFrame({
        "alpha": alphas,
        "mean_mse": mean_mse,
        "mean_rmse": np.sqrt(mean_mse),
    })

    return alpha_chosen, beta, intercept, scores_df


def _lambda_vec_from_alpha(block_of_col: np.ndarray, alpha: float, cfg: RunConfig) -> np.ndarray:
    """Per-column effective lambda given the chosen alpha and block ratios."""
    return alpha * _ratios_per_col(block_of_col, cfg)


def _lambda_vec_from_blocks(
    block_of_col: np.ndarray,
    lam_off: float,
    lam_def: float,
    lam_home: float,
    lam_rb: float,
    lam_playoff: float,
    lam_b2b: float,
    lam_coach: float,
) -> np.ndarray:
    """Per-column effective lambda from explicit per-block values."""
    lv = np.empty(block_of_col.shape, dtype=np.float64)
    lv[block_of_col == BLOCK_OFF] = lam_off
    lv[block_of_col == BLOCK_DEF] = lam_def
    lv[block_of_col == BLOCK_HOME] = lam_home
    lv[block_of_col == BLOCK_RB] = lam_rb
    lv[block_of_col == BLOCK_PLAYOFF] = lam_playoff
    lv[block_of_col == BLOCK_B2B] = lam_b2b
    lv[block_of_col == BLOCK_COACH] = lam_coach
    return lv


# =============================================================================
# Exhaustive per-block lambda grid search (CV)
# =============================================================================
def block_grid_search(
    dm: DesignMatrix,
    cfg: RunConfig,
) -> tuple[dict, np.ndarray, pd.DataFrame]:
    """Cartesian-product CV over (λ_off, λ_def) ONLY.

    Meta blocks (home, rb, playoff, b2b, coach) each have a single indicator
    column, so grid-searching their lambdas is overkill — they get fixed
    lambdas from cfg.lam_{home,rb,playoff,b2b,coach}. Only the player off/def
    blocks (hundreds of columns each) benefit from CV-chosen regularization.

    Cost ≈ |grid_off| × |grid_def| × K fits. Default: 4·4·3 = 48 fits.

    Returns (best_lambdas_dict, beta_at_best_on_full_data, scores_df).
    """
    import itertools
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=cfg.grid_cv_folds, shuffle=True, random_state=42)
    splits = list(kf.split(np.arange(dm.X.shape[0])))

    scores: list[dict] = []
    combos = list(itertools.product(cfg.grid_off, cfg.grid_def))
    total = len(combos)
    print(f"  grid combos: {total:,}   folds: {cfg.grid_cv_folds}   total fits: {total * cfg.grid_cv_folds:,}")
    print(f"  off grid: {list(cfg.grid_off)}")
    print(f"  def grid: {list(cfg.grid_def)}")
    print(f"  fixed meta lambdas: home={cfg.lam_home}, rb={cfg.lam_rb}, "
          f"playoff={cfg.lam_playoff}, b2b={cfg.lam_b2b}, coach={cfg.lam_coach}")

    t0 = time.time()
    for i, (lo, ld) in enumerate(combos):
        lam_vec = _lambda_vec_from_blocks(
            dm.block_of_col, lo, ld,
            cfg.lam_home, cfg.lam_rb, cfg.lam_playoff,
            cfg.lam_b2b, cfg.lam_coach,
        )
        fold_rmses: list[float] = []
        for tr_idx, va_idx in splits:
            X_tr = dm.X[tr_idx]
            X_va = dm.X[va_idx]
            y_tr = dm.y[tr_idx]
            y_va = dm.y[va_idx]
            w_tr = dm.base_weight[tr_idx]
            beta = fit_block_ridge(X_tr, y_tr, w_tr, lam_vec)
            y_mean = float(np.average(y_tr, weights=w_tr))
            y_pred = X_va @ beta + y_mean
            fold_rmses.append(float(np.sqrt(np.mean((y_va - y_pred) ** 2))))
        rec = {
            "lam_off": lo,
            "lam_def": ld,
            "mean_rmse": float(np.mean(fold_rmses)),
            "std_rmse": float(np.std(fold_rmses)),
        }
        scores.append(rec)
        if (i + 1) % max(1, total // 10) == 0 or i == 0:
            elapsed = time.time() - t0
            est_total = elapsed / (i + 1) * total
            print(f"    [{i + 1:>4}/{total}]  off={lo:.0f} def={ld:.0f}  "
                  f"rmse={rec['mean_rmse']:.5f}  ({elapsed:.0f}s / est {est_total:.0f}s)")

    scores_df = pd.DataFrame(scores).sort_values("mean_rmse").reset_index(drop=True)
    best = scores_df.iloc[0].to_dict()
    print(f"\n  best combo: off={best['lam_off']:.0f}  def={best['lam_def']:.0f}")
    print(f"  best rmse:  {best['mean_rmse']:.5f}  (±{best['std_rmse']:.5f} across folds)")

    lam_vec = _lambda_vec_from_blocks(
        dm.block_of_col,
        best["lam_off"], best["lam_def"],
        cfg.lam_home, cfg.lam_rb, cfg.lam_playoff, cfg.lam_b2b, cfg.lam_coach,
    )
    beta = fit_block_ridge(dm.X, dm.y, dm.base_weight, lam_vec)
    best_lambdas = {
        "lam_off": float(best["lam_off"]),
        "lam_def": float(best["lam_def"]),
        "lam_home": float(cfg.lam_home),
        "lam_rb": float(cfg.lam_rb),
        "lam_playoff": float(cfg.lam_playoff),
        "lam_b2b": float(cfg.lam_b2b),
        "lam_coach": float(cfg.lam_coach),
    }
    return best_lambdas, beta, scores_df


# =============================================================================
# MPG lookup + replacement-level shrinkage
# =============================================================================
def load_mpg_lookup(seasons: list[int]) -> dict[int, float]:
    """Return {PLAYER_ID -> MPG averaged across the window's seasons}.

    Uses playersheets/year_totals/{year}.csv files from the SPM data tree. If a
    file is missing we just skip that season; players present only in missing
    seasons will have no MPG entry and fall into the "low" tier by default.
    """
    mins: defaultdict[int, float] = defaultdict(float)
    games: defaultdict[int, float] = defaultdict(float)
    for yr in seasons:
        path = PLAYERSHEETS_YEAR_TOTALS / f"{yr}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["PLAYER_ID", "MIN", "GP"])
        except ValueError:
            continue
        df = df.dropna(subset=["PLAYER_ID"])
        for pid, m, gp in zip(df["PLAYER_ID"].astype(int), df["MIN"].astype(float), df["GP"].astype(float)):
            mins[pid] += float(m or 0.0)
            games[pid] += float(gp or 0.0)
    mpg: dict[int, float] = {}
    for pid, m in mins.items():
        g = games.get(pid, 0.0)
        if g > 0:
            mpg[pid] = m / g
    return mpg


def tier_of(mpg: float, edges: TierEdges) -> str:
    """Bucket MPG into a label like '0-5', '5-10', …, '35-40', '40+'.

    Bucket upper bounds come from `edges.edges`. A player with `mpg < edges[0]`
    goes into the first bucket; a player above the final edge falls into
    '{last}+'.
    """
    if mpg != mpg:  # NaN → lowest bucket
        mpg = 0.0
    lo = 0.0
    for hi in edges.edges:
        if mpg < hi:
            return f"{lo:g}-{hi:g}"
        lo = hi
    return f"{edges.edges[-1]:g}+"


def replacement_level_shrink(
    beta: np.ndarray,
    col_to_key: dict[int, str],
    mpg_lookup: dict[int, float],
    cfg: RunConfig,
) -> tuple[np.ndarray, dict[int, str], np.ndarray]:
    """Compute per-tier replacement target vector and return (beta_target, tier_map, used).

    `beta_target` has shape [n_cols]. Meta columns get 0. Player columns get
    their tier's replacement level (off/def computed separately). `used` is an
    indicator array for which columns got a non-zero target.

    Modes:
      - "off":     no shrinkage (zero target everywhere)
      - "tier":    target = mean of bottom-quartile of that (tier, side) subset
      - "uniform": target = mean of all low-tier players (off/def)
    """
    n_cols = beta.shape[0]
    target = np.zeros(n_cols, dtype=np.float64)
    tier_map: dict[int, str] = {}
    if cfg.replacement_mode == "off":
        return target, tier_map, np.zeros(n_cols, dtype=bool)

    # Gather player columns per tier and side
    off_by_tier: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    def_by_tier: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for col_idx, key in col_to_key.items():
        if not (key.endswith("_off") or key.endswith("_def")):
            continue
        pid_str = key.split("_")[0]
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        mpg = mpg_lookup.get(pid, 0.0)
        tier = tier_of(mpg, cfg.tier_edges)
        tier_map[pid] = tier
        if key.endswith("_off"):
            off_by_tier[tier].append((col_idx, beta[col_idx]))
        else:
            def_by_tier[tier].append((col_idx, beta[col_idx]))

    # Compute replacement per tier
    def _bottom_q_mean(values: list[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float64)
        q = np.quantile(arr, 0.25)
        return float(arr[arr <= q].mean()) if np.any(arr <= q) else float(arr.mean())

    all_tiers = sorted(set(off_by_tier.keys()) | set(def_by_tier.keys()))
    if cfg.replacement_mode == "uniform":
        # Lowest-MPG bucket = bucket whose upper bound is smallest
        lowest_tier = min(all_tiers, key=lambda t: float(t.split("-")[0]) if "-" in t else 0.0) if all_tiers else None
        off_target = _bottom_q_mean([v for _, v in off_by_tier.get(lowest_tier, [])]) if lowest_tier else 0.0
        def_target = _bottom_q_mean([v for _, v in def_by_tier.get(lowest_tier, [])]) if lowest_tier else 0.0
        for tier in all_tiers:
            for col_idx, _ in off_by_tier.get(tier, []):
                target[col_idx] = off_target
            for col_idx, _ in def_by_tier.get(tier, []):
                target[col_idx] = def_target
    else:  # tier
        for tier in all_tiers:
            off_tgt = _bottom_q_mean([v for _, v in off_by_tier.get(tier, [])])
            def_tgt = _bottom_q_mean([v for _, v in def_by_tier.get(tier, [])])
            for col_idx, _ in off_by_tier.get(tier, []):
                target[col_idx] = off_tgt
            for col_idx, _ in def_by_tier.get(tier, []):
                target[col_idx] = def_tgt

    used = target != 0.0
    return target, tier_map, used


# =============================================================================
# Diagnostics
# =============================================================================
def run_diagnostics(
    dm: DesignMatrix,
    beta: np.ndarray,
    lambda_vec: np.ndarray,
    cfg: RunConfig,
) -> dict:
    """Condition number + worst-correlated player pair."""
    out: dict = {"run_id": cfg.run_id}
    weights = dm.base_weight
    W_sqrt = np.sqrt(weights)
    Xw = dm.X.multiply(W_sqrt[:, None]).tocsr()
    XtWX = (Xw.T @ Xw).tocsc() + diags(lambda_vec, format="csc")
    try:
        s_max = float(svds(XtWX.asfptype(), k=1, which="LM", return_singular_vectors=False)[0])
        s_min = float(svds(XtWX.asfptype(), k=1, which="SM", return_singular_vectors=False)[0])
        cond = s_max / max(s_min, 1e-12)
    except Exception as e:
        s_max = s_min = float("nan")
        cond = float("nan")
        out["cond_error"] = str(e)
    out["sigma_max"] = s_max
    out["sigma_min"] = s_min
    out["condition_number"] = cond
    return out


def regularization_path(
    dm: DesignMatrix,
    cfg: RunConfig,
    alpha_chosen: float,
    beta_final: np.ndarray,
    top_k: int = 20,
) -> pd.DataFrame:
    """Refit at 10 alphas around alpha_chosen, record coefs for top-K players by final RAPM."""
    grid = np.geomspace(alpha_chosen / 4.0, alpha_chosen * 4.0, 10)

    records: list[tuple[int, int, float]] = []
    off_idx: dict[int, int] = {}
    def_idx: dict[int, int] = {}
    for idx, key in dm.col_to_key.items():
        if key.endswith("_off"):
            off_idx[int(key.split("_")[0])] = idx
        elif key.endswith("_def"):
            def_idx[int(key.split("_")[0])] = idx
    for pid, oi in off_idx.items():
        di = def_idx.get(pid)
        if di is None:
            continue
        rapm = beta_final[oi] - beta_final[di]
        records.append((pid, oi, rapm))
    records.sort(key=lambda t: -t[2])
    top = records[:top_k]

    rows_out: list[dict] = []
    weights = dm.base_weight
    for alpha in grid:
        lam_vec = _lambda_vec_from_alpha(dm.block_of_col, float(alpha), cfg)
        beta = fit_block_ridge(dm.X, dm.y, weights, lam_vec)
        for pid, oi, _ in top:
            rows_out.append({"alpha": float(alpha), "pid": pid, "beta_off": float(beta[oi])})
    return pd.DataFrame(rows_out)


def jacobs_standard_errors(
    dm: DesignMatrix,
    beta: np.ndarray,
    intercept: float,
    lambda_vec: np.ndarray,
) -> pd.DataFrame:
    """Closed-form standard errors following Jacobs (squared2020.com, RAPM Part IV).

    σ² = R'W R / (N − (P_off + P_def) − 1)
    Σ_β = σ² · (X' W X + diag(λ))⁻¹
    SE_j = √(diag Σ_β)_j

    For each player we also compute cov(β_off, β_def), so Var(Net) is exact
    rather than needing `Var(off) + Var(def)`.

    Returns a DataFrame with (col_idx, key, se, var). Player off-def covariance
    is returned via an extra pid-keyed dict attached as `df.attrs['cov_od']`.
    """
    X = dm.X
    y = dm.y
    W = dm.base_weight
    n, p = X.shape

    # Residuals on the *original* scale.
    resid = y - (X @ beta + intercept)
    rss_w = float(np.sum(W * resid ** 2))

    # Effective degrees of freedom: count unique players seen on offense and on defense.
    p_off = sum(1 for k in dm.col_to_key.values() if k.endswith("_off"))
    p_def = sum(1 for k in dm.col_to_key.values() if k.endswith("_def"))
    dof = max(1.0, n - p_off - p_def - 1)
    sigma2 = rss_w / dof

    # Form A = Xᵀ W X + diag(λ) and factor once.
    W_sqrt = np.sqrt(W)
    Xw = X.multiply(W_sqrt[:, None]).tocsr()
    A = (Xw.T @ Xw).tocsc() + diags(lambda_vec, format="csc")
    solver = splu(A)

    # Compute the columns of A⁻¹ corresponding to every player (off+def column).
    # This gives us diag(A⁻¹) for player cols and the cross covariance
    # cov(β_off_i, β_def_i) in one sweep.
    player_cols: list[tuple[int, str]] = [
        (i, k) for i, k in dm.col_to_key.items() if k.endswith("_off") or k.endswith("_def")
    ]
    n_player = len(player_cols)

    # Batch solves for memory safety (p × n_player dense can be large)
    diag_inv = np.zeros(p, dtype=np.float64)
    cov_od_by_pid: dict[int, float] = {}
    off_idx_by_pid: dict[int, int] = {}
    def_idx_by_pid: dict[int, int] = {}
    for col_idx, key in player_cols:
        pid = int(key.split("_")[0])
        if key.endswith("_off"):
            off_idx_by_pid[pid] = col_idx
        else:
            def_idx_by_pid[pid] = col_idx

    batch = 128
    player_col_indices = np.asarray([ci for ci, _ in player_cols], dtype=np.int64)
    for start in range(0, n_player, batch):
        end = min(n_player, start + batch)
        cols = player_col_indices[start:end]
        rhs = np.zeros((p, end - start), dtype=np.float64)
        for j, c in enumerate(cols):
            rhs[c, j] = 1.0
        sol = solver.solve(rhs)   # shape (p, end-start)
        for j, c in enumerate(cols):
            diag_inv[c] = sol[c, j]
        for j, c in enumerate(cols):
            key = dm.col_to_key[c]
            pid = int(key.split("_")[0])
            # If this column is the _off for player pid, read cov to their _def row
            if key.endswith("_off") and pid in def_idx_by_pid:
                cov_od_by_pid[pid] = float(sol[def_idx_by_pid[pid], j])

    # Also compute diag_inv for meta columns (optional but cheap)
    meta_cols = [i for i, k in dm.col_to_key.items() if not (k.endswith("_off") or k.endswith("_def"))]
    if meta_cols:
        rhs = np.zeros((p, len(meta_cols)), dtype=np.float64)
        for j, c in enumerate(meta_cols):
            rhs[c, j] = 1.0
        sol = solver.solve(rhs)
        for j, c in enumerate(meta_cols):
            diag_inv[c] = sol[c, j]

    var_beta = sigma2 * diag_inv
    var_beta = np.maximum(var_beta, 0.0)
    se_beta = np.sqrt(var_beta)

    df = pd.DataFrame({
        "col_idx": np.arange(p),
        "key": [dm.col_to_key[i] for i in range(p)],
        "var": var_beta,
        "se": se_beta,
    })
    # Attach cov(off, def) per pid so downstream writers can use Var(Net) exactly.
    df.attrs["cov_od"] = {pid: sigma2 * v for pid, v in cov_od_by_pid.items()}
    df.attrs["sigma2"] = sigma2
    df.attrs["dof"] = dof
    return df


# =============================================================================
# Result writers
# =============================================================================
def _load_name_lookup() -> dict[str, str]:
    df = pd.read_csv(ALL_NAMES_CSV)
    return dict(zip(df["PLAYER_ID"].astype(str), df["PLAYER_NAME"]))


def write_raw_dump(dm: DesignMatrix, beta: np.ndarray, cfg: RunConfig) -> Path:
    path = RAPM_CORE_DUMP / f"rapm_core_{cfg.window_label}_{format_season_list(cfg.seasons)}_{cfg.suffix}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["col_idx", "key", "coef"])
        for i, key in dm.col_to_key.items():
            w.writerow([i, key, float(beta[i])])
    print(f"Raw coefficients → {path}")
    return path


def write_human_readable(
    dm: DesignMatrix,
    beta_raw: np.ndarray,
    beta_target: np.ndarray,
    tier_map: dict[int, str],
    mpg_lookup: dict[int, float],
    cfg: RunConfig,
    se_df: pd.DataFrame | None,
    dm_for_counts: DesignMatrix | None = None,
) -> Path:
    names = _load_name_lookup()

    off_map: dict[int, int] = {}
    def_map: dict[int, int] = {}
    for idx, key in dm.col_to_key.items():
        if key.endswith("_off"):
            off_map[int(key.split("_")[0])] = idx
        elif key.endswith("_def"):
            def_map[int(key.split("_")[0])] = idx

    # Possession counts per column (X entries are +1 per player/possession, so column-sum == # possessions)
    if dm_for_counts is not None:
        col_sums = np.asarray(dm_for_counts.X.sum(axis=0)).ravel()
    else:
        col_sums = np.asarray(dm.X.sum(axis=0)).ravel()

    # Lookup SE + off/def covariance per player for Jacobs CIs.
    se_by_col: dict[int, float] = {}
    cov_od_by_pid: dict[int, float] = {}
    if se_df is not None:
        for _, row in se_df.iterrows():
            se_by_col[int(row["col_idx"])] = float(row["se"])
        cov_od_by_pid = se_df.attrs.get("cov_od", {})

    Z = 1.96   # ~95% for a Gaussian posterior (matches Jacobs' HPD interpretation)

    rows: list[dict] = []
    for pid, oi in off_map.items():
        di = def_map.get(pid)
        if di is None:
            continue
        off = float(beta_raw[oi]) * 100.0
        deff = float(beta_raw[di]) * 100.0
        replace_off = float(beta_target[oi]) * 100.0
        replace_def = float(beta_target[di]) * 100.0
        name = names.get(str(pid), str(pid))
        mpg = mpg_lookup.get(pid, float("nan"))
        tier = tier_map.get(pid, tier_of(0.0, cfg.tier_edges))
        poss_off = int(col_sums[oi])
        poss_def = int(col_sums[di])
        vorp_off = off - replace_off
        vorp_def = deff - replace_def
        rec = {
            "Name": name,
            "PLAYER_ID": pid,
            "MPG": round(mpg, 2) if mpg == mpg else None,
            "Tier": tier,
            "Poss_Off": poss_off,
            "Poss_Def": poss_def,
            "Off": round(off, 3),
            "Def": round(deff, 3),
            # RAPM convention: positive Off is good offense; positive Def means the player
            # "allowed" more points (worse defense). Net RAPM = Off - Def.
            "RAPM": round(off - deff, 3),
            "Replace_Off": round(replace_off, 3),
            "Replace_Def": round(replace_def, 3),
            "VORP_Off": round(vorp_off, 3),
            "VORP_Def": round(vorp_def, 3),
            "VORP": round(vorp_off - vorp_def, 3),
        }
        if se_df is not None:
            se_off = se_by_col.get(oi, float("nan")) * 100.0
            se_def = se_by_col.get(di, float("nan")) * 100.0
            cov_od = cov_od_by_pid.get(pid, 0.0) * (100.0 ** 2)
            var_net = se_off ** 2 + se_def ** 2 - 2.0 * cov_od
            se_net = math.sqrt(max(var_net, 0.0))
            rec["Off_SE"] = round(se_off, 3)
            rec["Def_SE"] = round(se_def, 3)
            rec["RAPM_SE"] = round(se_net, 3)
            rec["Off_CI_lo"] = round(off - Z * se_off, 3)
            rec["Off_CI_hi"] = round(off + Z * se_off, 3)
            rec["Def_CI_lo"] = round(deff - Z * se_def, 3)
            rec["Def_CI_hi"] = round(deff + Z * se_def, 3)
            rec["RAPM_CI_lo"] = round((off - deff) - Z * se_net, 3)
            rec["RAPM_CI_hi"] = round((off - deff) + Z * se_net, 3)
        rec["Season"] = format_season_list(cfg.seasons)
        rows.append(rec)

    df = pd.DataFrame(rows).sort_values("RAPM", ascending=False)
    # Everyone is kept — replacement-level shrinkage (MPG-bucketed) handles
    # small-sample players, so we don't need a possession floor.
    path = RAPM_CORE_RESULTS / f"Core_Rapm_{cfg.window_label}_{format_season_list(cfg.seasons)}.csv"
    df.to_csv(path, index=False)
    print(f"Human-readable → {path}  ({len(df):,} players)")
    print("Top 15:")
    print(df.head(15).to_string(index=False))
    return path


def append_combined(result_path: Path) -> None:
    """Append the latest Core_Rapm_*.csv into Combined_Rapm_core.csv."""
    df = pd.read_csv(result_path)
    header = not COMBINED_CORE.exists()
    df.to_csv(COMBINED_CORE, mode="a", header=header, index=False)
    print(f"Appended to {COMBINED_CORE}")


# =============================================================================
# Main orchestration
# =============================================================================
def resolve_seasons(window: str, end_season: int | None, start_season: int | None) -> tuple[list[int], str]:
    """Resolve CLI window flags to a concrete list of seasons."""
    if window == "custom":
        if start_season is None or end_season is None:
            raise SystemExit("--window custom requires --start-season and --end-season")
        return list(range(start_season, end_season + 1)), f"custom{start_season}-{end_season}"
    if window == "full":
        # Minimum useful full range given db dump
        lo = start_season or 1997
        hi = end_season or 2024
        return list(range(lo, hi + 1)), f"full"
    size = int(window)
    if end_season is None:
        end_season = 2024
    seasons = list(range(end_season - size + 1, end_season + 1))
    return seasons, f"{size}yr"


def run(cfg: RunConfig) -> dict:
    playoff_tag = " | PLAYOFFS ONLY" if cfg.playoff_only else ""
    print(f"\n=== rapm_core run {cfg.run_id} | {cfg.window_label} | seasons {cfg.seasons}{playoff_tag} ===")
    t0 = time.time()

    print("Fetching possessions from MySQL...")
    rows = fetch_possessions(cfg.seasons, playoff_only=cfg.playoff_only)
    mode_label = "playoff-only" if cfg.playoff_only else "all games"
    print(f"  fetched {len(rows):,} possessions ({mode_label}) in {time.time() - t0:.1f}s")

    print("Building game metadata (playoff + B2B)...")
    game_meta, b2b_pair = build_team_game_order(rows)

    print("Building design matrix...")
    dm = build_design_matrix(rows, cfg, game_meta, b2b_pair)
    print(f"  X shape = {dm.X.shape}  nnz = {dm.X.nnz:,}")

    if cfg.within_season_gamma is not None or cfg.cross_season_rate is not None:
        print("\nApplying time-decay weights "
              f"(within-season γ={cfg.within_season_gamma}, cross-season rate={cfg.cross_season_rate})...")
        td_w = compute_time_decay_weights(
            rows, cfg.within_season_gamma, cfg.cross_season_rate, game_meta,
        )
        dm.base_weight = dm.base_weight * td_w

    # Lambda selection
    intercept_pass1 = 0.0
    if cfg.search_mode == "grid":
        print("\nExhaustive per-block lambda grid search...")
        best_lambdas, beta_pass1, scores_df = block_grid_search(dm, cfg)
        scores_df.to_csv(DIAGNOSTICS_DIR / f"grid_scores_{cfg.run_id}.csv", index=False)
        lam_vec = _lambda_vec_from_blocks(
            dm.block_of_col,
            best_lambdas["lam_off"], best_lambdas["lam_def"], best_lambdas["lam_home"],
            best_lambdas["lam_rb"], best_lambdas["lam_playoff"],
            best_lambdas["lam_b2b"], best_lambdas["lam_coach"],
        )
        alpha_chosen = float("nan")   # not applicable in grid mode
    else:
        print("\nSelecting player alpha via sklearn RidgeCV (built-in CV)...")
        alpha_chosen, beta_pass1, intercept_pass1, scores_df = ridge_cv_select_alpha(dm, cfg)
        scores_df.to_csv(DIAGNOSTICS_DIR / f"cv_scores_{cfg.run_id}.csv", index=False)
        print("  alpha grid scores (RMSE):")
        for _, row in scores_df.iterrows():
            marker = "  *chosen*" if float(row["alpha"]) == alpha_chosen else ""
            print(f"    alpha={row['alpha']:>7.0f}   rmse={row['mean_rmse']:.5f}{marker}")
        print(f"\n→ chosen alpha = {alpha_chosen:.1f}  (applied to both off and def; meta blocks get α * ratio)")
        print(f"  meta effective lambdas: "
              f"home={alpha_chosen * cfg.ratio_home:.1f}, "
              f"rb={alpha_chosen * cfg.ratio_rb:.1f}, "
              f"playoff={alpha_chosen * cfg.ratio_playoff:.1f}, "
              f"b2b={alpha_chosen * cfg.ratio_b2b:.1f}")
        print("  (trim the grid by dropping alphas far from this value on subsequent runs.)")
        lam_vec = _lambda_vec_from_alpha(dm.block_of_col, alpha_chosen, cfg)
        best_lambdas = {
            "lam_off": alpha_chosen, "lam_def": alpha_chosen,
            "lam_home": alpha_chosen * cfg.ratio_home,
            "lam_rb": alpha_chosen * cfg.ratio_rb,
            "lam_playoff": alpha_chosen * cfg.ratio_playoff,
            "lam_b2b": alpha_chosen * cfg.ratio_b2b,
            "lam_coach": alpha_chosen * cfg.ratio_coach,
        }

    beta_target = np.zeros_like(beta_pass1)
    tier_map: dict[int, str] = {}
    mpg_lookup: dict[int, float] = {}
    intercept_final = intercept_pass1
    beta_final = beta_pass1
    if cfg.replacement_mode != "off":
        print("\nLoading MPG lookup and computing tier-based replacement levels...")
        mpg_lookup = load_mpg_lookup(cfg.seasons)
        beta_target, tier_map, _ = replacement_level_shrink(beta_pass1, dm.col_to_key, mpg_lookup, cfg)
        print("Refitting at chosen alpha toward replacement levels (pass 2)...")
        beta_final = fit_block_ridge(dm.X, dm.y, dm.base_weight, lam_vec, beta_target=beta_target)
    else:
        mpg_lookup = load_mpg_lookup(cfg.seasons)
        for pid, mpg in mpg_lookup.items():
            tier_map[pid] = tier_of(mpg, cfg.tier_edges)

    # Jacobs closed-form standard errors (replaces bootstrap CI).
    se_df = None
    if cfg.compute_std_errors:
        print("\nComputing Jacobs standard errors (closed-form)...")
        se_df = jacobs_standard_errors(dm, beta_final, intercept_final, lam_vec)
        se_df.to_csv(DIAGNOSTICS_DIR / f"standard_errors_{cfg.run_id}.csv", index=False)
        print(f"  σ² = {se_df.attrs['sigma2']:.5f}  dof = {se_df.attrs['dof']:.0f}")

    if cfg.run_diagnostics:
        print("\nDiagnostics: condition number + regularization path...")
        diag = run_diagnostics(dm, beta_final, lam_vec, cfg)
        (DIAGNOSTICS_DIR / f"condition_{cfg.run_id}.json").write_text(json.dumps(diag, indent=2))
        print(f"  condition number = {diag.get('condition_number'):.2e}")
        ref_alpha = alpha_chosen if not math.isnan(alpha_chosen) else best_lambdas["lam_off"]
        path_df = regularization_path(dm, cfg, ref_alpha, beta_final)
        path_df.to_csv(DIAGNOSTICS_DIR / f"reg_path_{cfg.run_id}.csv", index=False)
        print(f"  reg path → {DIAGNOSTICS_DIR / f'reg_path_{cfg.run_id}.csv'}")

    (DIAGNOSTICS_DIR / f"chosen_lambdas_{cfg.run_id}.json").write_text(
        json.dumps({
            "search_mode": cfg.search_mode,
            "alpha_chosen": alpha_chosen,
            "best_lambdas": best_lambdas,
            "seasons": cfg.seasons,
            "window_label": cfg.window_label,
        }, indent=2, default=float)
    )

    write_raw_dump(dm, beta_final, cfg)
    result_path = write_human_readable(
        dm, beta_final, beta_target, tier_map, mpg_lookup, cfg, se_df, dm_for_counts=dm,
    )
    append_combined(result_path)

    print(f"\nDone ({time.time() - t0:.1f}s). run_id={cfg.run_id}")
    print(f"  chosen lambdas: {best_lambdas}")

    return {
        "run_id": cfg.run_id,
        "result_path": str(result_path),
        "best_lambdas": best_lambdas,
        "alpha_chosen": alpha_chosen,
        "seasons": cfg.seasons,
        "window_label": cfg.window_label,
    }


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rapm_core — block-ridge RAPM with time-series CV")
    p.add_argument("--window", default="3", help="1, 3, 5, full, or custom")
    p.add_argument("--end-season", type=int, default=None)
    p.add_argument("--start-season", type=int, default=None)
    p.add_argument("--suffix", default="v1")

    # Meta toggles
    p.add_argument("--home", dest="use_home", action="store_true", default=True)
    p.add_argument("--no-home", dest="use_home", action="store_false")
    p.add_argument("--rubberband", dest="use_rubberband", action="store_true", default=True)
    p.add_argument("--no-rubberband", dest="use_rubberband", action="store_false")
    p.add_argument("--playoff", dest="use_playoff", action="store_true", default=True)
    p.add_argument("--no-playoff", dest="use_playoff", action="store_false")
    p.add_argument("--b2b", dest="use_b2b", action="store_true", default=False)
    p.add_argument("--coach", dest="use_coach", action="store_true", default=False)

    # Time decay (default off — opt in per run)
    p.add_argument("--within-season-decay", dest="within_season_gamma", type=float, default=None,
                   help="Gamma for exp(-gamma*games_back). Opt-in. e.g. 0.01.")
    p.add_argument("--cross-season-decay", dest="cross_season_rate", type=float, default=None,
                   help="Per-season multiplier. Opt-in. e.g. 0.7 or 0.85.")

    # Lambda-search mode
    p.add_argument("--search-mode", choices=["ridgecv", "grid"], default="ridgecv",
                   help="'ridgecv' = fast sklearn LOO-GCV with block ratios; "
                        "'grid' = exhaustive per-block lambda grid (use for the full optimal run).")

    # RidgeCV fast path
    p.add_argument("--alpha-grid", default="100,250,500,1000,1500,2000,3000,4500,6000,8000",
                   help="Player alpha grid for RidgeCV (ridgecv mode).")
    p.add_argument("--cv-method", choices=["gcv", "kfold"], default="gcv")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--ratio-home", type=float, default=0.05)
    p.add_argument("--ratio-rb", type=float, default=0.50)
    p.add_argument("--ratio-playoff", type=float, default=0.25)
    p.add_argument("--ratio-b2b", type=float, default=0.25)
    p.add_argument("--ratio-coach", type=float, default=1.00)

    # Block-grid mode — grid ONLY over player off/def. Meta columns are single
    # indicator variables (home, rubber-band, playoff, b2b, coach) so they get
    # fixed scalar lambdas tuned here (or left at defaults).
    p.add_argument("--grid-off", default="500,1500,3000,6000")
    p.add_argument("--grid-def", default="500,1500,3000,6000")
    p.add_argument("--grid-cv-folds", type=int, default=3)
    p.add_argument("--lam-home", type=float, default=200.0,
                   help="Fixed lambda for the home-court indicator column (grid mode).")
    p.add_argument("--lam-rb", type=float, default=1500.0,
                   help="Fixed lambda for the rubber-band (score-margin) column.")
    p.add_argument("--lam-playoff", type=float, default=500.0,
                   help="Fixed lambda for the playoff indicator column.")
    p.add_argument("--lam-b2b", type=float, default=500.0,
                   help="Fixed lambda for the back-to-back indicator column.")
    p.add_argument("--lam-coach", type=float, default=2000.0,
                   help="Fixed lambda for the coach block (unused in v1).")

    # Replacement / MPG buckets
    p.add_argument("--replacement-shrinkage", choices=["off", "tier", "uniform"], default="tier")
    p.add_argument("--mpg-edges", default="5,10,15,20,25,30,35,40",
                   help="Upper bounds of MPG buckets (comma-sep, ascending). Default = 5-minute buckets.")

    # Playoff-only mode
    p.add_argument("--playoff-only", dest="playoff_only", action="store_true", default=False,
                   help="Fetch only playoff possessions (uses per-season date overrides for lockout/COVID years).")

    # Diagnostics
    p.add_argument("--no-diagnostics", dest="run_diagnostics", action="store_false", default=True)
    p.add_argument("--no-std-errors", dest="compute_std_errors", action="store_false", default=True,
                   help="Skip Jacobs closed-form standard errors (faster).")

    return p.parse_args(argv)


def cfg_from_args(args: argparse.Namespace) -> RunConfig:
    seasons, window_label = resolve_seasons(args.window, args.end_season, args.start_season)

    def _parse_grid(s: str) -> tuple[float, ...]:
        return tuple(float(x.strip()) for x in s.split(",") if x.strip())

    cfg = RunConfig(
        seasons=seasons,
        window_label=window_label,
        suffix=args.suffix,
        use_home=args.use_home,
        use_rubberband=args.use_rubberband,
        use_playoff=args.use_playoff,
        use_b2b=args.use_b2b,
        use_coach=args.use_coach,
        within_season_gamma=args.within_season_gamma,
        cross_season_rate=args.cross_season_rate,
        search_mode=args.search_mode,
        cv_folds=args.cv_folds,
        cv_method=args.cv_method,
        alpha_grid=_parse_grid(args.alpha_grid),
        ratio_home=args.ratio_home,
        ratio_rb=args.ratio_rb,
        ratio_playoff=args.ratio_playoff,
        ratio_b2b=args.ratio_b2b,
        ratio_coach=args.ratio_coach,
        grid_off=_parse_grid(args.grid_off),
        grid_def=_parse_grid(args.grid_def),
        grid_cv_folds=args.grid_cv_folds,
        lam_home=args.lam_home,
        lam_rb=args.lam_rb,
        lam_playoff=args.lam_playoff,
        lam_b2b=args.lam_b2b,
        lam_coach=args.lam_coach,
        replacement_mode=args.replacement_shrinkage,
        tier_edges=TierEdges(edges=_parse_grid(args.mpg_edges)),
        playoff_only=args.playoff_only,
        run_diagnostics=args.run_diagnostics,
        compute_std_errors=args.compute_std_errors,
    )
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = cfg_from_args(args)
    run(cfg)


if __name__ == "__main__":
    main()
