#!/usr/bin/env python3
"""standard_rapm.py - rebuilt RAPM runner.

This script is intentionally self-contained: it fetches possessions, builds a
sparse RAPM design matrix, tunes block lambdas without a large Cartesian grid,
fits regular-season or playoff RAPM, and writes reproducible outputs.
"""
from __future__ import annotations
import os

import argparse
import csv
import json
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date as Date
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import cg, splu
from sklearn.model_selection import GroupKFold, KFold

from paths import (
    ALL_NAMES_CSV,
    STANDARD_RAPM_DIAGNOSTICS,
    STANDARD_RAPM_DUMP,
    STANDARD_RAPM_RESULTS,
    ensure_dirs,
)

ensure_dirs()


SeasonType = Literal["regular", "playoff", "all"]
PriorMode = Literal["zero", "target", "offset", "dummy"]


BLOCK_OFF = 0
BLOCK_DEF = 1
BLOCK_HOME = 2
BLOCK_RB = 3
BLOCK_SEASON = 4

BLOCK_NAMES = {
    BLOCK_OFF: "off",
    BLOCK_DEF: "def",
    BLOCK_HOME: "home",
    BLOCK_RB: "rubberband",
    BLOCK_SEASON: "season",
}


PLAYOFF_DATE_OVERRIDES: dict[int, tuple[Date, Date]] = {
    1999: (Date(1999, 5, 8), Date(1999, 6, 25)),
    2012: (Date(2012, 4, 28), Date(2012, 6, 21)),
    2020: (Date(2020, 8, 17), Date(2020, 10, 11)),
    2021: (Date(2021, 5, 22), Date(2021, 7, 20)),
}

GT_BASE = {1: 25, 2: 20, 3: 17, 4: 12}


@dataclass
class LambdaProfile:
    off: float = 3000.0
    defense: float = 3000.0
    meta: float = 300.0
    season: float = 100.0

    def as_dict(self) -> dict[str, float]:
        return {
            "lambda_off": float(self.off),
            "lambda_def": float(self.defense),
            "lambda_meta": float(self.meta),
            "lambda_season": float(self.season),
        }


@dataclass
class RunConfig:
    seasons: list[int]
    season_type: SeasonType = "regular"
    spec: str = "standard_rs_v1"
    prior_mode: PriorMode = "zero"
    include_home: bool = True
    include_rubberband: bool = True
    include_season_effects: bool = True
    garbage_time: bool = True
    standardize_rubberband: bool = True
    optimize_lambdas: bool = True
    cv_folds: int = 3
    lambda_search_iters: int = 3
    lambda_profile: LambdaProfile = field(default_factory=LambdaProfile)
    off_conf: float = 1000.0
    def_conf: float = 1000.0
    compute_intervals: bool = True
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class DesignMatrix:
    X: csr_matrix
    y: np.ndarray
    weights: np.ndarray
    col_to_key: dict[int, str]
    key_to_col: dict[str, int]
    block_of_col: np.ndarray
    gameids: np.ndarray
    row_seasons: np.ndarray
    kept_rows: int
    dropped_garbage_time: int
    meta: dict


@dataclass
class FitResult:
    beta: np.ndarray
    intercept: float
    lambda_profile: LambdaProfile
    zero_penalty: np.ndarray
    target_penalty: np.ndarray
    target: np.ndarray
    cv_scores: pd.DataFrame
    elapsed_seconds: float

    @property
    def total_penalty(self) -> np.ndarray:
        return self.zero_penalty + self.target_penalty


def format_season_list(seasons: list[int]) -> str:
    if not seasons:
        return "unknown"
    if len(seasons) == 1:
        return str(seasons[0])
    return f"{seasons[0]}-" + "-".join(str(y)[-2:] for y in seasons[1:])


def resolve_seasons(window: str, end_season: int | None, start_season: int | None) -> list[int]:
    if window == "custom":
        if start_season is None or end_season is None:
            raise SystemExit("--window custom requires --start-season and --end-season")
        if start_season > end_season:
            raise SystemExit("--start-season must be <= --end-season")
        return list(range(start_season, end_season + 1))
    size = int(window)
    if size <= 0:
        raise SystemExit("--window must be positive")
    if end_season is None:
        end_season = 2024
    return list(range(end_season - size + 1, end_season + 1))


def _ensure_mysql_driver():
    try:
        import MySQLdb  # type: ignore

        return MySQLdb
    except Exception:
        import pymysql  # type: ignore

        pymysql.install_as_MySQLdb()
        import MySQLdb  # type: ignore

        return MySQLdb


def connect_db():
    mysql = _ensure_mysql_driver()
    return mysql.connect(
        host="localhost",
        user="root",
        password=os.environ.get("NBA_DB_PASSWORD", ""),
        db="nba_api",
        unix_socket="/tmp/mysql.sock",
    )


def fetch_possessions(seasons: Iterable[int]) -> list[tuple]:
    seasons = list(seasons)
    placeholders = ",".join(["%s"] * len(seasons))
    query = f"""
        SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5,
               season, date, period, num, gameid
        FROM matchups
        WHERE season IN ({placeholders})
          AND pts IS NOT NULL
        ORDER BY season, date, gameid, period, num
    """
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(query, seasons)
    rows = list(cur.fetchall())
    cur.close()
    conn.close()
    return rows


def as_date(value) -> Date | None:
    if value is None:
        return None
    if isinstance(value, Date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def playoff_window(season: int) -> tuple[Date, Date]:
    return PLAYOFF_DATE_OVERRIDES.get(season, (Date(season, 4, 12), Date(season, 6, 30)))


def is_playoff_game(season: int, game_date) -> bool:
    parsed = as_date(game_date)
    if parsed is None:
        return False
    lo, hi = playoff_window(season)
    return lo <= parsed <= hi


def keep_for_season_type(row: tuple, season_type: SeasonType) -> bool:
    if season_type == "all":
        return True
    is_playoff = is_playoff_game(int(row[12]), row[13])
    return is_playoff if season_type == "playoff" else not is_playoff


def gt_threshold(period: int, progress: float) -> int:
    base = GT_BASE.get(period, 12)
    progress = max(0.0, min(1.0, float(progress)))
    return base + int(8 * (1.0 - progress))


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _filtered_possessions(rows: list[tuple], cfg: RunConfig) -> tuple[list[dict], int]:
    rows = [r for r in rows if keep_for_season_type(r, cfg.season_type)]
    max_num: defaultdict[tuple, int] = defaultdict(int)
    for r in rows:
        gameid = r[16]
        period = _safe_int(r[14], 1)
        num = _safe_int(r[15], 0)
        max_num[(gameid, period)] = max(max_num[(gameid, period)], num)

    game_score: defaultdict = defaultdict(lambda: [0, 0])
    kept: list[dict] = []
    dropped = 0
    for r in rows:
        home_poss = bool(r[0])
        pts = float(r[1])
        away = [int(r[i]) for i in range(2, 7)]
        home = [int(r[i]) for i in range(7, 12)]
        season = int(r[12])
        period = _safe_int(r[14], 1)
        num = _safe_int(r[15], 0)
        gameid = r[16]

        home_pts, away_pts = game_score[gameid]
        margin_home = home_pts - away_pts
        margin_from_offense = margin_home if home_poss else -margin_home
        denom = max(1, max_num[(gameid, period)])
        progress = min(1.0, num / float(denom))

        if cfg.garbage_time and abs(margin_home) >= gt_threshold(period, progress):
            dropped += 1
        else:
            kept.append(
                {
                    "pts": pts,
                    "off_players": home if home_poss else away,
                    "def_players": away if home_poss else home,
                    "home_poss": home_poss,
                    "margin_from_offense": margin_from_offense,
                    "season": season,
                    "date": r[13],
                    "period": period,
                    "num": num,
                    "gameid": gameid,
                }
            )

        if home_poss:
            game_score[gameid][0] += pts
        else:
            game_score[gameid][1] += pts

    return kept, dropped


def build_design_matrix(raw_rows: list[tuple], cfg: RunConfig) -> DesignMatrix:
    possessions, dropped = _filtered_possessions(raw_rows, cfg)
    if not possessions:
        raise RuntimeError("No possessions available after season/garbage-time filters.")

    players = sorted(
        {
            player
            for poss in possessions
            for player in [*poss["off_players"], *poss["def_players"]]
        }
    )
    col_to_key: dict[int, str] = {}
    key_to_col: dict[str, int] = {}

    def add_col(key: str) -> int:
        idx = len(col_to_key)
        col_to_key[idx] = key
        key_to_col[key] = idx
        return idx

    for player in players:
        add_col(f"{player}_off")
        add_col(f"{player}_def")

    if cfg.include_home:
        add_col("META_home")
    if cfg.include_rubberband:
        add_col("META_rubberband")
    if cfg.include_season_effects:
        for season in sorted(set(cfg.seasons))[1:]:
            add_col(f"META_season_{season}")

    block_of_col = np.empty(len(col_to_key), dtype=np.int32)
    for idx, key in col_to_key.items():
        if key.endswith("_off"):
            block_of_col[idx] = BLOCK_OFF
        elif key.endswith("_def"):
            block_of_col[idx] = BLOCK_DEF
        elif key == "META_home":
            block_of_col[idx] = BLOCK_HOME
        elif key == "META_rubberband":
            block_of_col[idx] = BLOCK_RB
        elif key.startswith("META_season_"):
            block_of_col[idx] = BLOCK_SEASON
        else:
            raise ValueError(f"Unknown column key: {key}")

    rb_values = np.asarray([p["margin_from_offense"] for p in possessions], dtype=np.float64)
    rb_mean = float(rb_values.mean()) if len(rb_values) else 0.0
    rb_scale = float(rb_values.std()) if len(rb_values) else 1.0
    if rb_scale <= 1e-12:
        rb_scale = 1.0

    row_idx: list[int] = []
    col_idx: list[int] = []
    vals: list[float] = []
    y = np.empty(len(possessions), dtype=np.float64)
    gameids = np.empty(len(possessions), dtype=object)
    row_seasons = np.empty(len(possessions), dtype=np.int32)

    for i, poss in enumerate(possessions):
        for player in poss["off_players"]:
            row_idx.append(i)
            col_idx.append(key_to_col[f"{player}_off"])
            vals.append(1.0)
        for player in poss["def_players"]:
            row_idx.append(i)
            col_idx.append(key_to_col[f"{player}_def"])
            vals.append(1.0)
        if cfg.include_home:
            row_idx.append(i)
            col_idx.append(key_to_col["META_home"])
            vals.append(1.0 if poss["home_poss"] else -1.0)
        if cfg.include_rubberband:
            rb = float(poss["margin_from_offense"])
            if cfg.standardize_rubberband:
                rb = (rb - rb_mean) / rb_scale
            row_idx.append(i)
            col_idx.append(key_to_col["META_rubberband"])
            vals.append(rb)
        if cfg.include_season_effects:
            key = f"META_season_{poss['season']}"
            if key in key_to_col:
                row_idx.append(i)
                col_idx.append(key_to_col[key])
                vals.append(1.0)

        y[i] = poss["pts"]
        gameids[i] = poss["gameid"]
        row_seasons[i] = int(poss["season"])

    X = csr_matrix(
        (
            np.asarray(vals, dtype=np.float64),
            (np.asarray(row_idx, dtype=np.int64), np.asarray(col_idx, dtype=np.int64)),
        ),
        shape=(len(possessions), len(col_to_key)),
    )
    return DesignMatrix(
        X=X,
        y=y,
        weights=np.ones(len(possessions), dtype=np.float64),
        col_to_key=col_to_key,
        key_to_col=key_to_col,
        block_of_col=block_of_col,
        gameids=gameids,
        row_seasons=row_seasons,
        kept_rows=len(possessions),
        dropped_garbage_time=dropped,
        meta={
            "rubberband_mean": rb_mean,
            "rubberband_scale": rb_scale,
            "season_type": cfg.season_type,
            "garbage_time": cfg.garbage_time,
        },
    )


def lambda_vector(dm: DesignMatrix, profile: LambdaProfile) -> np.ndarray:
    lv = np.empty(dm.X.shape[1], dtype=np.float64)
    lv[dm.block_of_col == BLOCK_OFF] = profile.off
    lv[dm.block_of_col == BLOCK_DEF] = profile.defense
    lv[dm.block_of_col == BLOCK_HOME] = profile.meta
    lv[dm.block_of_col == BLOCK_RB] = profile.meta
    lv[dm.block_of_col == BLOCK_SEASON] = profile.season
    return lv


def load_prior(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    if {"key", "coef"}.issubset(df.columns):
        return dict(zip(df["key"].astype(str), df["coef"].astype(float)))
    if {"Key", "Coefficient"}.issubset(df.columns):
        return dict(zip(df["Key"].astype(str), df["Coefficient"].astype(float)))
    if {"Player", "Ridge Coefficient"}.issubset(df.columns):
        return dict(zip(df["Player"].astype(str), df["Ridge Coefficient"].astype(float)))
    raise ValueError(f"Unsupported prior file format: {path}")


def prior_target_vector(dm: DesignMatrix, prior: dict[str, float]) -> np.ndarray:
    target = np.zeros(dm.X.shape[1], dtype=np.float64)
    for idx, key in dm.col_to_key.items():
        if key in prior:
            target[idx] = float(prior[key])
    return target


def prior_confidence_vector(dm: DesignMatrix, prior: dict[str, float], cfg: RunConfig) -> np.ndarray:
    conf = np.zeros(dm.X.shape[1], dtype=np.float64)
    for idx, key in dm.col_to_key.items():
        if key not in prior:
            continue
        if key.endswith("_off"):
            conf[idx] = cfg.off_conf
        elif key.endswith("_def"):
            conf[idx] = cfg.def_conf
    return conf


def penalty_vectors(
    dm: DesignMatrix,
    cfg: RunConfig,
    profile: LambdaProfile,
    prior: dict[str, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = lambda_vector(dm, profile)
    target = prior_target_vector(dm, prior or {})
    zeros = np.zeros_like(base)
    target_penalty = np.zeros_like(base)

    if cfg.prior_mode == "zero":
        zeros = base
    elif cfg.prior_mode in {"target", "offset"}:
        target_penalty = base
    elif cfg.prior_mode == "dummy":
        zeros = base
        target_penalty = prior_confidence_vector(dm, prior or {}, cfg)
    else:
        raise ValueError(f"Unknown prior mode: {cfg.prior_mode}")

    return zeros, target_penalty, target


def _cg_solve(a_mat: csc_matrix, rhs: np.ndarray) -> np.ndarray:
    try:
        sol, info = cg(a_mat.tocsr(), rhs, rtol=1e-6, maxiter=5000)
    except TypeError:
        sol, info = cg(a_mat.tocsr(), rhs, tol=1e-6, maxiter=5000)
    if info != 0:
        solver = splu(a_mat)
        sol = solver.solve(rhs)
    return sol


def solve_penalized_ridge(
    X: csr_matrix,
    y: np.ndarray,
    weights: np.ndarray,
    zero_penalty: np.ndarray,
    target_penalty: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    intercept = float(np.average(y, weights=weights)) if weights.sum() > 0 else float(y.mean())
    y_centered = y - intercept
    w_sqrt = np.sqrt(weights)
    Xw = X.multiply(w_sqrt[:, None]).tocsr()
    penalty = zero_penalty + target_penalty
    lhs = (Xw.T @ Xw).tocsc() + diags(penalty, format="csc")
    rhs = X.T @ (weights * y_centered) + target_penalty * target
    beta = _cg_solve(lhs, rhs)
    return beta, intercept


def predict(X: csr_matrix, beta: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(X @ beta).ravel() + intercept


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.ones_like(y_true, dtype=np.float64)
    return float(math.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.ones_like(y_true, dtype=np.float64)
    return float(np.average(np.abs(y_true - y_pred), weights=weights))


def grouped_splits(gameids: np.ndarray, n_folds: int):
    unique_games = np.unique(gameids)
    n_folds = max(2, min(int(n_folds), len(unique_games)))
    if len(unique_games) >= n_folds:
        splitter = GroupKFold(n_splits=n_folds)
        return list(splitter.split(np.arange(len(gameids)), groups=gameids))
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    return list(splitter.split(np.arange(len(gameids))))


def cv_rmse_for_profile(
    dm: DesignMatrix,
    cfg: RunConfig,
    profile: LambdaProfile,
    prior: dict[str, float] | None,
) -> float:
    splits = grouped_splits(dm.gameids, cfg.cv_folds)
    fold_scores: list[float] = []
    zero_full, target_full, target_vec = penalty_vectors(dm, cfg, profile, prior)
    for train_idx, valid_idx in splits:
        beta, intercept = solve_penalized_ridge(
            dm.X[train_idx],
            dm.y[train_idx],
            dm.weights[train_idx],
            zero_full,
            target_full,
            target_vec,
        )
        y_pred = predict(dm.X[valid_idx], beta, intercept)
        fold_scores.append(weighted_rmse(dm.y[valid_idx], y_pred, dm.weights[valid_idx]))
    return float(np.mean(fold_scores))


def tune_lambdas(
    dm: DesignMatrix,
    cfg: RunConfig,
    prior: dict[str, float] | None,
) -> tuple[LambdaProfile, pd.DataFrame]:
    if not cfg.optimize_lambdas:
        rec = cfg.lambda_profile.as_dict()
        rec["mean_rmse"] = cv_rmse_for_profile(dm, cfg, cfg.lambda_profile, prior)
        rec["stage"] = "fixed"
        return cfg.lambda_profile, pd.DataFrame([rec])

    current = LambdaProfile(
        off=cfg.lambda_profile.off,
        defense=cfg.lambda_profile.defense,
        meta=cfg.lambda_profile.meta,
        season=cfg.lambda_profile.season,
    )
    best_score = cv_rmse_for_profile(dm, cfg, current, prior)
    records = [{**current.as_dict(), "mean_rmse": best_score, "stage": "initial"}]

    fields = ["off", "defense", "meta", "season"]
    log_step = 0.75
    for iteration in range(cfg.lambda_search_iters):
        improved = False
        for field_name in fields:
            for direction in (-1.0, 1.0):
                candidate = LambdaProfile(
                    off=current.off,
                    defense=current.defense,
                    meta=current.meta,
                    season=current.season,
                )
                value = getattr(candidate, field_name)
                setattr(candidate, field_name, float(np.clip(value * (10.0 ** (direction * log_step)), 1.0, 100000.0)))
                score = cv_rmse_for_profile(dm, cfg, candidate, prior)
                records.append({
                    **candidate.as_dict(),
                    "mean_rmse": score,
                    "stage": f"iter_{iteration + 1}_{field_name}_{'up' if direction > 0 else 'down'}",
                })
                if score + 1e-8 < best_score:
                    current = candidate
                    best_score = score
                    improved = True
        if not improved:
            log_step *= 0.5

    scores = pd.DataFrame(records).sort_values("mean_rmse")
    best = scores.iloc[0]
    profile = LambdaProfile(
        off=float(best["lambda_off"]),
        defense=float(best["lambda_def"]),
        meta=float(best["lambda_meta"]),
        season=float(best["lambda_season"]),
    )
    return profile, scores


def fit_model(dm: DesignMatrix, cfg: RunConfig, prior: dict[str, float] | None = None) -> FitResult:
    start = time.time()
    if cfg.prior_mode != "zero" and not prior:
        raise ValueError(f"prior_mode={cfg.prior_mode} requires a prior file or prior dict")
    profile, cv_scores = tune_lambdas(dm, cfg, prior)
    zero_penalty, target_penalty, target = penalty_vectors(dm, cfg, profile, prior)
    beta, intercept = solve_penalized_ridge(
        dm.X,
        dm.y,
        dm.weights,
        zero_penalty,
        target_penalty,
        target,
    )
    return FitResult(
        beta=beta,
        intercept=intercept,
        lambda_profile=profile,
        zero_penalty=zero_penalty,
        target_penalty=target_penalty,
        target=target,
        cv_scores=cv_scores,
        elapsed_seconds=time.time() - start,
    )


def standard_errors(dm: DesignMatrix, fit: FitResult) -> pd.DataFrame:
    X = dm.X
    weights = dm.weights
    residuals = dm.y - predict(X, fit.beta, fit.intercept)
    p_players = int(np.sum((dm.block_of_col == BLOCK_OFF) | (dm.block_of_col == BLOCK_DEF)))
    dof = max(1.0, X.shape[0] - p_players - 1)
    sigma2 = float(np.sum(weights * residuals ** 2) / dof)

    Xw = X.multiply(np.sqrt(weights)[:, None]).tocsr()
    lhs = (Xw.T @ Xw).tocsc() + diags(fit.total_penalty, format="csc")
    solver = splu(lhs)
    n_cols = X.shape[1]

    diag_inv = np.zeros(n_cols, dtype=np.float64)
    off_idx: dict[int, int] = {}
    def_idx: dict[int, int] = {}
    for idx, key in dm.col_to_key.items():
        if key.endswith("_off"):
            off_idx[int(key.split("_")[0])] = idx
        elif key.endswith("_def"):
            def_idx[int(key.split("_")[0])] = idx

    player_cols = sorted(set(off_idx.values()) | set(def_idx.values()))
    cov_od: dict[int, float] = {}
    batch = 128
    for start in range(0, len(player_cols), batch):
        cols = player_cols[start:start + batch]
        rhs = np.zeros((n_cols, len(cols)), dtype=np.float64)
        for j, col in enumerate(cols):
            rhs[col, j] = 1.0
        sol = solver.solve(rhs)
        for j, col in enumerate(cols):
            diag_inv[col] = sol[col, j]
            key = dm.col_to_key[col]
            if key.endswith("_off"):
                pid = int(key.split("_")[0])
                if pid in def_idx:
                    cov_od[pid] = float(sigma2 * sol[def_idx[pid], j])

    meta_cols = [i for i, key in dm.col_to_key.items() if key.startswith("META_")]
    if meta_cols:
        rhs = np.zeros((n_cols, len(meta_cols)), dtype=np.float64)
        for j, col in enumerate(meta_cols):
            rhs[col, j] = 1.0
        sol = solver.solve(rhs)
        for j, col in enumerate(meta_cols):
            diag_inv[col] = sol[col, j]

    se = np.sqrt(np.maximum(0.0, sigma2 * diag_inv))
    df = pd.DataFrame({
        "col_idx": np.arange(n_cols),
        "key": [dm.col_to_key[i] for i in range(n_cols)],
        "se": se,
        "var": se ** 2,
    })
    df.attrs["cov_od"] = cov_od
    df.attrs["sigma2"] = sigma2
    df.attrs["dof"] = dof
    return df


def name_lookup() -> dict[str, str]:
    names = pd.read_csv(ALL_NAMES_CSV)
    return dict(zip(names["PLAYER_ID"].astype(str), names["PLAYER_NAME"]))


def player_table(dm: DesignMatrix, fit: FitResult, cfg: RunConfig, se_df: pd.DataFrame | None) -> pd.DataFrame:
    names = name_lookup()
    col_sums = np.asarray(dm.X.sum(axis=0)).ravel()
    se_by_col: dict[int, float] = {}
    cov_od: dict[int, float] = {}
    if se_df is not None:
        se_by_col = {int(row.col_idx): float(row.se) for row in se_df.itertuples(index=False)}
        cov_od = se_df.attrs.get("cov_od", {})

    off_cols: dict[int, int] = {}
    def_cols: dict[int, int] = {}
    for idx, key in dm.col_to_key.items():
        if key.endswith("_off"):
            off_cols[int(key.split("_")[0])] = idx
        elif key.endswith("_def"):
            def_cols[int(key.split("_")[0])] = idx

    rows: list[dict] = []
    z = 1.96
    for pid, off_col in off_cols.items():
        def_col = def_cols.get(pid)
        if def_col is None:
            continue
        off = float(fit.beta[off_col]) * 100.0
        defense = float(fit.beta[def_col]) * 100.0
        rapm = off - defense
        rec = {
            "Name": names.get(str(pid), str(pid)),
            "PLAYER_ID": pid,
            "Poss_Off": int(col_sums[off_col]),
            "Poss_Def": int(col_sums[def_col]),
            "Off": round(off, 3),
            "Def": round(defense, 3),
            "RAPM": round(rapm, 3),
            "Season": format_season_list(cfg.seasons),
            "Season_Type": cfg.season_type,
            "Prior_Mode": cfg.prior_mode,
            **fit.lambda_profile.as_dict(),
        }
        if se_df is not None:
            off_se = se_by_col.get(off_col, float("nan")) * 100.0
            def_se = se_by_col.get(def_col, float("nan")) * 100.0
            cov = cov_od.get(pid, 0.0) * (100.0 ** 2)
            rapm_se = math.sqrt(max(0.0, off_se ** 2 + def_se ** 2 - 2.0 * cov))
            rec.update({
                "Off_SE": round(off_se, 3),
                "Def_SE": round(def_se, 3),
                "RAPM_SE": round(rapm_se, 3),
                "Off_CI_lo": round(off - z * off_se, 3),
                "Off_CI_hi": round(off + z * off_se, 3),
                "Def_CI_lo": round(defense - z * def_se, 3),
                "Def_CI_hi": round(defense + z * def_se, 3),
                "RAPM_CI_lo": round(rapm - z * rapm_se, 3),
                "RAPM_CI_hi": round(rapm + z * rapm_se, 3),
            })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("RAPM", ascending=False)


def meta_table(dm: DesignMatrix, fit: FitResult, se_df: pd.DataFrame | None) -> pd.DataFrame:
    se_by_col: dict[int, float] = {}
    if se_df is not None:
        se_by_col = {int(row.col_idx): float(row.se) for row in se_df.itertuples(index=False)}
    rows = []
    for idx, key in dm.col_to_key.items():
        if not key.startswith("META_"):
            continue
        coef = float(fit.beta[idx])
        rows.append({
            "key": key,
            "coef": coef,
            "coef_per_100": coef * 100.0,
            "se": se_by_col.get(idx, float("nan")),
            "block": BLOCK_NAMES[int(dm.block_of_col[idx])],
        })
    return pd.DataFrame(rows)


def output_stem(cfg: RunConfig) -> str:
    return f"standard_rapm_{cfg.spec}_{cfg.season_type}_{format_season_list(cfg.seasons)}_{cfg.prior_mode}_{cfg.run_id}"


def write_outputs(dm: DesignMatrix, fit: FitResult, cfg: RunConfig, se_df: pd.DataFrame | None) -> dict[str, str]:
    stem = output_stem(cfg)
    raw_path = STANDARD_RAPM_DUMP / f"{stem}_coefficients.csv"
    players_path = STANDARD_RAPM_RESULTS / f"{stem}_players.csv"
    meta_path = STANDARD_RAPM_DUMP / f"{stem}_meta.csv"
    cv_path = STANDARD_RAPM_DIAGNOSTICS / f"{stem}_lambda_scores.csv"
    run_meta_path = STANDARD_RAPM_DIAGNOSTICS / f"{stem}_run.json"

    with open(raw_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["col_idx", "key", "coef", "block", "zero_penalty", "target_penalty", "target"])
        for idx, key in dm.col_to_key.items():
            writer.writerow([
                idx,
                key,
                float(fit.beta[idx]),
                BLOCK_NAMES[int(dm.block_of_col[idx])],
                float(fit.zero_penalty[idx]),
                float(fit.target_penalty[idx]),
                float(fit.target[idx]),
            ])

    players = player_table(dm, fit, cfg, se_df)
    players.to_csv(players_path, index=False)
    meta = meta_table(dm, fit, se_df)
    meta.to_csv(meta_path, index=False)
    fit.cv_scores.to_csv(cv_path, index=False)

    residuals = dm.y - predict(dm.X, fit.beta, fit.intercept)
    run_meta = {
        "run_id": cfg.run_id,
        "spec": cfg.spec,
        "seasons": cfg.seasons,
        "season_type": cfg.season_type,
        "prior_mode": cfg.prior_mode,
        "kept_possessions": dm.kept_rows,
        "dropped_garbage_time": dm.dropped_garbage_time,
        "n_columns": dm.X.shape[1],
        "intercept": fit.intercept,
        "lambda_profile": fit.lambda_profile.as_dict(),
        "rmse_in_sample": weighted_rmse(dm.y, predict(dm.X, fit.beta, fit.intercept), dm.weights),
        "mae_in_sample": weighted_mae(dm.y, predict(dm.X, fit.beta, fit.intercept), dm.weights),
        "residual_mean": float(residuals.mean()),
        "elapsed_seconds": fit.elapsed_seconds,
        "paths": {
            "raw_coefficients": str(raw_path),
            "players": str(players_path),
            "meta": str(meta_path),
            "lambda_scores": str(cv_path),
        },
        "design_meta": dm.meta,
    }
    with open(run_meta_path, "w") as handle:
        json.dump(run_meta, handle, indent=2, sort_keys=True)

    print(f"Players -> {players_path}")
    print(f"Raw coefficients -> {raw_path}")
    print(f"Diagnostics -> {run_meta_path}")
    print(players.head(15).to_string(index=False))
    return {
        "raw": str(raw_path),
        "players": str(players_path),
        "meta": str(meta_path),
        "lambda_scores": str(cv_path),
        "run_meta": str(run_meta_path),
    }


def coefficients_as_prior(dm: DesignMatrix, beta: np.ndarray) -> dict[str, float]:
    return {
        key: float(beta[idx])
        for idx, key in dm.col_to_key.items()
        if key.endswith("_off") or key.endswith("_def")
    }


def run_single(raw_rows: list[tuple], cfg: RunConfig, prior: dict[str, float] | None = None) -> tuple[DesignMatrix, FitResult, dict[str, str]]:
    print(
        f"\n=== {cfg.spec} | {cfg.season_type} | seasons {format_season_list(cfg.seasons)} "
        f"| prior={cfg.prior_mode} | run={cfg.run_id} ==="
    )
    dm = build_design_matrix(raw_rows, cfg)
    print(
        f"Design: {dm.X.shape[0]:,} possessions x {dm.X.shape[1]:,} columns "
        f"({dm.dropped_garbage_time:,} garbage-time possessions dropped)"
    )
    fit = fit_model(dm, cfg, prior)
    se_df = standard_errors(dm, fit) if cfg.compute_intervals else None
    paths = write_outputs(dm, fit, cfg, se_df)
    return dm, fit, paths


def run_playoff_suite(raw_rows: list[tuple], base_cfg: RunConfig) -> None:
    if base_cfg.season_type != "regular":
        print("Playoff suite always builds its regular-season prior from regular-season possessions.")

    regular_cfg = RunConfig(
        seasons=base_cfg.seasons,
        season_type="regular",
        spec=f"{base_cfg.spec}_regular_prior",
        prior_mode="zero",
        include_home=base_cfg.include_home,
        include_rubberband=base_cfg.include_rubberband,
        include_season_effects=base_cfg.include_season_effects,
        garbage_time=base_cfg.garbage_time,
        standardize_rubberband=base_cfg.standardize_rubberband,
        optimize_lambdas=base_cfg.optimize_lambdas,
        cv_folds=base_cfg.cv_folds,
        lambda_search_iters=base_cfg.lambda_search_iters,
        lambda_profile=base_cfg.lambda_profile,
        compute_intervals=base_cfg.compute_intervals,
    )
    reg_dm, reg_fit, reg_paths = run_single(raw_rows, regular_cfg)
    prior = coefficients_as_prior(reg_dm, reg_fit.beta)

    prior_cache = STANDARD_RAPM_DUMP / f"standard_rapm_prior_{base_cfg.spec}_{format_season_list(base_cfg.seasons)}_{regular_cfg.run_id}.csv"
    with open(prior_cache, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", "coef"])
        for key, coef in sorted(prior.items()):
            writer.writerow([key, coef])
    print(f"Regular-season prior cache -> {prior_cache}")
    print(f"Regular-season prior source -> {reg_paths['raw']}")

    for mode in ("zero", "offset", "dummy"):
        playoff_cfg = RunConfig(
            seasons=base_cfg.seasons,
            season_type="playoff",
            spec=f"{base_cfg.spec}_playoff_{mode}",
            prior_mode=mode,  # type: ignore[arg-type]
            include_home=base_cfg.include_home,
            include_rubberband=base_cfg.include_rubberband,
            include_season_effects=base_cfg.include_season_effects,
            garbage_time=base_cfg.garbage_time,
            standardize_rubberband=base_cfg.standardize_rubberband,
            optimize_lambdas=base_cfg.optimize_lambdas,
            cv_folds=base_cfg.cv_folds,
            lambda_search_iters=base_cfg.lambda_search_iters,
            lambda_profile=base_cfg.lambda_profile,
            off_conf=base_cfg.off_conf,
            def_conf=base_cfg.def_conf,
            compute_intervals=base_cfg.compute_intervals,
        )
        run_single(raw_rows, playoff_cfg, prior if mode != "zero" else None)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuilt standard RAPM runner.")
    parser.add_argument("--window", default="3", help="Window length, or 'custom'.")
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--season-type", choices=["regular", "playoff", "all"], default="regular")
    parser.add_argument("--spec", default="standard_rs_v1")
    parser.add_argument("--prior-mode", choices=["zero", "target", "offset", "dummy"], default="zero")
    parser.add_argument("--prior-file", default=None)
    parser.add_argument("--run-playoff-suite", action="store_true")

    parser.add_argument("--no-home", dest="include_home", action="store_false", default=True)
    parser.add_argument("--no-rubberband", dest="include_rubberband", action="store_false", default=True)
    parser.add_argument("--no-season-effects", dest="include_season_effects", action="store_false", default=True)
    parser.add_argument("--no-garbage-time", dest="garbage_time", action="store_false", default=True)
    parser.add_argument("--no-standardize-rubberband", dest="standardize_rubberband", action="store_false", default=True)

    parser.add_argument("--no-optimize-lambdas", dest="optimize_lambdas", action="store_false", default=True)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--lambda-search-iters", type=int, default=3)
    parser.add_argument("--lambda-off", type=float, default=3000.0)
    parser.add_argument("--lambda-def", type=float, default=3000.0)
    parser.add_argument("--lambda-meta", type=float, default=300.0)
    parser.add_argument("--lambda-season", type=float, default=100.0)
    parser.add_argument("--off-conf", type=float, default=1000.0)
    parser.add_argument("--def-conf", type=float, default=1000.0)
    parser.add_argument("--no-intervals", dest="compute_intervals", action="store_false", default=True)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunConfig:
    seasons = resolve_seasons(args.window, args.end_season, args.start_season)
    return RunConfig(
        seasons=seasons,
        season_type=args.season_type,
        spec=args.spec,
        prior_mode=args.prior_mode,
        include_home=args.include_home,
        include_rubberband=args.include_rubberband,
        include_season_effects=args.include_season_effects,
        garbage_time=args.garbage_time,
        standardize_rubberband=args.standardize_rubberband,
        optimize_lambdas=args.optimize_lambdas,
        cv_folds=args.cv_folds,
        lambda_search_iters=args.lambda_search_iters,
        lambda_profile=LambdaProfile(
            off=args.lambda_off,
            defense=args.lambda_def,
            meta=args.lambda_meta,
            season=args.lambda_season,
        ),
        off_conf=args.off_conf,
        def_conf=args.def_conf,
        compute_intervals=args.compute_intervals,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = config_from_args(args)
    raw_rows = fetch_possessions(cfg.seasons)
    print(f"Fetched {len(raw_rows):,} possessions for seasons {format_season_list(cfg.seasons)}")
    if args.run_playoff_suite:
        run_playoff_suite(raw_rows, cfg)
        return
    prior = load_prior(args.prior_file) if args.prior_file else None
    run_single(raw_rows, cfg, prior)


if __name__ == "__main__":
    main()
