#!/usr/bin/env python3
"""Simple weighted 3-year RAPM.

This is intentionally stripped down from the larger RAPM scripts:
- 2022-2024 possessions, including playoffs
- player offense/defense indicators plus home, playoff, clipped rubber band, and season columns
- uniform possession weights
- fixed lambdas from the prior coordinate-search run
- closed-form ridge standard errors / 95% confidence intervals
- age-at-mid-2024-25 and age-adjusted columns using the existing aging curve
"""
from __future__ import annotations
import os

import csv
import json
import math
import sys
import time
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, lil_matrix
from scipy.sparse.linalg import cg, splu


RAPM_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RAPM_ROOT.parent
SRC_DIR = RAPM_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import ALL_NAMES_CSV, PLAYERSHEETS_YEAR_TOTALS  # noqa: E402


SEASONS = [2022, 2023, 2024]
SEASON_WEIGHTS = {2022: 1.0, 2023: 1.0, 2024: 1.0}
ALPHAS = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]
RUBBER_BAND_LAMBDAS = [10, 25, 50, 100, 250, 500, 1000, 2000]
CONTEXT_LAMBDAS = [10, 25, 50, 100, 250, 500, 1000, 2000, 5000]
RUBBER_BAND_CLIP = 25.0
FIXED_LAMBDAS = {
    "offense": 2500.0,
    "defense": 4000.0,
    "home": 5000.0,
    "playoff": 5000.0,
    "rubber_band": 2000.0,
    "season": 5000.0,
}
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PLAYOFF_OVERRIDES = {
    2020: (Date(2020, 8, 17), Date(2020, 10, 11)),
    2021: (Date(2021, 5, 22), Date(2021, 7, 20)),
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_mysql_driver():
    try:
        import MySQLdb  # type: ignore

        return MySQLdb
    except Exception:
        import pymysql  # type: ignore

        pymysql.install_as_MySQLdb()
        import MySQLdb  # type: ignore

        return MySQLdb


def connect_db():
    mysql = ensure_mysql_driver()
    return mysql.connect(
        host="localhost",
        user="root",
        password=os.environ.get("NBA_DB_PASSWORD", ""),
        db="nba_api",
        unix_socket="/tmp/mysql.sock",
    )


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
    return PLAYOFF_OVERRIDES.get(season, (Date(season, 4, 12), Date(season, 6, 30)))


def is_playoff_game(season: int, game_date) -> bool:
    parsed = as_date(game_date)
    if parsed is None:
        return False
    lo, hi = playoff_window(season)
    return lo <= parsed <= hi


def fetch_possessions() -> list[tuple]:
    placeholders = ",".join(["%s"] * len(SEASONS))
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
    cur.execute(query, SEASONS)
    rows = list(cur.fetchall())
    cur.close()
    conn.close()
    return rows


def rubber_band_values(data: list[tuple]) -> np.ndarray:
    game_score: dict = {}
    values = np.zeros(len(data), dtype=np.float64)
    for counter, item in enumerate(data):
        home_poss = bool(item[0])
        pts = float(item[1])
        gameid = item[16]
        if gameid not in game_score:
            game_score[gameid] = [0.0, 0.0]
        home_pts, away_pts = game_score[gameid]
        margin_home = home_pts - away_pts
        margin_from_offense = margin_home if home_poss else -margin_home
        values[counter] = np.clip(margin_from_offense, -RUBBER_BAND_CLIP, RUBBER_BAND_CLIP)
        if home_poss:
            game_score[gameid][0] += pts
        else:
            game_score[gameid][1] += pts
    return values


def build_matrix(data: list[tuple]) -> tuple[csr_matrix, np.ndarray, np.ndarray, dict[int, str], dict]:
    all_players = {}
    for item in data:
        for i in range(2, 12):
            all_players[int(item[i])] = 1

    player_to_col: dict[str, int] = {}
    col_to_player: dict[int, str] = {}
    for player in sorted(all_players):
        for side in ("off", "def"):
            key = f"{player}_{side}"
            number = len(player_to_col)
            player_to_col[key] = number
            col_to_player[number] = key

    rubber_band_col = len(col_to_player)
    col_to_player[rubber_band_col] = "rubber_band"
    home_col = len(col_to_player)
    col_to_player[home_col] = "home_effect"
    playoff_col = len(col_to_player)
    col_to_player[playoff_col] = "playoff_indicator"
    season_cols = {}
    for season in SEASONS[1:]:
        col = len(col_to_player)
        season_cols[season] = col
        col_to_player[col] = f"season_{season}"

    rubber_band_raw = rubber_band_values(data)
    rubber_band_mean = float(rubber_band_raw.mean())
    rubber_band_std = float(rubber_band_raw.std())
    if rubber_band_std <= 1e-12:
        rubber_band_std = 1.0
    rubber_band_z = (rubber_band_raw - rubber_band_mean) / rubber_band_std

    X = lil_matrix((len(data), len(col_to_player)), dtype=np.float64)
    y = np.zeros(len(data), dtype=np.float64)
    sample_weights = np.zeros(len(data), dtype=np.float64)

    for counter, item in enumerate(data):
        home_poss = bool(item[0])
        pts = float(item[1])
        season = int(item[12])
        game_date = item[13]
        away_list = [int(item[i]) for i in range(2, 7)]
        home_list = [int(item[i]) for i in range(7, 12)]
        off_list, def_list = (home_list, away_list) if home_poss else (away_list, home_list)

        for player in off_list:
            X[counter, player_to_col[f"{player}_off"]] = 1.0
        for player in def_list:
            X[counter, player_to_col[f"{player}_def"]] = 1.0

        X[counter, home_col] = 1.0 if home_poss else -1.0
        if is_playoff_game(season, game_date):
            X[counter, playoff_col] = 1.0
        X[counter, rubber_band_col] = rubber_band_z[counter]
        if season in season_cols:
            X[counter, season_cols[season]] = 1.0
        y[counter] = pts
        sample_weights[counter] = SEASON_WEIGHTS[season]

    rubber_band_meta = {
        "rubber_band_col": rubber_band_col,
        "home_col": home_col,
        "playoff_col": playoff_col,
        "rubber_band_clip": RUBBER_BAND_CLIP,
        "rubber_band_mean": rubber_band_mean,
        "rubber_band_std": rubber_band_std,
        "season_cols": season_cols,
        "season_baseline": SEASONS[0],
    }
    return X.tocsr(), y, sample_weights, col_to_player, rubber_band_meta


def lambda_vector(col_to_player: dict[int, str], lambdas: dict[str, float]) -> np.ndarray:
    penalty = np.zeros(len(col_to_player), dtype=np.float64)
    for col, key in col_to_player.items():
        if key.endswith("_off"):
            penalty[col] = lambdas["offense"]
        elif key.endswith("_def"):
            penalty[col] = lambdas["defense"]
        elif key == "rubber_band":
            penalty[col] = lambdas["rubber_band"]
        elif key == "home_effect":
            penalty[col] = lambdas["home"]
        elif key == "playoff_indicator":
            penalty[col] = lambdas["playoff"]
        elif key.startswith("season_"):
            penalty[col] = lambdas["season"]
        else:
            raise ValueError(f"Unknown column key: {key}")
    return penalty


def fit_block_ridge(
    X: csr_matrix,
    y_centered: np.ndarray,
    sample_weights: np.ndarray,
    col_to_player: dict[int, str],
    lambdas: dict[str, float],
) -> np.ndarray:
    Xw = X.multiply(np.sqrt(sample_weights)[:, None]).tocsr()
    lhs = (Xw.T @ Xw).tocsc() + diags(lambda_vector(col_to_player, lambdas), format="csc")
    rhs = X.T @ (sample_weights * y_centered)
    try:
        beta, info = cg(lhs.tocsr(), rhs, rtol=1e-6, maxiter=5000)
    except TypeError:
        beta, info = cg(lhs.tocsr(), rhs, tol=1e-6, maxiter=5000)
    if info != 0:
        beta = splu(lhs).solve(rhs)
    return beta


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(math.sqrt(np.average((y_true - y_pred) ** 2, weights=weights)))


def time_order_splits(n_rows: int, n_folds: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_rows)
    folds = np.array_split(indices, n_folds)
    splits = []
    for fold in folds:
        valid = fold
        train = np.setdiff1d(indices, valid, assume_unique=True)
        splits.append((train, valid))
    return splits


def score_lambdas(
    X: csr_matrix,
    y_centered: np.ndarray,
    sample_weights: np.ndarray,
    col_to_player: dict[int, str],
    lambdas: dict[str, float],
) -> float:
    scores = []
    for train_idx, valid_idx in time_order_splits(X.shape[0], n_folds=4):
        beta = fit_block_ridge(
            X[train_idx],
            y_centered[train_idx],
            sample_weights[train_idx],
            col_to_player,
            lambdas,
        )
        y_pred = np.asarray(X[valid_idx] @ beta).ravel()
        scores.append(weighted_rmse(y_centered[valid_idx], y_pred, sample_weights[valid_idx]))
    return float(np.mean(scores))


def coordinate_search_lambdas(
    X: csr_matrix,
    y_centered: np.ndarray,
    sample_weights: np.ndarray,
    col_to_player: dict[int, str],
) -> tuple[dict[str, float], pd.DataFrame]:
    candidates = {
        "home": CONTEXT_LAMBDAS,
        "season": CONTEXT_LAMBDAS,
        "playoff": CONTEXT_LAMBDAS,
    }
    lambdas = dict(FIXED_LAMBDAS)
    records = []
    for iteration in range(3):
        changed = False
        for block in ("home", "season", "playoff"):
            best_value = lambdas[block]
            best_score = None
            for candidate in candidates[block]:
                trial = dict(lambdas)
                trial[block] = float(candidate)
                score = score_lambdas(X, y_centered, sample_weights, col_to_player, trial)
                records.append({
                    "iteration": iteration + 1,
                    "tuned_block": block,
                    "lambda_offense": trial["offense"],
                    "lambda_defense": trial["defense"],
                    "lambda_home": trial["home"],
                    "lambda_playoff": trial["playoff"],
                    "lambda_rubber_band": trial["rubber_band"],
                    "lambda_season": trial["season"],
                    "cv_rmse": score,
                })
                if best_score is None or score < best_score:
                    best_score = score
                    best_value = float(candidate)
            if lambdas[block] != best_value:
                changed = True
            lambdas[block] = best_value
        if not changed:
            break
    scores = pd.DataFrame(records).sort_values("cv_rmse")
    return lambdas, scores


def run_ridge_model(
    X: csr_matrix,
    y_centered: np.ndarray,
    sample_weights: np.ndarray,
    col_to_player: dict[int, str],
):
    lambdas, scores = coordinate_search_lambdas(X, y_centered, sample_weights, col_to_player)
    print(
        "LAMBDAS:",
        f"offense={lambdas['offense']}",
        f"defense={lambdas['defense']}",
        f"home={lambdas['home']}",
        f"playoff={lambdas['playoff']}",
        f"rubber_band={lambdas['rubber_band']}",
        f"season={lambdas['season']}",
    )
    beta = fit_block_ridge(X, y_centered, sample_weights, col_to_player, lambdas)
    return beta, lambdas, scores


def ridge_standard_errors(
    X: csr_matrix,
    y: np.ndarray,
    sample_weights: np.ndarray,
    beta: np.ndarray,
    intercept: float,
    lambdas: dict[str, float],
    col_to_player: dict[int, str],
) -> tuple[np.ndarray, dict[int, float]]:
    residuals = y - (X @ beta + intercept)
    p_players = sum(1 for key in col_to_player.values() if key.endswith("_off") or key.endswith("_def"))
    dof = max(1.0, X.shape[0] - p_players - 1)
    sigma2 = float(np.sum(sample_weights * residuals**2) / dof)

    Xw = X.multiply(np.sqrt(sample_weights)[:, None]).tocsr()
    lhs = (Xw.T @ Xw).tocsc() + diags(lambda_vector(col_to_player, lambdas), format="csc")
    solver = splu(lhs)

    se = np.zeros(X.shape[1], dtype=np.float64)
    off_cols: dict[int, int] = {}
    def_cols: dict[int, int] = {}
    for col, key in col_to_player.items():
        if key in {"rubber_band", "home_effect", "playoff_indicator"} or key.startswith("season_"):
            continue
        player_id, side = key.split("_")
        if side == "off":
            off_cols[int(player_id)] = col
        else:
            def_cols[int(player_id)] = col

    cov_od: dict[int, float] = {}
    columns = np.arange(X.shape[1])
    batch_size = 128
    for start in range(0, len(columns), batch_size):
        cols = columns[start:start + batch_size]
        rhs = np.zeros((X.shape[1], len(cols)), dtype=np.float64)
        for j, col in enumerate(cols):
            rhs[col, j] = 1.0
        sol = solver.solve(rhs)
        for j, col in enumerate(cols):
            se[col] = math.sqrt(max(0.0, sigma2 * sol[col, j]))
            key = col_to_player[col]
            if key in {"rubber_band", "home_effect", "playoff_indicator"} or key.startswith("season_"):
                continue
            player_id, side = key.split("_")
            pid = int(player_id)
            if side == "off" and pid in def_cols:
                cov_od[pid] = float(sigma2 * sol[def_cols[pid], j])
    return se, cov_od


def load_names() -> dict[int, str]:
    names = pd.read_csv(ALL_NAMES_CSV)
    return dict(zip(names["PLAYER_ID"].astype(int), names["PLAYER_NAME"].astype(str)))


def load_age_lookup() -> dict[int, float]:
    path = PLAYERSHEETS_YEAR_TOTALS / "2024.csv"
    df = pd.read_csv(path, usecols=["PLAYER_ID", "AGE"])
    df = df.dropna(subset=["PLAYER_ID", "AGE"])
    # 2024 is the 2023-24 season. Mid-season 2024-25 is roughly one year later.
    return dict(zip(df["PLAYER_ID"].astype(int), df["AGE"].astype(float) + 1.0))


def load_aging_curve() -> pd.DataFrame | None:
    path = RAPM_ROOT / "outputs" / "aging" / "aging_curve.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def curve_offset(curve: pd.DataFrame | None, age: float | None, col: str, ref_age: int = 27) -> float:
    if curve is None or age is None or np.isnan(age):
        return 0.0
    ages = curve["age"].astype(int).to_numpy()
    lo, hi = int(ages.min()), int(ages.max())
    age_i = int(np.clip(round(age), lo, hi))
    ref_i = int(np.clip(ref_age, lo, hi))
    by_age = dict(zip(curve["age"].astype(int), curve[col].astype(float)))
    return float(by_age[ref_i] - by_age[age_i])


def write_outputs(
    X: csr_matrix,
    beta: np.ndarray,
    lambdas: dict[str, float],
    intercept: float,
    y: np.ndarray,
    sample_weights: np.ndarray,
    col_to_player: dict[int, str],
    rubber_band_meta: dict,
    lambda_scores: pd.DataFrame,
) -> tuple[Path, Path, Path, Path]:
    names = load_names()
    ages = load_age_lookup()
    curve = load_aging_curve()
    se, cov_od = ridge_standard_errors(X, y, sample_weights, beta, intercept, lambdas, col_to_player)
    col_sums = np.asarray(X.sum(axis=0)).ravel()

    raw_path = OUTPUT_DIR / "simple_rapm_with_playoff_indicator_coefficients_2022_2024.csv"
    players_path = OUTPUT_DIR / "simple_rapm_with_playoff_indicator_players_2022_2024.csv"
    meta_path = OUTPUT_DIR / "simple_rapm_with_playoff_indicator_run_2022_2024.json"
    lambda_path = OUTPUT_DIR / "simple_rapm_with_playoff_indicator_lambda_scores_2022_2024.csv"

    with open(raw_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["col", "player_side", "coef", "se", "poss"])
        for col, key in col_to_player.items():
            writer.writerow([col, key, float(beta[col]), float(se[col]), float(col_sums[col])])

    off_cols: dict[int, int] = {}
    def_cols: dict[int, int] = {}
    for col, key in col_to_player.items():
        if key in {"rubber_band", "home_effect", "playoff_indicator"} or key.startswith("season_"):
            continue
        player_id, side = key.split("_")
        if side == "off":
            off_cols[int(player_id)] = col
        else:
            def_cols[int(player_id)] = col

    rows = []
    z = 1.96
    for pid, off_col in off_cols.items():
        def_col = def_cols.get(pid)
        if def_col is None:
            continue
        off = float(beta[off_col]) * 100.0
        defense = float(beta[def_col]) * 100.0
        rapm = off - defense
        off_se = float(se[off_col]) * 100.0
        def_se = float(se[def_col]) * 100.0
        cov = cov_od.get(pid, 0.0) * (100.0**2)
        rapm_se = math.sqrt(max(0.0, off_se**2 + def_se**2 - 2.0 * cov))

        age_2025_mid = ages.get(pid, np.nan)
        off_age_adj = off + curve_offset(curve, age_2025_mid, "cumOff")
        def_age_adj = defense + curve_offset(curve, age_2025_mid, "cumDef")
        rapm_age_adj = rapm + curve_offset(curve, age_2025_mid, "cumRAPM")

        rows.append({
            "Name": names.get(pid, str(pid)),
            "PLAYER_ID": pid,
            "Poss_Off": int(col_sums[off_col]),
            "Poss_Def": int(col_sums[def_col]),
            "Age_2024_25_Mid": round(float(age_2025_mid), 1) if not np.isnan(age_2025_mid) else None,
            "Off": round(off, 3),
            "Def": round(defense, 3),
            "RAPM": round(rapm, 3),
            "Off_SE": round(off_se, 3),
            "Def_SE": round(def_se, 3),
            "RAPM_SE": round(rapm_se, 3),
            "Off_CI_lo": round(off - z * off_se, 3),
            "Off_CI_hi": round(off + z * off_se, 3),
            "Def_CI_lo": round(defense - z * def_se, 3),
            "Def_CI_hi": round(defense + z * def_se, 3),
            "RAPM_CI_lo": round(rapm - z * rapm_se, 3),
            "RAPM_CI_hi": round(rapm + z * rapm_se, 3),
            "Off_AgeAdj_To27": round(off_age_adj, 3),
            "Def_AgeAdj_To27": round(def_age_adj, 3),
            "RAPM_AgeAdj_To27": round(rapm_age_adj, 3),
        })

    players = pd.DataFrame(rows).sort_values("RAPM", ascending=False)
    players.to_csv(players_path, index=False)

    run_meta = {
        "seasons": SEASONS,
        "season_weights": SEASON_WEIGHTS,
        "includes_playoffs": True,
        "lambdas": lambdas,
        "intercept": intercept,
        "n_possessions": int(X.shape[0]),
        "n_columns": int(X.shape[1]),
        "y_mean": float(np.average(y, weights=sample_weights)),
        "aging_curve_used": str(RAPM_ROOT / "outputs" / "aging" / "aging_curve.csv") if curve is not None else None,
        "rubber_band": {
            **rubber_band_meta,
            "coefficient": float(beta[rubber_band_meta["rubber_band_col"]]),
            "coefficient_per_100": float(beta[rubber_band_meta["rubber_band_col"]] * 100.0),
        },
        "home_effect": {
            "coefficient": float(beta[rubber_band_meta["home_col"]]),
            "coefficient_per_100": float(beta[rubber_band_meta["home_col"]] * 100.0),
            "coding": "+1 home offense, -1 away offense",
        },
        "playoff_indicator": {
            "coefficient": float(beta[rubber_band_meta["playoff_col"]]),
            "coefficient_per_100": float(beta[rubber_band_meta["playoff_col"]] * 100.0),
            "coding": "1 playoff possession, 0 otherwise",
        },
        "season_effects": {
            key: {
                "coefficient": float(beta[col]),
                "coefficient_per_100": float(beta[col] * 100.0),
            }
            for col, key in col_to_player.items()
            if key.startswith("season_")
        },
        "outputs": {
            "raw": str(raw_path),
            "players": str(players_path),
            "lambda_scores": str(lambda_path),
        },
    }
    with open(meta_path, "w") as handle:
        json.dump(run_meta, handle, indent=2, sort_keys=True)

    print(players.head(20).to_string(index=False))
    lambda_scores.to_csv(lambda_path, index=False)
    return raw_path, players_path, meta_path, lambda_path


def main() -> None:
    ensure_dirs()
    start = time.time()
    print("Fetching 2022-2024 possessions, including playoffs...")
    data = fetch_possessions()
    print(f"Fetched {len(data):,} possessions")

    print("Building sparse player matrix plus rubber band column...")
    X, y, sample_weights, col_to_player, rubber_band_meta = build_matrix(data)
    y_av = float(np.average(y, weights=sample_weights))

    print("Running coordinate lambda search...")
    beta, lambdas, lambda_scores = run_ridge_model(X, y - y_av, sample_weights, col_to_player)

    print("Writing coefficients, confidence intervals, and age-adjusted output...")
    raw_path, players_path, meta_path, lambda_path = write_outputs(
        X,
        beta,
        lambdas,
        y_av,
        y,
        sample_weights,
        col_to_player,
        rubber_band_meta,
        lambda_scores,
    )

    print(f"Raw coefficients -> {raw_path}")
    print(f"Players -> {players_path}")
    print(f"Run metadata -> {meta_path}")
    print(f"Lambda scores -> {lambda_path}")
    print(f"Elapsed seconds: {time.time() - start:.1f}")


if __name__ == "__main__":
    main()
