#!/usr/bin/env python3
"""Train 2022-2024 simple RAPM and retrodict the next season."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
RAPM_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import simple_weighted_rapm as simple_rapm  # noqa: E402
from simple_weighted_rapm import (  # noqa: E402
    FIXED_LAMBDAS,
    OUTPUT_DIR,
    PLAYOFF_OVERRIDES,
    RUBBER_BAND_CLIP,
    build_matrix,
    connect_db,
    fit_block_ridge,
    is_playoff_game,
    rubber_band_values,
    weighted_rmse,
)


TRAIN_SEASONS = [2021, 2022, 2023]
TEST_SEASON = 2024


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_possessions_for_seasons(seasons: list[int]) -> list[tuple]:
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


def build_matrix_for_existing_columns(
    data: list[tuple],
    col_to_player: dict[int, str],
    train_meta: dict,
) -> tuple[csr_matrix, np.ndarray, np.ndarray, dict]:
    key_to_col = {key: col for col, key in col_to_player.items()}
    X = lil_matrix((len(data), len(col_to_player)), dtype=np.float64)
    y = np.zeros(len(data), dtype=np.float64)
    weights = np.ones(len(data), dtype=np.float64)

    rubber_raw = rubber_band_values(data)
    rubber_z = (rubber_raw - float(train_meta["rubber_band_mean"])) / float(train_meta["rubber_band_std"])

    unseen_player_entries = 0
    playoff_possessions = 0
    for counter, item in enumerate(data):
        home_poss = bool(item[0])
        pts = float(item[1])
        season = int(item[12])
        game_date = item[13]
        away_list = [int(item[i]) for i in range(2, 7)]
        home_list = [int(item[i]) for i in range(7, 12)]
        off_list, def_list = (home_list, away_list) if home_poss else (away_list, home_list)

        for player in off_list:
            col = key_to_col.get(f"{player}_off")
            if col is None:
                unseen_player_entries += 1
            else:
                X[counter, col] = 1.0
        for player in def_list:
            col = key_to_col.get(f"{player}_def")
            if col is None:
                unseen_player_entries += 1
            else:
                X[counter, col] = 1.0

        X[counter, key_to_col["home_effect"]] = 1.0 if home_poss else -1.0
        if is_playoff_game(season, game_date):
            X[counter, key_to_col["playoff_indicator"]] = 1.0
            playoff_possessions += 1
        X[counter, key_to_col["rubber_band"]] = rubber_z[counter]

        season_col = key_to_col.get(f"season_{season}")
        if season_col is not None:
            X[counter, season_col] = 1.0

        y[counter] = pts

    meta = {
        "unseen_player_entries": unseen_player_entries,
        "playoff_possessions": playoff_possessions,
        "rubber_band_clip": RUBBER_BAND_CLIP,
        "rubber_band_mean_from_train": float(train_meta["rubber_band_mean"]),
        "rubber_band_std_from_train": float(train_meta["rubber_band_std"]),
    }
    return X.tocsr(), y, weights, meta


def mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.abs(y_true - y_pred), weights=weights))


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"actual": y_true, "pred": y_pred, "weight": weights})
    df["pred_bin"] = pd.qcut(df["pred"], q=10, duplicates="drop")
    return (
        df.groupby("pred_bin", observed=True)
        .apply(
            lambda g: pd.Series({
                "n": len(g),
                "pred_mean": np.average(g["pred"], weights=g["weight"]),
                "actual_mean": np.average(g["actual"], weights=g["weight"]),
            }),
            include_groups=False,
        )
        .reset_index()
    )


def main() -> None:
    ensure_dirs()
    start = time.time()
    simple_rapm.SEASONS = TRAIN_SEASONS
    simple_rapm.SEASON_WEIGHTS = {season: 1.0 for season in TRAIN_SEASONS}
    train_rows = fetch_possessions_for_seasons(TRAIN_SEASONS)
    test_rows = fetch_possessions_for_seasons([TEST_SEASON])
    if not test_rows:
        raise RuntimeError(f"No possessions found for test season {TEST_SEASON}")

    X_train, y_train, w_train, col_to_player, train_meta = build_matrix(train_rows)
    y_mean = float(np.average(y_train, weights=w_train))
    beta = fit_block_ridge(X_train, y_train - y_mean, w_train, col_to_player, dict(FIXED_LAMBDAS))

    X_test, y_test, w_test, test_meta = build_matrix_for_existing_columns(test_rows, col_to_player, train_meta)
    y_pred = np.asarray(X_test @ beta).ravel() + y_mean

    rmse_value = weighted_rmse(y_test, y_pred, w_test)
    mae_value = mae(y_test, y_pred, w_test)
    baseline_pred = np.full_like(y_test, y_mean)
    baseline_rmse = weighted_rmse(y_test, baseline_pred, w_test)
    baseline_mae = mae(y_test, baseline_pred, w_test)

    pred_path = OUTPUT_DIR / "simple_rapm_retrodict_train_2021_2023_test_2024_predictions.csv"
    summary_path = OUTPUT_DIR / "simple_rapm_retrodict_train_2021_2023_test_2024_summary.json"
    calibration_path = OUTPUT_DIR / "simple_rapm_retrodict_train_2021_2023_test_2024_calibration.csv"

    pd.DataFrame({
        "actual_pts": y_test,
        "pred_pts": y_pred,
        "residual": y_test - y_pred,
    }).to_csv(pred_path, index=False)
    calibration_table(y_test, y_pred, w_test).to_csv(calibration_path, index=False)

    summary = {
        "train_seasons": TRAIN_SEASONS,
        "test_season": TEST_SEASON,
        "lambdas": FIXED_LAMBDAS,
        "train_possessions": int(X_train.shape[0]),
        "test_possessions": int(X_test.shape[0]),
        "test_rmse": rmse_value,
        "test_mae": mae_value,
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "rmse_improvement_vs_mean": baseline_rmse - rmse_value,
        "mae_improvement_vs_mean": baseline_mae - mae_value,
        "test_meta": test_meta,
        "playoff_overrides": {str(k): [str(v[0]), str(v[1])] for k, v in PLAYOFF_OVERRIDES.items()},
        "elapsed_seconds": time.time() - start,
        "outputs": {
            "predictions": str(pred_path),
            "calibration": str(calibration_path),
        },
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Predictions -> {pred_path}")
    print(f"Calibration -> {calibration_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
