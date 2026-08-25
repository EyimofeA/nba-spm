#!/usr/bin/env python3
"""Quick nonlinear probe: LightGBM on top of frozen linear RAPM ratings.

Question: does a nonlinear model (which can learn a proper, margin x time
"rubberband" instead of one linear column) predict next-season game margins
better than plain linear RAPM?

Design:
- Fit linear RAPM (player + home, GT-filtered) on the train window.
- Per possession features: lineup off-rating sum, lineup def-rating sum,
  home flag, live score margin, period.
- Train LightGBM on train-window possessions, predict 2024 possessions,
  aggregate to game margins, compare to the linear retrodiction.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from paths import STANDARD_RAPM_DIAGNOSTICS, ensure_dirs
from standard_rapm import build_design_matrix, fetch_possessions, format_season_list
from validate_ratings import fit_beta, make_cfg

ensure_dirs()


def possession_features(rows: list[tuple], cfg, key_beta: dict[str, float]):
    """Features straight from raw rows using the engine's own filtering."""
    dm = build_design_matrix(rows, cfg)
    n_cols = dm.X.shape[1]
    off_vec = np.zeros(n_cols)
    def_vec = np.zeros(n_cols)
    for j in range(n_cols):
        key = dm.col_to_key[j]
        if key.endswith("_off"):
            off_vec[j] = key_beta.get(key, 0.0)
        elif key.endswith("_def"):
            def_vec[j] = key_beta.get(key, 0.0)

    X = dm.X
    off_sum = X.maximum(0).dot(off_vec)          # +1 entries pick offense players
    def_sum = X.minimum(0).dot(def_vec) * -1.0   # -1 entries pick defenders
    # Live margin from the rubberband column if present, else zeros.
    rb_col = next((j for j, k in dm.col_to_key.items() if k == "META_rubberband"), None)
    margin = np.asarray(X[:, rb_col].todense()).ravel() if rb_col is not None else np.zeros(X.shape[0])
    home = dm.row_home_off.astype(float)
    feats = pd.DataFrame({
        "off_sum": off_sum,
        "def_sum": def_sum,
        "home": home,
        "margin": margin,
    })
    return feats, dm


def game_margin(dm, y_pred: np.ndarray) -> pd.DataFrame:
    sign = np.where(dm.row_home_off, 1.0, -1.0)
    df = pd.DataFrame({"gameid": dm.gameids, "t": dm.y * sign, "p": y_pred * sign})
    return df.groupby("gameid", observed=True).agg(t=("t", "sum"), p=("p", "sum"))


def score(g: pd.DataFrame) -> tuple[float, float]:
    rmse = float(np.sqrt(np.mean((g["t"] - g["p"]) ** 2)))
    corr = float(np.corrcoef(g["t"], g["p"])[0, 1])
    return rmse, corr


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", type=int, default=2021)
    parser.add_argument("--train-end", type=int, default=2023)
    parser.add_argument("--next-season", type=int, default=2024)
    parser.add_argument("--lambda-off", type=float, default=3000.0)
    parser.add_argument("--lambda-def", type=float, default=3000.0)
    parser.add_argument("--lambda-meta", type=float, default=300.0)
    parser.add_argument("--lambda-season", type=float, default=100.0)
    args = parser.parse_args(argv)

    import lightgbm as lgb

    train_seasons = list(range(args.train_start, args.train_end + 1))
    train_rows = fetch_possessions(train_seasons)
    next_rows = fetch_possessions([args.next_season])

    # 1) Frozen linear ratings from the winning spec.
    t0 = time.time()
    lin_cfg = make_cfg(train_seasons, "home", args)
    lin_dm = build_design_matrix(train_rows, lin_cfg)
    beta, intercept = fit_beta(lin_dm, lin_cfg)
    key_beta = {k: float(beta[j]) for j, k in lin_dm.col_to_key.items()}
    print(f"linear RAPM fit: {time.time()-t0:.0f}s", flush=True)

    # 2) Features (need rubberband column for the live margin -> use rb spec for feature dm).
    feat_cfg = make_cfg(train_seasons, "home_rubberband", args)
    tr_feats, tr_dm = possession_features(train_rows, feat_cfg, key_beta)
    nx_cfg = make_cfg([args.next_season], "home_rubberband", args)
    nx_feats, nx_dm = possession_features(next_rows, nx_cfg, key_beta)

    # 3) Linear baseline on the same 2024 games (map frozen betas).
    lin_nx_cfg = make_cfg([args.next_season], "home", args)
    lin_nx_dm = build_design_matrix(next_rows, lin_nx_cfg)
    mapped = np.array([key_beta.get(lin_nx_dm.col_to_key[j], 0.0) for j in range(lin_nx_dm.X.shape[1])])
    lin_pred = lin_nx_dm.X @ mapped + intercept
    lin_rmse, lin_corr = score(game_margin(lin_nx_dm, lin_pred))
    print(f"LINEAR baseline: margin rmse={lin_rmse:.2f} corr={lin_corr:.3f}", flush=True)

    # 4) LightGBM.
    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.9,
        random_state=7, verbose=-1,
    )
    model.fit(tr_feats, tr_dm.y)
    gbm_pred = model.predict(nx_feats)
    gbm_rmse, gbm_corr = score(game_margin(nx_dm, gbm_pred))
    print(f"LIGHTGBM (with margin/home feats): margin rmse={gbm_rmse:.2f} corr={gbm_corr:.3f} "
          f"[fit {time.time()-t0:.0f}s]", flush=True)

    # 5) LightGBM WITHOUT the live margin (fair forward-prediction variant).
    cols = ["off_sum", "def_sum", "home"]
    model2 = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=200, subsample=0.8, colsample_bytree=0.9,
        random_state=7, verbose=-1,
    )
    model2.fit(tr_feats[cols], tr_dm.y)
    gbm2_pred = model2.predict(nx_feats[cols])
    g2_rmse, g2_corr = score(game_margin(nx_dm, gbm2_pred))
    print(f"LIGHTGBM (no live margin): margin rmse={g2_rmse:.2f} corr={g2_corr:.3f}", flush=True)

    imp = pd.Series(model.feature_importances_, index=tr_feats.columns).sort_values(ascending=False)
    print("\nfeature importances (with-margin model):")
    print(imp.to_string())

    out = pd.DataFrame([
        {"model": "linear_rapm", "margin_rmse": lin_rmse, "margin_corr": lin_corr},
        {"model": "lgbm_with_margin", "margin_rmse": gbm_rmse, "margin_corr": gbm_corr},
        {"model": "lgbm_no_margin", "margin_rmse": g2_rmse, "margin_corr": g2_corr},
    ])
    path = STANDARD_RAPM_DIAGNOSTICS / f"nonlinear_probe_{format_season_list(train_seasons)}_next{args.next_season}.csv"
    out.to_csv(path, index=False)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
