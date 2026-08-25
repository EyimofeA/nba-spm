#!/usr/bin/env python3
"""Step 3: full SPM prior (box + tracking features) under clean semantics.

Stage 1: for each fold's train window, regress pass-1 player coefficients
(off/def separately) on the window's feature vector Z (from
data/spm_features_windows.parquet), weighted ridge, OOF by player.
Report OOF R^2 per side. tau^2 from OOF residuals -> per-player prior pull.

Stage 2: pass-2 refit (champion penalty + c*sigma^2/tau^2 pull to SPM center),
c grid around the minutes-prior optimum. Gate: both folds + anchors.
Comparisons logged: champion, minutes-prior best, SPM prior.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from experiments import (
    anchor_check,
    append_result,
    base_cfg,
    decay_weights,
    done_names,
    evaluate_on_next,
)
from paths import DATA, ensure_dirs
from standard_rapm import (
    build_design_matrix,
    fetch_possessions,
    lambda_vector,
    predict,
    solve_penalized_ridge,
)

ensure_dirs()
HL = 250.0
FOLDS = [
    ("f24", [2021, 2022, 2023], 2024),
    ("f23", [2020, 2021, 2022], 2023),
]
# OnOffRtg/OnDefRtg are computed from the same possessions as the RAPM labels:
# they share the label's noise (leakage through the feature, not the CV), which
# inflated v1's OOF R2 while poisoning the prior. Excluded as of v1.1.
EXCLUDE = {"PLAYER_ID", "Window_End", "MIN", "GP", "OnOffRtg", "OnDefRtg"}


def champion_fit(dm, cfg, w):
    base = lambda_vector(dm, cfg.lambda_profile)
    return solve_penalized_ridge(dm.X, dm.y, w, base, np.zeros_like(base), np.zeros(dm.X.shape[1]))


def spm_prior(dm, beta, feats: pd.DataFrame, n_folds=5, seed=7, alpha=10.0):
    """OOF weighted-ridge SPM per side. Returns target_vec, lam0_unit, tau2, r2 dict."""
    col_sums = np.asarray(np.abs(dm.X).sum(axis=0)).ravel()
    pids, cols, sides = [], [], []
    for j, k in dm.col_to_key.items():
        if k.endswith("_off") or k.endswith("_def"):
            pids.append(int(k.split("_")[0]))
            cols.append(j)
            sides.append(k.endswith("_off"))
    info = pd.DataFrame({"pid": pids, "col": cols, "is_off": sides})
    info["y"] = beta[info["col"].values]
    info["w"] = col_sums[info["col"].values]

    feats = feats.copy()
    # log-poss shape features: the minutes-prior lesson — linear poss can't
    # represent the coach-trust curve, log + quadratic can.
    for pc in ("OffPoss", "DefPoss"):
        if pc in feats.columns:
            feats[f"log_{pc}"] = np.log1p(feats[pc])
            feats[f"log_{pc}_sq"] = feats[f"log_{pc}"] ** 2
    fcols = [c for c in feats.columns if c not in EXCLUDE and feats[c].dtype != object]
    f = feats.set_index("PLAYER_ID")[fcols]
    # standardize + median-impute
    f = (f - f.mean()) / f.std().replace(0, 1)
    f = f.fillna(0.0)
    info = info.merge(f, left_on="pid", right_index=True, how="left")
    have = info[fcols].notna().all(axis=1)
    info.loc[~have, fcols] = 0.0

    rng = np.random.default_rng(seed)
    pid_list = info["pid"].unique()
    fold_of_pid = dict(zip(pid_list, rng.integers(0, n_folds, len(pid_list))))
    info["fold"] = info["pid"].map(fold_of_pid)

    pred = np.zeros(len(info))
    r2 = {}
    for side in (True, False):
        s = info["is_off"] == side
        for k in range(n_folds):
            tr = s & (info["fold"] != k)
            te = s & (info["fold"] == k)
            X = info.loc[tr, fcols].values
            yv = info.loc[tr, "y"].values
            wv = info.loc[tr, "w"].values
            Xb = np.column_stack([np.ones(tr.sum()), X])
            Ws = np.sqrt(wv)
            A = (Xb * Ws[:, None]).T @ (Xb * Ws[:, None]) + alpha * np.eye(Xb.shape[1])
            b = (Xb * Ws[:, None]).T @ (yv * Ws)
            coef = np.linalg.solve(A, b)
            Xt = np.column_stack([np.ones(te.sum()), info.loc[te, fcols].values])
            pred[te.values] = Xt @ coef
        yv, wv, pv = info.loc[s, "y"], info.loc[s, "w"], pred[s.values]
        ss_res = float(np.average((yv - pv) ** 2, weights=wv))
        ss_tot = float(np.average((yv - np.average(yv, weights=wv)) ** 2, weights=wv))
        r2["off" if side else "def"] = 1 - ss_res / ss_tot

    resid = info["y"].values - pred
    # per-side tau^2: the defense SPM is far weaker than offense; each side's
    # pull strength must reflect its own measured reliability.
    target_vec = np.zeros(dm.X.shape[1])
    lam0_unit = np.zeros(dm.X.shape[1])
    tau2_side = {}
    for side in (True, False):
        s = (info["is_off"] == side).values
        tau2_side["off" if side else "def"] = float(
            np.average(resid[s] ** 2, weights=info["w"].values[s]))
        target_vec[info["col"].values[s]] = pred[s]
        lam0_unit[info["col"].values[s]] = 1.0 / tau2_side["off" if side else "def"]
    tau2 = float(np.average(resid ** 2, weights=info["w"].values))
    return target_vec, lam0_unit, tau2, {**r2, **{f"tau2_{k}": v for k, v in tau2_side.items()}}


def main() -> None:
    c_grid = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1.0, 2.0, 4.0]
    feats_all = pd.read_parquet(DATA / "spm_features_windows.parquet")
    already = done_names()

    for tag, train_seasons, test_season in FOLDS:
        cfg = base_cfg(train_seasons)
        dm = build_design_matrix(fetch_possessions(train_seasons), cfg)
        w = dm.weights * decay_weights(dm, HL)
        next_dm = build_design_matrix(fetch_possessions([test_season]), base_cfg([test_season]))

        beta1, ic1 = champion_fit(dm, cfg, w)
        resid = dm.y - predict(dm.X, beta1, ic1)
        sigma2 = float(np.average(resid ** 2, weights=w))

        feats = feats_all[feats_all["Window_End"] == train_seasons[-1]]
        target, lam0_unit, tau2, r2 = spm_prior(dm, beta1, feats)
        print(f"[{tag}] SPM: OOF R2 off={r2['off']:.3f} def={r2['def']:.3f} "
              f"tau={np.sqrt(tau2)*100:.2f}/100 implied lam0={sigma2/tau2:.0f}", flush=True)

        base = lambda_vector(dm, cfg.lambda_profile)
        for c in c_grid:
            name = f"spmprior12_c{c}_{tag}"
            if name in already:
                print(f"SKIP {name}", flush=True)
                continue
            t0 = time.time()
            beta2, ic2 = solve_penalized_ridge(dm.X, dm.y, w, base, c * sigma2 * lam0_unit, target)
            kb = {k: float(beta2[j]) for j, k in dm.col_to_key.items()}
            rmse, corr, _ = evaluate_on_next(next_dm, kb, ic2)
            ok, note = anchor_check(dm, beta2)
            append_result({"name": name,
                           "params": json.dumps({"c": c, "tau2": tau2, "r2_off": r2["off"], "r2_def": r2["def"]}),
                           "margin_rmse": round(rmse, 3), "margin_corr": round(corr, 4), "n_games": 0,
                           "anchors_ok": ok, "anchor_note": note, "ess": None, "n_train_rows": dm.X.shape[0],
                           "elapsed_s": round(time.time() - t0, 1), "ts": pd.Timestamp.now().isoformat()})
            print(f"EXPERIMENT_DONE {name}: corr={corr:.4f} rmse={rmse:.2f} anchors={ok}", flush=True)

    print("SPMV1_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
