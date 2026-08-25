#!/usr/bin/env python3
"""Step 1: minutes-only prior (the one-feature SPM) under CLEAN prior semantics.

Stage 1: pass-1 RAPM (champion config) gives labels beta_hat_j. Regress them on
log(possessions) out-of-fold across players -> prior center beta_0_j, and
measure tau^2 = weighted OOF residual variance (= how wrong this prior is).

Stage 2: refit with the champion zero-penalty UNCHANGED plus a separate
per-player pull lambda_0 = c * sigma^2/tau^2 toward beta_0. c in {0.5, 1, 2}.
Controls: c=0 (champion itself) and the raw 2018-20 stale prior at c=1 under
the same clean semantics (re-baseline of the overnight result).

Gate: both folds (2021-23 -> 2024, 2020-22 -> 2023). Everything appends to
outputs/diagnostics/experiments.csv (fold tagged in the name).
"""
from __future__ import annotations

import json
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
from paths import DIAGNOSTICS_DIR, ensure_dirs
from standard_rapm import (
    build_design_matrix,
    coefficients_as_prior,
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


def champion_fit(dm, cfg, w):
    """Zero-prior champion: base lambda toward zero, no target term."""
    base = lambda_vector(dm, cfg.lambda_profile)
    zeros_target = np.zeros(dm.X.shape[1])
    return solve_penalized_ridge(dm.X, dm.y, w, base, np.zeros_like(base), zeros_target)


def prior_fit(dm, cfg, w, lam0_vec, target_vec):
    """Clean semantics: champion zero-penalty UNCHANGED + separate pull to target."""
    base = lambda_vector(dm, cfg.lambda_profile)
    return solve_penalized_ridge(dm.X, dm.y, w, base, lam0_vec, target_vec)


def minutes_prior(dm, beta, n_folds=5, seed=7):
    """OOF regression of pass-1 player coefficients on log(decayed possessions).
    Returns (target_vec, lam0_unit_vec, tau2, oof_r2). lam0_unit = sigma2/tau2
    for player cols, 0 for meta cols (they keep only the base penalty)."""
    col_sums = np.asarray(np.abs(dm.X).sum(axis=0)).ravel()
    player_cols = np.array([j for j, k in dm.col_to_key.items()
                            if k.endswith("_off") or k.endswith("_def")])
    is_off = np.array([dm.col_to_key[j].endswith("_off") for j in player_cols])
    x = np.log1p(col_sums[player_cols])
    y = beta[player_cols]
    w = col_sums[player_cols]

    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, n_folds, len(player_cols))
    pred = np.zeros_like(y)
    for f in range(n_folds):
        tr = fold_of != f
        for side in (True, False):  # off and def separately
            m = tr & (is_off == side)
            t = (fold_of == f) & (is_off == side)
            if m.sum() < 10 or t.sum() == 0:
                continue
            X1 = np.column_stack([np.ones(m.sum()), x[m], x[m] ** 2])
            Ws = np.sqrt(w[m])
            coef, *_ = np.linalg.lstsq(X1 * Ws[:, None], y[m] * Ws, rcond=None)
            Xt = np.column_stack([np.ones(t.sum()), x[t], x[t] ** 2])
            pred[t] = Xt @ coef

    resid = y - pred
    tau2 = float(np.average(resid ** 2, weights=w))
    ss_tot = float(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
    oof_r2 = 1.0 - tau2 / ss_tot

    target_vec = np.zeros(dm.X.shape[1])
    lam0_unit = np.zeros(dm.X.shape[1])
    target_vec[player_cols] = pred
    lam0_unit[player_cols] = 1.0 / tau2  # sigma2 multiplied in later
    return target_vec, lam0_unit, tau2, oof_r2


def stale_prior_vectors(dm, prev_prior: dict[str, float], tau2: float):
    """Raw 2018-20 prior mapped onto this window's columns, same lam0 scale."""
    target_vec = np.zeros(dm.X.shape[1])
    lam0_unit = np.zeros(dm.X.shape[1])
    for j, k in dm.col_to_key.items():
        if k.endswith("_off") or k.endswith("_def"):
            if k in prev_prior:
                target_vec[j] = prev_prior[k]
                lam0_unit[j] = 1.0 / tau2
    return target_vec, lam0_unit


def main() -> None:
    import sys
    c_grid = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.5, 1.0, 2.0]
    already = done_names()
    for tag, train_seasons, test_season in FOLDS:
        cfg = base_cfg(train_seasons)
        dm = build_design_matrix(fetch_possessions(train_seasons), cfg)
        w = dm.weights * decay_weights(dm, HL)
        next_dm = build_design_matrix(fetch_possessions([test_season]), base_cfg([test_season]))

        t0 = time.time()
        beta1, ic1 = champion_fit(dm, cfg, w)
        resid = dm.y - predict(dm.X, beta1, ic1)
        sigma2 = float(np.average(resid ** 2, weights=w))
        print(f"[{tag}] pass1 fit {time.time()-t0:.0f}s sigma2={sigma2:.3f}", flush=True)

        kb1 = {k: float(beta1[j]) for j, k in dm.col_to_key.items()}
        rmse, corr, _ = evaluate_on_next(next_dm, kb1, ic1)
        print(f"[{tag}] champion: corr={corr:.4f} rmse={rmse:.2f}", flush=True)

        target, lam0_unit, tau2, oof_r2 = minutes_prior(dm, beta1)
        print(f"[{tag}] minutes prior: tau={np.sqrt(tau2)*100:.2f}/100 OOF_R2={oof_r2:.3f} "
              f"implied lam0={sigma2/tau2:.0f}", flush=True)

        for c in c_grid:
            name = f"mprior_c{c}_{tag}"
            if name in already:
                print(f"SKIP {name}", flush=True)
                continue
            t0 = time.time()
            beta2, ic2 = prior_fit(dm, cfg, w, c * sigma2 * lam0_unit, target)
            kb2 = {k: float(beta2[j]) for j, k in dm.col_to_key.items()}
            rmse, corr, _ = evaluate_on_next(next_dm, kb2, ic2)
            ok, note = anchor_check(dm, beta2)
            append_result({"name": name, "params": json.dumps({"c": c, "tau2": tau2, "oof_r2": oof_r2,
                                                               "lam0": c * sigma2 / tau2}),
                           "margin_rmse": round(rmse, 3), "margin_corr": round(corr, 4), "n_games": 0,
                           "anchors_ok": ok, "anchor_note": note, "ess": None, "n_train_rows": dm.X.shape[0],
                           "elapsed_s": round(time.time() - t0, 1), "ts": pd.Timestamp.now().isoformat()})
            print(f"EXPERIMENT_DONE {name}: corr={corr:.4f} rmse={rmse:.2f} anchors={ok}", flush=True)

        # stale-prior control (clean semantics re-baseline), fold 1 only
        if tag == "f24" and len(sys.argv) <= 1:
            prev_seasons = [s - 3 for s in train_seasons]
            pcfg = base_cfg(prev_seasons)
            pdm = build_design_matrix(fetch_possessions(prev_seasons), pcfg)
            pw = pdm.weights * decay_weights(pdm, HL)
            pbeta, _ = champion_fit(pdm, pcfg, pw)
            prev_prior = coefficients_as_prior(pdm, pbeta)
            st_target, st_lam0 = stale_prior_vectors(dm, prev_prior, tau2)
            name = f"staleprior_clean_c1_{tag}"
            if name not in already:
                t0 = time.time()
                beta3, ic3 = prior_fit(dm, cfg, w, sigma2 * st_lam0, st_target)
                kb3 = {k: float(beta3[j]) for j, k in dm.col_to_key.items()}
                rmse, corr, _ = evaluate_on_next(next_dm, kb3, ic3)
                ok, note = anchor_check(dm, beta3)
                append_result({"name": name, "params": json.dumps({"c": 1.0, "prior": "2018-20-clean"}),
                               "margin_rmse": round(rmse, 3), "margin_corr": round(corr, 4), "n_games": 0,
                               "anchors_ok": ok, "anchor_note": note, "ess": None,
                               "n_train_rows": dm.X.shape[0], "elapsed_s": round(time.time() - t0, 1),
                               "ts": pd.Timestamp.now().isoformat()})
                print(f"EXPERIMENT_DONE {name}: corr={corr:.4f} rmse={rmse:.2f} anchors={ok}", flush=True)

    print("MPRIOR_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
