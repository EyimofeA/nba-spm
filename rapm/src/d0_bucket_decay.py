#!/usr/bin/env python3
"""D0: model-free decay fit.

Split the train window (2021-23) into 6 equal time buckets, give each bucket a
free possession weight (most-recent bucket pinned to 1.0), and tune the other
five log-weights with Nelder-Mead to maximize 2024 game-margin correlation.
The learned weights ARE the empirical decay function; D1 reads its shape.

Every objective evaluation is appended to outputs/diagnostics/decay_bucket_evals.csv
(crash-safe). Warm start: weights implied by the hl=365 exponential champion.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments import NEXT_SEASON, TRAIN_SEASONS, base_cfg, evaluate_on_next, fit
from paths import DIAGNOSTICS_DIR, ensure_dirs
from standard_rapm import build_design_matrix, fetch_possessions

ensure_dirs()
EVALS_CSV = DIAGNOSTICS_DIR / "decay_bucket_evals.csv"
N_BUCKETS = 6


def main() -> None:
    train_rows = fetch_possessions(TRAIN_SEASONS)
    next_rows = fetch_possessions([NEXT_SEASON])
    cfg = base_cfg(TRAIN_SEASONS)
    dm = build_design_matrix(train_rows, cfg)
    next_dm = build_design_matrix(next_rows, base_cfg([NEXT_SEASON]))

    dates = pd.to_datetime(pd.Series(dm.row_dates)).values
    age_days = ((dates.max() - dates) / np.timedelta64(1, "D")).astype(float)
    span = age_days.max() + 1e-9
    bucket = np.minimum((age_days / span * N_BUCKETS).astype(int), N_BUCKETS - 1)
    # bucket 0 = most recent, pinned to weight 1.0
    mids = np.array([(b + 0.5) * span / N_BUCKETS for b in range(N_BUCKETS)])
    print(f"rows={len(age_days):,} span={span:.0f}d bucket_mid_days={mids.round(0).tolist()}", flush=True)

    n_eval = [0]
    best = {"corr": -np.inf}

    def objective(logw5: np.ndarray) -> float:
        w_buckets = np.concatenate([[1.0], np.exp(logw5)])
        w = dm.weights * w_buckets[bucket]
        t0 = time.time()
        beta, intercept = fit(dm, cfg, weights=w)
        key_beta = {k: float(beta[j]) for j, k in dm.col_to_key.items()}
        rmse, corr, _ = evaluate_on_next(next_dm, key_beta, intercept)
        n_eval[0] += 1
        rec = {"eval": n_eval[0], "weights": json.dumps(w_buckets.round(4).tolist()),
               "corr": round(corr, 4), "rmse": round(rmse, 3),
               "elapsed_s": round(time.time() - t0, 1), "ts": pd.Timestamp.now().isoformat()}
        pd.DataFrame([rec]).to_csv(EVALS_CSV, mode="a", header=not EVALS_CSV.exists(), index=False)
        if corr > best["corr"]:
            best.update({"corr": corr, "rmse": rmse, "weights": w_buckets.tolist()})
        print(f"EVAL {n_eval[0]}: w={w_buckets.round(3).tolist()} corr={corr:.4f} rmse={rmse:.2f} "
              f"[{rec['elapsed_s']}s]", flush=True)
        return -corr

    # warm start at hl=365 exponential
    x0 = np.log(np.power(0.5, mids[1:] / 365.0))
    res = minimize(objective, x0, method="Nelder-Mead",
                   options={"maxfev": 55, "xatol": 0.05, "fatol": 1e-4})

    out = {"best_weights": best["weights"], "best_corr": best["corr"], "best_rmse": best["rmse"],
           "bucket_mid_days": mids.tolist(), "n_evals": n_eval[0], "converged": bool(res.success)}
    (DIAGNOSTICS_DIR / "decay_buckets_best.json").write_text(json.dumps(out, indent=2))
    print("D0_BUCKETS_DONE", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
