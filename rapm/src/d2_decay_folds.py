#!/usr/bin/env python3
"""D1+D2: read the shape of the learned bucket weights, then confirm decay
variants on TWO chronological folds. A variant only wins if both folds agree.

Folds: train 2021-23 -> test 2024, train 2020-22 -> test 2023.
Variants: no decay, exp hl365 (champion), exp hl250, learned buckets (from D0),
exponential + power-law fits to the buckets.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from experiments import base_cfg, evaluate_on_next, fit
from paths import DIAGNOSTICS_DIR, ensure_dirs
from standard_rapm import build_design_matrix, fetch_possessions

ensure_dirs()

FOLDS = [
    ("2021-23->2024", [2021, 2022, 2023], 2024),
    ("2020-22->2023", [2020, 2021, 2022], 2023),
]


def exp_w(t, hl):
    return np.power(0.5, t / hl)


def pow_w(t, tau, alpha):
    return np.power(1.0 + t / tau, -alpha)


def main() -> None:
    d0 = json.loads((DIAGNOSTICS_DIR / "decay_buckets_best.json").read_text())
    mids = np.array(d0["bucket_mid_days"])
    wts = np.array(d0["best_weights"])

    # --- D1: fit families to the learned buckets (weight recent buckets more) ---
    sigma = 1.0 / np.sqrt(np.arange(len(wts), 0, -1))
    fits = {}
    try:
        (hl_fit,), _ = curve_fit(exp_w, mids, wts, p0=[365], sigma=sigma, maxfev=5000)
        r2 = 1 - np.sum((wts - exp_w(mids, hl_fit)) ** 2) / np.sum((wts - wts.mean()) ** 2)
        fits["exp"] = {"hl": float(hl_fit), "r2": float(r2)}
    except Exception as e:
        fits["exp"] = {"error": str(e)}
    try:
        (tau_fit, a_fit), _ = curve_fit(pow_w, mids, wts, p0=[300, 1.0], sigma=sigma, maxfev=5000)
        r2 = 1 - np.sum((wts - pow_w(mids, tau_fit, a_fit)) ** 2) / np.sum((wts - wts.mean()) ** 2)
        fits["powerlaw"] = {"tau": float(tau_fit), "alpha": float(a_fit), "r2": float(r2)}
    except Exception as e:
        fits["powerlaw"] = {"error": str(e)}
    print("D1 family fits:", json.dumps(fits, indent=2), flush=True)

    bucket_edges_frac = np.linspace(0, 1, len(wts) + 1)

    def variants(age_days: np.ndarray, span: float) -> dict[str, np.ndarray | None]:
        v = {
            "no_decay": None,
            "exp_hl365": exp_w(age_days, 365.0),
            "exp_hl250": exp_w(age_days, 250.0),
        }
        b = np.minimum((age_days / (span + 1e-9) * len(wts)).astype(int), len(wts) - 1)
        v["buckets_learned"] = wts[b]
        if "hl" in fits.get("exp", {}):
            v["exp_fitted"] = exp_w(age_days, fits["exp"]["hl"])
        if "tau" in fits.get("powerlaw", {}):
            v["pow_fitted"] = pow_w(age_days, fits["powerlaw"]["tau"], fits["powerlaw"]["alpha"])
        return v

    rows = []
    for fold_name, train_seasons, test_season in FOLDS:
        cfg = base_cfg(train_seasons)
        dm = build_design_matrix(fetch_possessions(train_seasons), cfg)
        next_dm = build_design_matrix(fetch_possessions([test_season]), base_cfg([test_season]))
        dates = pd.to_datetime(pd.Series(dm.row_dates)).values
        age_days = ((dates.max() - dates) / np.timedelta64(1, "D")).astype(float)
        span = age_days.max()
        for name, w_extra in variants(age_days, span).items():
            w = dm.weights if w_extra is None else dm.weights * w_extra
            beta, intercept = fit(dm, cfg, weights=w)
            kb = {k: float(beta[j]) for j, k in dm.col_to_key.items()}
            rmse, corr, _ = evaluate_on_next(next_dm, kb, intercept)
            rows.append({"fold": fold_name, "variant": name,
                         "corr": round(corr, 4), "rmse": round(rmse, 3)})
            print(f"{fold_name} {name}: corr={corr:.4f} rmse={rmse:.2f}", flush=True)

    df = pd.DataFrame(rows)
    piv = df.pivot(index="variant", columns="fold", values="corr")
    piv["mean_corr"] = piv.mean(axis=1)
    piv = piv.sort_values("mean_corr", ascending=False)
    out = DIAGNOSTICS_DIR / "decay_fold_confirmation.csv"
    df.to_csv(out, index=False)
    print("\n" + piv.to_string())
    print(f"D2_FOLDS_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
