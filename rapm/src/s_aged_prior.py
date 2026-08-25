#!/usr/bin/env python3
"""S: age-translated previous-window prior.

Last night the raw 2018-20 prior LOST (0.589-0.613 vs 0.632 baseline) because
it's stale. Fix under test: before using an old rating as a shrinkage target,
translate it through the aging curve (off/def separately) for the years elapsed.

Variants (train 2021-23, decay hl250 weights, test 2024):
  zero prior (new champion)      -- the bar
  raw 2018-20 prior, strength 2  -- best stale-prior result from last night
  aged 2018-20 prior, strengths {1, 2, 4}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments import base_cfg, decay_weights, evaluate_on_next, fit
from paths import AGING_DIR, CAREER_RAPM_CSV, DIAGNOSTICS_DIR, ensure_dirs
from standard_rapm import build_design_matrix, coefficients_as_prior, fetch_possessions

ensure_dirs()

PREV = [2018, 2019, 2020]
TRAIN = [2021, 2022, 2023]
TEST = 2024
GAP_YEARS = 3  # midpoint of PREV to midpoint of TRAIN


def aging_translation() -> tuple[dict[int, float], dict[int, float]]:
    curve = pd.read_csv(AGING_DIR / "aging_curve_delta.csv")
    f_off = dict(zip(curve["Age"], curve["f_off"]))
    f_def = dict(zip(curve["Age"], curve["f_def"]))

    def delta(f: dict, age0: int) -> float:
        a1 = min(age0 + GAP_YEARS, max(f))
        a0 = max(min(age0, max(f)), min(f))
        return f.get(a1, f[max(f)]) - f.get(a0, 0.0)

    ages = pd.read_csv(CAREER_RAPM_CSV)
    ages = ages[ages["Season"] == 2019].dropna(subset=["Age"])
    age_of = dict(zip(ages["PLAYER_ID"].astype(int), ages["Age"].round().astype(int)))
    d_off = {pid: delta(f_off, a) for pid, a in age_of.items()}
    d_def = {pid: delta(f_def, a) for pid, a in age_of.items()}
    return d_off, d_def


def aged(prior: dict[str, float], d_off: dict[int, float], d_def: dict[int, float]) -> dict[str, float]:
    out = {}
    for key, val in prior.items():
        pid_s, side = key.rsplit("_", 1)
        pid = int(pid_s)
        if side == "off":
            out[key] = val + d_off.get(pid, 0.0) / 100.0
        else:
            # Def column: positive coef = bad defense; aging worsening defense adds positively
            out[key] = val + d_def.get(pid, 0.0) / 100.0
    return out


def main() -> None:
    prev_cfg = base_cfg(PREV)
    prev_dm = build_design_matrix(fetch_possessions(PREV), prev_cfg)
    prev_beta, _ = fit(prev_dm, prev_cfg)
    raw_prior = coefficients_as_prior(prev_dm, prev_beta)
    d_off, d_def = aging_translation()
    aged_prior = aged(raw_prior, d_off, d_def)
    covered = sum(1 for k in raw_prior if int(k.rsplit('_', 1)[0]) in d_off)
    print(f"prior players: {len(raw_prior)//2}, age-covered: {covered//2}", flush=True)

    train_rows = fetch_possessions(TRAIN)
    next_dm = build_design_matrix(fetch_possessions([TEST]), base_cfg([TEST]))

    results = []

    def run(name, prior_mode, prior, strength):
        cfg = base_cfg(TRAIN, lam_off=3000 * strength, lam_def=3000 * strength, prior_mode=prior_mode)
        dm = build_design_matrix(train_rows, cfg)
        w = dm.weights * decay_weights(dm, 250.0)
        beta, intercept = fit(dm, cfg, prior=prior, weights=w)
        kb = {k: float(beta[j]) for j, k in dm.col_to_key.items()}
        rmse, corr, _ = evaluate_on_next(next_dm, kb, intercept)
        results.append({"name": name, "corr": round(corr, 4), "rmse": round(rmse, 3)})
        print(f"{name}: corr={corr:.4f} rmse={rmse:.2f}", flush=True)

    run("zero_prior_hl250", "zero", None, 1.0)
    run("raw_prior_s2_hl250", "target", raw_prior, 2.0)
    for s in (1.0, 2.0, 4.0):
        run(f"aged_prior_s{s}_hl250", "target", aged_prior, s)

    df = pd.DataFrame(results)
    out = DIAGNOSTICS_DIR / "aged_prior_results.csv"
    df.to_csv(out, index=False)
    print("\n" + df.sort_values("corr", ascending=False).to_string(index=False))
    print(f"S_AGED_PRIOR_DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
