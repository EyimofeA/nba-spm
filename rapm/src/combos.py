#!/usr/bin/env python3
"""Combination round: merge the winning knobs from experiments.py and verify
the combo beats each ingredient before it becomes the final config."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from experiments import (
    NEXT_SEASON,
    RESULTS_CSV,
    TRAIN_SEASONS,
    anchor_check,
    append_result,
    base_cfg,
    decay_weights,
    done_names,
    ess,
    evaluate_on_next,
    fit,
)
from standard_rapm import build_design_matrix, fetch_possessions

COMBOS = [
    ("combo_decay365_asym2000_4500", {"hl": 365, "lam_off": 2000, "lam_def": 4500}),
    ("combo_decay365_asym1500_6000", {"hl": 365, "lam_off": 1500, "lam_def": 6000}),
    ("decay_hl250", {"hl": 250, "lam_off": 3000, "lam_def": 3000}),
    ("decay_hl500", {"hl": 500, "lam_off": 3000, "lam_def": 3000}),
    ("combo_decay500_asym2000_4500", {"hl": 500, "lam_off": 2000, "lam_def": 4500}),
]


def main() -> None:
    train_rows = fetch_possessions(TRAIN_SEASONS)
    next_rows = fetch_possessions([NEXT_SEASON])
    next_dm = build_design_matrix(next_rows, base_cfg([NEXT_SEASON]))
    already = done_names()

    for name, p in COMBOS:
        if name in already:
            print(f"SKIP {name}", flush=True)
            continue
        t0 = time.time()
        cfg = base_cfg(TRAIN_SEASONS, lam_off=p["lam_off"], lam_def=p["lam_def"])
        dm = build_design_matrix(train_rows, cfg)
        w = dm.weights * decay_weights(dm, p["hl"])
        beta, intercept = fit(dm, cfg, weights=w)
        key_beta = {k: float(beta[j]) for j, k in dm.col_to_key.items()}
        anchors_ok, note = anchor_check(dm, beta)
        rmse, corr, n_games = evaluate_on_next(next_dm, key_beta, intercept)
        append_result({
            "name": name, "params": json.dumps(p), "margin_rmse": round(rmse, 3),
            "margin_corr": round(corr, 4), "n_games": n_games, "anchors_ok": anchors_ok,
            "anchor_note": note, "ess": round(ess(w), 0), "n_train_rows": dm.X.shape[0],
            "elapsed_s": round(time.time() - t0, 1), "ts": pd.Timestamp.now().isoformat(),
        })
        print(f"EXPERIMENT_DONE {name}: corr={corr:.4f} rmse={rmse:.2f} anchors={anchors_ok} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    print("COMBOS_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
