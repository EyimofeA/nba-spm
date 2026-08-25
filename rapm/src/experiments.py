#!/usr/bin/env python3
"""Overnight experiment queue for regular-season RAPM.

One process, sequential experiments, each appends a row to
outputs/diagnostics/experiments.csv the moment it finishes (crash-safe:
rerunning skips names already present). Gate metric: next-season (2024)
game-margin corr/RMSE from ratings frozen on the train window (2021-23).

Includes: Gobert/Jokic sign anchors, ESS logging on weighted fits,
walk-forward-only priors. No playoffs, no SEs (ideation mode).
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from paths import DIAGNOSTICS_DIR, ensure_dirs
from standard_rapm import (
    LambdaProfile,
    RunConfig,
    build_design_matrix,
    coefficients_as_prior,
    fetch_possessions,
    name_lookup,
    penalty_vectors,
    solve_penalized_ridge,
)

ensure_dirs()
RESULTS_CSV = DIAGNOSTICS_DIR / "experiments.csv"
POOL_SENTINEL = 9_999_999

TRAIN_SEASONS = [2021, 2022, 2023]
NEXT_SEASON = 2024


# ---------- config helpers ----------

def base_cfg(seasons, *, garbage_time=True, lam_off=3000.0, lam_def=3000.0, prior_mode="zero") -> RunConfig:
    return RunConfig(
        seasons=list(seasons),
        season_type="regular",
        spec="exp",
        prior_mode=prior_mode,
        include_home=True,
        include_rubberband=False,
        include_season_effects=False,
        garbage_time=garbage_time,
        optimize_lambdas=False,
        compute_intervals=False,
        lambda_profile=LambdaProfile(off=lam_off, defense=lam_def, meta=300.0, season=100.0),
    )


def fit(dm, cfg, prior=None, weights=None):
    zero_p, target_p, target = penalty_vectors(dm, cfg, cfg.lambda_profile, prior)
    w = dm.weights if weights is None else weights
    return solve_penalized_ridge(dm.X, dm.y, w, zero_p, target_p, target)


# ---------- evaluation ----------

def margin_metrics(dm, y_pred, mask=None):
    idx = np.arange(dm.X.shape[0]) if mask is None else np.where(mask)[0]
    sign = np.where(dm.row_home_off[idx], 1.0, -1.0)
    df = pd.DataFrame({"g": dm.gameids[idx], "t": dm.y[idx] * sign, "p": y_pred[idx] * sign})
    g = df.groupby("g", observed=True).agg(t=("t", "sum"), p=("p", "sum"))
    rmse = float(np.sqrt(np.mean((g["t"] - g["p"]) ** 2)))
    corr = float(np.corrcoef(g["t"], g["p"])[0, 1])
    return rmse, corr, len(g)


def map_beta(next_dm, key_beta: dict[str, float]) -> np.ndarray:
    return np.array([key_beta.get(next_dm.col_to_key[j], 0.0) for j in range(next_dm.X.shape[1])])


def evaluate_on_next(next_dm, key_beta, intercept):
    y_pred = next_dm.X @ map_beta(next_dm, key_beta) + intercept
    return margin_metrics(next_dm, y_pred)


def anchor_check(train_dm, beta) -> tuple[bool, str]:
    """Gobert test: known anchors must have the right sign."""
    names = name_lookup()  # pid(str) -> name
    want = {"jokic": None, "gobert": None}
    pid_of = {}
    for pid, nm in names.items():
        low = str(nm).lower()
        for k in want:
            if k in low:
                pid_of[k] = str(pid)
    notes = []
    ok = True
    key_to_col = train_dm.key_to_col
    for k, pid in pid_of.items():
        oc, dc = key_to_col.get(f"{pid}_off"), key_to_col.get(f"{pid}_def")
        if oc is None or dc is None:
            continue
        total = (beta[oc] - beta[dc]) * 100.0
        notes.append(f"{k}={total:+.2f}")
        if total <= 0:
            ok = False
    if "gobert" in pid_of:
        dc = key_to_col.get(f"{pid_of['gobert']}_def")
        if dc is not None and beta[dc] * 100.0 >= 0.5:  # defense coef should be negative-ish
            ok = False
            notes.append("gobert_def_sign_flip")
    return ok, ";".join(notes)


def ess(w: np.ndarray) -> float:
    return float((w.sum() ** 2) / np.maximum((w ** 2).sum(), 1e-12))


# ---------- experiment implementations ----------

def pooled_rows(rows, min_poss):
    counts = {}
    for r in rows:
        for pid in list(r[2:12]):
            counts[pid] = counts.get(pid, 0) + 1
    low = {pid for pid, c in counts.items() if c < min_poss}
    out = []
    for r in rows:
        r = list(r)
        for i in range(2, 12):
            if r[i] in low:
                r[i] = POOL_SENTINEL
        out.append(tuple(r))
    return out, len(low)


def decay_weights(dm, half_life_days):
    dates = pd.to_datetime(pd.Series(dm.row_dates)).values
    age_days = (dates.max() - dates) / np.timedelta64(1, "D")
    return np.power(0.5, age_days.astype(float) / half_life_days)


def soft_gt_weights(dm):
    """Competitiveness weight: logistic decline around the garbage threshold."""
    from standard_rapm import GT_BASE
    thr = np.array([GT_BASE.get(min(int(p), 4), 12) for p in dm.row_periods], dtype=float)
    return 1.0 / (1.0 + np.exp((np.abs(dm.row_margin_off) - thr) / 3.0))


# ---------- runner ----------

def append_result(row: dict):
    df = pd.DataFrame([row])
    header = not RESULTS_CSV.exists()
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def done_names() -> set[str]:
    if not RESULTS_CSV.exists():
        return set()
    return set(pd.read_csv(RESULTS_CSV)["name"].astype(str))


def main():
    train_rows = fetch_possessions(TRAIN_SEASONS)
    next_rows = fetch_possessions([NEXT_SEASON])
    print(f"train: {len(train_rows):,} | next: {len(next_rows):,}", flush=True)

    eval_cfg = base_cfg([NEXT_SEASON])
    next_dm = build_design_matrix(next_rows, eval_cfg)
    already = done_names()

    def run(name, params, train_rows_v=None, cfg=None, weight_fn=None, prior=None, key_beta=None, intercept=None):
        """key_beta+intercept given -> skip fitting (pre-computed)."""
        if name in already:
            print(f"SKIP {name} (already done)", flush=True)
            return None
        t0 = time.time()
        try:
            if key_beta is None:
                rows_v = train_rows_v if train_rows_v is not None else train_rows
                dm = build_design_matrix(rows_v, cfg)
                w = dm.weights.copy()
                if weight_fn is not None:
                    w = w * weight_fn(dm)
                beta, intercept = fit(dm, cfg, prior=prior, weights=w)
                key_beta = {k: float(beta[j]) for j, k in dm.col_to_key.items()}
                anchors_ok, anchor_note = anchor_check(dm, beta)
                sample_ess = ess(w)
                n_rows = dm.X.shape[0]
            else:
                anchors_ok, anchor_note, sample_ess, n_rows = True, "precomputed", float("nan"), 0
            rmse, corr, n_games = evaluate_on_next(next_dm, key_beta, intercept)
            row = {
                "name": name, "params": json.dumps(params), "margin_rmse": round(rmse, 3),
                "margin_corr": round(corr, 4), "n_games": n_games, "anchors_ok": anchors_ok,
                "anchor_note": anchor_note, "ess": round(sample_ess, 0), "n_train_rows": n_rows,
                "elapsed_s": round(time.time() - t0, 1), "ts": pd.Timestamp.now().isoformat(),
            }
            append_result(row)
            print(f"EXPERIMENT_DONE {name}: corr={corr:.4f} rmse={rmse:.2f} anchors={anchors_ok} "
                  f"ess={sample_ess:,.0f} [{row['elapsed_s']}s]", flush=True)
            return key_beta, intercept
        except Exception as e:
            append_result({"name": name, "params": json.dumps(params), "margin_rmse": None,
                           "margin_corr": None, "n_games": 0, "anchors_ok": False,
                           "anchor_note": f"ERROR: {e}", "ess": None, "n_train_rows": 0,
                           "elapsed_s": round(time.time() - t0, 1), "ts": pd.Timestamp.now().isoformat()})
            print(f"EXPERIMENT_FAILED {name}: {e}", flush=True)
            return None

    # --- baseline ---
    run("baseline_home", {}, cfg=base_cfg(TRAIN_SEASONS))

    # --- Phase A: recency decay ---
    for hl in (365, 730):
        run(f"decay_hl{hl}", {"half_life_days": hl}, cfg=base_cfg(TRAIN_SEASONS),
            weight_fn=lambda dm, hl=hl: decay_weights(dm, hl))

    # --- Phase A: soft garbage weighting (keep all rows, weight by competitiveness) ---
    run("soft_gt", {"w": "logistic(|margin|-thr)/3"}, cfg=base_cfg(TRAIN_SEASONS, garbage_time=False),
        weight_fn=soft_gt_weights)

    # --- Phase A: replacement pooling ---
    for n in (250, 500, 1000):
        rows_v, n_pooled = pooled_rows(train_rows, n)
        run(f"pool_{n}", {"min_poss": n, "n_pooled_players": n_pooled}, train_rows_v=rows_v,
            cfg=base_cfg(TRAIN_SEASONS))

    # --- Phase A: asymmetric lambdas (agnostic grid, both directions) ---
    for lo, ld in ((1500, 6000), (6000, 1500), (2000, 4500), (4500, 2000)):
        run(f"asym_off{lo}_def{ld}", {"lam_off": lo, "lam_def": ld},
            cfg=base_cfg(TRAIN_SEASONS, lam_off=lo, lam_def=ld))

    # --- Phase A: elastic net via SGD ---
    def elastic(alpha, l1_ratio):
        from sklearn.linear_model import SGDRegressor
        cfg = base_cfg(TRAIN_SEASONS)
        dm = build_design_matrix(train_rows, cfg)
        m = SGDRegressor(penalty="elasticnet", alpha=alpha, l1_ratio=l1_ratio,
                         max_iter=15, tol=None, random_state=7, learning_rate="adaptive", eta0=0.01)
        y_c = dm.y - dm.y.mean()
        m.fit(dm.X, y_c)
        kb = {k: float(m.coef_[j]) for j, k in dm.col_to_key.items()}
        sparsity = float(np.mean(np.abs(m.coef_) < 1e-8))
        return kb, float(dm.y.mean()), sparsity

    for alpha, l1r in ((1e-6, 0.15), (1e-5, 0.5)):
        name = f"enet_a{alpha}_l1{l1r}"
        if name not in already:
            try:
                t0 = time.time()
                kb, ic, sp = elastic(alpha, l1r)
                rmse, corr, n_games = evaluate_on_next(next_dm, kb, ic)
                append_result({"name": name, "params": json.dumps({"alpha": alpha, "l1_ratio": l1r, "sparsity": sp}),
                               "margin_rmse": round(rmse, 3), "margin_corr": round(corr, 4), "n_games": n_games,
                               "anchors_ok": True, "anchor_note": f"sparsity={sp:.2f}", "ess": None,
                               "n_train_rows": 0, "elapsed_s": round(time.time() - t0, 1),
                               "ts": pd.Timestamp.now().isoformat()})
                print(f"EXPERIMENT_DONE {name}: corr={corr:.4f} rmse={rmse:.2f} sparsity={sp:.2f}", flush=True)
            except Exception as e:
                print(f"EXPERIMENT_FAILED {name}: {e}", flush=True)

    # --- Phase B: previous-window prior (walk-forward: 2018-20 never sees 2021-23) ---
    prev_rows = fetch_possessions([2018, 2019, 2020])
    prev_cfg = base_cfg([2018, 2019, 2020])
    prev_dm = build_design_matrix(prev_rows, prev_cfg)
    prev_beta, _ = fit(prev_dm, prev_cfg)
    prev_prior = coefficients_as_prior(prev_dm, prev_beta)
    print("previous-window prior fit done", flush=True)

    for strength in (0.5, 1.0, 2.0):
        run(f"prior_prev_s{strength}", {"prior": "2018-20", "strength_mult": strength},
            cfg=base_cfg(TRAIN_SEASONS, lam_off=3000 * strength, lam_def=3000 * strength,
                         prior_mode="target"),
            prior=prev_prior)

    # --- Phase B: infinite RAPM (walk-forward chain 2012-14 -> 2015-17 -> 2018-20 -> prior) ---
    chain_prior = None
    for window in ([2012, 2013, 2014], [2015, 2016, 2017], [2018, 2019, 2020]):
        rows_w = fetch_possessions(window)
        cfg_w = base_cfg(window, prior_mode="zero" if chain_prior is None else "target")
        dm_w = build_design_matrix(rows_w, cfg_w)
        beta_w, _ = fit(dm_w, cfg_w, prior=chain_prior)
        chain_prior = coefficients_as_prior(dm_w, beta_w)
        print(f"chain fit {window} done", flush=True)
    run("prior_chain_depth3", {"chain": "2012->2015->2018->2021"},
        cfg=base_cfg(TRAIN_SEASONS, prior_mode="target"), prior=chain_prior)

    # --- Phase B: window lengths (singles; ensemble judged in interpretation) ---
    for name, seasons in (("win1_2023", [2023]), ("win5_2019_23", [2019, 2020, 2021, 2022, 2023])):
        run(name, {"seasons": seasons}, train_rows_v=fetch_possessions(seasons), cfg=base_cfg(seasons))

    print("EXPERIMENTS_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
