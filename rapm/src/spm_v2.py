#!/usr/bin/env python3
"""SPM v2: pooled-window training, CV alpha, heteroskedastic tau, residual SPM."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from experiments import anchor_check, append_result, base_cfg, decay_weights, done_names, evaluate_on_next
from feature_eval import FOLDS, HL, champion_fit, minutes_prior_pack, prior_fit, team_layer_metrics
from paths import DATA, RAPM_RESULTS, ensure_dirs
from standard_rapm import build_design_matrix, fetch_possessions, predict

ensure_dirs()

PANEL = RAPM_RESULTS / "final_20260703_hl250" / "rapm_all_windows.csv"
EXCLUDE = {"PLAYER_ID", "Window_End", "MIN", "GP", "OnOffRtg", "OnDefRtg"}
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]


def pool_feats_for_fold(feats: pd.DataFrame, fold: str) -> pd.DataFrame:
    spec = FOLDS[fold]
    blocked = set(spec["train"]) | {spec["test"]}
    return feats[~feats["Window_End"].isin(blocked)].copy()


def merge_labels(feats: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    p = panel[["PLAYER_ID", "Window_End", "Off", "Def"]].copy()
    p["PLAYER_ID"] = p["PLAYER_ID"].astype(int)
    return feats.merge(p, on=["PLAYER_ID", "Window_End"], how="inner", suffixes=("", "_label"))


def feature_matrix(feats: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    f = feats.copy()
    for pc in ("OffPoss", "DefPoss"):
        if pc in f.columns:
            f[f"log_{pc}"] = np.log1p(f[pc])
            f[f"log_{pc}_sq"] = f[f"log_{pc}"] ** 2
    cols = [c for c in f.columns if c not in EXCLUDE and c not in ("Off", "Def") and f[c].dtype != object]
    X = f[cols].astype(float)
    X = (X - X.mean()) / X.std().replace(0, 1)
    return X.fillna(0.0), cols


def oof_ridge_side(
    info: pd.DataFrame,
    fcols: list[str],
    side: bool,
    alpha: float,
    y_col: str = "y",
    n_folds: int = 5,
    seed: int = 7,
) -> tuple[np.ndarray, float, float]:
    s = info["is_off"] == side
    sub = info.loc[s].copy()
    if sub.empty:
        return np.zeros(len(info)), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    pids = sub["pid"].unique()
    fold_map = dict(zip(pids, rng.integers(0, n_folds, len(pids))))
    sub["fold"] = sub["pid"].map(fold_map)
    pred = np.zeros(len(sub))

    for k in range(n_folds):
        tr = sub["fold"] != k
        te = sub["fold"] == k
        X = sub.loc[tr, fcols].values
        yv = sub.loc[tr, y_col].values
        wv = sub.loc[tr, "w"].values
        Xb = np.column_stack([np.ones(tr.sum()), X])
        Ws = np.sqrt(wv)
        A = (Xb * Ws[:, None]).T @ (Xb * Ws[:, None]) + alpha * np.eye(Xb.shape[1])
        b = (Xb * Ws[:, None]).T @ (yv * Ws)
        coef = np.linalg.solve(A, b)
        Xt = np.column_stack([np.ones(te.sum()), sub.loc[te, fcols].values])
        pred[te.values] = Xt @ coef

    yv, wv, pv = sub["y"].values, sub["w"].values, pred
    ss_res = float(np.average((yv - pv) ** 2, weights=wv))
    ss_tot = float(np.average((yv - np.average(yv, weights=wv)) ** 2, weights=wv))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    full_pred = np.zeros(len(info))
    full_pred[s.values] = pred
    resid = info.loc[s, "y"].values - pred
    tau2 = float(np.average(resid ** 2, weights=info.loc[s, "w"].values))
    return full_pred, r2, tau2


def hetero_lam0_scale(log_poss: np.ndarray, resid2: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Smooth tau^2(log poss) -> per-player 1/tau^2 multiplier (median-normalized)."""
    lp = np.asarray(log_poss, float)
    r2 = np.asarray(resid2, float)
    w = np.asarray(weights, float)
    order = np.argsort(lp)
    lp, r2, w = lp[order], r2[order], w[order]
    # binned weighted mean of squared residuals
    n_bins = min(20, max(5, len(lp) // 50))
    edges = np.quantile(lp, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = lp.min() - 1e-9, lp.max() + 1e-9
    tau2 = np.zeros(len(lp))
    for i in range(n_bins):
        m = (lp >= edges[i]) & (lp < edges[i + 1])
        if m.sum() == 0:
            continue
        tau2[m] = float(np.average(r2[m], weights=w[m]))
    tau2 = np.maximum(tau2, 1e-8)
    inv = 1.0 / tau2
    inv *= np.median(tau2)  # normalize so median pull ~ homoskedastic
    out = np.zeros(len(lp))
    out[order] = inv
    return out


def build_spm_prior_pooled(
    dm,
    beta,
    pooled_feats: pd.DataFrame,
    *,
    alpha: float = 10.0,
    residual: bool = False,
    n_folds: int = 5,
    feature_cols: list[str] | None = None,
):
    from feature_eval import PriorPack

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
    info["log_poss"] = np.log1p(info["w"])

    if residual:
        mp = minutes_prior_pack(dm, beta)
        info["y"] = info["y"] - mp.target[info["col"].values]

    # attach window-end features from pooled table (use latest window per player for train fold)
    Xdf, fcols = feature_matrix(pooled_feats)
    pf = pooled_feats.copy()
    pf["PLAYER_ID"] = pf["PLAYER_ID"].astype(int)
    pf_feat, fcols = feature_matrix(pf)
    if feature_cols is not None:
        fcols = [c for c in fcols if c in feature_cols]
    for c in fcols:
        pf[c] = pf_feat[c].values
    feat_mean = pf.groupby("PLAYER_ID")[fcols].mean()
    info = info.merge(feat_mean, left_on="pid", right_index=True, how="left")
    info[fcols] = info[fcols].fillna(0.0)

    pred_off, r2_off, tau2_off = oof_ridge_side(info, fcols, True, alpha)
    pred_def, r2_def, tau2_def = oof_ridge_side(info, fcols, False, alpha)
    pred = np.where(info["is_off"].values, pred_off, pred_def)

    if residual:
        pred = pred + minutes_prior_pack(dm, beta).target[info["col"].values]

    target_vec = np.zeros(dm.X.shape[1])
    lam0_unit = np.zeros(dm.X.shape[1])
    for side, tau2, r2 in ((True, tau2_off, r2_off), (False, tau2_def, r2_def)):
        s = info["is_off"] == side
        target_vec[info.loc[s, "col"].values] = pred[s.values]
        resid2 = (info.loc[s, "y"].values - pred[s.values]) ** 2
        scale = hetero_lam0_scale(info.loc[s, "log_poss"].values, resid2, info.loc[s, "w"].values)
        base_inv = 1.0 / max(tau2, 1e-8)
        lam0_unit[info.loc[s, "col"].values] = base_inv * scale

    return PriorPack(
        target=target_vec,
        lam0_unit=lam0_unit,
        tau2_off=tau2_off,
        tau2_def=tau2_def,
        oof_r2_off=r2_off,
        oof_r2_def=r2_def,
        meta={"alpha": alpha, "residual": residual, "n_pool": len(pooled_feats), "feature_cols": fcols},
    )


def cv_alpha(pooled_feats: pd.DataFrame, panel: pd.DataFrame, dm, beta) -> float:
    merged = merge_labels(pool_feats_for_fold(pooled_feats, "f24"), panel)
    if merged.empty:
        return 10.0
    best_alpha, best_score = 10.0, -np.inf
    for alpha in ALPHA_GRID:
        pack = build_spm_prior_pooled(dm, beta, merged, alpha=alpha)
        score = (pack.oof_r2_off or 0) + (pack.oof_r2_def or 0)
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    c_grid = [2.0, 4.0]
    feats_all = pd.read_parquet(DATA / "spm_features_windows.parquet")
    panel = pd.read_csv(PANEL)
    already = done_names()

    for fold in ("f24", "f23"):
        spec = FOLDS[fold]
        train_seasons, test_season = spec["train"], spec["test"]
        cfg = base_cfg(train_seasons)
        dm = build_design_matrix(fetch_possessions(train_seasons), cfg)
        w = dm.weights * decay_weights(dm, HL)
        next_dm = build_design_matrix(fetch_possessions([test_season]), base_cfg([test_season]))
        beta1, ic1 = champion_fit(dm, cfg, w)
        resid = dm.y - predict(dm.X, beta1, ic1)
        sigma2 = float(np.average(resid ** 2, weights=w))

        pooled = pool_feats_for_fold(feats_all, fold)
        merged = merge_labels(pooled, panel)
        alpha = cv_alpha(feats_all, panel, dm, beta1)
        print(f"[{fold}] pooled rows={len(merged)} cv_alpha={alpha}", flush=True)

        for residual in (False, True) if mode == "full" else (False,):
            tag = "residual" if residual else "pooled"
            pack = build_spm_prior_pooled(dm, beta1, merged, alpha=alpha, residual=residual)
            print(
                f"[{fold}] {tag} OOF off={pack.oof_r2_off:.3f} def={pack.oof_r2_def:.3f}",
                flush=True,
            )
            lam0 = sigma2 * pack.lam0_unit
            for c in c_grid:
                name = f"spmv2_{tag}_a{alpha:g}_c{c:g}_{fold}"
                if name in already:
                    continue
                t0 = time.time()
                beta2, ic2 = prior_fit(dm, cfg, w, c * lam0, pack.target)
                kb = {k: float(beta2[j]) for j, k in dm.col_to_key.items()}
                rmse, corr, ng = evaluate_on_next(next_dm, kb, ic2)
                ok, note = anchor_check(dm, beta2)
                t_net, t_wins = team_layer_metrics(kb, test_season)
                append_result({
                    "name": name,
                    "params": json.dumps({
                        "alpha": alpha, "c": c, "residual": residual,
                        "r2_off": pack.oof_r2_off, "r2_def": pack.oof_r2_def,
                        "n_pool": len(merged),
                    }),
                    "margin_rmse": round(rmse, 3),
                    "margin_corr": round(corr, 4),
                    "n_games": ng,
                    "anchors_ok": ok,
                    "anchor_note": note,
                    "ess": None,
                    "n_train_rows": dm.X.shape[0],
                    "elapsed_s": round(time.time() - t0, 1),
                    "ts": pd.Timestamp.now().isoformat(),
                })
                print(f"EXPERIMENT_DONE {name}: corr={corr:.4f} rmse={rmse:.2f} team={t_net:.3f}", flush=True)

    print("SPMV2_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
