#!/usr/bin/env python3
"""RAPM × SPM evaluation harness — two-pass splice + L1 gate + L2 team report.

Frozen backbone: hl250, λ=3000 symmetric, home, garbage-time filtered.
Tunable: prior strength c (and future SPM feature sets).

Entry: run_splice(fold, prior_fn, c_grid) -> list[SpliceResult]
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from experiments import anchor_check, base_cfg, decay_weights, evaluate_on_next, ess
from paths import FEATURES_DIR, PLAYERSHEETS_YEAR_TOTALS, PROJECT_ROOT, ensure_dirs
from standard_rapm import (
    build_design_matrix,
    fetch_possessions,
    lambda_vector,
    predict,
    solve_penalized_ridge,
)

ensure_dirs()

HL = 250.0
C_GRID_DEFAULT = (1.0, 2.0, 4.0, 8.0)

FOLDS = {
    "f24": {"train": [2021, 2022, 2023], "test": 2024},
    "f23": {"train": [2020, 2021, 2022], "test": 2023},
    "vault": {"train": [2015, 2016, 2017], "test": 2018},
}

TEAM_ABBR_MAP = {"PHX": "PHO"}
TEAM_RATINGS = PROJECT_ROOT / "zts" / "data" / "processed" / "team_ratings.csv"


@dataclass
class PriorPack:
    target: np.ndarray
    lam0_unit: np.ndarray
    tau2_off: float
    tau2_def: float
    oof_r2_off: float
    oof_r2_def: float
    meta: dict = field(default_factory=dict)


@dataclass
class SpliceResult:
    fold: str
    name: str
    c: float
    margin_corr: float
    margin_rmse: float
    n_games: int
    anchors_ok: bool
    anchor_note: str
    sample_ess: float
    oof_r2_off: float
    oof_r2_def: float
    tau2_off: float
    tau2_def: float
    team_net_corr: float
    team_wins_corr: float
    elapsed_s: float
    params: dict = field(default_factory=dict)


def champion_fit(dm, cfg, w):
    base = lambda_vector(dm, cfg.lambda_profile)
    zeros = np.zeros(dm.X.shape[1])
    return solve_penalized_ridge(dm.X, dm.y, w, base, np.zeros_like(base), zeros)


def prior_fit(dm, cfg, w, lam0_vec, target_vec):
    base = lambda_vector(dm, cfg.lambda_profile)
    return solve_penalized_ridge(dm.X, dm.y, w, base, lam0_vec, target_vec)


def minutes_prior_pack(dm, beta, n_folds=5, seed=7) -> PriorPack:
    """Minutes-only prior — same as spm_minutes_prior.minutes_prior, per-side tau2."""
    col_sums = np.asarray(np.abs(dm.X).sum(axis=0)).ravel()
    player_cols = np.array(
        [j for j, k in dm.col_to_key.items() if k.endswith("_off") or k.endswith("_def")]
    )
    is_off = np.array([dm.col_to_key[j].endswith("_off") for j in player_cols])
    x = np.log1p(col_sums[player_cols])
    y = beta[player_cols]
    w = col_sums[player_cols]

    rng = np.random.default_rng(seed)
    fold_of = rng.integers(0, n_folds, len(player_cols))
    pred = np.zeros_like(y)

    for f in range(n_folds):
        tr = fold_of != f
        for side in (True, False):
            m = tr & (is_off == side)
            t = (fold_of == f) & (is_off == side)
            if m.sum() < 10 or t.sum() == 0:
                continue
            X1 = np.column_stack([np.ones(m.sum()), x[m], x[m] ** 2])
            Ws = np.sqrt(w[m])
            coef, *_ = np.linalg.lstsq(X1 * Ws[:, None], y[m] * Ws, rcond=None)
            Xt = np.column_stack([np.ones(t.sum()), x[t], x[t] ** 2])
            pred[t] = Xt @ coef

    target_vec = np.zeros(dm.X.shape[1])
    lam0_unit = np.zeros(dm.X.shape[1])
    target_vec[player_cols] = pred

    def side_stats(side: bool) -> tuple[float, float, float]:
        m = is_off == side
        resid = y[m] - pred[m]
        ww = w[m]
        tau2 = float(np.average(resid ** 2, weights=ww))
        ss_tot = float(np.average((y[m] - np.average(y[m], weights=ww)) ** 2, weights=ww))
        r2 = 1.0 - tau2 / ss_tot if ss_tot > 0 else float("nan")
        return tau2, r2, 1.0 / max(tau2, 1e-12)

    tau2_off, r2_off, lam_off = side_stats(True)
    tau2_def, r2_def, lam_def = side_stats(False)
    for j, side in zip(player_cols, is_off):
        lam0_unit[j] = lam_off if side else lam_def

    return PriorPack(
        target=target_vec,
        lam0_unit=lam0_unit,
        tau2_off=tau2_off,
        tau2_def=tau2_def,
        oof_r2_off=r2_off,
        oof_r2_def=r2_def,
        meta={"prior": "minutes"},
    )


def key_beta_to_player_df(key_beta: dict[str, float], names: dict[str, str] | None = None) -> pd.DataFrame:
    """Player-level off/def/rapm from coefficient dict."""
    rows = []
    seen = set()
    for k, v in key_beta.items():
        if not k.endswith("_off"):
            continue
        pid = k[:-4]
        if pid in seen:
            continue
        dk = f"{pid}_def"
        if dk not in key_beta:
            continue
        seen.add(pid)
        off, df = float(v), float(key_beta[dk])
        rows.append({
            "PLAYER_ID": int(pid) if pid.isdigit() else pid,
            "Off": off * 100.0,
            "Def": df * 100.0,
            "RAPM": (off - df) * 100.0,
            "Name": (names or {}).get(pid, pid),
        })
    return pd.DataFrame(rows)


def load_team_minutes(season: int) -> pd.DataFrame:
    path = PLAYERSHEETS_YEAR_TOTALS / f"{season}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["team", "PLAYER_ID", "minutes"])
    df = pd.read_csv(path, low_memory=False)
    minute_col = "MIN" if "MIN" in df.columns and df["MIN"].notna().any() else "Minutes"
    df = df.rename(columns={minute_col: "minutes", "TEAM_ABBREVIATION": "team"})
    df = df.dropna(subset=["PLAYER_ID", "team", "minutes"])
    df["team"] = df["team"].astype(str).replace(TEAM_ABBR_MAP)
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
    df["minutes"] = df["minutes"].astype(float)
    return (
        df.groupby(["team", "PLAYER_ID"], as_index=False)
        .agg(minutes=("minutes", "sum"))
    )


def load_actual_teams(season: int) -> pd.DataFrame:
    if not TEAM_RATINGS.exists():
        return pd.DataFrame(columns=["team", "actual_net_rating", "actual_wins"])
    df = pd.read_csv(TEAM_RATINGS)
    df = df[df["Season"] == season].copy()
    df["team"] = df["Team"].astype(str).replace(TEAM_ABBR_MAP)
    df["actual_net_rating"] = df["team_ortg"] - df["team_drtg"]
    return df.rename(columns={"team_wins": "actual_wins"})[
        ["team", "actual_net_rating", "actual_wins"]
    ]


def team_layer_metrics(key_beta: dict[str, float], test_season: int) -> tuple[float, float]:
    """L2: minute-weighted team RAPM vs actual net rating / wins."""
    players = key_beta_to_player_df(key_beta)
    if players.empty:
        return float("nan"), float("nan")
    minutes = load_team_minutes(test_season)
    actual = load_actual_teams(test_season)
    if minutes.empty or actual.empty:
        return float("nan"), float("nan")

    merged = minutes.merge(players, on="PLAYER_ID", how="left")
    merged["RAPM"] = merged["RAPM"].fillna(0.0)

    def team_pred(g: pd.DataFrame) -> pd.Series:
        w = g["minutes"].to_numpy(float)
        if w.sum() <= 0:
            return pd.Series({"pred_net_rating": np.nan})
        return pd.Series({"pred_net_rating": float(np.average(g["RAPM"], weights=w))})

    pred = merged.groupby("team").apply(team_pred, include_groups=False).reset_index()
    out = pred.merge(actual, on="team", how="inner")
    if len(out) < 5:
        return float("nan"), float("nan")

    def safe_corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(a.corr(b))

    return (
        safe_corr(out["pred_net_rating"], out["actual_net_rating"]),
        safe_corr(out["pred_net_rating"], out["actual_wins"]),
    )


def run_splice(
    fold: str,
    prior_fn: Callable | None = None,
    *,
    c_grid: tuple[float, ...] = C_GRID_DEFAULT,
    name_prefix: str = "splice",
) -> list[SpliceResult]:
    """Full two-pass splice for one fold. prior_fn(dm, beta1) -> PriorPack; default minutes."""
    spec = FOLDS[fold]
    train_seasons, test_season = spec["train"], spec["test"]
    cfg = base_cfg(train_seasons)
    dm = build_design_matrix(fetch_possessions(train_seasons), cfg)
    w = dm.weights * decay_weights(dm, HL)
    next_dm = build_design_matrix(fetch_possessions([test_season]), base_cfg([test_season]))

    t0 = time.time()
    beta1, ic1 = champion_fit(dm, cfg, w)
    resid = dm.y - predict(dm.X, beta1, ic1)
    sigma2 = float(np.average(resid ** 2, weights=w))

    if prior_fn is None:
        prior_fn = minutes_prior_pack

    pack = prior_fn(dm, beta1)
    lam0 = sigma2 * pack.lam0_unit
    results: list[SpliceResult] = []

    for c in c_grid:
        t1 = time.time()
        beta2, ic2 = prior_fit(dm, cfg, w, c * lam0, pack.target)
        kb2 = {k: float(beta2[j]) for j, k in dm.col_to_key.items()}
        rmse, corr, n_games = evaluate_on_next(next_dm, kb2, ic2)
        ok, note = anchor_check(dm, beta2)
        t_net, t_wins = team_layer_metrics(kb2, test_season)
        results.append(
            SpliceResult(
                fold=fold,
                name=f"{name_prefix}_c{c:g}_{fold}",
                c=c,
                margin_corr=round(corr, 4),
                margin_rmse=round(rmse, 3),
                n_games=n_games,
                anchors_ok=ok,
                anchor_note=note,
                sample_ess=round(ess(w), 0),
                oof_r2_off=round(pack.oof_r2_off, 4),
                oof_r2_def=round(pack.oof_r2_def, 4),
                tau2_off=pack.tau2_off,
                tau2_def=pack.tau2_def,
                team_net_corr=round(t_net, 4) if np.isfinite(t_net) else float("nan"),
                team_wins_corr=round(t_wins, 4) if np.isfinite(t_wins) else float("nan"),
                elapsed_s=round(time.time() - t1, 1),
                params={"sigma2": sigma2, "prior_meta": pack.meta, "pass1_s": round(time.time() - t0, 1)},
            )
        )
    return results


def append_results_tsv(rows: list[dict], path=None):
    path = path or (FEATURES_DIR / "results.tsv")
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size:
        old = pd.read_csv(path, sep="\t")
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, sep="\t", index=False)


def splice_to_tsv_row(r: SpliceResult, source: str = "harness:baseline", *, features_chosen: str = "") -> dict:
    gate_col = f"gate_{r.fold}"
    row = {
        "id": r.name,
        "source": source,
        "description": f"minutes prior c={r.c} {r.fold}",
        "features_chosen": features_chosen,
        "tier_reached": 2,
        "status": "keep" if r.anchors_ok else "discard",
        "gate_f24": r.margin_corr if r.fold == "f24" else "",
        "gate_f23": r.margin_corr if r.fold == "f23" else "",
        "gate_vault": r.margin_corr if r.fold == "vault" else "",
        "rmse_f24": r.margin_rmse if r.fold == "f24" else "",
        "oof_r2_off": r.oof_r2_off,
        "oof_r2_def": r.oof_r2_def,
        "anchor_ok": r.anchors_ok,
        "team_net_corr": r.team_net_corr,
        "team_wins_corr": r.team_wins_corr,
        "complexity_score": 1,
        "ts": pd.Timestamp.now().isoformat(),
    }
    return row


def run_baseline_repro() -> dict:
    """Reproduce minutes prior; return best-by-corr per fold."""
    from experiments import append_result
    from run_lock import rapm_run_lock

    summary = {}
    with rapm_run_lock("baseline_repro"):
        for fold in ("f24", "f23"):
            res = run_splice(fold, name_prefix="mprior_repro")
            for r in res:
                append_result({
                    "name": r.name,
                    "params": json.dumps({"c": r.c, **r.params}),
                    "margin_rmse": r.margin_rmse,
                    "margin_corr": r.margin_corr,
                    "n_games": r.n_games,
                    "anchors_ok": r.anchors_ok,
                    "anchor_note": r.anchor_note,
                    "ess": r.sample_ess,
                    "n_train_rows": 0,
                    "elapsed_s": r.elapsed_s,
                    "ts": pd.Timestamp.now().isoformat(),
                })
                append_results_tsv([splice_to_tsv_row(
                    r, features_chosen="minutes: log_poss + log_poss² (per side)"
                )])
            best = max(res, key=lambda x: x.margin_corr)
            summary[fold] = {"best_c": best.c, "corr": best.margin_corr, "rmse": best.margin_rmse}
            print(
                f"BASELINE_REPRO {fold}: best c={best.c} corr={best.margin_corr} rmse={best.margin_rmse} "
                f"team_net={best.team_net_corr} team_wins={best.team_wins_corr}",
                flush=True,
            )
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "baseline":
        out = run_baseline_repro()
        print("BASELINE_SUMMARY", json.dumps(out), flush=True)
    elif len(sys.argv) > 1 and sys.argv[1] == "vault":
        for r in run_splice("vault", name_prefix="mprior_repro"):
            print(asdict(r), flush=True)
    else:
        for fold in ("f24", "f23"):
            for r in run_splice(fold, c_grid=(2.0,), name_prefix="mprior_quick"):
                print(f"{r.name}: corr={r.margin_corr}", flush=True)
