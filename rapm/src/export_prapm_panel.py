#!/usr/bin/env python3
"""Build single-season pRAPM panel: 1yr possession RAPM + SPM prior pull.

Per season Y:
  pass-1 zero-prior RAPM on season Y possessions (hl250 decay)
  prior center = SPM Off/Def for that player-season (from spm_panel)
  pass-2 ridge with strength c · σ² / τ²_spm
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pathlib import Path

from experiments import decay_weights
from export_spm_panel import build_spm_panel
from feature_eval import HL, PriorPack, champion_fit, prior_fit
from paths import DATA, RAPM_ALL_WINDOWS_CSV, RAPM_RESULTS, ensure_dirs
from standard_rapm import (
    FitResult,
    LambdaProfile,
    RunConfig,
    build_design_matrix,
    fetch_possessions,
    lambda_vector,
    player_table,
    predict,
    standard_errors,
)

ensure_dirs()

DEFAULT_C = 4.0
SPM_PANEL = DATA / "spm_panel_windows.csv"
PRAPM_PANEL = DATA / "prapm_panel_windows.csv"


def _load_run_config() -> dict:
    meta_path = RAPM_ALL_WINDOWS_CSV.parent / "run_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"decay_hl": HL, "lambda_off": 3000, "lambda_def": 3000}


def make_cfg(season: int, run_meta: dict) -> RunConfig:
    return RunConfig(
        seasons=[season],
        season_type="regular",
        spec="prapm_season_spm",
        prior_mode="target",
        include_home=True,
        include_rubberband=False,
        include_season_effects=False,
        garbage_time=True,
        optimize_lambdas=False,
        compute_intervals=True,
        lambda_profile=LambdaProfile(
            off=float(run_meta.get("lambda_off", 3000)),
            defense=float(run_meta.get("lambda_def", 3000)),
            meta=300.0,
            season=100.0,
        ),
    )


def spm_prior_pack(dm, beta1: np.ndarray, spm_season: pd.DataFrame) -> PriorPack:
    """Prior center from SPM panel; τ² from pass-1 vs SPM misfit (per side)."""
    spm = spm_season.set_index("PLAYER_ID")
    target_vec = np.zeros(dm.X.shape[1])
    col_sums = np.asarray(np.abs(dm.X).sum(axis=0)).ravel()

    pids, cols, sides, y1, yt, ww = [], [], [], [], [], []
    for j, key in dm.col_to_key.items():
        if not (key.endswith("_off") or key.endswith("_def")):
            continue
        pid = int(key.split("_")[0])
        if pid not in spm.index:
            continue
        side_off = key.endswith("_off")
        t = float(spm.loc[pid, "Off" if side_off else "Def"]) / 100.0
        target_vec[j] = t
        pids.append(pid)
        cols.append(j)
        sides.append(side_off)
        y1.append(float(beta1[j]))
        yt.append(t)
        ww.append(float(col_sums[j]))

    if not cols:
        return PriorPack(
            target=target_vec,
            lam0_unit=np.zeros(dm.X.shape[1]),
            tau2_off=float("nan"),
            tau2_def=float("nan"),
            oof_r2_off=float("nan"),
            oof_r2_def=float("nan"),
            meta={"prior": "spm", "n": 0},
        )

    y1 = np.array(y1)
    yt = np.array(yt)
    ww = np.array(ww)
    sides = np.array(sides)
    lam0_unit = np.zeros(dm.X.shape[1])

    def side_stats(side: bool) -> tuple[float, float]:
        m = sides == side
        if not m.any():
            return float("nan"), 0.0
        resid = y1[m] - yt[m]
        tau2 = float(np.average(resid ** 2, weights=ww[m]))
        inv = 1.0 / max(tau2, 1e-8)
        for j, s in zip(cols, sides):
            if s == side:
                lam0_unit[j] = inv
        return tau2, inv

    tau2_off, _ = side_stats(True)
    tau2_def, _ = side_stats(False)
    return PriorPack(
        target=target_vec,
        lam0_unit=lam0_unit,
        tau2_off=tau2_off,
        tau2_def=tau2_def,
        oof_r2_off=float("nan"),
        oof_r2_def=float("nan"),
        meta={"prior": "spm", "n": len(set(pids))},
    )


def confidence_tiers(df: pd.DataFrame) -> pd.DataFrame:
    if "RAPM_SE" not in df.columns:
        df["Tier"] = "?"
        return df
    try:
        df["Tier"] = pd.qcut(df["RAPM_SE"], q=3, labels=["A", "B", "C"])
    except ValueError:
        df["Tier"] = "B"
    return df


def fit_season_prapm(season: int, spm_season: pd.DataFrame, *, c: float, run_meta: dict) -> pd.DataFrame:
    cfg = make_cfg(season, run_meta)
    dm = build_design_matrix(fetch_possessions([season]), cfg)
    hl = float(run_meta.get("decay_hl", HL))
    w = dm.weights * decay_weights(dm, hl)

    beta1, ic1 = champion_fit(dm, cfg, w)
    resid = dm.y - predict(dm.X, beta1, ic1)
    sigma2 = float(np.average(resid ** 2, weights=w))

    pack = spm_prior_pack(dm, beta1, spm_season)
    lam0 = sigma2 * pack.lam0_unit
    beta2, ic2 = prior_fit(dm, cfg, w, c * lam0, pack.target)

    base = lambda_vector(dm, cfg.lambda_profile)
    fit = FitResult(
        beta=beta2,
        intercept=ic2,
        lambda_profile=cfg.lambda_profile,
        zero_penalty=base,
        target_penalty=c * lam0,
        target=pack.target,
        cv_scores=pd.DataFrame(),
        elapsed_seconds=0.0,
    )
    se_df = standard_errors(dm, fit)
    table = player_table(dm, fit, cfg, se_df)
    table = confidence_tiers(table)
    table["Window_End"] = int(season)
    table["Season"] = str(season)
    table["Prior_Mode"] = f"spm_c{c:g}"
    return table


def build_prapm_panel(c: float = DEFAULT_C, *, spm_df: pd.DataFrame | None = None) -> pd.DataFrame:
    run_meta = _load_run_config()
    if spm_df is None:
        if SPM_PANEL.exists():
            spm_df = pd.read_csv(SPM_PANEL)
        else:
            spm_df = build_spm_panel()
            spm_df.to_csv(SPM_PANEL, index=False)

    spm_df["Window_End"] = spm_df["Window_End"].astype(int)
    seasons = sorted(spm_df["Window_End"].unique())
    out = []
    for season in seasons:
        sub = spm_df[spm_df["Window_End"] == season]
        if sub.empty:
            continue
        try:
            table = fit_season_prapm(season, sub, c=c, run_meta=run_meta)
            out.append(table)
            print(f"PRAPM_SEASON_DONE {season}: {len(table)} players", flush=True)
        except Exception as e:
            print(f"PRAPM_SEASON_FAIL {season}: {e}", flush=True)

    if not out:
        raise RuntimeError("pRAPM panel empty")
    return pd.concat(out, ignore_index=True)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--c", type=float, default=DEFAULT_C)
    p.add_argument("--season", type=int, default=None, help="single season smoke test")
    p.add_argument("--tag", default="prapm_season_spm_c4")
    args = p.parse_args()

    run_meta = _load_run_config()
    if args.season is not None:
        spm_df = pd.read_csv(SPM_PANEL) if SPM_PANEL.exists() else build_spm_panel()
        sub = spm_df[spm_df["Window_End"] == args.season]
        df = fit_season_prapm(args.season, sub, c=args.c, run_meta=run_meta)
    else:
        df = build_prapm_panel(c=args.c)

    df.to_csv(PRAPM_PANEL, index=False)
    out_dir = RAPM_RESULTS / f"final_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    long_path = out_dir / "rapm_all_windows.csv"
    df.to_csv(long_path, index=False)
    meta = {
        "tag": args.tag,
        "spec": "prapm_season_spm",
        "window": 1,
        "decay_hl": run_meta.get("decay_hl", HL),
        "lambda_off": run_meta.get("lambda_off", 3000),
        "lambda_def": run_meta.get("lambda_def", 3000),
        "prior_c": args.c,
        "prior_type": "spm",
        "windows": sorted(df["Window_End"].unique().astype(int).tolist()),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"PRAPM_PANEL_DONE rows={len(df)} -> {PRAPM_PANEL}", flush=True)

    if args.season is None:
        try:
            import subprocess
            import sys

            subprocess.run(
                [sys.executable, str(Path(__file__).resolve().parent / "build_human_viewer.py"), "--no-agent", "--skip-spm"],
                check=False,
            )
        except Exception as e:
            print(f"VIEWER_REBUILD_SKIP {e}", flush=True)


if __name__ == "__main__":
    main()
