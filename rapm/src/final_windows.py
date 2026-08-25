#!/usr/bin/env python3
"""Final production RAPM: rolling 3-year windows across history.

Writes ONE dated folder (outputs/rapm_results/final_<tag>/) containing one CSV
per window plus a combined long table with confidence tiers. SEs/CIs on.

Optional --chain-prior runs windows chronologically and shrinks each window
toward the previous window's coefficients (walk-forward, never sees itself).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from paths import RAPM_RESULTS, ensure_dirs
from standard_rapm import (
    FitResult,
    LambdaProfile,
    RunConfig,
    build_design_matrix,
    coefficients_as_prior,
    fetch_possessions,
    fit_model,
    format_season_list,
    player_table,
    standard_errors,
)

ensure_dirs()


def make_cfg(seasons: list[int], args: argparse.Namespace, prior_mode: str) -> RunConfig:
    return RunConfig(
        seasons=seasons,
        season_type="regular",
        spec=args.spec,
        prior_mode=prior_mode,  # type: ignore[arg-type]
        include_home=True,
        include_rubberband=False,
        include_season_effects=False,
        garbage_time=True,
        optimize_lambdas=False,
        compute_intervals=True,
        lambda_profile=LambdaProfile(off=args.lambda_off, defense=args.lambda_def,
                                     meta=300.0, season=100.0),
    )


def confidence_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """A/B/C tiers by RAPM_SE terciles within each window (A = most certain)."""
    if "RAPM_SE" not in df.columns:
        df["Tier"] = "?"
        return df
    try:
        df["Tier"] = pd.qcut(df["RAPM_SE"], q=3, labels=["A", "B", "C"])
    except ValueError:
        df["Tier"] = "B"
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-end-season", type=int, default=1999)
    parser.add_argument("--last-end-season", type=int, default=2024)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--spec", default="final_v3")
    parser.add_argument("--lambda-off", type=float, default=3000.0)
    parser.add_argument("--lambda-def", type=float, default=3000.0)
    parser.add_argument("--chain-prior", action="store_true",
                        help="walk-forward: shrink each window toward the previous window's fit")
    parser.add_argument("--decay-hl", type=float, default=None,
                        help="exponential recency half-life in days (e.g. 365)")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    tag = args.tag or pd.Timestamp.now().strftime("%Y%m%d")
    out_dir = RAPM_RESULTS / f"final_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = []
    prior = None
    t_all = time.time()
    for end in range(args.first_end_season, args.last_end_season + 1):
        seasons = list(range(end - args.window + 1, end + 1))
        label = format_season_list(seasons)
        out_csv = out_dir / f"rapm_{label}.csv"
        if out_csv.exists():
            print(f"SKIP {label} (exists)", flush=True)
            combined.append(pd.read_csv(out_csv))
            continue
        t0 = time.time()
        rows = fetch_possessions(seasons)
        mode = "target" if (args.chain_prior and prior) else "zero"
        cfg = make_cfg(seasons, args, mode)
        dm = build_design_matrix(rows, cfg)
        if args.decay_hl:
            from experiments import decay_weights
            dm.weights = dm.weights * decay_weights(dm, args.decay_hl)
        fit: FitResult = fit_model(dm, cfg, prior if mode == "target" else None)
        se_df = standard_errors(dm, fit)
        table = player_table(dm, fit, cfg, se_df)
        table = confidence_tiers(table)
        table["Window_End"] = end
        table.to_csv(out_csv, index=False)
        combined.append(table)
        if args.chain_prior:
            prior = coefficients_as_prior(dm, fit.beta)
        print(f"WINDOW_DONE {label}: {len(table)} players, "
              f"{dm.X.shape[0]:,} poss [{time.time()-t0:.0f}s]", flush=True)

    long_df = pd.concat(combined, ignore_index=True)
    long_path = out_dir / "rapm_all_windows.csv"
    long_df.to_csv(long_path, index=False)
    meta = {
        "tag": tag, "spec": args.spec, "window": args.window,
        "chain_prior": args.chain_prior,
        "decay_hl": args.decay_hl,
        "lambda_off": args.lambda_off, "lambda_def": args.lambda_def,
        "windows": [int(e) for e in sorted(long_df["Window_End"].unique())],
        "elapsed_s": round(time.time() - t_all, 1),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"ALL_WINDOWS_DONE -> {long_path}", flush=True)


if __name__ == "__main__":
    main()
