#!/usr/bin/env python3
"""Rating-level validation for RAPM specs.

Two tests that possession RMSE cannot do:

1. Split-half reliability: split a window's GAMES into two random halves,
   fit RAPM on each half, correlate player ratings across halves
   (min-possession filtered). Noisy models -> low correlation.

2. Next-season retrodiction: fit RAPM through season S, then predict every
   game margin of season S+1 using only those frozen coefficients (players
   unseen in training get 0). Scores whether the RATINGS carry real signal
   forward, which is the product we ship.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from paths import STANDARD_RAPM_DIAGNOSTICS, ensure_dirs
from standard_rapm import (
    LambdaProfile,
    RunConfig,
    build_design_matrix,
    fetch_possessions,
    format_season_list,
    penalty_vectors,
    solve_penalized_ridge,
)

ensure_dirs()

SPECS: dict[str, dict] = {
    "player_only": dict(include_home=False, include_rubberband=False, include_season_effects=False),
    "home": dict(include_home=True, include_rubberband=False, include_season_effects=False),
    "home_rubberband": dict(include_home=True, include_rubberband=True, include_season_effects=False),
    "home_rubberband_season": dict(include_home=True, include_rubberband=True, include_season_effects=True),
}


def make_cfg(seasons: list[int], spec_name: str, args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        seasons=seasons,
        season_type="regular",
        spec=f"val_{spec_name}",
        prior_mode="zero",
        garbage_time=True,
        optimize_lambdas=False,
        compute_intervals=False,
        lambda_profile=LambdaProfile(
            off=args.lambda_off, defense=args.lambda_def,
            meta=args.lambda_meta, season=args.lambda_season,
        ),
        **SPECS[spec_name],
    )


def fit_beta(dm, cfg) -> tuple[np.ndarray, float]:
    zero_penalty, target_penalty, target = penalty_vectors(dm, cfg, cfg.lambda_profile, None)
    return solve_penalized_ridge(dm.X, dm.y, dm.weights, zero_penalty, target_penalty, target)


def player_ratings(dm, beta: np.ndarray) -> pd.DataFrame:
    """Per-player total RAPM (off - def, per 100) with possession counts."""
    col_sums = np.asarray(np.abs(dm.X).sum(axis=0)).ravel()
    off, dfn, poss = {}, {}, {}
    for idx, key in dm.col_to_key.items():
        if key.endswith("_off"):
            pid = int(key.split("_")[0])
            off[pid] = float(beta[idx]) * 100.0
            poss[pid] = poss.get(pid, 0) + int(col_sums[idx])
        elif key.endswith("_def"):
            pid = int(key.split("_")[0])
            dfn[pid] = float(beta[idx]) * 100.0
            poss[pid] = poss.get(pid, 0) + int(col_sums[idx])
    rows = [
        {"pid": pid, "rapm": off[pid] - dfn.get(pid, 0.0), "poss": poss.get(pid, 0)}
        for pid in off
        if pid in dfn
    ]
    return pd.DataFrame(rows)


def split_half_reliability(raw_rows: list[tuple], cfg: RunConfig, n_repeats: int, min_poss: int, seed: int = 7) -> dict:
    dm = build_design_matrix(raw_rows, cfg)
    games = pd.unique(dm.gameids)
    rng = np.random.default_rng(seed)
    corrs = []
    for _ in range(n_repeats):
        perm = rng.permutation(len(games))
        half_a = set(games[perm[: len(games) // 2]])
        in_a = np.fromiter((g in half_a for g in dm.gameids), dtype=bool, count=len(dm.gameids))
        ratings = []
        for mask in (in_a, ~in_a):
            idx = np.where(mask)[0]
            sub_X, sub_y, sub_w = dm.X[idx], dm.y[idx], dm.weights[idx]
            zero_penalty, target_penalty, target = penalty_vectors(dm, cfg, cfg.lambda_profile, None)
            beta, _ = solve_penalized_ridge(sub_X, sub_y, sub_w, zero_penalty, target_penalty, target)
            half_dm = replace(dm, X=sub_X)
            ratings.append(player_ratings(half_dm, beta))
        merged = ratings[0].merge(ratings[1], on="pid", suffixes=("_a", "_b"))
        merged = merged[(merged["poss_a"] >= min_poss) & (merged["poss_b"] >= min_poss)]
        if len(merged) >= 10:
            corrs.append(float(np.corrcoef(merged["rapm_a"], merged["rapm_b"])[0, 1]))
    return {
        "split_half_corr": float(np.mean(corrs)) if corrs else float("nan"),
        "split_half_corr_sd": float(np.std(corrs)) if corrs else float("nan"),
        "n_repeats": len(corrs),
    }


def next_season_retrodiction(train_rows: list[tuple], next_rows: list[tuple], cfg: RunConfig) -> dict:
    """Fit on train seasons, predict next season's game margins with frozen betas."""
    train_dm = build_design_matrix(train_rows, cfg)
    beta, intercept = fit_beta(train_dm, cfg)
    key_beta = {key: float(beta[idx]) for idx, key in train_dm.col_to_key.items()}

    next_cfg = replace(cfg, seasons=[max(r[12] for r in next_rows)], include_season_effects=False)
    next_dm = build_design_matrix(next_rows, next_cfg)
    mapped = np.array([key_beta.get(next_dm.col_to_key[j], 0.0) for j in range(next_dm.X.shape[1])])
    y_pred = next_dm.X @ mapped + intercept

    known = sum(
        1 for j, key in next_dm.col_to_key.items()
        if key.endswith("_off") and key in key_beta
    )
    total = sum(1 for key in next_dm.col_to_key.values() if key.endswith("_off"))

    sign = np.where(next_dm.row_home_off, 1.0, -1.0)
    df = pd.DataFrame({
        "gameid": next_dm.gameids,
        "m_true": next_dm.y * sign,
        "m_pred": y_pred * sign,
    })
    g = df.groupby("gameid", observed=True).agg(m_true=("m_true", "sum"), m_pred=("m_pred", "sum"))
    rmse = float(np.sqrt(np.mean((g["m_true"] - g["m_pred"]) ** 2)))
    corr = float(np.corrcoef(g["m_true"], g["m_pred"])[0, 1])
    # Naive baseline: predict every margin as the training-mean margin (~home edge).
    base_rmse = float(np.sqrt(np.mean((g["m_true"] - g["m_true"].mean()) ** 2)))
    return {
        "next_margin_rmse": rmse,
        "next_margin_corr": corr,
        "baseline_margin_rmse": base_rmse,
        "n_next_games": int(len(g)),
        "player_coverage": round(known / max(total, 1), 3),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Rating-level RAPM validation.")
    parser.add_argument("--train-start", type=int, default=2021)
    parser.add_argument("--train-end", type=int, default=2023)
    parser.add_argument("--next-season", type=int, default=2024)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--min-poss", type=int, default=2000)
    parser.add_argument("--specs", default="player_only,home,home_rubberband,home_rubberband_season")
    parser.add_argument("--lambda-off", type=float, default=3000.0)
    parser.add_argument("--lambda-def", type=float, default=3000.0)
    parser.add_argument("--lambda-meta", type=float, default=300.0)
    parser.add_argument("--lambda-season", type=float, default=100.0)
    parser.add_argument("--tag", default="ratings_v1")
    args = parser.parse_args(argv)

    train_seasons = list(range(args.train_start, args.train_end + 1))
    train_rows = fetch_possessions(train_seasons)
    next_rows = fetch_possessions([args.next_season])
    print(f"train {format_season_list(train_seasons)}: {len(train_rows):,} rows | "
          f"next {args.next_season}: {len(next_rows):,} rows")

    results = []
    for spec_name in args.specs.split(","):
        spec_name = spec_name.strip()
        t0 = time.time()
        cfg = make_cfg(train_seasons, spec_name, args)
        rel = split_half_reliability(train_rows, cfg, args.n_repeats, args.min_poss)
        retro = next_season_retrodiction(train_rows, next_rows, cfg)
        row = {"spec": spec_name, **rel, **retro, "elapsed_s": round(time.time() - t0, 1)}
        results.append(row)
        print(f"{spec_name}: split-half r={rel['split_half_corr']:.3f} | "
              f"next-season margin r={retro['next_margin_corr']:.3f} "
              f"rmse={retro['next_margin_rmse']:.2f} (baseline {retro['baseline_margin_rmse']:.2f}) "
              f"[{row['elapsed_s']}s]")

    out = pd.DataFrame(results)
    stem = f"rating_validation_{args.tag}_{format_season_list(train_seasons)}_next{args.next_season}"
    path = STANDARD_RAPM_DIAGNOSTICS / f"{stem}.csv"
    out.to_csv(path, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
