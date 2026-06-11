#!/usr/bin/env python3
"""Evaluate standard RAPM specs with grouped OOS and retrodiction tests."""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import replace

import numpy as np
import pandas as pd

from paths import STANDARD_RAPM_DIAGNOSTICS, ensure_dirs
from standard_rapm import (
    LambdaProfile,
    RunConfig,
    build_design_matrix,
    coefficients_as_prior,
    fetch_possessions,
    fit_model,
    format_season_list,
    grouped_splits,
    penalty_vectors,
    predict,
    resolve_seasons,
    solve_penalized_ridge,
    tune_lambdas,
    weighted_mae,
    weighted_rmse,
)

ensure_dirs()


def base_config(args: argparse.Namespace, season_type: str = "regular") -> RunConfig:
    seasons = resolve_seasons(args.window, args.end_season, args.start_season)
    return RunConfig(
        seasons=seasons,
        season_type=season_type,  # type: ignore[arg-type]
        spec=args.spec,
        prior_mode="zero",
        include_home=True,
        include_rubberband=True,
        include_season_effects=True,
        garbage_time=True,
        optimize_lambdas=args.optimize_lambdas,
        cv_folds=args.cv_folds,
        lambda_search_iters=args.lambda_search_iters,
        lambda_profile=LambdaProfile(
            off=args.lambda_off,
            defense=args.lambda_def,
            meta=args.lambda_meta,
            season=args.lambda_season,
        ),
        compute_intervals=False,
    )


def regular_spec_configs(cfg: RunConfig) -> list[tuple[str, RunConfig]]:
    return [
        (
            "player_only",
            replace(
                cfg,
                spec=f"{cfg.spec}_eval_player_only",
                include_home=False,
                include_rubberband=False,
                include_season_effects=False,
                garbage_time=True,
            ),
        ),
        (
            "home",
            replace(
                cfg,
                spec=f"{cfg.spec}_eval_home",
                include_home=True,
                include_rubberband=False,
                include_season_effects=False,
                garbage_time=True,
            ),
        ),
        (
            "home_rubberband",
            replace(
                cfg,
                spec=f"{cfg.spec}_eval_home_rubberband",
                include_home=True,
                include_rubberband=True,
                include_season_effects=False,
                garbage_time=True,
            ),
        ),
        (
            "home_rubberband_season",
            replace(
                cfg,
                spec=f"{cfg.spec}_eval_home_rubberband_season",
                include_home=True,
                include_rubberband=True,
                include_season_effects=True,
                garbage_time=True,
            ),
        ),
        (
            "home_rubberband_season_no_gt",
            replace(
                cfg,
                spec=f"{cfg.spec}_eval_home_rubberband_season_no_gt",
                include_home=True,
                include_rubberband=True,
                include_season_effects=True,
                garbage_time=False,
            ),
        ),
    ]


def chronological_splits(dm) -> list[tuple[np.ndarray, np.ndarray]]:
    seasons = sorted(set(int(s) for s in dm.row_seasons))
    if len(seasons) >= 2:
        latest = seasons[-1]
        train_idx = np.where(dm.row_seasons < latest)[0]
        valid_idx = np.where(dm.row_seasons == latest)[0]
        if len(train_idx) and len(valid_idx):
            return [(train_idx, valid_idx)]

    n = dm.X.shape[0]
    cut = max(1, int(n * 0.8))
    if cut >= n:
        cut = n - 1
    return [(np.arange(cut), np.arange(cut, n))]


def score_splits(raw_rows: list[tuple], cfg: RunConfig, prior: dict[str, float] | None, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dm = build_design_matrix(raw_rows, cfg)
    if cfg.optimize_lambdas:
        profile, lambda_scores = tune_lambdas(dm, cfg, prior)
    else:
        profile = cfg.lambda_profile
        lambda_scores = pd.DataFrame([{**profile.as_dict(), "mean_rmse": np.nan, "stage": "fixed"}])

    zero_penalty, target_penalty, target = penalty_vectors(dm, cfg, profile, prior)
    splits = grouped_splits(dm.gameids, cfg.cv_folds) if method == "game_grouped_cv" else chronological_splits(dm)

    rows = []
    calibration_rows = []
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        beta, intercept = solve_penalized_ridge(
            dm.X[train_idx],
            dm.y[train_idx],
            dm.weights[train_idx],
            zero_penalty,
            target_penalty,
            target,
        )
        y_pred = predict(dm.X[valid_idx], beta, intercept)
        y_true = dm.y[valid_idx]
        weights = dm.weights[valid_idx]
        rows.append({
            "spec": cfg.spec,
            "season_type": cfg.season_type,
            "prior_mode": cfg.prior_mode,
            "method": method,
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "rmse": weighted_rmse(y_true, y_pred, weights),
            "mae": weighted_mae(y_true, y_pred, weights),
            **profile.as_dict(),
            "kept_possessions": dm.kept_rows,
            "dropped_garbage_time": dm.dropped_garbage_time,
        })

        cal = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "weight": weights})
        try:
            cal["pred_bin"] = pd.qcut(cal["y_pred"], q=min(10, cal["y_pred"].nunique()), duplicates="drop")
            grouped = cal.groupby("pred_bin", observed=True).apply(
                lambda g: pd.Series({
                    "n": len(g),
                    "pred_mean": np.average(g["y_pred"], weights=g["weight"]),
                    "actual_mean": np.average(g["y_true"], weights=g["weight"]),
                }),
                include_groups=False,
            ).reset_index()
            grouped["spec"] = cfg.spec
            grouped["method"] = method
            grouped["fold"] = fold
            calibration_rows.extend(grouped.to_dict("records"))
        except Exception:
            pass

    fold_scores = pd.DataFrame(rows)
    fold_scores["lambda_source_rows"] = len(lambda_scores)
    return fold_scores, pd.DataFrame(calibration_rows)


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    summary = (
        scores.groupby(["spec", "season_type", "prior_mode", "method"], as_index=False)
        .agg(
            folds=("fold", "count"),
            total_valid=("n_valid", "sum"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            lambda_off=("lambda_off", "mean"),
            lambda_def=("lambda_def", "mean"),
            lambda_meta=("lambda_meta", "mean"),
            lambda_season=("lambda_season", "mean"),
            dropped_garbage_time=("dropped_garbage_time", "max"),
        )
        .sort_values(["method", "rmse"])
    )
    baseline = summary.groupby("method")["rmse"].transform("first")
    summary["rmse_delta_vs_best"] = summary["rmse"] - baseline
    return summary


def playoff_prior_scores(raw_rows: list[tuple], cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    regular_cfg = replace(cfg, season_type="regular", prior_mode="zero", spec=f"{cfg.spec}_eval_regular_prior")
    reg_dm = build_design_matrix(raw_rows, regular_cfg)
    reg_fit = fit_model(reg_dm, regular_cfg)
    prior = coefficients_as_prior(reg_dm, reg_fit.beta)

    all_scores = []
    all_calibration = []
    for mode in ("zero", "offset", "dummy"):
        playoff_cfg = replace(
            cfg,
            season_type="playoff",
            prior_mode=mode,  # type: ignore[arg-type]
            spec=f"{cfg.spec}_eval_playoff_{mode}",
        )
        for method in ("game_grouped_cv", "chronological_retrodiction"):
            scores, cal = score_splits(raw_rows, playoff_cfg, prior if mode != "zero" else None, method)
            all_scores.append(scores)
            all_calibration.append(cal)
    return pd.concat(all_scores, ignore_index=True), pd.concat(all_calibration, ignore_index=True)


def run(args: argparse.Namespace) -> dict[str, str]:
    start = time.time()
    cfg = base_config(args, season_type="regular")
    raw_rows = fetch_possessions(cfg.seasons)
    print(f"Fetched {len(raw_rows):,} possessions for seasons {format_season_list(cfg.seasons)}")

    score_frames = []
    calibration_frames = []
    for name, spec_cfg in regular_spec_configs(cfg):
        print(f"\nEvaluating regular spec: {name}")
        for method in ("game_grouped_cv", "chronological_retrodiction"):
            scores, calibration = score_splits(raw_rows, spec_cfg, None, method)
            scores["short_spec"] = name
            score_frames.append(scores)
            calibration_frames.append(calibration)

    if args.include_playoff_prior:
        print("\nEvaluating playoff no-prior vs regular-season-prior variants")
        scores, calibration = playoff_prior_scores(raw_rows, cfg)
        score_frames.append(scores)
        calibration_frames.append(calibration)

    fold_scores = pd.concat(score_frames, ignore_index=True)
    summary = summarize(fold_scores)
    calibration = pd.concat(calibration_frames, ignore_index=True) if calibration_frames else pd.DataFrame()

    run_id = uuid.uuid4().hex[:8]
    stem = f"standard_rapm_eval_{args.spec}_{format_season_list(cfg.seasons)}_{run_id}"
    fold_path = STANDARD_RAPM_DIAGNOSTICS / f"{stem}_fold_scores.csv"
    summary_path = STANDARD_RAPM_DIAGNOSTICS / f"{stem}_summary.csv"
    calibration_path = STANDARD_RAPM_DIAGNOSTICS / f"{stem}_calibration.csv"
    meta_path = STANDARD_RAPM_DIAGNOSTICS / f"{stem}_run.json"

    fold_scores.to_csv(fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    with open(meta_path, "w") as handle:
        json.dump(
            {
                "run_id": run_id,
                "spec": args.spec,
                "seasons": cfg.seasons,
                "elapsed_seconds": time.time() - start,
                "include_playoff_prior": args.include_playoff_prior,
                "paths": {
                    "fold_scores": str(fold_path),
                    "summary": str(summary_path),
                    "calibration": str(calibration_path),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print("\nEvaluation summary:")
    print(summary.to_string(index=False))
    print(f"\nSummary -> {summary_path}")
    return {
        "fold_scores": str(fold_path),
        "summary": str(summary_path),
        "calibration": str(calibration_path),
        "run_meta": str(meta_path),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate standard RAPM model specs.")
    parser.add_argument("--window", default="3", help="Window length, or 'custom'.")
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--start-season", type=int, default=None)
    parser.add_argument("--spec", default="standard_rs_v1")
    parser.add_argument("--include-playoff-prior", action="store_true")
    parser.add_argument("--optimize-lambdas", action="store_true", default=False)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--lambda-search-iters", type=int, default=2)
    parser.add_argument("--lambda-off", type=float, default=3000.0)
    parser.add_argument("--lambda-def", type=float, default=3000.0)
    parser.add_argument("--lambda-meta", type=float, default=300.0)
    parser.add_argument("--lambda-season", type=float, default=100.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
