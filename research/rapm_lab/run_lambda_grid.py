"""Select rolling five-year RAPM penalties from stored sufficient statistics."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import (
    score_stored_evaluation,
    solve_stored_ridge,
    stored_evaluation_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "research" / "experiments" / "rolling_5y_lambda_grid_v1.json"
)
DEFAULT_MATRIX_RUN = (
    REPO_ROOT
    / "research"
    / "rapm_lab"
    / "outputs"
    / "rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77"
)


def _parameter_grid(contract: dict) -> list[dict[str, float]]:
    grid = contract["grid"]
    return [
        {
            "lambda_off": float(lambda_off),
            "lambda_def": float(lambda_def),
            "lambda_home": float(lambda_home),
        }
        for lambda_off, lambda_def, lambda_home in itertools.product(
            grid["lambda_off"], grid["lambda_def"], grid["lambda_home"]
        )
    ]


def _config_id(config: dict[str, float]) -> str:
    return (
        f"off_{config['lambda_off']:g}__def_{config['lambda_def']:g}"
        f"__home_{config['lambda_home']:g}"
    )


def _equal_season_summary(folds: pd.DataFrame, seasons: list[int]) -> dict[str, float]:
    scoped = folds.loc[folds["evaluation_season"].isin(seasons)]
    if sorted(scoped["evaluation_season"].astype(int).tolist()) != sorted(seasons):
        raise ValueError(f"Incomplete evaluation scope: expected {seasons}")
    return {
        "mean_correlation": float(scoped["margin_correlation"].mean()),
        "mean_mae": float(scoped["margin_mae"].mean()),
        "mean_rmse": float(scoped["margin_rmse"].mean()),
    }


def select_candidate(
    summaries: pd.DataFrame,
    *,
    correlation_tolerance: float,
) -> pd.Series:
    """Apply the preregistered correlation-first, MAE tie-break rule."""
    best_correlation = float(summaries["selection_mean_correlation"].max())
    eligible = summaries.loc[
        summaries["selection_mean_correlation"] >= best_correlation - correlation_tolerance
    ]
    return eligible.sort_values(
        ["selection_mean_mae", "lambda_off", "lambda_def", "lambda_home"],
        kind="stable",
    ).iloc[0]


def paired_bootstrap_mse_improvement(
    predictions: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Bootstrap baseline-minus-candidate MSE by whole game within season."""
    required = {"season", "actual_margin", "baseline_prediction", "candidate_prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Bootstrap predictions missing columns: {missing}")
    rng = np.random.default_rng(seed)
    season_frames = [frame.reset_index(drop=True) for _, frame in predictions.groupby("season")]
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        baseline_squared: list[np.ndarray] = []
        candidate_squared: list[np.ndarray] = []
        for frame in season_frames:
            indices = rng.integers(0, len(frame), size=len(frame))
            actual = frame["actual_margin"].to_numpy()[indices]
            baseline = frame["baseline_prediction"].to_numpy()[indices]
            candidate = frame["candidate_prediction"].to_numpy()[indices]
            baseline_squared.append((actual - baseline) ** 2)
            candidate_squared.append((actual - candidate) ** 2)
        values[draw] = float(
            np.mean(np.concatenate(baseline_squared))
            - np.mean(np.concatenate(candidate_squared))
        )
    frame = pd.DataFrame(
        {
            "draw": np.arange(draws, dtype=np.int64),
            "baseline_minus_candidate_mse": values,
        }
    )
    summary: dict[str, float | int] = {
        "draws": draws,
        "seed": seed,
        "mean_mse_improvement": float(values.mean()),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "probability_mse_improvement": float(np.mean(values > 0.0)),
    }
    return frame, summary


def run_grid(contract_path: Path, matrix_run: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Lambda research requires a frozen experiment contract.")
    if matrix_run.name != contract["matrix_run_id"]:
        raise ValueError("Matrix run does not match the frozen contract.")
    catalog = pd.read_parquet(matrix_run / "lambda_matrix_catalog.parquet")
    catalog = catalog.loc[catalog["evaluation_status"].eq("complete")].copy()
    catalog["evaluation_season"] = catalog["evaluation_season"].astype(int)
    expected_seasons = sorted(
        contract["selection_seasons"] + contract["reused_diagnostic_seasons"]
    )
    observed_seasons = sorted(catalog["evaluation_season"].tolist())
    if observed_seasons != expected_seasons:
        raise ValueError(
            f"Stored evaluation seasons {observed_seasons} do not match {expected_seasons}."
        )
    if int(contract["untouched_confirmation_season"]) in observed_seasons:
        raise ValueError("Untouched confirmation season appeared in the matrix catalog.")

    identity_payload = {
        "contract_hash": sha256_file(contract_path),
        "script_hash": sha256_file(Path(__file__)),
        "matrix_run_id": matrix_run.name,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = matrix_run / f"lambda_grid_v1_{identity}"
    completed_path = output / "run.json"
    if completed_path.exists():
        return json.loads(completed_path.read_text())
    checkpoints = output / "config_checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    matrix_dirs = {
        int(row.evaluation_season): REPO_ROOT / Path(row.manifest).parent
        for row in catalog.itertuples(index=False)
    }
    configs = _parameter_grid(contract)
    fold_frames: list[pd.DataFrame] = []
    for index, config in enumerate(configs, start=1):
        config_id = _config_id(config)
        checkpoint = checkpoints / f"{config_id}.parquet"
        if checkpoint.exists():
            folds = pd.read_parquet(checkpoint)
        else:
            rows: list[dict] = []
            for season in expected_seasons:
                matrix_dir = matrix_dirs[season]
                beta, intercept, _ = solve_stored_ridge(matrix_dir, **config)
                metrics = score_stored_evaluation(matrix_dir, beta, intercept)
                rows.append(
                    {
                        "config_id": config_id,
                        **config,
                        "evaluation_season": season,
                        **metrics,
                    }
                )
            folds = pd.DataFrame(rows)
            folds.to_parquet(checkpoint, index=False)
        fold_frames.append(folds)
        if index == 1 or index % 10 == 0 or index == len(configs):
            print(f"grid {index}/{len(configs)}: {config_id}", flush=True)

    fold_results = pd.concat(fold_frames, ignore_index=True)
    fold_results.to_parquet(output / "fold_results.parquet", index=False)
    summary_rows: list[dict] = []
    for config_id, folds in fold_results.groupby("config_id", sort=False):
        first = folds.iloc[0]
        selection = _equal_season_summary(folds, contract["selection_seasons"])
        diagnostic = _equal_season_summary(
            folds, contract["reused_diagnostic_seasons"]
        )
        summary_rows.append(
            {
                "config_id": config_id,
                "lambda_off": float(first["lambda_off"]),
                "lambda_def": float(first["lambda_def"]),
                "lambda_home": float(first["lambda_home"]),
                **{f"selection_{key}": value for key, value in selection.items()},
                **{f"diagnostic_{key}": value for key, value in diagnostic.items()},
            }
        )
    summaries = pd.DataFrame(summary_rows)
    candidate = select_candidate(
        summaries,
        correlation_tolerance=float(
            contract["selection"]["tie_tolerance_correlation"]
        ),
    )
    baseline_config = {key: float(value) for key, value in contract["baseline"].items()}
    baseline_id = _config_id(baseline_config)
    baseline = summaries.loc[summaries["config_id"].eq(baseline_id)].iloc[0]
    summaries["selected_candidate"] = summaries["config_id"].eq(candidate["config_id"])
    summaries["baseline"] = summaries["config_id"].eq(baseline_id)
    summaries.sort_values(
        ["selection_mean_correlation", "selection_mean_mae"],
        ascending=[False, True],
    ).to_parquet(output / "config_summary.parquet", index=False)

    diagnostic_predictions: list[pd.DataFrame] = []
    candidate_config = {
        key: float(candidate[key]) for key in ("lambda_off", "lambda_def", "lambda_home")
    }
    for season in contract["reused_diagnostic_seasons"]:
        matrix_dir = matrix_dirs[int(season)]
        baseline_beta, baseline_intercept, _ = solve_stored_ridge(
            matrix_dir, **baseline_config
        )
        candidate_beta, candidate_intercept, _ = solve_stored_ridge(
            matrix_dir, **candidate_config
        )
        baseline_predictions = stored_evaluation_predictions(
            matrix_dir, baseline_beta, baseline_intercept
        ).rename(columns={"predicted_margin": "baseline_prediction"})
        candidate_predictions = stored_evaluation_predictions(
            matrix_dir, candidate_beta, candidate_intercept
        ).rename(columns={"predicted_margin": "candidate_prediction"})
        joined = baseline_predictions.merge(
            candidate_predictions[
                ["game_id", "actual_margin", "candidate_prediction"]
            ],
            on=["game_id", "actual_margin"],
            how="inner",
            validate="one_to_one",
        )
        joined.insert(0, "season", int(season))
        diagnostic_predictions.append(joined)
    prediction_frame = pd.concat(diagnostic_predictions, ignore_index=True)
    prediction_frame.to_parquet(output / "diagnostic_game_predictions.parquet", index=False)
    gate = contract["diagnostic_gate"]
    bootstrap_draws, bootstrap_summary = paired_bootstrap_mse_improvement(
        prediction_frame,
        draws=int(gate["paired_bootstrap_draws"]),
        seed=int(gate["paired_bootstrap_seed"]),
    )
    bootstrap_draws.to_parquet(output / "diagnostic_bootstrap_draws.parquet", index=False)

    comparisons = {
        "selection_correlation_gain": float(
            candidate["selection_mean_correlation"]
            - baseline["selection_mean_correlation"]
        ),
        "selection_mae_change": float(
            candidate["selection_mean_mae"] - baseline["selection_mean_mae"]
        ),
        "diagnostic_correlation_gain": float(
            candidate["diagnostic_mean_correlation"]
            - baseline["diagnostic_mean_correlation"]
        ),
        "diagnostic_mae_change": float(
            candidate["diagnostic_mean_mae"] - baseline["diagnostic_mean_mae"]
        ),
    }
    checks = {
        "selection_correlation": comparisons["selection_correlation_gain"]
        >= float(gate["minimum_selection_correlation_gain"]),
        "diagnostic_correlation": comparisons["diagnostic_correlation_gain"]
        >= float(gate["minimum_diagnostic_correlation_gain"]),
        "diagnostic_mae": comparisons["diagnostic_mae_change"]
        <= float(gate["maximum_diagnostic_mae_increase"]),
        "diagnostic_bootstrap": bootstrap_summary["probability_mse_improvement"]
        >= float(gate["required_mse_improvement_probability"]),
    }
    passed = bool(all(checks.values()))
    metric_keys = (
        "selection_mean_correlation",
        "selection_mean_mae",
        "selection_mean_rmse",
        "diagnostic_mean_correlation",
        "diagnostic_mean_mae",
        "diagnostic_mean_rmse",
    )
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_challenger" if passed else "research_null",
        "contract": str(contract_path.relative_to(REPO_ROOT)),
        "contract_hash": identity_payload["contract_hash"],
        "script_hash": identity_payload["script_hash"],
        "matrix_run_id": matrix_run.name,
        "grid_configurations": len(configs),
        "scored_folds": int(len(fold_results)),
        "selection_seasons": contract["selection_seasons"],
        "reused_diagnostic_seasons": contract["reused_diagnostic_seasons"],
        "untouched_confirmation_season": contract["untouched_confirmation_season"],
        "candidate": {
            "config_id": str(candidate["config_id"]),
            **candidate_config,
            **{key: float(candidate[key]) for key in metric_keys},
        },
        "baseline": {
            "config_id": baseline_id,
            **baseline_config,
            **{key: float(baseline[key]) for key in metric_keys},
        },
        "comparison": comparisons,
        "diagnostic_bootstrap": bootstrap_summary,
        "gate_checks": checks,
        "gate_passed": passed,
        "artifacts": {
            "fold_results": "fold_results.parquet",
            "config_summary": "config_summary.parquet",
            "diagnostic_game_predictions": "diagnostic_game_predictions.parquet",
            "diagnostic_bootstrap_draws": "diagnostic_bootstrap_draws.parquet",
        },
        "season_2027_loaded": False,
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, completed_path)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--matrix-run", type=Path, default=DEFAULT_MATRIX_RUN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_grid(args.contract.resolve(), args.matrix_run.resolve())
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
