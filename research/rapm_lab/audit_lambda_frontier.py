"""Recheck GCV randomness and paired-game losses for a lambda frontier run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import (
    solve_stored_generalized_ridge,
    stored_evaluation_predictions,
    stored_training_diagnostics,
)

from run_lambda_frontier import _build_penalty, _candidate_id


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "research" / "experiments" / "rolling_5y_lambda_frontier_v1.json"
)
DEFAULT_MATRIX_RUN = (
    REPO_ROOT
    / "research"
    / "rapm_lab"
    / "outputs"
    / "rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77"
)


def _matrix_directories(matrix_run: Path) -> dict[int, Path]:
    catalog = pd.read_parquet(matrix_run / "lambda_matrix_catalog.parquet")
    complete = catalog.loc[catalog["evaluation_status"].eq("complete")].copy()
    complete["evaluation_season"] = complete["evaluation_season"].astype(int)
    return {
        int(row.evaluation_season): REPO_ROOT / Path(row.manifest).parent
        for row in complete.itertuples(index=False)
    }


def _unique_finalists(run: dict) -> dict[str, dict]:
    candidates: dict[str, dict] = {}
    for finalist in run["finalists"]:
        candidate = finalist["candidate"]
        candidates[_candidate_id(candidate)] = candidate
    return candidates


def _gcv_reproducibility(
    run: dict,
    contract: dict,
    matrix_dirs: dict[int, Path],
) -> pd.DataFrame:
    candidates = {
        finalist["candidate_id"]: finalist["candidate"]
        for finalist in run["finalists"]
        if finalist["label"] == "baseline" or finalist["label"].endswith("training_gcv")
    }
    eb_cache: dict[str, dict] = {}
    rows: list[dict] = []
    for candidate_id, candidate in candidates.items():
        for season in contract["selection_seasons"]:
            matrix_dir = matrix_dirs[int(season)]
            penalty, _ = _build_penalty(matrix_dir, candidate, contract, eb_cache)
            solution = solve_stored_generalized_ridge(matrix_dir, penalty)
            diagnostic = stored_training_diagnostics(
                matrix_dir,
                solution,
                penalty,
                probes=32,
                seed=20260927 + int(season),
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "family": candidate["family"],
                    "evaluation_season": int(season),
                    "gcv_recheck": float(diagnostic["gcv"]),
                    "effective_df_recheck": float(diagnostic["effective_df"]),
                }
            )
    frame = pd.DataFrame(rows)
    means = (
        frame.assign(log_gcv=np.log(frame["gcv_recheck"]))
        .groupby(["candidate_id", "family"], as_index=False)
        .agg(mean_log_gcv_recheck=("log_gcv", "mean"))
        .sort_values("mean_log_gcv_recheck")
    )
    means["gcv_recheck_rank"] = np.arange(1, len(means) + 1)
    return frame.merge(means, on=["candidate_id", "family"], validate="many_to_one")


def _paired_game_audit(
    run: dict,
    contract: dict,
    matrix_dirs: dict[int, Path],
    *,
    draws: int = 2000,
    seed: int = 20260928,
) -> pd.DataFrame:
    candidates = _unique_finalists(run)
    baseline = next(
        finalist["candidate"]
        for finalist in run["finalists"]
        if finalist["label"] == "baseline"
    )
    baseline_id = _candidate_id(baseline)
    eb_cache: dict[str, dict] = {}
    prediction_frames: dict[str, list[pd.DataFrame]] = {
        candidate_id: [] for candidate_id in candidates
    }
    for season in contract["reused_diagnostic_seasons"]:
        matrix_dir = matrix_dirs[int(season)]
        for candidate_id, candidate in candidates.items():
            penalty, _ = _build_penalty(matrix_dir, candidate, contract, eb_cache)
            solution = solve_stored_generalized_ridge(matrix_dir, penalty)
            predictions = stored_evaluation_predictions(
                matrix_dir, solution.beta, solution.intercept
            )
            predictions.insert(0, "season", int(season))
            prediction_frames[candidate_id].append(predictions)
    combined = {
        candidate_id: pd.concat(frames, ignore_index=True)
        for candidate_id, frames in prediction_frames.items()
    }
    baseline_frame = combined[baseline_id]
    rng = np.random.default_rng(seed)
    sampled_indices = {
        int(season): np.vstack(
            [
                rng.integers(0, int((baseline_frame["season"] == season).sum()), size=int((baseline_frame["season"] == season).sum()))
                for _ in range(draws)
            ]
        )
        for season in contract["reused_diagnostic_seasons"]
    }
    rows: list[dict] = []
    for candidate_id, candidate_frame in combined.items():
        joined = baseline_frame.merge(
            candidate_frame[["season", "game_id", "actual_margin", "predicted_margin"]],
            on=["season", "game_id", "actual_margin"],
            suffixes=("_baseline", "_candidate"),
            validate="one_to_one",
        )
        mse_differences = np.zeros(draws, dtype=np.float64)
        mae_differences = np.zeros(draws, dtype=np.float64)
        for season in contract["reused_diagnostic_seasons"]:
            season_frame = joined.loc[joined["season"].eq(season)].reset_index(drop=True)
            indices = sampled_indices[int(season)]
            actual = season_frame["actual_margin"].to_numpy()[indices]
            baseline_prediction = season_frame["predicted_margin_baseline"].to_numpy()[indices]
            candidate_prediction = season_frame["predicted_margin_candidate"].to_numpy()[indices]
            mse_differences += np.mean(
                (actual - baseline_prediction) ** 2
                - (actual - candidate_prediction) ** 2,
                axis=1,
            ) / len(contract["reused_diagnostic_seasons"])
            mae_differences += np.mean(
                np.abs(actual - baseline_prediction)
                - np.abs(actual - candidate_prediction),
                axis=1,
            ) / len(contract["reused_diagnostic_seasons"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": candidates[candidate_id]["family"],
                "games": int(len(joined)),
                "draws": draws,
                "mean_baseline_minus_candidate_mse": float(mse_differences.mean()),
                "mse_lower_95": float(np.quantile(mse_differences, 0.025)),
                "mse_upper_95": float(np.quantile(mse_differences, 0.975)),
                "probability_candidate_improves_mse": float(np.mean(mse_differences > 0)),
                "mean_baseline_minus_candidate_mae": float(mae_differences.mean()),
                "probability_candidate_improves_mae": float(np.mean(mae_differences > 0)),
            }
        )
    return pd.DataFrame(rows)


def audit(run_dir: Path, contract_path: Path, matrix_run: Path) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    contract = json.loads(contract_path.read_text())
    if run["season_2027_loaded"] or int(contract["untouched_confirmation_season"]) == 2027 and 2027 in run["selection_seasons"] + run["reused_diagnostic_seasons"]:
        raise ValueError("Season 2027 exposure detected.")
    matrix_dirs = _matrix_directories(matrix_run)
    gcv = _gcv_reproducibility(run, contract, matrix_dirs)
    paired = _paired_game_audit(run, contract, matrix_dirs)
    gcv.to_parquet(run_dir / "gcv_reproducibility_audit.parquet", index=False)
    paired.to_parquet(run_dir / "paired_game_audit.parquet", index=False)
    gcv_ranking = (
        gcv[["candidate_id", "family", "mean_log_gcv_recheck", "gcv_recheck_rank"]]
        .drop_duplicates()
        .sort_values("gcv_recheck_rank")
        .to_dict("records")
    )
    result = {
        "status": "complete",
        "frontier_run_id": run["run_id"],
        "gcv_recheck_probes": 32,
        "gcv_ranking": gcv_ranking,
        "paired_game_rows": paired.to_dict("records"),
        "season_2027_loaded": False,
    }
    write_json_atomic(result, run_dir / "audit.json")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--matrix-run", type=Path, default=DEFAULT_MATRIX_RUN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(
        args.run_dir.resolve(), args.contract.resolve(), args.matrix_run.resolve()
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
