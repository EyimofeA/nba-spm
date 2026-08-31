#!/usr/bin/env python3
"""Separate AIO prior mean calibration from prior precision."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import diags
from scipy.sparse.linalg import cg, spsolve

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
try:
    from run_aio_prior_canonical_followup import _center, _remap_annual
    from run_full_spm_history_ablation import _annual_bundles
except ModuleNotFoundError:  # Imported as research.run_* by tests.
    from research.run_aio_prior_canonical_followup import _center, _remap_annual
    from research.run_full_spm_history_ablation import _annual_bundles


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "aio_prior_calibration_precision_v1"
CONTRACT = ROOT / "research/experiments/aio_prior_calibration_precision_v1.yml"
SOURCE_RUN = ROOT / (
    "artifacts/research/compact_spm_comparison/"
    "compact_spm_comparison_v1_2a0f8a6f31"
)
TARGETS = ROOT / (
    "artifacts/models/five_year_target_spm/"
    "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
MATRIX_ROOT = ROOT / (
    "research/rapm_lab/outputs/rolling_5y_2014_2026/"
    "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
CANDIDATES = ("box_pipm", "full_spm", "compact_spm")
RATING_SEASONS = tuple(range(2021, 2026))
PENALTIES = (1500.0, 3000.0, 4500.0, 6000.0)


def select_configuration(history: pd.DataFrame, parameter_columns: list[str]) -> dict:
    """Select the lowest equal-season MSE with a deterministic conservative tie-break."""
    grouped = (
        history.groupby(parameter_columns + ["test_season"], as_index=False)
        .agg(mse=("squared_error", "mean"))
        .groupby(parameter_columns, as_index=False)
        .agg(equal_season_mse=("mse", "mean"))
    )
    if grouped.empty:
        raise ValueError("Calibration selection needs at least one earlier fold.")
    grouped["distance_from_unit"] = sum(
        (grouped[column] - 3000.0).abs() for column in parameter_columns
    )
    winner = grouped.sort_values(
        ["equal_season_mse", "distance_from_unit", *parameter_columns]
    ).iloc[0]
    return {column: float(winner[column]) for column in parameter_columns}


def fit_prior_affine(
    history: pd.DataFrame, current: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Calibrate each prior side on earlier out-of-fold player targets."""
    output = current.copy()
    parameters: dict[str, float] = {}
    for side in ("offense", "defense"):
        prediction = history[f"prior_{side}_per_100"].to_numpy(dtype=float)
        target = history[f"target_{side}"].to_numpy(dtype=float)
        weights = history["sample_weight"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(history)), prediction])
        weighted = design * np.sqrt(weights)[:, None]
        fitted = np.linalg.lstsq(weighted, target * np.sqrt(weights), rcond=None)[0]
        intercept, slope = map(float, fitted)
        output[f"prior_{side}_per_100"] = (
            intercept + slope * output[f"prior_{side}_per_100"]
        )
        parameters[f"prior_{side}_intercept"] = intercept
        parameters[f"prior_{side}_slope"] = slope
    output["prior_net_per_100"] = (
        output["prior_offense_per_100"] + output["prior_defense_per_100"]
    )
    return output, parameters


def _solve(
    annual,
    center: np.ndarray,
    *,
    offense_penalty: float = 3000.0,
    defense_penalty: float = 3000.0,
) -> tuple[np.ndarray, float]:
    n = len(annual.players)
    penalty = np.concatenate(
        [
            np.full(n, offense_penalty),
            np.full(n, defense_penalty),
            np.asarray([300.0]),
        ]
    )
    lhs = annual.xtx + diags(penalty, format="csr")
    rhs = annual.xty_centered + penalty * center
    try:
        raw, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        raw, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        raw = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(raw)
    off_mean = float(np.average(beta[:n], weights=annual.off_possessions))
    def_mean = float(np.average(beta[n : 2 * n], weights=annual.def_possessions))
    beta[:n] -= off_mean
    beta[n : 2 * n] -= def_mean
    intercept = annual.base_intercept + 5.0 * (off_mean + def_mean)
    return beta, intercept


def _predictions(
    matrix_dir: Path,
    annual,
    center: np.ndarray,
    *,
    offense_penalty: float = 3000.0,
    defense_penalty: float = 3000.0,
) -> pd.DataFrame:
    beta, intercept = _solve(
        annual,
        center,
        offense_penalty=offense_penalty,
        defense_penalty=defense_penalty,
    )
    frame = stored_evaluation_predictions(matrix_dir, beta, intercept)
    frame["squared_error"] = (
        frame["actual_margin"] - frame["predicted_margin"]
    ) ** 2
    return frame


def _affine(history: pd.DataFrame, current: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts = history.groupby("test_season")["game_id"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(history)), history["predicted_margin"].to_numpy(dtype=float)]
    )
    weighted = design * np.sqrt(weights)[:, None]
    target = history["actual_margin"].to_numpy(dtype=float) * np.sqrt(weights)
    intercept, slope = np.linalg.lstsq(weighted, target, rcond=None)[0]
    output = current.copy()
    output["predicted_margin"] = intercept + slope * output["predicted_margin"]
    output["squared_error"] = (
        output["actual_margin"] - output["predicted_margin"]
    ) ** 2
    return output, {"affine_intercept": float(intercept), "affine_slope": float(slope)}


def _metrics(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (candidate, arm, test_season), frame in games.groupby(
        ["candidate", "arm", "test_season"], sort=True
    ):
        actual = frame["actual_margin"].to_numpy(dtype=float)
        predicted = frame["predicted_margin"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(predicted, actual, 1)
        rows.append(
            {
                "candidate": candidate,
                "arm": arm,
                "test_season": int(test_season),
                "games": len(frame),
                "mse": float(np.mean((actual - predicted) ** 2)),
                "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
                "calibration_slope": float(slope),
                "calibration_intercept": float(intercept),
            }
        )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby(["candidate", "arm"], as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            equal_season_mse=("mse", "mean"),
            mean_correlation=("correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            mean_calibration_intercept=("calibration_intercept", "mean"),
        )
        .sort_values("equal_season_mse")
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    return folds, summary


def _bootstrap(games: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for candidate in CANDIDATES:
        unit = games.loc[
            games["candidate"].eq(candidate) & games["arm"].eq("unit")
        ]
        for arm in (
            "prior_affine_calibrated",
            "precision_calibrated",
            "prior_affine_and_precision",
            "game_affine_diagnostic",
        ):
            challenger = games.loc[
                games["candidate"].eq(candidate) & games["arm"].eq(arm)
            ]
            deltas = []
            point = []
            season_arrays = []
            for season in sorted(unit["test_season"].unique()):
                left = unit.loc[unit["test_season"].eq(season)].set_index("game_id")
                right = challenger.loc[
                    challenger["test_season"].eq(season)
                ].set_index("game_id")
                if not left.index.equals(right.index):
                    raise ValueError("Calibration arms must score identical games.")
                values = np.column_stack(
                    [left["squared_error"].to_numpy(), right["squared_error"].to_numpy()]
                )
                season_arrays.append(values)
                point.append(float(np.mean(values[:, 0] - values[:, 1])))
            for _ in range(draws):
                draw = []
                for values in season_arrays:
                    sampled = values[rng.integers(0, len(values), len(values))]
                    draw.append(float(np.mean(sampled[:, 0] - sampled[:, 1])))
                deltas.append(float(np.mean(draw)))
            low, high = np.quantile(deltas, [0.025, 0.975])
            rows.append(
                {
                    "candidate": candidate,
                    "reference_arm": "unit",
                    "candidate_arm": arm,
                    "mean_mse_improvement": float(np.mean(point)),
                    "bootstrap_95_low": float(low),
                    "bootstrap_95_high": float(high),
                    "probability_candidate_lower_mse": float(np.mean(np.asarray(deltas) > 0)),
                    "draws": draws,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Experiment contract does not match the runner.")
    priors = pd.read_parquet(SOURCE_RUN / "priors.parquet")
    priors = priors.loc[priors["candidate"].isin(CANDIDATES)].copy()
    targets = pd.read_parquet(TARGETS)
    target_columns = [
        "PLAYER_ID",
        "Window_End",
        "target_offense",
        "target_defense",
        "Poss_Off",
        "Poss_Def",
    ]
    calibration = priors.merge(
        targets[target_columns], on=["PLAYER_ID", "Window_End"], how="inner"
    )
    calibration["sample_weight"] = np.sqrt(
        calibration[["Poss_Off", "Poss_Def"]].min(axis=1)
    )
    annual, reconstruction = _annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)

    grid_rows = []
    centers = {}
    annual_by_season = {}
    for rating_season in RATING_SEASONS:
        matrix_dir = MATRIX_ROOT / f"5y_end_{rating_season}"
        players = np.load(matrix_dir / "player_ids.npy")
        bundle = _remap_annual(annual[rating_season], players)
        annual_by_season[rating_season] = bundle
        for candidate in CANDIDATES:
            prior = priors.loc[
                priors["candidate"].eq(candidate)
                & priors["Window_End"].eq(rating_season)
            ]
            center, _ = _center(prior, bundle)
            centers[(rating_season, candidate)] = center
            for offense_penalty, defense_penalty in itertools.product(PENALTIES, repeat=2):
                frame = _predictions(
                    matrix_dir,
                    bundle,
                    center,
                    offense_penalty=offense_penalty,
                    defense_penalty=defense_penalty,
                )
                frame = frame.assign(
                    candidate=candidate,
                    rating_season=rating_season,
                    test_season=rating_season + 1,
                    offense_penalty=offense_penalty,
                    defense_penalty=defense_penalty,
                )
                grid_rows.append(frame)
    grid = pd.concat(grid_rows, ignore_index=True)

    selections = []
    scored = []
    for rating_season in contract["evaluation"]["scored_rating_seasons"]:
        matrix_dir = MATRIX_ROOT / f"5y_end_{rating_season}"
        for candidate in CANDIDATES:
            history = grid.loc[
                grid["candidate"].eq(candidate)
                & grid["rating_season"].lt(rating_season)
            ]
            precision_choice = select_configuration(
                history,
                ["offense_penalty", "defense_penalty"],
            )
            prior_history = calibration.loc[
                calibration["candidate"].eq(candidate)
                & calibration["Window_End"].lt(rating_season)
            ]
            current_prior = priors.loc[
                priors["candidate"].eq(candidate)
                & priors["Window_End"].eq(rating_season)
            ]
            affine_prior, affine_parameters = fit_prior_affine(
                prior_history, current_prior
            )
            affine_center, _ = _center(
                affine_prior, annual_by_season[rating_season]
            )
            choices = {
                "unit": (centers[(rating_season, candidate)], {}),
                "prior_affine_calibrated": (affine_center, {}),
                "precision_calibrated": (
                    centers[(rating_season, candidate)],
                    precision_choice,
                ),
                "prior_affine_and_precision": (affine_center, precision_choice),
            }
            current_unit = None
            for arm, (arm_center, parameters) in choices.items():
                frame = _predictions(
                    matrix_dir,
                    annual_by_season[rating_season],
                    arm_center,
                    **parameters,
                ).assign(
                    candidate=candidate,
                    arm=arm,
                    rating_season=rating_season,
                    test_season=rating_season + 1,
                )
                scored.append(frame)
                if arm == "unit":
                    current_unit = frame
                selections.append(
                    {
                        "candidate": candidate,
                        "rating_season": rating_season,
                        "arm": arm,
                        "selection_rating_seasons": ",".join(
                            map(str, sorted(history["rating_season"].unique()))
                        ),
                        "offense_penalty": parameters.get("offense_penalty", 3000.0),
                        "defense_penalty": parameters.get("defense_penalty", 3000.0),
                        **(
                            affine_parameters
                            if arm in {"prior_affine_calibrated", "prior_affine_and_precision"}
                            else {}
                        ),
                    }
                )
            unit_history = history.loc[
                history["offense_penalty"].eq(3000.0)
                & history["defense_penalty"].eq(3000.0)
            ]
            affine, affine_parameters = _affine(unit_history, current_unit)
            affine["arm"] = "game_affine_diagnostic"
            scored.append(affine)
            selections.append(
                {
                    "candidate": candidate,
                    "rating_season": rating_season,
                    "arm": "game_affine_diagnostic",
                    "selection_rating_seasons": ",".join(
                        map(str, sorted(history["rating_season"].unique()))
                    ),
                    **affine_parameters,
                }
            )

    games = pd.concat(scored, ignore_index=True).sort_values(
        ["candidate", "arm", "test_season", "game_id"]
    )
    folds, summary = _metrics(games)
    intervals = _bootstrap(
        games,
        draws=int(contract["evaluation"]["uncertainty"]["draws"]),
        seed=int(contract["evaluation"]["uncertainty"]["seed"]),
    )

    source_paths = {
        "contract": CONTRACT,
        "runner": Path(__file__),
        "source_manifest": SOURCE_RUN / "run.json",
        "source_priors": SOURCE_RUN / "priors.parquet",
        "targets": TARGETS,
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "candidates": list(CANDIDATES),
        "rating_seasons": list(RATING_SEASONS),
        "scored_rating_seasons": contract["evaluation"]["scored_rating_seasons"],
        "player_penalty_grid": list(PENALTIES),
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / "artifacts/research/aio_prior_calibration_precision" / (
        f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "selections.parquet": pd.DataFrame(selections),
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "paired_bootstrap.parquet": intervals,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "identical_games_within_candidate": True,
            "season_2027_loaded": False,
        },
        "files": {},
        "forbidden_interpretation": "Reused diagnostic folds cannot promote a public model.",
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "rows": len(frame),
            "sha256": sha256_file(output / name),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print("\nUnit-reference improvements")
    print(intervals.to_string(index=False))


if __name__ == "__main__":
    main()
