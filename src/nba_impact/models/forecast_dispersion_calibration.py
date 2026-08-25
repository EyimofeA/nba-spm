"""Predeclared dispersion calibration (experiment forecast_dispersion_calibration_v1).

Adds exposure-aware predictive standard deviations on top of the promoted
backbone combination without changing any point prediction. Variance model:
sd_i^2 = max(a_side + b_side / n_i, floor), parameters fit once per side on
selection-fold squared residuals against 1/n via weighted least squares.
Confirmation seasons are scored exactly once with frozen parameters.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

PARAMETER_FIT_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
CONFIRMATION_SEASONS = (2025, 2026)
DISPERSION_RANGE = (0.85, 1.15)
COVERAGE_TOLERANCE = 0.05
NOMINAL_COVERAGE_68 = 0.6826894921370859
VARIANCE_FLOOR = 1e-6
MINIMUM_SIDE_POSSESSIONS = 1000.0

_REQUIRED = {
    "PLAYER_ID",
    "Target_Season",
    "panel_target_offense",
    "panel_target_defense",
    "panel_target_net",
    "raw_offense",
    "raw_defense",
    "state_space_offense",
    "state_space_defense",
    "Poss_Off",
    "Poss_Def",
}

_SIDES = ("offense", "defense", "net")


def _fit_variance_parameters(
    residuals: np.ndarray, exposures: np.ndarray, weights: np.ndarray
) -> tuple[float, float]:
    """Weighted least squares of squared residuals on 1/n, clipped at zero."""
    design = np.column_stack([np.ones_like(exposures), 1.0 / exposures])
    scale = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(design * scale[:, None], (residuals**2) * scale, rcond=None)
    intercept, slope = float(max(coefficients[0], 0.0)), float(max(coefficients[1], 0.0))
    return intercept, slope


def _weighted_second_moment(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float(np.sqrt(np.sum(weights * values**2) / total)) if total > 0 else float("nan")


def _gate_summary(
    frame: pd.DataFrame, side_forecast: str, side_target: str, side_sd: str
) -> dict:
    residual = (frame[side_forecast] - frame[side_target]).to_numpy()
    sd = frame[side_sd].to_numpy()
    weights = frame["weight"].to_numpy()
    rmse = _weighted_second_moment(residual, weights)
    spread = _weighted_second_moment(sd, weights)
    dispersion = rmse / spread if spread > 0 else float("nan")
    coverage = float(np.mean(np.abs(residual) <= sd))
    return {
        "players": int(len(frame)),
        "weighted_rmse": rmse,
        "weighted_mean_sd": spread,
        "dispersion": dispersion,
        "dispersion_pass": bool(DISPERSION_RANGE[0] <= dispersion <= DISPERSION_RANGE[1]),
        "coverage_68": coverage,
        "coverage_pass": bool(abs(coverage - NOMINAL_COVERAGE_68) <= COVERAGE_TOLERANCE),
    }


def build_forecast_dispersion_calibration(
    scored_rows_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    """Fit the frozen variance model on selection folds, score confirmation once."""
    rows = pd.read_parquet(scored_rows_path)
    if missing := sorted(_REQUIRED - set(rows.columns)):
        raise ValueError(f"Backbone scored rows are missing columns: {missing}")
    rows = rows.loc[
        np.minimum(rows["Poss_Off"], rows["Poss_Def"]).ge(MINIMUM_SIDE_POSSESSIONS)
    ].copy()
    if rows.empty:
        raise ValueError("Calibration received zero primary-population rows.")

    rows["forecast_offense"] = 0.5 * (rows["raw_offense"] + rows["state_space_offense"])
    rows["forecast_defense"] = 0.5 * (rows["raw_defense"] + rows["state_space_defense"])
    rows["forecast_net"] = rows["forecast_offense"] + rows["forecast_defense"]

    fit_mask = rows["Target_Season"].isin(PARAMETER_FIT_SEASONS)
    fit_frame = rows.loc[fit_mask]
    parameters: dict[str, dict[str, float]] = {}
    side_columns = {
        "offense": ("forecast_offense", "panel_target_offense", "Poss_Off"),
        "defense": ("forecast_defense", "panel_target_defense", "Poss_Def"),
    }
    for side, (forecast, target, possession) in side_columns.items():
        residual = (fit_frame[forecast] - fit_frame[target]).to_numpy()
        intercept, slope = _fit_variance_parameters(
            residual,
            fit_frame[possession].to_numpy(),
            fit_frame["weight"].to_numpy(),
        )
        parameters[side] = {"intercept": intercept, "slope": slope}
    for side in ("offense", "defense"):
        possession = "Poss_Off" if side == "offense" else "Poss_Def"
        variance = np.maximum(
            parameters[side]["intercept"] + parameters[side]["slope"] / rows[possession].to_numpy(),
            VARIANCE_FLOOR,
        )
        rows[f"sd_{side}"] = np.sqrt(variance)
    rows["sd_net"] = np.sqrt(rows["sd_offense"] ** 2 + rows["sd_defense"] ** 2)

    bound_columns: list[str] = []
    for z in (1.0, 1.96):
        tag = int(z * 100)
        for side in _SIDES:
            rows[f"{side}_lo{tag}"] = rows[f"forecast_{side}"] - z * rows[f"sd_{side}"]
            rows[f"{side}_hi{tag}"] = rows[f"forecast_{side}"] + z * rows[f"sd_{side}"]
            bound_columns.extend([f"{side}_lo{tag}", f"{side}_hi{tag}"])

    confirmation = rows.loc[rows["Target_Season"].isin(CONFIRMATION_SEASONS)]
    gates = {
        side: _gate_summary(
            confirmation,
            f"forecast_{side}",
            f"panel_target_{side}",
            f"sd_{side}",
        )
        for side in _SIDES
    }
    diagnostics = {
        str(int(season)): {
            side: _gate_summary(
                rows.loc[rows["Target_Season"].eq(season)],
                f"forecast_{side}",
                f"panel_target_{side}",
                f"sd_{side}",
            )
            for side in _SIDES
        }
        for season in sorted(rows.loc[fit_mask, "Target_Season"].unique())
    }
    all_pass = all(item["dispersion_pass"] and item["coverage_pass"] for item in gates.values())
    decision = "intervals_promoted" if all_pass else "intervals_blocked_" + "_".join(
        side for side, item in gates.items() if not (item["dispersion_pass"] and item["coverage_pass"])
    )

    input_hash = sha256_file(scored_rows_path)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = hashlib.sha256(f"{input_hash}|{source_hash}".encode()).hexdigest()[:10]
    run_id = f"forecast_dispersion_calibration_v1_{identity}"
    output = Path(artifact_root) / "models" / "forecast_dispersion_calibration" / run_id
    output.mkdir(parents=True, exist_ok=False)

    columns = [
        "PLAYER_ID",
        "Target_Season",
        "weight",
        "forecast_offense",
        "forecast_defense",
        "forecast_net",
        "panel_target_offense",
        "panel_target_defense",
        "panel_target_net",
        *[f"sd_{side}" for side in _SIDES],
    ] + bound_columns
    rows[columns].to_parquet(output / "intervals.parquet", index=False)

    run = {
        "run_id": run_id,
        "model_family": "next_season_backbone_dispersion_calibration",
        "estimand": "single_regular_season_normal_rapm_offense_defense_and_net_points_per_100",
        "status": "research_predeclared_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_snapshot_id": None,
        "artifact_path": str(output),
        "inputs": {"scored_rows": input_hash, "source_code": source_hash},
        "config": {
            "experiment_id": "forecast_dispersion_calibration_v1",
            "variance_model": "sd^2 = max(intercept + slope / side_possessions, floor)",
            "parameter_fit_seasons": list(PARAMETER_FIT_SEASONS),
            "confirmation_seasons": list(CONFIRMATION_SEASONS),
            "dispersion_range": list(DISPERSION_RANGE),
            "coverage_tolerance": COVERAGE_TOLERANCE,
            "variance_floor": VARIANCE_FLOOR,
            "net_combination": "independent sides",
            "population_rule": "minimum side possessions >= 1000",
        },
        "metrics": {
            "parameters": parameters,
            "confirmation_gates": gates,
            "in_sample_diagnostics": diagnostics,
            "decision": decision,
        },
    }
    write_json_atomic(run, output / "run.json")
    return run
