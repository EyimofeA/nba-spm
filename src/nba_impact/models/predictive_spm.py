"""Predeclared next-season predictive SPM (experiment predictive_spm_v1).

Trains the frozen annual_spm_v1 learners on consecutive-season pairs
(features at season s, single-season normal RAPM targets at season s+1) and
forecasts season T using only target seasons strictly before T. Compared on
identical rows against last-season RAPM persistence.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.single_season_spm import _selected_single_season_features
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_model_comparison import _fit_model

SIDES = ("offense", "defense", "net")
EXPECTED_SCHEMA_VERSION = "experiment_preregistration_v1"
EXPECTED_EXPERIMENT_ID = "predictive_spm_v1"


def _artifact_run_id(path: str | Path) -> str:
    """Return the declared run ID for an artifact file or directory."""
    candidate = Path(path)
    artifact = candidate if candidate.is_dir() else candidate.parent
    manifest = artifact / "run.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"Artifact manifest {manifest} has no run_id.")
        return run_id
    return artifact.name


def _load_contract(
    contract_path: str | Path,
    output_seasons: tuple[int, ...],
) -> dict:
    """Parse and enforce the frozen predictive-SPM contract before data access."""
    payload = yaml.safe_load(Path(contract_path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Predictive SPM contract must be a mapping.")
    expected = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "experiment_id": EXPECTED_EXPERIMENT_ID,
        "status": "preregistered",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"Predictive SPM contract {field} must be {value!r}; "
                f"found {payload.get(field)!r}."
            )
    if not payload.get("estimand_id"):
        raise ValueError("Predictive SPM contract requires estimand_id.")
    data_contract = payload.get("data_contract")
    if not isinstance(data_contract, dict):
        raise ValueError("Predictive SPM contract requires data_contract.")
    required = {
        "features",
        "targets",
        "feature_reference_run",
        "diagnostic_folds",
        "confirmation_folds",
        "untouched_confirmation_seasons",
    }
    if missing := sorted(required - set(data_contract)):
        raise ValueError(f"Predictive SPM data contract is missing {missing}.")

    untouched = {int(value) for value in data_contract["untouched_confirmation_seasons"]}
    if reserved := sorted(untouched & set(output_seasons)):
        raise ValueError(f"Reserved Season {reserved} must remain untouched.")
    declared_folds = tuple(
        int(value)
        for value in (
            *data_contract["diagnostic_folds"],
            *data_contract["confirmation_folds"],
        )
    )
    if output_seasons != declared_folds:
        raise ValueError(
            "output_seasons must exactly match the contract folds "
            f"{declared_folds}; found {output_seasons}."
        )
    if tuple(sorted(set(declared_folds))) != declared_folds:
        raise ValueError("Predictive SPM contract folds must be unique and increasing.")
    return payload


def _validate_pinned_inputs(
    contract: dict,
    features_path: str | Path,
    targets_path: str | Path,
    reference_run_path: str | Path,
) -> None:
    pins = contract["data_contract"]
    actual = {
        "features": _artifact_run_id(features_path),
        "targets": _artifact_run_id(targets_path),
        "feature_reference_run": _artifact_run_id(reference_run_path),
    }
    for field, actual_run_id in actual.items():
        if actual_run_id != pins[field]:
            raise ValueError(
                f"Predictive SPM {field} must be {pins[field]!r}; "
                f"found {actual_run_id!r}."
            )


def _weighted_affine(
    prediction: np.ndarray, actual: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    """Weighted least-squares slope/intercept of actual on prediction."""
    sw = weight.sum()
    sx = float(weight @ prediction)
    sy = float(weight @ actual)
    sxx = float(weight @ (prediction * prediction))
    sxy = float(weight @ (prediction * actual))
    denom = sxx - sx * sx / sw
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan"), float("nan")
    slope = (sxy - sx * sy / sw) / denom
    return float(slope), float((sy - slope * sx) / sw)


def _weighted_mean_variance(
    values: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0:
        return float("nan"), float("nan")
    mean = float(weight @ values / total)
    variance = float(weight @ ((values - mean) ** 2) / total)
    return mean, max(variance, 0.0)


def _weighted_correlation(
    actual: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> float:
    actual_mean, actual_variance = _weighted_mean_variance(actual, weight)
    prediction_mean, prediction_variance = _weighted_mean_variance(prediction, weight)
    if (
        not np.isfinite(actual_variance)
        or not np.isfinite(prediction_variance)
        or actual_variance <= 0
        or prediction_variance <= 0
    ):
        return float("nan")
    covariance = float(
        weight @ ((actual - actual_mean) * (prediction - prediction_mean))
        / weight.sum()
    )
    return float(covariance / np.sqrt(actual_variance * prediction_variance))


def _dispersion(
    actual: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> float:
    _, actual_variance = _weighted_mean_variance(actual, weight)
    _, prediction_variance = _weighted_mean_variance(prediction, weight)
    if not np.isfinite(actual_variance) or actual_variance <= 0:
        return float("nan")
    return float(np.sqrt(prediction_variance / actual_variance))


def _predictive_metrics(
    actual: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    if len(actual) == 0:
        return {
            "weighted_rmse": float("nan"),
            "weighted_correlation": float("nan"),
            "correlation": float("nan"),
            "dispersion_ratio": float("nan"),
            "calibration_slope": float("nan"),
            "calibration_intercept": float("nan"),
        }
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(prediction).all()
        or not np.isfinite(weight).all()
        or (weight <= 0).any()
    ):
        raise ValueError("Predictive metrics require finite values and positive weights.")
    slope, intercept = _weighted_affine(prediction, actual, weight)
    unweighted = (
        float(np.corrcoef(actual, prediction)[0, 1])
        if np.std(actual) > 0 and np.std(prediction) > 0
        else float("nan")
    )
    return {
        "weighted_rmse": float(np.sqrt(np.average((actual - prediction) ** 2, weights=weight))),
        "weighted_correlation": _weighted_correlation(actual, prediction, weight),
        "correlation": unweighted,
        "dispersion_ratio": _dispersion(actual, prediction, weight),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def _score_fold(
    forecasts: pd.DataFrame,
    *,
    season: int,
    train_seasons: tuple[int, ...],
    calibration_ok: bool,
) -> tuple[pd.DataFrame, list[dict]]:
    """Score every valid arm on one common player-season intersection."""
    required = [
        "sample_weight",
        "target_offense",
        "target_defense",
        "target_net",
        "persistence_net",
        "raw_offense",
        "raw_defense",
        "raw_net",
    ]
    if calibration_ok:
        required.extend(
            ["calibrated_offense", "calibrated_defense", "calibrated_net"]
        )
    finite = np.isfinite(forecasts[required].to_numpy(dtype=float)).all(axis=1)
    forecasts = forecasts.copy()
    forecasts["scored_common"] = finite
    scored = forecasts.loc[finite]
    if scored.empty:
        raise ValueError(f"Forecast season {season} has no common comparator rows.")

    arms = {
        "persistence": {"net": "persistence_net"},
        "raw": {side: f"raw_{side}" for side in SIDES},
    }
    if calibration_ok:
        arms["calibrated"] = {side: f"calibrated_{side}" for side in SIDES}

    rows: list[dict] = []
    for arm, columns in arms.items():
        for side, column in columns.items():
            actual = scored[f"target_{side}"].to_numpy(dtype=float)
            predicted = scored[column].to_numpy(dtype=float)
            weight = scored["sample_weight"].to_numpy(dtype=float)
            rows.append(
                {
                    "forecast_season": season,
                    "arm": arm,
                    "component": side,
                    "training_start": train_seasons[0],
                    "training_end": train_seasons[-1],
                    "training_seasons": len(train_seasons),
                    "rows": len(scored),
                    "evaluation_rows_before_common_filter": len(forecasts),
                    "persistence_coverage": float(
                        forecasts["persistence_net"].notna().mean()
                    ),
                    **_predictive_metrics(actual, predicted, weight),
                }
            )
    return forecasts, rows


def _fit_predict(
    train: pd.DataFrame,
    predict_frame: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Raw forecasts for one fold, one row per predicted player."""
    out = predict_frame[["PLAYER_ID", "Target_Season"]].copy()
    for side in ("offense", "defense"):
        model = _fit_model(_frozen_model(side), train, selected[side], f"target_{side}")
        out[f"raw_{side}"] = model.predict(predict_frame.loc[:, selected[side]])
    out["raw_net"] = out["raw_offense"] + out["raw_defense"]
    return out


def _calibrate(
    train: pd.DataFrame,
    predictions: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Affine recalibration from leave-one-target-season-out training predictions."""
    params: dict[str, dict[str, float]] = {}
    oof_frames = []
    seasons = sorted(train["Target_Season"].unique())
    if len(seasons) < 2:
        raise ValueError("Calibration needs at least two training target seasons.")
    for held_out in seasons:
        inner_train = train.loc[train["Target_Season"].ne(held_out)]
        inner_eval = train.loc[train["Target_Season"].eq(held_out)]
        oof_frames.append(
            _fit_predict(inner_train, inner_eval, selected).merge(
                inner_eval[
                    [
                        "PLAYER_ID",
                        "Target_Season",
                        "sample_weight",
                        "target_offense",
                        "target_defense",
                    ]
                ],
                on=["PLAYER_ID", "Target_Season"],
                validate="one_to_one",
            )
        )
    oof = pd.concat(oof_frames, ignore_index=True)
    out = predictions.copy()
    for side in ("offense", "defense"):
        slope, intercept = _weighted_affine(
            oof[f"raw_{side}"].to_numpy(),
            oof[f"target_{side}"].to_numpy(),
            oof["sample_weight"].to_numpy(),
        )
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError(
                f"{side} calibration slope {slope} is invalid; arm fails this fold."
            )
        params[side] = {"slope": slope, "intercept": intercept}
        out[f"calibrated_{side}"] = slope * out[f"raw_{side}"].to_numpy() + intercept
    out["calibrated_net"] = out["calibrated_offense"] + out["calibrated_defense"]
    return out, params


def build_predictive_spm(
    features_path: str | Path,
    targets_path: str | Path,
    reference_run_path: str | Path,
    contract_path: str | Path,
    *,
    artifact_root: str | Path,
    output_seasons: tuple[int, ...] = tuple(range(2019, 2027)),
    minimum_training_seasons: int = 3,
) -> dict:
    if not output_seasons or tuple(sorted(set(output_seasons))) != output_seasons:
        raise ValueError("output_seasons must be unique and increasing.")

    contract = _load_contract(contract_path, output_seasons)
    _validate_pinned_inputs(
        contract, features_path, targets_path, reference_run_path
    )
    features = pd.read_parquet(features_path).rename(columns={"Window_End": "Season"})
    targets = pd.read_parquet(targets_path)
    if features.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Predictive SPM feature keys must be unique.")
    if targets.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Predictive SPM target keys must be unique.")

    selected = _selected_single_season_features(reference_run_path)
    required = {feature for side in ("offense", "defense") for feature in selected[side]}
    if missing := sorted(required - set(features.columns)):
        raise ValueError(f"Predictive SPM features are missing {missing}.")

    # Consecutive-season pair panel: features at season s, targets at season s+1.
    left = features[["PLAYER_ID", "Season", *required]]
    right = targets[
        [
            "PLAYER_ID",
            "Season",
            "target_offense",
            "target_defense",
            "target_net",
            "Poss_Off",
            "Poss_Def",
        ]
    ].rename(columns={"Season": "Target_Season"})
    panel = left.merge(right, on="PLAYER_ID")
    panel = panel.loc[panel["Target_Season"].eq(panel["Season"] + 1)].copy()
    if panel.duplicated(["PLAYER_ID", "Target_Season"]).any():
        raise ValueError("Predictive SPM pair keys must be unique.")
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )

    # Persistence baseline source: each evaluated player's previous-season RAPM.
    prior_targets = targets.rename(columns={"Season": "Prior_Season"})[
        ["PLAYER_ID", "Prior_Season", "target_net"]
    ]
    prior_targets["Target_Season"] = prior_targets.pop("Prior_Season") + 1

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    calibration_log: dict[str, object] = {}
    for season in output_seasons:
        train = panel.loc[panel["Target_Season"].lt(season)].copy()
        evaluate = panel.loc[panel["Target_Season"].eq(season)].copy()
        train_seasons = tuple(sorted(int(v) for v in train["Target_Season"].unique()))
        if len(train_seasons) < minimum_training_seasons:
            raise ValueError(
                f"Forecast season {season} has {len(train_seasons)} training "
                f"seasons; at least {minimum_training_seasons} are required."
            )
        if evaluate.empty:
            raise ValueError(f"Forecast season {season} has no evaluation rows.")

        forecasts = _fit_predict(train, evaluate, selected)
        try:
            forecasts, cal_params = _calibrate(train, forecasts, selected)
            calibration_ok = True
        except ValueError as error:
            calibration_ok = False
            cal_params = {"error": str(error)}
            for side in SIDES:
                forecasts[f"calibrated_{side}"] = np.nan
        calibration_log[str(season)] = cal_params

        forecasts = forecasts.merge(
            evaluate[
                ["PLAYER_ID", "Target_Season", "sample_weight", "target_offense", "target_defense", "target_net"]
            ],
            on=["PLAYER_ID", "Target_Season"],
            how="left",
            validate="one_to_one",
        ).merge(
            prior_targets,
            on=["PLAYER_ID", "Target_Season"],
            how="left",
            validate="one_to_one",
        ).rename(columns={"target_net_x": "target_net", "target_net_y": "persistence_net"})
        forecasts["Window_End"] = season
        forecasts, rows = _score_fold(
            forecasts,
            season=season,
            train_seasons=train_seasons,
            calibration_ok=calibration_ok,
        )
        prediction_rows.append(forecasts)
        metric_rows.extend(rows)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)

    identity = uuid.uuid5(
        uuid.NAMESPACE_URL,
        json.dumps(
            {
                "features_sha256": sha256_file(features_path),
                "targets_sha256": sha256_file(targets_path),
                "reference_run": sha256_file(Path(reference_run_path) / "run.json"),
                "contract_sha256": sha256_file(contract_path),
                "source_sha256": sha256_file(Path(__file__)),
                "output_seasons": list(output_seasons),
            },
            sort_keys=True,
        ),
    ).hex[:10]
    run_id = f"predictive_spm_v1_{identity}"
    output = Path(artifact_root) / "models" / "predictive_spm" / run_id
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)

    summary = {}
    net = metrics.loc[metrics["component"].eq("net")]
    for arm in ("persistence", "raw", "calibrated"):
        rows = net.loc[net["arm"].eq(arm)]
        summary[f"{arm}_mean_weighted_rmse"] = (
            float(rows["weighted_rmse"].mean()) if not rows.empty else float("nan")
        )
    run = {
        "run_id": run_id,
        "model_family": "next_season_predictive_statistical_plus_minus",
        "estimand_id": contract["estimand_id"],
        "status": "research_predeclared_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment_id": contract["experiment_id"],
            "schema_version": contract["schema_version"],
            "output_seasons": list(output_seasons),
            "minimum_training_seasons": minimum_training_seasons,
            "training_rule": "pairs with target season strictly before the forecast season",
            "features": {side: list(selected[side]) for side in ("offense", "defense")},
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "reference_run": sha256_file(Path(reference_run_path) / "run.json"),
                "contract": sha256_file(contract_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "quality": {
            "pair_panel_rows": len(panel),
            "prediction_rows": len(predictions),
            "players": int(predictions["PLAYER_ID"].nunique()),
            "duplicate_keys": int(
                predictions.duplicated(["PLAYER_ID", "Window_End"]).sum()
            ),
            "common_scored_rows": int(predictions["scored_common"].sum()),
            "all_comparators_use_common_rows": True,
            "untouched_seasons_loaded": [],
        },
        "metrics": {"summary": summary, "calibration_parameters": calibration_log},
        "predictions_path": str((output / "predictions.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "Rookies and returning-after-gap players are out of scope by design.",
            "Targets are noisy one-season normal RAPM labels, not ground truth.",
            "The target panel keeps its explicit legacy/canonical provenance boundary.",
            "Confirmation folds 2025 and 2026 are scored once; do not retune on them.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
