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

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.single_season_spm import _selected_single_season_features
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_impact import _metrics
from nba_impact.models.statistical_model_comparison import _fit_model

SIDES = ("offense", "defense", "net")


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


def _dispersion(actual: np.ndarray, prediction: np.ndarray) -> float:
    if np.std(actual) == 0:
        return float("nan")
    return float(np.std(prediction) / np.std(actual))


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

    contract = json.loads(Path(contract_path).read_text())
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
        persistence_rows = int(forecasts["persistence_net"].notna().sum())
        forecasts["Window_End"] = season
        prediction_rows.append(forecasts)

        common = forecasts.dropna(subset=["persistence_net"])
        arms = {
            "persistence": {side: f"persistence_{side}" for side in ("net",)},
            "raw": {side: f"raw_{side}" for side in SIDES},
            "calibrated": {side: f"calibrated_{side}" for side in SIDES},
        }
        for arm, columns in arms.items():
            frame = common if arm == "persistence" else forecasts
            for side, column in columns.items():
                actual = frame[f"target_{side}"].to_numpy()
                predicted = frame[column].to_numpy()
                finite = np.isfinite(predicted)
                metric_rows.append(
                    {
                        "forecast_season": season,
                        "arm": arm,
                        "component": side,
                        "training_start": train_seasons[0],
                        "training_end": train_seasons[-1],
                        "training_seasons": len(train_seasons),
                        "rows": int(finite.sum()),
                        **_metrics(
                            actual[finite],
                            predicted[finite],
                            frame["sample_weight"].to_numpy()[finite],
                        ),
                        "dispersion_ratio": _dispersion(
                            actual[finite], predicted[finite]
                        ),
                    }
                )

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
        summary[f"{arm}_mean_weighted_rmse"] = float(rows["weighted_rmse"].mean())
    run = {
        "run_id": run_id,
        "model_family": "next_season_predictive_statistical_plus_minus",
        "estimand": contract.get("estimand"),
        "status": "research_predeclared_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "experiment_id": "predictive_spm_v1",
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
