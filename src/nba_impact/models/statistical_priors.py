"""Leakage-safe cross-fitted statistical priors for normal RAPM."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_impact import _load_panel, _metrics
from nba_impact.models.statistical_model_comparison import _fit_model


KEYS = ("PLAYER_ID", "Window_End")
PRIOR_COLUMNS = (
    "prior_offense_per_100",
    "prior_defense_per_100",
    "prior_net_per_100",
)


def _reference_contract(
    reference_run_path: str | Path,
    features_path: str | Path,
    targets_path: str | Path,
) -> tuple[dict, dict[str, tuple[str, ...]]]:
    reference_path = Path(reference_run_path)
    reference = json.loads((reference_path / "run.json").read_text())
    hashes = reference.get("config", {}).get("source_hashes", {})
    expected = {
        "features": sha256_file(features_path),
        "targets": sha256_file(targets_path),
    }
    for source, actual_hash in expected.items():
        if hashes.get(source) != actual_hash:
            raise ValueError(
                f"Cross-fitted prior {source} input does not match the reference run."
            )
    selected = reference.get("selected_features")
    if not isinstance(selected, dict):
        raise ValueError("Reference run does not contain selected_features.")
    features = {
        side: tuple(selected.get(side, ())) for side in ("offense", "defense")
    }
    if any(not values for values in features.values()):
        raise ValueError("Reference run has an empty offense or defense feature set.")
    return reference, features


def build_cross_fitted_statistical_priors(
    features_path: str | Path,
    targets_path: str | Path,
    reference_run_path: str | Path,
    *,
    artifact_root: str | Path,
    prediction_window_ends: tuple[int, ...] | None = None,
    target_window_seasons: int = 3,
) -> dict:
    """Predict each eligible window using only non-overlapping older labels."""
    if target_window_seasons < 1:
        raise ValueError("target_window_seasons must be positive.")
    reference, selected_features = _reference_contract(
        reference_run_path, features_path, targets_path
    )
    feature_frame = pd.read_parquet(features_path)
    if feature_frame.duplicated(list(KEYS)).any():
        raise ValueError("Statistical-prior feature keys must be unique.")
    required = set(KEYS)
    required.update(selected_features["offense"])
    required.update(selected_features["defense"])
    if missing := sorted(required - set(feature_frame.columns)):
        raise ValueError(f"Statistical-prior features are missing {missing}.")

    panel = _load_panel(features_path, targets_path)
    available_ends = tuple(sorted(int(value) for value in feature_frame["Window_End"].unique()))
    earliest = min(available_ends)
    eligible_ends = tuple(
        end for end in available_ends if end - target_window_seasons >= earliest
    )
    if prediction_window_ends is None:
        prediction_window_ends = eligible_ends
    else:
        prediction_window_ends = tuple(sorted(set(prediction_window_ends)))
    if not prediction_window_ends:
        raise ValueError("No prediction windows were requested.")
    if invalid := sorted(set(prediction_window_ends) - set(eligible_ends)):
        raise ValueError(
            "Prediction windows need at least one non-overlapping older target "
            f"window; invalid windows: {invalid}."
        )

    prior_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    coverage_rows: list[dict] = []
    fold_models: dict[str, dict[str, dict]] = {}
    fitted_models: dict[tuple[int, str], object] = {}
    for test_end in prediction_window_ends:
        train_max = test_end - target_window_seasons
        train = panel.loc[panel["Window_End"] <= train_max].copy()
        test_features = feature_frame.loc[
            feature_frame["Window_End"].eq(test_end)
        ].copy()
        evaluation = panel.loc[panel["Window_End"].eq(test_end)].copy()
        if min(len(train), len(test_features), len(evaluation)) == 0:
            raise ValueError(f"Cross-fitted prior fold {test_end} has an empty partition.")
        if int(train["Window_End"].max()) > train_max:
            raise AssertionError("Training target windows overlap the prediction window.")

        fold = test_features.loc[:, list(KEYS)].copy()
        for side in ("offense", "defense"):
            model = _fit_model(
                _frozen_model(side),
                train,
                selected_features[side],
                f"target_{side}",
            )
            fitted_models[(test_end, side)] = model
            fold[f"prior_{side}_per_100"] = model.predict(
                test_features.loc[:, selected_features[side]]
            )
        fold["prior_net_per_100"] = (
            fold["prior_offense_per_100"] + fold["prior_defense_per_100"]
        )
        fold["train_max_window_end"] = train_max
        fold["train_rows"] = len(train)
        fold["train_target_windows"] = train["Window_End"].nunique()
        prior_rows.append(fold)

        scored = evaluation.loc[
            :, [*KEYS, "target_offense", "target_defense", "target_net", "sample_weight"]
        ].merge(fold, on=list(KEYS), how="left", validate="one_to_one")
        if scored[list(PRIOR_COLUMNS)].isna().any().any():
            raise ValueError(f"Fold {test_end} is missing priors for labeled players.")
        for side in ("offense", "defense", "net"):
            metric_rows.append(
                {
                    "test_window_end": test_end,
                    "target": side,
                    "train_max_window_end": train_max,
                    "train_rows": len(train),
                    "train_target_windows": int(train["Window_End"].nunique()),
                    "test_feature_rows": len(test_features),
                    "test_labeled_rows": len(evaluation),
                    **_metrics(
                        scored[f"target_{side}"].to_numpy(),
                        scored[f"prior_{side}_per_100"].to_numpy(),
                        scored["sample_weight"].to_numpy(),
                    ),
                }
            )
        coverage_rows.append(
            {
                "test_window_end": test_end,
                "feature_rows": len(test_features),
                "labeled_rows": len(evaluation),
                "unlabeled_prior_rows": len(test_features) - len(evaluation),
                "label_coverage": len(evaluation) / len(test_features),
            }
        )

    priors = pd.concat(prior_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    coverage = pd.DataFrame(coverage_rows)
    if priors.duplicated(list(KEYS)).any():
        raise ValueError("Cross-fitted prior keys are not unique.")
    if priors[list(PRIOR_COLUMNS)].isna().any().any():
        raise ValueError("Cross-fitted priors contain missing predictions.")
    if not np.isfinite(priors[list(PRIOR_COLUMNS)].to_numpy()).all():
        raise ValueError("Cross-fitted priors contain non-finite predictions.")
    expected_rows = int(
        feature_frame["Window_End"].isin(prediction_window_ends).sum()
    )
    if len(priors) != expected_rows:
        raise ValueError("Cross-fitted prior output does not cover every eligible feature row.")

    summary = (
        metrics.groupby("target", as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
            folds=("test_window_end", "nunique"),
        )
        .sort_values("target", kind="stable")
    )
    run_id = f"statistical_priors_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "statistical_priors" / run_id
    output.mkdir(parents=True, exist_ok=False)
    priors.to_parquet(output / "priors.parquet", index=False)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    coverage.to_parquet(output / "coverage.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    for (test_end, side), model in fitted_models.items():
        path = output / f"model_window_{test_end}_{side}.joblib"
        joblib.dump(model, path)
        fold_models.setdefault(str(test_end), {})[side] = {
            "path": str(path.resolve()),
            "features": list(selected_features[side]),
            "feature_count": len(selected_features[side]),
        }

    run = {
        "run_id": run_id,
        "model_family": "cross_fitted_statistical_priors",
        "estimand": "same_window_three_season_normal_rapm_offense_defense_and_net",
        "status": "validated_research_priors",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "prediction_time": "end_of_three_season_window_after_features_are_observed",
            "intended_use": "retrodictive_statistical_center_for_prior_informed_rapm",
            "not_a_forecast": True,
            "target_window_seasons": target_window_seasons,
            "prediction_window_ends": list(prediction_window_ends),
            "purge_rule": "train Window_End <= prediction Window_End - target_window_seasons",
            "reference_run_id": reference["run_id"],
            "reference_run_path": str(Path(reference_run_path).resolve()),
            "selected_features": {
                side: list(values) for side, values in selected_features.items()
            },
            "model_configs": {
                "offense": reference["config"]["offense_model"],
                "defense": reference["config"]["defense_model"],
            },
            "sample_weight": "sqrt(min(Poss_Off, Poss_Def)); training and evaluation only",
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "reference_run": sha256_file(Path(reference_run_path) / "run.json"),
                "source_code": sha256_file(Path(__file__)),
                "frozen_model_factory": sha256_file(
                    Path(__file__).with_name("statistical_feature_ablation.py")
                ),
                "model_fit_helper": sha256_file(
                    Path(__file__).with_name("statistical_model_comparison.py")
                ),
            },
        },
        "quality": {
            "rows": len(priors),
            "players": int(priors["PLAYER_ID"].nunique()),
            "windows": int(priors["Window_End"].nunique()),
            "duplicate_keys": 0,
            "missing_predictions": 0,
            "nonfinite_predictions": 0,
            "all_eligible_feature_rows_scored": len(priors) == expected_rows,
            "minimum_label_coverage": float(coverage["label_coverage"].min()),
        },
        "metrics": {
            "summary": summary.to_dict(orient="records"),
            "folds": metrics.to_dict(orient="records"),
        },
        "models": fold_models,
        "priors_path": str((output / "priors.parquet").resolve()),
        "caveats": [
            "These priors use statistics from the same window they describe; they are retrodictive, not forecasts.",
            "The first three feature windows cannot be cross-fitted because no non-overlapping older tracking-era labels exist.",
            "The 2022-2024 target windows informed earlier feature research, so their metrics are not untouched promotion evidence.",
            "The historical normal-RAPM labels end in 2024 and may not be the final target run.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
