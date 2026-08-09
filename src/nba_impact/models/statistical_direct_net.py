"""Compare direct nonlinear net RAPM with decomposed statistical impact."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_impact import _load_panel, _metrics
from nba_impact.models.statistical_model_comparison import (
    _candidate_models,
    _fit_model,
)

DEFAULT_HISTOGRAM_GRID = (
    (0.03, 7, 1.0),
    (0.03, 7, 10.0),
    (0.03, 15, 1.0),
    (0.03, 15, 10.0),
    (0.07, 7, 1.0),
    (0.07, 7, 10.0),
    (0.07, 15, 1.0),
    (0.07, 15, 10.0),
)


def run_statistical_direct_net_comparison(
    features_path: str | Path,
    targets_path: str | Path,
    component_run_path: str | Path,
    *,
    artifact_root: str | Path,
    test_window_ends: tuple[int, ...] = (2022, 2023, 2024),
    first_complete_tracking_window: int = 2016,
    target_window_seasons: int = 3,
    histogram_grid: tuple[tuple[float, int, float], ...] = DEFAULT_HISTOGRAM_GRID,
) -> dict:
    """Tune direct net HGB inside each fold and compare saved component predictions."""
    component_path = Path(component_run_path)
    component_run = json.loads((component_path / "run.json").read_text())
    feature_hash = sha256_file(features_path)
    target_hash = sha256_file(targets_path)
    parent_hashes = component_run["config"]["source_hashes"]
    if parent_hashes["features"] != feature_hash or parent_hashes["targets"] != target_hash:
        raise ValueError("Direct-net inputs do not match the component run sources.")
    features = tuple(component_run["config"]["features"])
    component_predictions = pd.read_parquet(component_path / "fold_predictions.parquet")
    required_component_columns = {
        "PLAYER_ID",
        "Window_End",
        "prediction_ridge_defense",
        "prediction_ridge_offense",
        "prediction_histogram_gbm_offense",
    }
    if missing := sorted(required_component_columns - set(component_predictions.columns)):
        raise ValueError(f"Component predictions are missing {missing}.")

    panel = _load_panel(features_path, targets_path)
    panel = panel.loc[panel["Window_End"] >= first_complete_tracking_window].copy()
    histogram_candidates = _candidate_models(
        (1.0,), ((0.1, 0.5),), histogram_grid
    )["histogram_gbm"]
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    selected: dict[int, tuple[dict, object]] = {}

    for test_end in test_window_ends:
        train_max = test_end - target_window_seasons
        inner_validation_end = train_max
        inner_train_max = inner_validation_end - target_window_seasons
        outer_train = panel.loc[panel["Window_End"] <= train_max]
        test = panel.loc[panel["Window_End"] == test_end]
        inner_train = outer_train.loc[outer_train["Window_End"] <= inner_train_max]
        inner_validation = outer_train.loc[
            outer_train["Window_End"] == inner_validation_end
        ]
        if min(map(len, (outer_train, test, inner_train, inner_validation))) == 0:
            raise ValueError(f"Purged fold ending {test_end} has an empty partition.")

        best = None
        for config, candidate in histogram_candidates:
            fitted = _fit_model(candidate, inner_train, features, "target_net")
            prediction = fitted.predict(inner_validation.loc[:, features])
            score = _metrics(
                inner_validation["target_net"].to_numpy(),
                prediction,
                inner_validation["sample_weight"].to_numpy(),
            )["weighted_rmse"]
            if best is None or score < best[0]:
                best = (score, config, candidate)
        if best is None:
            raise RuntimeError(f"No direct-net candidate selected for {test_end}.")
        _, config, candidate = best
        direct_model = _fit_model(candidate, outer_train, features, "target_net")
        selected[test_end] = (config, direct_model)

        fold = test[
            ["PLAYER_ID", "Window_End", "target_net", "sample_weight"]
        ].merge(
            component_predictions.loc[
                component_predictions["Window_End"].eq(test_end),
                list(required_component_columns),
            ],
            on=["PLAYER_ID", "Window_End"],
            how="inner",
            validate="one_to_one",
        )
        if len(fold) != len(test):
            raise ValueError(f"Component coverage differs for fold {test_end}.")
        fold["prediction_ridge_components"] = (
            fold["prediction_ridge_offense"] + fold["prediction_ridge_defense"]
        )
        fold["prediction_hybrid_components"] = (
            fold["prediction_histogram_gbm_offense"]
            + fold["prediction_ridge_defense"]
        )
        fold["prediction_direct_histogram_gbm"] = direct_model.predict(
            test.loc[:, features]
        )
        for variant, column in {
            "ridge_components": "prediction_ridge_components",
            "histogram_offense_plus_ridge_defense": "prediction_hybrid_components",
            "direct_histogram_gbm": "prediction_direct_histogram_gbm",
        }.items():
            metric_rows.append(
                {
                    "test_window_end": test_end,
                    "variant": variant,
                    "selected_config": (
                        json.dumps(config, sort_keys=True)
                        if variant == "direct_histogram_gbm"
                        else None
                    ),
                    **_metrics(
                        fold["target_net"].to_numpy(),
                        fold[column].to_numpy(),
                        fold["sample_weight"].to_numpy(),
                    ),
                }
            )
        prediction_rows.append(fold)

    fold_metrics = pd.DataFrame(metric_rows)
    summary = (
        fold_metrics.groupby("variant", as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
            folds=("test_window_end", "nunique"),
        )
        .sort_values("mean_weighted_rmse", kind="stable")
    )
    pivot = fold_metrics.pivot(
        index="test_window_end", columns="variant", values="weighted_rmse"
    )
    direct_wins = int(
        (
            pivot["direct_histogram_gbm"]
            < pivot["histogram_offense_plus_ridge_defense"]
        ).sum()
    )

    run_id = f"statistical_direct_net_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "statistical_direct_net" / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        output / "fold_predictions.parquet", index=False
    )
    fold_metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)

    last_config, _ = selected[max(test_window_ends)]
    matching = [
        candidate
        for candidate_config, candidate in histogram_candidates
        if candidate_config == last_config
    ]
    final_model = _fit_model(matching[0], panel, features, "target_net")
    model_path = output / "model_direct_histogram_gbm.joblib"
    joblib.dump(final_model, model_path)
    run = {
        "run_id": run_id,
        "model_family": "statistical_direct_net_comparison",
        "estimand": "three_season_normal_rapm_net",
        "status": "research_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "features_path": str(Path(features_path).resolve()),
            "targets_path": str(Path(targets_path).resolve()),
            "parent_component_run": component_run["run_id"],
            "parent_component_path": str(component_path.resolve()),
            "test_window_ends": list(test_window_ends),
            "target_window_seasons": target_window_seasons,
            "selection_metric": "inner_validation_weighted_rmse",
            "source_hashes": {
                "features": feature_hash,
                "targets": target_hash,
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {
            "summary": summary.to_dict(orient="records"),
            "direct_net_fold_wins_vs_hybrid": direct_wins,
        },
        "model": {
            "path": str(model_path.resolve()),
            "config": last_config,
        },
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
