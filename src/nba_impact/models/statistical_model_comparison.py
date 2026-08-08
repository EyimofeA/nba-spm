"""Chronological model-family comparison for three-season statistical impact."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_impact import (
    FORBIDDEN_PRIMARY_FEATURES,
    _load_panel,
    _metrics,
)


def _candidate_models(
    ridge_alphas: tuple[float, ...],
    elastic_grid: tuple[tuple[float, float], ...],
    histogram_grid: tuple[tuple[float, int, float], ...],
) -> dict[str, list[tuple[dict, Pipeline]]]:
    candidates: dict[str, list[tuple[dict, Pipeline]]] = {
        "ridge": [],
        "elastic_net": [],
        "histogram_gbm": [],
    }
    for alpha in ridge_alphas:
        config = {"alpha": alpha}
        candidates["ridge"].append(
            (
                config,
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                        ("model", Ridge(alpha=alpha)),
                    ]
                ),
            )
        )
    for alpha, l1_ratio in elastic_grid:
        config = {"alpha": alpha, "l1_ratio": l1_ratio}
        candidates["elastic_net"].append(
            (
                config,
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                        (
                            "model",
                            ElasticNet(
                                alpha=alpha,
                                l1_ratio=l1_ratio,
                                max_iter=20_000,
                                tol=1e-5,
                                selection="cyclic",
                            ),
                        ),
                    ]
                ),
            )
        )
    for learning_rate, max_leaf_nodes, l2_regularization in histogram_grid:
        config = {
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "l2_regularization": l2_regularization,
        }
        candidates["histogram_gbm"].append(
            (
                config,
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                learning_rate=learning_rate,
                                max_iter=250,
                                max_leaf_nodes=max_leaf_nodes,
                                min_samples_leaf=30,
                                l2_regularization=l2_regularization,
                                early_stopping=False,
                                random_state=20260808,
                            ),
                        ),
                    ]
                ),
            )
        )
    if any(not models for models in candidates.values()):
        raise ValueError("Every model family must have at least one candidate.")
    return candidates


def _fit_model(
    model: Pipeline,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
) -> Pipeline:
    model.fit(
        frame.loc[:, features],
        frame[target],
        model__sample_weight=frame["sample_weight"],
    )
    return model


def run_statistical_model_comparison(
    features_path: str | Path,
    targets_path: str | Path,
    *,
    artifact_root: str | Path,
    features: tuple[str, ...] | None = None,
    test_window_ends: tuple[int, ...] = (2022, 2023, 2024),
    first_complete_tracking_window: int = 2016,
    target_window_seasons: int = 3,
    ridge_alphas: tuple[float, ...] = (30.0, 100.0, 300.0, 1000.0, 3000.0),
    elastic_grid: tuple[tuple[float, float], ...] = (
        (0.001, 0.1),
        (0.001, 0.5),
        (0.001, 0.9),
        (0.01, 0.1),
        (0.01, 0.5),
        (0.01, 0.9),
        (0.1, 0.1),
        (0.1, 0.5),
        (0.1, 0.9),
        (1.0, 0.1),
        (1.0, 0.5),
        (1.0, 0.9),
    ),
    histogram_grid: tuple[tuple[float, int, float], ...] = (
        (0.03, 7, 1.0),
        (0.03, 7, 10.0),
        (0.03, 15, 1.0),
        (0.03, 15, 10.0),
        (0.07, 7, 1.0),
        (0.07, 7, 10.0),
        (0.07, 15, 1.0),
        (0.07, 15, 10.0),
    ),
) -> dict:
    """Compare deterministic model families on identical purged outer folds."""
    panel = _load_panel(features_path, targets_path)
    panel = panel.loc[panel["Window_End"] >= first_complete_tracking_window].copy()
    if features is None:
        features = tuple(
            column
            for column in pd.read_parquet(features_path).columns
            if column not in FORBIDDEN_PRIMARY_FEATURES
        )
    if missing := sorted(set(features) - set(panel.columns)):
        raise ValueError(f"Model comparison is missing features {missing}.")
    families = _candidate_models(ridge_alphas, elastic_grid, histogram_grid)
    targets = {"offense": "target_offense", "defense": "target_defense"}
    metrics: list[dict] = []
    predictions: list[pd.DataFrame] = []
    selected: dict[tuple[int, str, str], tuple[dict, Pipeline]] = {}

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
        fold = test[
            [
                "PLAYER_ID",
                "Window_End",
                "target_offense",
                "target_defense",
                "target_net",
                "sample_weight",
            ]
        ].copy()

        for family, candidates in families.items():
            for target_name, target_column in targets.items():
                best: tuple[float, dict, Pipeline] | None = None
                for config, candidate in candidates:
                    fitted = _fit_model(candidate, inner_train, features, target_column)
                    inner_prediction = fitted.predict(inner_validation.loc[:, features])
                    score = _metrics(
                        inner_validation[target_column].to_numpy(),
                        inner_prediction,
                        inner_validation["sample_weight"].to_numpy(),
                    )["weighted_rmse"]
                    if best is None or score < best[0]:
                        best = (score, config, candidate)
                if best is None:
                    raise RuntimeError(f"No candidate selected for {family} {target_name}.")
                _, config, candidate = best
                fitted = _fit_model(candidate, outer_train, features, target_column)
                selected[(test_end, family, target_name)] = (config, fitted)
                prediction_column = f"prediction_{family}_{target_name}"
                fold[prediction_column] = fitted.predict(test.loc[:, features])
                metrics.append(
                    {
                        "test_window_end": test_end,
                        "family": family,
                        "target": target_name,
                        "selected_config": json.dumps(config, sort_keys=True),
                        "train_max_window_end": train_max,
                        "test_players": len(test),
                        **_metrics(
                            test[target_column].to_numpy(),
                            fold[prediction_column].to_numpy(),
                            test["sample_weight"].to_numpy(),
                        ),
                    }
                )
            net_column = f"prediction_{family}_net_from_components"
            fold[net_column] = (
                fold[f"prediction_{family}_offense"]
                + fold[f"prediction_{family}_defense"]
            )
            metrics.append(
                {
                    "test_window_end": test_end,
                    "family": family,
                    "target": "net_from_components",
                    "selected_config": None,
                    "train_max_window_end": train_max,
                    "test_players": len(test),
                    **_metrics(
                        test["target_net"].to_numpy(),
                        fold[net_column].to_numpy(),
                        test["sample_weight"].to_numpy(),
                    ),
                }
            )
        predictions.append(fold)

    fold_metrics = pd.DataFrame(metrics)
    summary = (
        fold_metrics.groupby(["family", "target"], as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
            folds=("test_window_end", "nunique"),
        )
        .sort_values(["target", "mean_weighted_rmse"], kind="stable")
    )
    net = fold_metrics.loc[fold_metrics["target"].eq("net_from_components")]
    net_pivot = net.pivot(index="test_window_end", columns="family", values="weighted_rmse")
    fold_wins_vs_ridge = {
        family: int((net_pivot[family] < net_pivot["ridge"]).sum())
        for family in ("elastic_net", "histogram_gbm")
    }

    run_id = f"statistical_model_comparison_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "statistical_model_comparison" / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(predictions, ignore_index=True).to_parquet(
        output / "fold_predictions.parquet", index=False
    )
    fold_metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)

    final_models = {}
    last_end = max(test_window_ends)
    for family in families:
        for target_name, target_column in targets.items():
            config, _ = selected[(last_end, family, target_name)]
            matching = [
                candidate
                for candidate_config, candidate in families[family]
                if candidate_config == config
            ]
            final = _fit_model(matching[0], panel, features, target_column)
            model_path = output / f"model_{family}_{target_name}.joblib"
            joblib.dump(final, model_path)
            final_models[f"{family}_{target_name}"] = {
                "path": str(model_path.resolve()),
                "config": config,
            }

    run = {
        "run_id": run_id,
        "model_family": "statistical_impact_family_comparison",
        "estimand": "three_season_normal_rapm_offense_defense_and_net",
        "status": "research_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "features_path": str(Path(features_path).resolve()),
            "targets_path": str(Path(targets_path).resolve()),
            "feature_count": len(features),
            "features": list(features),
            "test_window_ends": list(test_window_ends),
            "target_window_seasons": target_window_seasons,
            "purge_windows_between_train_and_test": target_window_seasons - 1,
            "selection_metric": "inner_validation_weighted_rmse",
            "families": list(families),
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {
            "joined_rows": len(panel),
            "players": panel["PLAYER_ID"].nunique(),
            "summary": summary.to_dict(orient="records"),
            "net_rmse_fold_wins_vs_ridge": fold_wins_vs_ridge,
        },
        "models": final_models,
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
