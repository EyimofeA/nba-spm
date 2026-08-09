"""Grouped feature ablations for the frozen decomposed statistical AIO."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import joblib

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_impact import FORBIDDEN_PRIMARY_FEATURES, _load_panel, _metrics
from nba_impact.models.statistical_model_comparison import _candidate_models, _fit_model

CORE_BOX = {
    "PTS_p100", "AST_p100", "TOV_p100", "STL_p100", "BLK_p100",
    "OREB_p100", "DREB_p100", "PF_p100", "PFD_p100", "FTA_p100",
    "FTM_p100", "FG2A_p100", "FG2M_p100", "FG3A_p100", "FG3M_p100",
    "usage_events_p100", "true_shooting_pct", "fg2_pct", "fg3_pct", "ft_pct",
}


def default_feature_groups(features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {
        "core_box": [],
        "shot_profile": [],
        "creation_role": [],
        "turnover_detail": [],
        "rebound_defense_tracking": [],
    }
    for feature in features:
        if feature in CORE_BOX:
            groups["core_box"].append(feature)
        elif any(token in feature for token in ("turnover", "travels", "offensive_fouls")):
            groups["turnover_detail"].append(feature)
        elif any(
            token in feature
            for token in (
                "rebound", "dreb_", "recovered_blocks", "contest", "chances"
            )
        ):
            groups["rebound_defense_tracking"].append(feature)
        elif any(
            token in feature
            for token in (
                "accuracy", "frequency", "catch_shoot", "pull_up", "at_rim",
                "short_mid", "long_mid", "corner3", "arc3", "open_", "tight_",
                "shot_quality", "shooting_fouls",
            )
        ):
            groups["shot_profile"].append(feature)
        elif any(
            token in feature
            for token in (
                "drive", "touch", "passes", "assist", "time_of_possession",
                "avg_seconds", "avg_dribbles", "paint_", "post_", "elbow_",
            )
        ):
            groups["creation_role"].append(feature)
    return {name: tuple(values) for name, values in groups.items() if values}


def _frozen_model(target: str):
    candidates = _candidate_models(
        (3000.0,),
        ((0.1, 0.1),),
        ((0.03, 7, 1.0),),
    )
    family = "histogram_gbm" if target == "offense" else "ridge"
    return candidates[family][0][1]


def run_statistical_feature_ablation(
    features_path: str | Path,
    targets_path: str | Path,
    *,
    artifact_root: str | Path,
    test_window_ends: tuple[int, ...] = (2022, 2023, 2024),
    first_complete_tracking_window: int = 2016,
    target_window_seasons: int = 3,
) -> dict:
    panel = _load_panel(features_path, targets_path)
    panel = panel.loc[panel["Window_End"] >= first_complete_tracking_window].copy()
    feature_columns = tuple(
        column
        for column in pd.read_parquet(features_path).columns
        if column not in FORBIDDEN_PRIMARY_FEATURES
    )
    groups = default_feature_groups(feature_columns)
    variants = {"full": feature_columns}
    for group, dropped in groups.items():
        variants[f"drop_{group}"] = tuple(
            feature for feature in feature_columns if feature not in dropped
        )

    prediction_rows = []
    component_metrics = []
    net_metrics = []
    for test_end in test_window_ends:
        train_max = test_end - target_window_seasons
        train = panel.loc[panel["Window_End"] <= train_max]
        test = panel.loc[panel["Window_End"] == test_end]
        if min(len(train), len(test)) == 0:
            raise ValueError(f"Feature-ablation fold {test_end} has an empty partition.")
        fold = test[
            ["PLAYER_ID", "Window_End", "target_offense", "target_defense", "target_net", "sample_weight"]
        ].copy()
        for variant, columns in variants.items():
            for target_name, target_column in {
                "offense": "target_offense",
                "defense": "target_defense",
            }.items():
                model = _fit_model(_frozen_model(target_name), train, columns, target_column)
                prediction_column = f"prediction_{target_name}_{variant}"
                fold[prediction_column] = model.predict(test.loc[:, columns])
                component_metrics.append(
                    {
                        "test_window_end": test_end,
                        "target": target_name,
                        "variant": variant,
                        "feature_count": len(columns),
                        **_metrics(
                            test[target_column].to_numpy(),
                            fold[prediction_column].to_numpy(),
                            test["sample_weight"].to_numpy(),
                        ),
                    }
                )
        for side in ("offense", "defense"):
            other = "defense" if side == "offense" else "offense"
            for variant in variants:
                prediction = (
                    fold[f"prediction_{side}_{variant}"]
                    + fold[f"prediction_{other}_full"]
                )
                net_metrics.append(
                    {
                        "test_window_end": test_end,
                        "ablated_side": side,
                        "variant": variant,
                        **_metrics(
                            test["target_net"].to_numpy(),
                            prediction.to_numpy(),
                            test["sample_weight"].to_numpy(),
                        ),
                    }
                )
        prediction_rows.append(fold)

    component = pd.DataFrame(component_metrics)
    net = pd.DataFrame(net_metrics)
    full_component = component.loc[component["variant"].eq("full"), [
        "test_window_end", "target", "weighted_rmse"
    ]].rename(columns={"weighted_rmse": "full_rmse"})
    decisions = component.loc[~component["variant"].eq("full")].merge(
        full_component, on=["test_window_end", "target"], validate="many_to_one"
    )
    decisions["rmse_delta_vs_full"] = decisions["weighted_rmse"] - decisions["full_rmse"]
    decision_summary = (
        decisions.groupby(["target", "variant"], as_index=False)
        .agg(
            mean_rmse_delta_vs_full=("rmse_delta_vs_full", "mean"),
            fold_wins=("rmse_delta_vs_full", lambda values: int((values < 0).sum())),
            folds=("test_window_end", "nunique"),
        )
    )
    decision_summary["robust_drop"] = (
        decision_summary["mean_rmse_delta_vs_full"].lt(0)
        & decision_summary["fold_wins"].ge(2)
    )

    confirmation_end = max(test_window_ends)
    discovery = decisions.loc[decisions["test_window_end"].lt(confirmation_end)]
    discovery_summary = (
        discovery.groupby(["target", "variant"], as_index=False)
        .agg(
            mean_rmse_delta_vs_full=("rmse_delta_vs_full", "mean"),
            fold_wins=("rmse_delta_vs_full", lambda values: int((values < 0).sum())),
            folds=("test_window_end", "nunique"),
        )
    )
    discovery_summary["selected_drop"] = (
        discovery_summary["mean_rmse_delta_vs_full"].lt(0)
        & discovery_summary["fold_wins"].eq(discovery_summary["folds"])
    )
    selected_groups: dict[str, list[str]] = {"offense": [], "defense": []}
    optimized_features = {}
    for target_name in selected_groups:
        selected_variants = discovery_summary.loc[
            discovery_summary["target"].eq(target_name)
            & discovery_summary["selected_drop"],
            "variant",
        ]
        selected_groups[target_name] = [
            variant.removeprefix("drop_") for variant in selected_variants
        ]
        dropped = {
            feature
            for group in selected_groups[target_name]
            for feature in groups[group]
        }
        optimized_features[target_name] = tuple(
            feature for feature in feature_columns if feature not in dropped
        )

    confirmation_train = panel.loc[
        panel["Window_End"] <= confirmation_end - target_window_seasons
    ]
    confirmation_test = panel.loc[panel["Window_End"] == confirmation_end]
    confirmation_predictions = confirmation_test[
        ["PLAYER_ID", "Window_End", "target_offense", "target_defense", "target_net", "sample_weight"]
    ].copy()
    confirmation_metrics = []
    for target_name, target_column in {
        "offense": "target_offense",
        "defense": "target_defense",
    }.items():
        columns = optimized_features[target_name]
        model = _fit_model(
            _frozen_model(target_name), confirmation_train, columns, target_column
        )
        prediction = model.predict(confirmation_test.loc[:, columns])
        confirmation_predictions[f"prediction_{target_name}_optimized"] = prediction
        full_prediction = prediction_rows[-1][f"prediction_{target_name}_full"].to_numpy()
        for variant, values in {"full": full_prediction, "optimized": prediction}.items():
            confirmation_metrics.append(
                {
                    "target": target_name,
                    "variant": variant,
                    **_metrics(
                        confirmation_test[target_column].to_numpy(),
                        values,
                        confirmation_test["sample_weight"].to_numpy(),
                    ),
                }
            )
    optimized_net = (
        confirmation_predictions["prediction_offense_optimized"]
        + confirmation_predictions["prediction_defense_optimized"]
    )
    full_net = (
        prediction_rows[-1]["prediction_offense_full"]
        + prediction_rows[-1]["prediction_defense_full"]
    )
    for variant, values in {"full": full_net, "optimized": optimized_net}.items():
        confirmation_metrics.append(
            {
                "target": "net",
                "variant": variant,
                **_metrics(
                    confirmation_test["target_net"].to_numpy(),
                    values.to_numpy(),
                    confirmation_test["sample_weight"].to_numpy(),
                ),
            }
        )
    confirmation_metrics_frame = pd.DataFrame(confirmation_metrics)

    run_id = f"statistical_feature_ablation_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "statistical_feature_ablation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(output / "fold_predictions.parquet", index=False)
    component.to_parquet(output / "component_metrics.parquet", index=False)
    net.to_parquet(output / "net_metrics.parquet", index=False)
    decision_summary.to_parquet(output / "decision_summary.parquet", index=False)
    discovery_summary.to_parquet(output / "discovery_summary.parquet", index=False)
    confirmation_predictions.to_parquet(
        output / "confirmation_predictions.parquet", index=False
    )
    confirmation_metrics_frame.to_parquet(
        output / "confirmation_metrics.parquet", index=False
    )
    run = {
        "run_id": run_id,
        "model_family": "frozen_decomposed_statistical_aio_feature_ablation",
        "status": "research_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "offense_model": {"family": "histogram_gbm", "learning_rate": 0.03, "max_leaf_nodes": 7, "l2_regularization": 1.0},
            "defense_model": {"family": "ridge", "alpha": 3000.0},
            "feature_groups": {name: list(values) for name, values in groups.items()},
            "ungrouped_features": sorted(set(feature_columns) - {feature for values in groups.values() for feature in values}),
            "test_window_ends": list(test_window_ends),
            "target_window_seasons": target_window_seasons,
            "feature_selection_folds": list(test_window_ends[:-1]),
            "feature_confirmation_fold": confirmation_end,
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {
            "decisions": decision_summary.to_dict(orient="records"),
            "robust_drops": decision_summary.loc[decision_summary["robust_drop"]].to_dict(orient="records"),
            "selected_groups_from_discovery": selected_groups,
            "optimized_feature_counts": {
                target: len(values) for target, values in optimized_features.items()
            },
            "confirmation": confirmation_metrics_frame.to_dict(orient="records"),
        },
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run


def fit_optimized_statistical_aio(
    features_path: str | Path,
    targets_path: str | Path,
    ablation_run_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    ablation_path = Path(ablation_run_path)
    ablation = json.loads((ablation_path / "run.json").read_text())
    hashes = ablation["config"]["source_hashes"]
    if hashes["features"] != sha256_file(features_path) or hashes["targets"] != sha256_file(targets_path):
        raise ValueError("Optimized AIO inputs do not match the ablation run.")
    panel = _load_panel(features_path, targets_path)
    all_features = tuple(
        column
        for column in pd.read_parquet(features_path).columns
        if column not in FORBIDDEN_PRIMARY_FEATURES
    )
    groups = ablation["config"]["feature_groups"]
    selected = ablation["metrics"]["selected_groups_from_discovery"]
    optimized_features = {}
    for target_name in ("offense", "defense"):
        dropped = {
            feature
            for group in selected[target_name]
            for feature in groups[group]
        }
        optimized_features[target_name] = tuple(
            feature for feature in all_features if feature not in dropped
        )

    run_id = f"statistical_aio_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "statistical_aio" / run_id
    output.mkdir(parents=True, exist_ok=False)
    models = {}
    for target_name, target_column in {
        "offense": "target_offense",
        "defense": "target_defense",
    }.items():
        model = _fit_model(
            _frozen_model(target_name),
            panel,
            optimized_features[target_name],
            target_column,
        )
        path = output / f"model_{target_name}.joblib"
        joblib.dump(model, path)
        models[target_name] = {
            "path": str(path.resolve()),
            "features": list(optimized_features[target_name]),
            "feature_count": len(optimized_features[target_name]),
        }
    run = {
        "run_id": run_id,
        "model_family": "decomposed_statistical_aio",
        "status": "research_challenger",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "offense_model": ablation["config"]["offense_model"],
            "defense_model": ablation["config"]["defense_model"],
            "selected_feature_drops": selected,
            "parent_ablation_run": ablation["run_id"],
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "confirmation": ablation["metrics"]["confirmation"],
        "models": models,
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
