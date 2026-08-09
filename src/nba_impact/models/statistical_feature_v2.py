"""Feature-block comparison for the frozen decomposed statistical all-in-one."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_impact import _load_panel, _metrics
from nba_impact.models.statistical_model_comparison import _fit_model


OFFENSE_TEMPORAL = {
    "PTS_p100", "AST_p100", "TOV_p100", "FTA_p100", "FG3A_p100",
    "drives_p100", "touches_p100", "potential_assists_p100",
    "true_shooting_pct", "at_rim_frequency", "arc3_frequency",
    "shot_quality_average",
}
DEFENSE_TEMPORAL = {
    "STL_p100", "BLK_p100", "DREB_p100", "PF_p100",
    "rebound_contests_p100", "rebound_chances_p100", "recovered_blocks_p100",
}

SCORING_TOPOLOGY = (
    "self_created_point_share", "assisted_three_share", "pull_up_attempt_share",
    "foul_pressure_per_fga", "shot_profile_entropy", "effective_shot_zones",
    "rim_and_three_frequency", "midrange_frequency", "expected_zone_points",
)
CREATION_QUALITY = (
    "potential_assist_conversion", "assist_points_per_touch",
    "total_points_created_per_touch", "drive_pass_rate", "creation_load_p100",
    "live_ball_turnovers_per_creation", "interior_role_load",
)
DEFENSIVE_INTERACTIONS = (
    "stocks_p100_def", "defensive_activity_p100", "rebound_contest_share",
    "dreb_contested_share", "block_recovery_rate", "stocks_per_foul",
)
ROLE_CONTEXT = (
    "self_created_point_share", "pull_up_attempt_share", "shot_profile_entropy",
    "effective_shot_zones", "rim_and_three_frequency", "midrange_frequency",
    "creation_load_p100", "interior_role_load",
)


def _suffix_features(
    columns: set[str], roots: set[str], suffixes: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            column
            for column in columns
            if any(column == f"{root}_{suffix}" for root in roots for suffix in suffixes)
        )
    )


def candidate_feature_blocks(columns: tuple[str, ...]) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return predeclared basketball feature blocks for each component target."""
    available = set(columns)
    stable = tuple(sorted(column for column in available if column.endswith("_eb")))
    offense_relative = tuple(
        sorted(
            f"{root}_relative"
            for root in OFFENSE_TEMPORAL
            if f"{root}_relative" in available
        )
    )
    defense_relative = tuple(
        sorted(
            f"{root}_relative"
            for root in DEFENSE_TEMPORAL
            if f"{root}_relative" in available
        )
    )
    return {
        "offense": {
            "stabilized_ratios": stable,
            "era_relative": offense_relative,
            "recent_level": _suffix_features(available, OFFENSE_TEMPORAL, ("latest",)),
            "temporal_dynamics": _suffix_features(
                available, OFFENSE_TEMPORAL, ("trend", "volatility")
            ),
            "scoring_topology": tuple(
                feature for feature in SCORING_TOPOLOGY if feature in available
            ),
            "creation_quality": tuple(
                feature for feature in CREATION_QUALITY if feature in available
            ),
        },
        "defense": {
            "era_relative": defense_relative,
            "recent_level": _suffix_features(available, DEFENSE_TEMPORAL, ("latest",)),
            "temporal_dynamics": _suffix_features(
                available, DEFENSE_TEMPORAL, ("trend", "volatility")
            ),
            "defensive_interactions": tuple(
                feature for feature in DEFENSIVE_INTERACTIONS if feature in available
            ),
            "role_context": tuple(feature for feature in ROLE_CONTEXT if feature in available),
        },
    }


def _evaluate_side(
    panel: pd.DataFrame,
    *,
    target_name: str,
    features: tuple[str, ...],
    other_features: tuple[str, ...],
    test_ends: tuple[int, ...],
    target_window_seasons: int,
    candidate_predictions: dict[int, object] | None = None,
    other_predictions: dict[int, object] | None = None,
) -> pd.DataFrame:
    target_column = f"target_{target_name}"
    other_name = "defense" if target_name == "offense" else "offense"
    other_column = f"target_{other_name}"
    rows: list[dict] = []
    for test_end in test_ends:
        train = panel.loc[panel["Window_End"] <= test_end - target_window_seasons]
        test = panel.loc[panel["Window_End"] == test_end]
        if min(len(train), len(test)) == 0:
            raise ValueError(f"Feature-v2 fold {test_end} has an empty partition.")
        if candidate_predictions is None:
            candidate = _fit_model(_frozen_model(target_name), train, features, target_column)
            candidate_prediction = candidate.predict(test.loc[:, features])
        else:
            candidate_prediction = candidate_predictions[test_end]
        if other_predictions is None:
            other = _fit_model(_frozen_model(other_name), train, other_features, other_column)
            other_prediction = other.predict(test.loc[:, other_features])
        else:
            other_prediction = other_predictions[test_end]
        component = _metrics(
            test[target_column].to_numpy(),
            candidate_prediction,
            test["sample_weight"].to_numpy(),
        )
        net = _metrics(
            test["target_net"].to_numpy(),
            candidate_prediction + other_prediction,
            test["sample_weight"].to_numpy(),
        )
        rows.append(
            {
                "test_window_end": test_end,
                "target": target_name,
                "feature_count": len(features),
                "component_rmse": component["weighted_rmse"],
                "component_correlation": component["correlation"],
                "net_rmse": net["weighted_rmse"],
                "net_correlation": net["correlation"],
            }
        )
    return pd.DataFrame(rows)


def _strictly_improves(candidate: pd.DataFrame, baseline: pd.DataFrame) -> bool:
    merged = candidate.merge(
        baseline[["test_window_end", "component_rmse"]],
        on="test_window_end",
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    delta = merged["component_rmse"] - merged["component_rmse_baseline"]
    return bool(delta.lt(0).all() and delta.mean() < 0)


def _fit_final_models(
    panel: pd.DataFrame,
    selected_features: dict[str, tuple[str, ...]],
    output: Path,
) -> dict[str, dict]:
    models: dict[str, dict] = {}
    for target_name in ("offense", "defense"):
        features = selected_features[target_name]
        model = _fit_model(
            _frozen_model(target_name), panel, features, f"target_{target_name}"
        )
        path = output / f"model_{target_name}.joblib"
        joblib.dump(model, path)
        models[target_name] = {
            "path": str(path.resolve()),
            "features": list(features),
            "feature_count": len(features),
        }
    return models


def run_statistical_feature_v2_comparison(
    features_path: str | Path,
    targets_path: str | Path,
    baseline_run_path: str | Path,
    *,
    artifact_root: str | Path,
    discovery_window_ends: tuple[int, ...] = (2022, 2023),
    confirmation_window_end: int = 2024,
    first_complete_tracking_window: int = 2016,
    target_window_seasons: int = 3,
) -> dict:
    """Select v2 blocks on discovery folds and score confirmation once."""
    baseline_path = Path(baseline_run_path)
    baseline = json.loads((baseline_path / "run.json").read_text())
    baseline_features = {
        target: tuple(baseline["models"][target]["features"])
        for target in ("offense", "defense")
    }
    feature_columns = tuple(pd.read_parquet(features_path).columns)
    panel = _load_panel(features_path, targets_path)
    panel = panel.loc[panel["Window_End"] >= first_complete_tracking_window].copy()
    blocks = candidate_feature_blocks(feature_columns)
    if missing := sorted(
        set(baseline_features["offense"] + baseline_features["defense"])
        - set(panel.columns)
    ):
        raise ValueError(f"Feature-v2 panel is missing baseline features {missing}.")

    block_rows: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    selected_features = dict(baseline_features)
    selected_blocks: dict[str, list[str]] = {"offense": [], "defense": []}

    baseline_predictions: dict[str, dict[int, object]] = {
        "offense": {},
        "defense": {},
    }
    for target_name in ("offense", "defense"):
        for test_end in discovery_window_ends:
            train = panel.loc[
                panel["Window_End"] <= test_end - target_window_seasons
            ]
            test = panel.loc[panel["Window_End"] == test_end]
            model = _fit_model(
                _frozen_model(target_name),
                train,
                baseline_features[target_name],
                f"target_{target_name}",
            )
            baseline_predictions[target_name][test_end] = model.predict(
                test.loc[:, baseline_features[target_name]]
            )

    for target_name in ("offense", "defense"):
        other_name = "defense" if target_name == "offense" else "offense"
        current_features = selected_features[target_name]
        current_metrics = _evaluate_side(
            panel,
            target_name=target_name,
            features=current_features,
            other_features=baseline_features[other_name],
            test_ends=discovery_window_ends,
            target_window_seasons=target_window_seasons,
            candidate_predictions=baseline_predictions[target_name],
            other_predictions=baseline_predictions[other_name],
        )
        current_metrics["block"] = "baseline"
        current_metrics["stage"] = "discovery"
        block_rows.append(current_metrics)
        baseline_metrics = current_metrics.copy()

        one_at_a_time: list[tuple[float, str, tuple[str, ...], pd.DataFrame]] = []
        for block_name, additions in blocks[target_name].items():
            candidate_features = tuple(dict.fromkeys((*current_features, *additions)))
            metrics = _evaluate_side(
                panel,
                target_name=target_name,
                features=candidate_features,
                other_features=baseline_features[other_name],
                test_ends=discovery_window_ends,
                target_window_seasons=target_window_seasons,
                other_predictions=baseline_predictions[other_name],
            )
            metrics["block"] = block_name
            metrics["stage"] = "discovery_single_block"
            block_rows.append(metrics)
            comparison = metrics.merge(
                current_metrics[["test_window_end", "component_rmse"]],
                on="test_window_end",
                suffixes=("", "_baseline"),
                validate="one_to_one",
            )
            mean_delta = float(
                (comparison["component_rmse"] - comparison["component_rmse_baseline"]).mean()
            )
            one_at_a_time.append((mean_delta, block_name, additions, metrics))
            accepted = _strictly_improves(metrics, current_metrics)
            selection_rows.append(
                {
                    "target": target_name,
                    "block": block_name,
                    "accepted": accepted,
                    "feature_count_before": len(current_features),
                    "feature_count_after": len(candidate_features),
                    "mean_component_rmse_before": float(current_metrics["component_rmse"].mean()),
                    "mean_component_rmse_after": float(metrics["component_rmse"].mean()),
                }
            )
            if accepted:
                selected_blocks[target_name].append(block_name)

        selected_additions = tuple(
            feature
            for block_name in selected_blocks[target_name]
            for feature in blocks[target_name][block_name]
        )
        current_features = tuple(dict.fromkeys((*current_features, *selected_additions)))
        if selected_additions:
            current_metrics = _evaluate_side(
                panel,
                target_name=target_name,
                features=current_features,
                other_features=baseline_features[other_name],
                test_ends=discovery_window_ends,
                target_window_seasons=target_window_seasons,
                other_predictions=baseline_predictions[other_name],
            )
            if not _strictly_improves(current_metrics, baseline_metrics):
                best = min(
                    (
                        candidate
                        for candidate in one_at_a_time
                        if _strictly_improves(candidate[3], baseline_metrics)
                    ),
                    default=None,
                    key=lambda candidate: candidate[0],
                )
                if best is None:
                    selected_blocks[target_name] = []
                    current_features = baseline_features[target_name]
                else:
                    _, block_name, additions, current_metrics = best
                    selected_blocks[target_name] = [block_name]
                    current_features = tuple(
                        dict.fromkeys((*baseline_features[target_name], *additions))
                    )
            current_metrics["block"] = "selected_combination"
            current_metrics["stage"] = "discovery_combination"
            block_rows.append(current_metrics.copy())
        selected_features[target_name] = current_features

    confirmation_metrics: list[dict] = []
    confirmation_test = panel.loc[
        panel["Window_End"].eq(confirmation_window_end)
    ].copy()
    prediction_frame = confirmation_test[
        [
            "PLAYER_ID",
            "Window_End",
            "target_offense",
            "target_defense",
            "target_net",
            "sample_weight",
        ]
    ].copy()
    train = panel.loc[
        panel["Window_End"] <= confirmation_window_end - target_window_seasons
    ]
    for target_name in ("offense", "defense"):
        for variant, features in {
            "baseline": baseline_features[target_name],
            "v2": selected_features[target_name],
        }.items():
            model = _fit_model(
                _frozen_model(target_name), train, features, f"target_{target_name}"
            )
            prediction = model.predict(confirmation_test.loc[:, features])
            prediction_frame[f"prediction_{target_name}_{variant}"] = prediction
            confirmation_metrics.append(
                {
                    "target": target_name,
                    "variant": variant,
                    "feature_count": len(features),
                    **_metrics(
                        prediction_frame[f"target_{target_name}"].to_numpy(),
                        prediction,
                        prediction_frame["sample_weight"].to_numpy(),
                    ),
                }
            )
    for variant in ("baseline", "v2"):
        prediction = (
            prediction_frame[f"prediction_offense_{variant}"]
            + prediction_frame[f"prediction_defense_{variant}"]
        )
        confirmation_metrics.append(
            {
                "target": "net",
                "variant": variant,
                "feature_count": sum(
                    len(values)
                    for values in (
                        baseline_features.values()
                        if variant == "baseline"
                        else selected_features.values()
                    )
                ),
                **_metrics(
                    prediction_frame["target_net"].to_numpy(),
                    prediction.to_numpy(),
                    prediction_frame["sample_weight"].to_numpy(),
                ),
            }
        )
    confirmation = pd.DataFrame(confirmation_metrics)
    net_scores = confirmation.loc[confirmation["target"].eq("net")].set_index("variant")
    confirmed = bool(
        net_scores.loc["v2", "weighted_rmse"]
        < net_scores.loc["baseline", "weighted_rmse"]
    )

    run_id = f"statistical_feature_v2_comparison_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "statistical_feature_v2" / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.concat(block_rows, ignore_index=True).to_parquet(
        output / "block_metrics.parquet", index=False
    )
    pd.DataFrame(selection_rows).to_parquet(output / "selection_trace.parquet", index=False)
    confirmation.to_parquet(output / "confirmation_metrics.parquet", index=False)
    prediction_frame.to_parquet(output / "confirmation_predictions.parquet", index=False)
    models = _fit_final_models(panel, selected_features, output) if confirmed else {}

    run = {
        "run_id": run_id,
        "model_family": "frozen_decomposed_statistical_aio_feature_v2",
        "estimand": "three_season_normal_rapm_offense_defense_and_net",
        "status": "research_challenger" if confirmed else "rejected",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "offense_model": {
                "family": "histogram_gbm",
                "learning_rate": 0.03,
                "max_leaf_nodes": 7,
                "l2_regularization": 1.0,
            },
            "defense_model": {"family": "ridge", "alpha": 3000.0},
            "discovery_window_ends": list(discovery_window_ends),
            "confirmation_window_end": confirmation_window_end,
            "target_window_seasons": target_window_seasons,
            "input_window_seasons": 3,
            "features_path": str(Path(features_path).resolve()),
            "targets_path": str(Path(targets_path).resolve()),
            "baseline_run_path": str(baseline_path.resolve()),
            "sample_weight": "sqrt(min(Poss_Off, Poss_Def)); not an input feature",
            "forbidden_primary_inputs": [
                "age", "experience", "height", "listed_position", "minutes", "games",
                "on_off_rating", "plus_minus",
            ],
            "selection_rule": "accept block only when component RMSE improves every discovery fold",
            "candidate_blocks": {
                target: {name: list(values) for name, values in target_blocks.items()}
                for target, target_blocks in blocks.items()
            },
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "baseline_run": sha256_file(baseline_path / "run.json"),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "selected_blocks": selected_blocks,
        "selected_features": {target: list(values) for target, values in selected_features.items()},
        "confirmation": confirmation.to_dict(orient="records"),
        "confirmed": confirmed,
        "metrics": {
            "joined_rows": int(len(panel)),
            "players": int(panel["PLAYER_ID"].nunique()),
            "selected_blocks": selected_blocks,
            "confirmation": confirmation.to_dict(orient="records"),
            "confirmed": confirmed,
        },
        "models": models,
        "caveats": [
            (
                "The 2024 fold was held out from this feature-block selection, "
                "but earlier model-family work already inspected it."
            ),
            (
                "The historical normal-RAPM labels end in 2024 and may not be "
                "the best available RAPM target run."
            ),
            (
                "Public box and tracking inputs contain much less direct "
                "defensive information than offensive information."
            ),
            (
                "The v2 feature contract neutral-fills missing engineered values "
                "and does not expose age, minutes, games, or listed position."
            ),
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
