"""Out-of-sample grouped permutation audit for a frozen statistical AIO."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features_v2 import PRIMARY_PUBLIC_INSPIRED_FEATURES
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_impact import _load_panel, _metrics
from nba_impact.models.statistical_model_comparison import _fit_model


def feature_mechanism(feature: str) -> str:
    """Assign one mutually exclusive basketball mechanism to a feature."""
    lower = feature.lower()
    if feature in PRIMARY_PUBLIC_INSPIRED_FEATURES:
        return "public_composites"
    if any(token in lower for token in ("turnover", "tov", "travels", "offensive_foul")):
        return "ball_security"
    if any(token in lower for token in ("oreb", "dreb", "rebound", "boxout", "screen_")):
        return "rebounding_screening"
    if any(
        token in lower
        for token in ("stl", "blk", "block", "deflection", "charge", "loose_ball", "stocks")
    ):
        return "defensive_disruption"
    if any(token in lower for token in ("fta", "ftm", "foul_pressure", "shooting_foul", "pfd")):
        return "rim_pressure_free_throws"
    if any(
        token in lower
        for token in (
            "ast", "assist", "creation", "drive", "touch", "pass", "on_ball",
            "time_of_possession", "dribb", "paint_", "post_", "elbow_", "interior_role",
        )
    ):
        return "creation_passing_role"
    if any(
        token in lower
        for token in (
            "pts", "fg2", "fg3", "accuracy", "frequency", "shoot", "shot_", "rim_", "mid",
            "corner3", "arc3", "open", "tight", "catch", "pull_up", "spacing", "efg",
        )
    ):
        return "shooting_scoring_spacing"
    if any(token in lower for token in ("pf", "defensive_activity")):
        return "defensive_activity_fouls"
    return "other"


def grouped_permutation_importance(
    model,
    features: pd.DataFrame,
    actual: np.ndarray,
    weights: np.ndarray,
    groups: dict[str, tuple[str, ...]],
    *,
    repeats: int = 20,
    seed: int = 20260817,
) -> pd.DataFrame:
    """Jointly permute each correlated feature family and measure score loss."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    baseline_prediction = model.predict(features)
    baseline = _metrics(actual, baseline_prediction, weights)
    rows: list[dict] = []
    for group, columns in groups.items():
        if not columns:
            continue
        deltas = []
        correlation_deltas = []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 1009 * repeat + len(rows))
            order = rng.permutation(len(features))
            permuted = features.copy()
            permuted.loc[:, list(columns)] = features.iloc[order][list(columns)].to_numpy()
            score = _metrics(actual, model.predict(permuted), weights)
            deltas.append(score["weighted_rmse"] - baseline["weighted_rmse"])
            correlation_deltas.append(score["correlation"] - baseline["correlation"])
        rows.append(
            {
                "group": group,
                "feature_count": len(columns),
                "rmse_delta_mean": float(np.mean(deltas)),
                "rmse_delta_std": float(np.std(deltas, ddof=0)),
                "correlation_delta_mean": float(np.mean(correlation_deltas)),
                "baseline_rmse": baseline["weighted_rmse"],
                "baseline_correlation": baseline["correlation"],
            }
        )
    return pd.DataFrame(rows).sort_values("rmse_delta_mean", ascending=False).reset_index(drop=True)


def _mechanism_groups(features: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for feature in features:
        grouped.setdefault(feature_mechanism(feature), []).append(feature)
    return {name: tuple(values) for name, values in grouped.items()}


def run_statistical_interpretability(
    features_path: str | Path,
    targets_path: str | Path,
    reference_run_path: str | Path,
    *,
    artifact_root: str | Path,
    test_window_end: int = 2024,
    target_window_seasons: int = 3,
    first_complete_tracking_window: int = 2016,
    group_repeats: int = 20,
    individual_repeats: int = 3,
    seed: int = 20260817,
) -> dict:
    """Audit a frozen AIO on one reused diagnostic fold without changing it."""
    reference_path = Path(reference_run_path)
    reference = json.loads((reference_path / "run.json").read_text())
    selected = {
        side: tuple(reference["selected_features"][side]) for side in ("offense", "defense")
    }
    panel = _load_panel(features_path, targets_path)
    panel = panel.loc[panel["Window_End"].ge(first_complete_tracking_window)].copy()
    train = panel.loc[panel["Window_End"].le(test_window_end - target_window_seasons)]
    test = panel.loc[panel["Window_End"].eq(test_window_end)]
    if min(len(train), len(test)) == 0:
        raise ValueError("Interpretability fold has an empty train or test partition.")

    grouped_outputs = []
    individual_outputs = []
    baseline_rows = []
    for side in ("offense", "defense"):
        columns = selected[side]
        if missing := sorted(set(columns) - set(panel.columns)):
            raise ValueError(f"Interpretability input is missing {side} features {missing}.")
        model = _fit_model(_frozen_model(side), train, columns, f"target_{side}")
        x_test = test.loc[:, columns]
        actual = test[f"target_{side}"].to_numpy()
        weights = test["sample_weight"].to_numpy()
        baseline = _metrics(actual, model.predict(x_test), weights)
        baseline_rows.append({"target": side, **baseline})
        grouped = grouped_permutation_importance(
            model, x_test, actual, weights, _mechanism_groups(columns),
            repeats=group_repeats, seed=seed + (0 if side == "offense" else 100_000),
        )
        grouped.insert(0, "target", side)
        grouped_outputs.append(grouped)
        individual = grouped_permutation_importance(
            model, x_test, actual, weights,
            {feature: (feature,) for feature in columns},
            repeats=individual_repeats,
            seed=seed + (200_000 if side == "offense" else 300_000),
        ).rename(columns={"group": "feature"})
        individual.insert(0, "target", side)
        individual["mechanism"] = individual["feature"].map(feature_mechanism)
        individual_outputs.append(individual)

    group_frame = pd.concat(grouped_outputs, ignore_index=True)
    individual_frame = pd.concat(individual_outputs, ignore_index=True)
    baseline_frame = pd.DataFrame(baseline_rows)
    run_id = f"statistical_interpretability_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "statistical_interpretability" / run_id
    output.mkdir(parents=True, exist_ok=False)
    group_frame.to_parquet(output / "group_importance.parquet", index=False)
    individual_frame.to_parquet(output / "feature_importance.parquet", index=False)
    baseline_frame.to_parquet(output / "baseline_metrics.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "frozen_statistical_aio_interpretability_v1",
        "status": "diagnostic_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "test_window_end": test_window_end,
            "train_window_end_max": test_window_end - target_window_seasons,
            "target_window_seasons": target_window_seasons,
            "group_repeats": group_repeats,
            "individual_repeats": individual_repeats,
            "seed": seed,
            "source_hashes": {
                "features": sha256_file(features_path),
                "targets": sha256_file(targets_path),
                "reference_run": sha256_file(reference_path / "run.json"),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "baseline": baseline_frame.to_dict(orient="records"),
        },
        "selected_features": {side: list(values) for side, values in selected.items()},
        "group_importance_path": str(output / "group_importance.parquet"),
        "feature_importance_path": str(output / "feature_importance.parquet"),
        "baseline_metrics_path": str(output / "baseline_metrics.parquet"),
        "artifact_path": str(output),
        "caveats": [
            "The 2024 fold was already inspected and is diagnostic, not promotion evidence.",
            "Permutation importance measures dependence of this fitted model, not causal credit.",
            "Whole-family permutation preserves within-family rows but correlated families can share importance.",
            "Negative importance is possible from sampling noise or overfit features.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
