"""Chronological, player-neutral expected-points baseline for possessions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.possession_context import CAUSAL_FEATURE_COLUMNS


CATEGORICAL_FEATURES = (
    "period",
    "is_overtime",
    "offense_is_home",
    "previous_possession_points",
    "is_first_possession",
)
NUMERIC_FEATURES = (
    "seconds_remaining_period_start",
    "regulation_seconds_remaining_start",
    "offense_score_diff_start",
)
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FORBIDDEN_MODEL_FEATURES = {
    "points",
    "offense_team_id",
    "defense_team_id",
    "home_team_id",
    "away_team_id",
    "lineup_ready",
}


def _validate_context(context: pd.DataFrame) -> pd.DataFrame:
    required = {
        "possession_id", "game_id", "season_start", "season_type", "points",
        *CAUSAL_FEATURE_COLUMNS,
    }
    if missing := sorted(required - set(context.columns)):
        raise ValueError(f"Possession-start context is missing columns: {missing}.")
    if context.duplicated("possession_id").any():
        raise ValueError("Possession-start context has duplicate possession IDs.")
    if not set(MODEL_FEATURES).issubset(CAUSAL_FEATURE_COLUMNS):
        raise AssertionError("Expected-points model inputs escape the causal feature contract.")
    if forbidden := sorted(set(MODEL_FEATURES) & FORBIDDEN_MODEL_FEATURES):
        raise AssertionError(f"Expected-points model contains forbidden inputs: {forbidden}.")
    frame = context.copy()
    frame["season_start"] = pd.to_numeric(frame["season_start"], errors="raise").astype(int)
    values = ["points", *NUMERIC_FEATURES]
    for column in values:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[values].to_numpy()).all():
        raise ValueError("Possession-start context has non-finite expected-points fields.")
    if frame["points"].lt(0).any():
        raise ValueError("Possession-point targets must be non-negative.")
    return frame


def _feature_frame(context: pd.DataFrame) -> pd.DataFrame:
    features = context.loc[:, MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(str)
    return features


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    positive = np.clip(prediction, 1e-9, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
        "mean_poisson_deviance": float(mean_poisson_deviance(actual, positive)),
        "mean_actual_points": float(np.mean(actual)),
        "mean_expected_points": float(np.mean(prediction)),
        "mean_bias": float(np.mean(prediction - actual)),
    }


def _pipeline(alpha: float, max_iter: int) -> object:
    return make_pipeline(
        ColumnTransformer(
            [
                ("categorical", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL_FEATURES)),
                ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            ]
        ),
        PoissonRegressor(alpha=alpha, max_iter=max_iter, tol=1e-8),
    )


def build_expected_possession_points(
    context_path: str | Path,
    *,
    artifact_root: str | Path,
    test_seasons: tuple[int, ...] = (2024, 2025),
    alpha: float = 0.01,
    max_iter: int = 300,
) -> dict:
    """Cross-fit expected possession points in chronological whole-season folds.

    Training is limited to earlier regular seasons. The model never receives a
    player, team, lineup, or current-possession outcome feature.
    """
    if alpha < 0:
        raise ValueError("Poisson alpha must be non-negative.")
    source_path = Path(context_path)
    context = _validate_context(pd.read_parquet(source_path))
    regular = context.loc[context["season_type"].eq("regular")].copy()
    available_seasons = set(regular["season_start"])
    if missing := sorted(set(test_seasons) - available_seasons):
        raise ValueError(f"Expected-points test seasons are missing: {missing}.")
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict] = []
    for test_season in test_seasons:
        train = regular.loc[regular["season_start"].lt(test_season)].copy()
        test = regular.loc[regular["season_start"].eq(test_season)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Chronological fold for {test_season} has empty train or test rows.")
        model = _pipeline(alpha, max_iter)
        features_train = _feature_frame(train)
        features_test = _feature_frame(test)
        model.fit(features_train, train["points"])
        expected = model.predict(features_test)
        constant = np.full(len(test), float(train["points"].mean()))
        actual = test["points"].to_numpy(dtype=float)
        context_metrics = _metrics(actual, expected)
        constant_metrics = _metrics(actual, constant)
        fold_rows.append(
            {
                "test_season": int(test_season),
                "train_seasons": list(sorted(train["season_start"].unique())),
                "train_games": int(train["game_id"].nunique()),
                "test_games": int(test["game_id"].nunique()),
                "train_possessions": int(len(train)),
                "test_possessions": int(len(test)),
                "context": context_metrics,
                "constant": constant_metrics,
                "poisson_iterations": int(model[-1].n_iter_),
                "converged": bool(model[-1].n_iter_ < max_iter),
            }
        )
        output = test.loc[
            :, ["possession_id", "game_id", "season_start", "points", "lineup_ready"]
        ].copy()
        output["expected_points_context"] = expected
        output["expected_points_constant"] = constant
        output["residual_points_context"] = output["points"] - expected
        output["fold_test_season"] = test_season
        predictions.append(output)
    cross_fitted = pd.concat(predictions, ignore_index=True)
    if cross_fitted.duplicated("possession_id").any():
        raise ValueError("Cross-fitted expected-point predictions have duplicate possession IDs.")
    if not np.isfinite(
        cross_fitted[["expected_points_context", "expected_points_constant", "residual_points_context"]].to_numpy()
    ).all():
        raise ValueError("Cross-fitted expected-point predictions contain non-finite values.")
    if cross_fitted["expected_points_context"].le(0).any():
        raise ValueError("Poisson expected points must be positive.")
    folds = pd.DataFrame(
        [
            {
                "test_season": row["test_season"],
                "train_games": row["train_games"],
                "test_games": row["test_games"],
                "train_possessions": row["train_possessions"],
                "test_possessions": row["test_possessions"],
                "context_rmse": row["context"]["rmse"],
                "constant_rmse": row["constant"]["rmse"],
                "context_mae": row["context"]["mae"],
                "constant_mae": row["constant"]["mae"],
                "context_poisson_deviance": row["context"]["mean_poisson_deviance"],
                "constant_poisson_deviance": row["constant"]["mean_poisson_deviance"],
                "context_mean_bias": row["context"]["mean_bias"],
                "constant_mean_bias": row["constant"]["mean_bias"],
                "poisson_iterations": row["poisson_iterations"],
                "converged": row["converged"],
            }
            for row in fold_rows
        ]
    )
    config = {
        "test_seasons": list(test_seasons),
        "alpha": alpha,
        "max_iter": max_iter,
        "model_features": list(MODEL_FEATURES),
        "source_sha256": sha256_file(source_path),
        "builder_sha256": sha256_file(Path(__file__)),
        "season_type": "regular",
        "cross_fit": "train strictly on seasons before each test season",
    }
    run_id = "expected_possession_points_v1_" + hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:10]
    output_dir = Path(artifact_root) / "models" / "expected_possession_points" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    cross_fitted.to_parquet(output_dir / "cross_fitted_predictions.parquet", index=False)
    folds.to_parquet(output_dir / "fold_metrics.parquet", index=False)
    metrics = {
        "mean_context_rmse": float(folds["context_rmse"].mean()),
        "mean_constant_rmse": float(folds["constant_rmse"].mean()),
        "mean_context_mae": float(folds["context_mae"].mean()),
        "mean_constant_mae": float(folds["constant_mae"].mean()),
        "mean_context_poisson_deviance": float(folds["context_poisson_deviance"].mean()),
        "mean_constant_poisson_deviance": float(folds["constant_poisson_deviance"].mean()),
        "maximum_absolute_context_bias": float(folds["context_mean_bias"].abs().max()),
        "all_folds_converged": bool(folds["converged"].all()),
        "predicted_possessions": int(len(cross_fitted)),
        "predicted_games": int(cross_fitted["game_id"].nunique()),
    }
    run = {
        "run_id": run_id,
        "model_family": "player_neutral_possession_start_poisson",
        "estimand": "cross_fitted_expected_points_for_a_canonical_possession",
        "estimand_id": "player_neutral_expected_possession_points_v1",
        "status": "research_null",
        "evidence_status": "chronological_cross_fit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
        "artifact_path": str(output_dir.resolve()),
        "cross_fitted_predictions_path": str(
            (output_dir / "cross_fitted_predictions.parquet").resolve()
        ),
        "caveats": [
            "This is a possession-start expected-points baseline, not optical-tracking EPV.",
            "Expected points are player-neutral; no player, team, or lineup identity is modeled.",
            "Residual RAPM is not fit by this run and needs an identical-game comparison with normal RAPM.",
            "The source lake covers 2023-25 only and Season 2027 is not included.",
        ],
        "forbidden_interpretation": "Player value, shot quality, causal player credit, or residual-RAPM result.",
    }
    write_json_atomic(run, output_dir / "run.json")
    return run
