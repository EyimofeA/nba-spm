"""Frozen Stage 1 nonlinear comparison for starter-free win probability."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.win_probability import (
    CHECKPOINTS,
    _checkpoint_rows,
    _metrics,
    sample_game_states,
)
from nba_impact.models.win_probability_ablation import _fit, build_pregame_elo
from nba_impact.models.win_probability_lineup import (
    _paired_bootstrap,
    build_pregame_team_context,
    make_rolling_context_features,
)


FOLDS = (("2023-24", "2024-25"), ("2024-25", "2025-26"))


def build_stage1_models(*, seed: int) -> dict[str, object]:
    """Return fixed candidates; no held-out-season tuning is permitted."""
    gam = Pipeline(
        [
            (
                "splines",
                SplineTransformer(
                    n_knots=5,
                    degree=3,
                    knots="quantile",
                    include_bias=False,
                ),
            ),
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
        ]
    )
    gbm = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        max_depth=6,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    return {"logistic": None, "spline_gam": gam, "hist_gbm": gbm}


def _build_states(
    event_states_path: str | Path, game_dim_path: str | Path, interval: int
) -> pd.DataFrame:
    event_columns = [
        "event_id",
        "game_id",
        "season_label",
        "actionId",
        "period",
        "seconds_remaining_period",
        "regulation_seconds_remaining",
        "seconds_elapsed_game",
        "is_overtime",
        "home_score_diff_after",
        "home_win",
        "is_terminal_event",
    ]
    events = pd.read_parquet(event_states_path, columns=event_columns)
    games = pd.read_parquet(game_dim_path)
    elo = build_pregame_elo(games)
    context = build_pregame_team_context(games)
    return (
        sample_game_states(events, interval_seconds=interval)
        .merge(
            elo[["game_id", "pregame_elo_diff"]], on="game_id", validate="many_to_one"
        )
        .merge(context, on="game_id", validate="many_to_one")
    )


def run_win_probability_stage1_comparison(
    event_states_path: str | Path,
    game_dim_path: str | Path,
    *,
    artifact_root: str | Path,
    interval_seconds: int = 30,
    bootstrap_repetitions: int = 5000,
    seed: int = 7,
) -> dict:
    states = _build_states(event_states_path, game_dim_path, interval_seconds)
    states = states.loc[~states["is_terminal_event"]].copy()
    run_id = f"wp_stage1_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability_stage1" / run_id
    output.mkdir(parents=True, exist_ok=False)
    fold_metrics: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_columns: list[str] | None = None

    for fold_number, (train_season, test_season) in enumerate(FOLDS, start=1):
        train = states.loc[states["season_label"].eq(train_season)].copy()
        test = states.loc[states["season_label"].eq(test_season)].copy()
        if train.empty or test.empty:
            raise ValueError(f"Fold {train_season} -> {test_season} has empty states.")
        features_train = make_rolling_context_features(train)
        features_test = make_rolling_context_features(test)
        if list(features_train.columns) != list(features_test.columns):
            raise ValueError("Train and test feature columns differ.")
        feature_columns = list(features_train.columns)
        outcome_train = train["home_win"].astype(int).to_numpy()
        outcome_test = test["home_win"].astype(int).to_numpy()
        models = build_stage1_models(seed=seed)
        timing: dict[str, float] = {}
        start = time.perf_counter()
        models["logistic"] = _fit(features_train, outcome_train)
        timing["logistic"] = float(time.perf_counter() - start)
        for name in ("spline_gam", "hist_gbm"):
            start = time.perf_counter()
            models[name].fit(features_train, outcome_train)
            timing[name] = float(time.perf_counter() - start)

        predictions = test[["game_id", "season_label", "home_win"]].copy()
        metrics: dict[str, dict] = {}
        for name, model in models.items():
            probability = model.predict_proba(features_test)[:, 1]
            predictions[f"probability_{name}"] = probability
            metrics[name] = _metrics(outcome_test, probability)
            joblib.dump(model, output / f"fold_{fold_number}_{name}.joblib")

        paired = {
            name: _paired_bootstrap(
                predictions,
                "probability_logistic",
                f"probability_{name}",
                repetitions=bootstrap_repetitions,
                seed=seed,
            )
            for name in ("spline_gam", "hist_gbm")
        }
        checkpoints = []
        for checkpoint, remaining in CHECKPOINTS.items():
            rows = _checkpoint_rows(
                test.assign(**predictions.filter(like="probability_")), remaining
            )
            checkpoints.append(
                {
                    "checkpoint": checkpoint,
                    **{
                        name: _metrics(
                            rows["home_win"].astype(int).to_numpy(),
                            rows[f"probability_{name}"].to_numpy(),
                        )
                        for name in models
                    },
                }
            )
        predictions["outer_fold"] = fold_number
        prediction_frames.append(predictions)
        fold_metrics.append(
            {
                "outer_fold": fold_number,
                "train_season": train_season,
                "test_season": test_season,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_games": int(train["game_id"].nunique()),
                "test_games": int(test["game_id"].nunique()),
                "variants": metrics,
                "paired_vs_logistic": paired,
                "fit_seconds": timing,
                "checkpoints": checkpoints,
            }
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled = {
        name: _paired_bootstrap(
            all_predictions,
            "probability_logistic",
            f"probability_{name}",
            repetitions=bootstrap_repetitions,
            seed=seed,
        )
        for name in ("spline_gam", "hist_gbm")
    }
    all_predictions.to_parquet(output / "test_predictions.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "win_probability_stage1_nonlinear_comparison",
        "estimand": "post_action_home_win_probability_on_frozen_starter_free_states",
        "status": "two_outer_fold_research_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "folds": [{"train": train, "test": test} for train, test in FOLDS],
            "interval_seconds": interval_seconds,
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
            "features": feature_columns,
            "spline_gam": {"n_knots": 5, "degree": 3, "C": 1.0},
            "hist_gbm": {
                "learning_rate": 0.05,
                "max_iter": 200,
                "max_leaf_nodes": 15,
                "max_depth": 6,
                "min_samples_leaf": 50,
                "l2_regularization": 1.0,
                "early_stopping": False,
            },
            "source_hashes": {
                "event_states": sha256_file(event_states_path),
                "game_dim": sha256_file(game_dim_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {"folds": fold_metrics, "pooled_paired_vs_logistic": pooled},
        "caveats": [
            "Hyperparameters are fixed and not selected on either outer test season.",
            "This compares architecture on identical inputs; it does not test sequence history.",
            "Uncalibrated candidates must satisfy the same calibration gate as logistic.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
