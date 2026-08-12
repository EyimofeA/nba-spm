"""Fixed-seed feed-forward MLP parity test for win probability."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.win_probability import _metrics
from nba_impact.models.win_probability_ablation import _fit
from nba_impact.models.win_probability_lineup import (
    _paired_bootstrap,
    make_rolling_context_features,
)
from nba_impact.models.win_probability_stage1 import FOLDS, _build_states


SEEDS = (7, 17, 29, 43, 71)


def build_mlp(*, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=1024,
                    learning_rate_init=1e-3,
                    max_iter=100,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=8,
                    random_state=seed,
                ),
            ),
        ]
    )


def run_win_probability_mlp_comparison(
    event_states_path: str | Path,
    game_dim_path: str | Path,
    *,
    artifact_root: str | Path,
    interval_seconds: int = 30,
    bootstrap_repetitions: int = 5000,
) -> dict:
    states = _build_states(event_states_path, game_dim_path, interval_seconds)
    states = states.loc[~states["is_terminal_event"]].copy()
    run_id = f"wp_mlp_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability_mlp" / run_id
    output.mkdir(parents=True, exist_ok=False)
    fold_results = []
    pooled_predictions = []

    for fold_number, (train_season, test_season) in enumerate(FOLDS, start=1):
        train = states.loc[states["season_label"].eq(train_season)].copy()
        test = states.loc[states["season_label"].eq(test_season)].copy()
        x_train = make_rolling_context_features(train)
        x_test = make_rolling_context_features(test)
        y_train = train["home_win"].astype(int).to_numpy()
        y_test = test["home_win"].astype(int).to_numpy()
        logistic = _fit(x_train, y_train)
        logistic_probability = logistic.predict_proba(x_test)[:, 1]
        joblib.dump(logistic, output / f"fold_{fold_number}_logistic.joblib")

        seed_probabilities = []
        seed_results = []
        for seed in SEEDS:
            model = build_mlp(seed=seed)
            start = time.perf_counter()
            model.fit(x_train, y_train)
            elapsed = float(time.perf_counter() - start)
            probability = model.predict_proba(x_test)[:, 1]
            seed_probabilities.append(probability)
            seed_results.append(
                {
                    "seed": seed,
                    "fit_seconds": elapsed,
                    "iterations": int(model.named_steps["mlp"].n_iter_),
                    "metrics": _metrics(y_test, probability),
                }
            )
            joblib.dump(model, output / f"fold_{fold_number}_mlp_seed_{seed}.joblib")

        ensemble_probability = np.mean(np.vstack(seed_probabilities), axis=0)
        predictions = test[["game_id", "season_label", "home_win"]].copy()
        predictions["probability_logistic"] = logistic_probability
        predictions["probability_mlp_ensemble"] = ensemble_probability
        predictions["outer_fold"] = fold_number
        pooled_predictions.append(predictions)
        fold_results.append(
            {
                "outer_fold": fold_number,
                "train_season": train_season,
                "test_season": test_season,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "logistic": _metrics(y_test, logistic_probability),
                "mlp_ensemble": _metrics(y_test, ensemble_probability),
                "mlp_seeds": seed_results,
                "paired_mlp_ensemble_vs_logistic": _paired_bootstrap(
                    predictions,
                    "probability_logistic",
                    "probability_mlp_ensemble",
                    repetitions=bootstrap_repetitions,
                    seed=7,
                ),
            }
        )

    pooled = pd.concat(pooled_predictions, ignore_index=True)
    pooled.to_parquet(output / "test_predictions.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "win_probability_feed_forward_mlp_parity",
        "estimand": "post_action_home_win_probability_on_frozen_starter_free_states",
        "status": "two_outer_fold_five_seed_research_comparison",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "folds": [{"train": train, "test": test} for train, test in FOLDS],
            "seeds": list(SEEDS),
            "interval_seconds": interval_seconds,
            "bootstrap_repetitions": bootstrap_repetitions,
            "architecture": {
                "hidden_layers": [64, 64],
                "activation": "relu",
                "residual_connections": False,
                "batch_size": 1024,
                "max_iter": 100,
                "early_stopping": True,
            },
            "source_hashes": {
                "event_states": sha256_file(event_states_path),
                "game_dim": sha256_file(game_dim_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {
            "folds": fold_results,
            "pooled_paired_mlp_ensemble_vs_logistic": _paired_bootstrap(
                pooled,
                "probability_logistic",
                "probability_mlp_ensemble",
                repetitions=bootstrap_repetitions,
                seed=7,
            ),
        },
        "caveats": [
            "Seeds quantify optimizer variability and are not independent outer folds.",
            "This is a feed-forward MLP, not the preregistered residual MLP, because PyTorch is unavailable.",
            "No hyperparameter is selected on either outer test season.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
