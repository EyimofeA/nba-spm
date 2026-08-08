"""Chronological win-probability ablations with time-safe pregame Elo."""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.win_probability import (
    CHECKPOINTS,
    _checkpoint_rows,
    _metrics,
    make_features,
    sample_game_states,
)


ELO_FEATURE_COLUMNS = ("pregame_elo_diff", "pregame_elo_remaining")


def build_pregame_elo(
    games: pd.DataFrame,
    *,
    initial_rating: float = 1500.0,
    k_factor: float = 20.0,
    home_advantage: float = 60.0,
    offseason_regression: float = 0.25,
) -> pd.DataFrame:
    """Calculate ratings before each date, then update the whole date as a batch."""
    required = {
        "game_id", "game_date", "season_start", "home_team_id", "away_team_id", "home_win"
    }
    missing = required - set(games.columns)
    if missing:
        raise ValueError(f"Game dimension is missing Elo columns: {sorted(missing)}")
    ordered = games.sort_values(["season_start", "game_date", "game_id"], kind="stable").copy()
    ratings: defaultdict[int, float] = defaultdict(lambda: initial_rating)
    rows: list[dict] = []
    active_season: int | None = None
    for (season_start, game_date), date_games in ordered.groupby(
        ["season_start", "game_date"], sort=True
    ):
        season_start = int(season_start)
        if active_season is None:
            active_season = season_start
        elif season_start != active_season:
            for team_id in list(ratings):
                ratings[team_id] = initial_rating + (1.0 - offseason_regression) * (
                    ratings[team_id] - initial_rating
                )
            active_season = season_start

        daily_deltas: defaultdict[int, float] = defaultdict(float)
        for game in date_games.itertuples(index=False):
            home_id = int(game.home_team_id)
            away_id = int(game.away_team_id)
            home_rating = float(ratings[home_id])
            away_rating = float(ratings[away_id])
            difference = home_rating - away_rating
            expected = 1.0 / (1.0 + 10.0 ** (-(difference + home_advantage) / 400.0))
            outcome = float(bool(game.home_win))
            delta = k_factor * (outcome - expected)
            daily_deltas[home_id] += delta
            daily_deltas[away_id] -= delta
            rows.append(
                {
                    "game_id": str(game.game_id),
                    "game_date": game_date,
                    "pregame_home_elo": home_rating,
                    "pregame_away_elo": away_rating,
                    "pregame_elo_diff": difference / 400.0,
                }
            )
        for team_id, delta in daily_deltas.items():
            ratings[team_id] += delta
    return pd.DataFrame(rows)


def make_elo_features(states: pd.DataFrame) -> pd.DataFrame:
    features = make_features(states)
    regulation_remaining = pd.to_numeric(states["regulation_seconds_remaining"], errors="raise")
    period_remaining = pd.to_numeric(states["seconds_remaining_period"], errors="raise")
    effective_remaining = np.where(states["is_overtime"].astype(bool), period_remaining, regulation_remaining)
    elo = pd.to_numeric(states["pregame_elo_diff"], errors="raise").astype(float)
    features["pregame_elo_diff"] = elo
    features["pregame_elo_remaining"] = elo * np.sqrt(np.maximum(effective_remaining, 0.0) / 2880.0)
    return features


def _fit(features: pd.DataFrame, outcome: np.ndarray) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
        ]
    )
    return model.fit(features, outcome)


def _paired_game_bootstrap(
    predictions: pd.DataFrame, *, repetitions: int, seed: int
) -> dict[str, float | list[float]]:
    rows = predictions.copy()
    rows["state_loss"] = (rows["home_win"] - rows["probability_state_only"]) ** 2
    rows["elo_loss"] = (rows["home_win"] - rows["probability_state_plus_elo"]) ** 2
    game_delta = rows.groupby("game_id").apply(
        lambda group: float(group["elo_loss"].mean() - group["state_loss"].mean()),
        include_groups=False,
    ).to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        draws[index] = rng.choice(game_delta, size=len(game_delta), replace=True).mean()
    return {
        "games": int(len(game_delta)),
        "mean_game_brier_delta_elo_minus_state": float(game_delta.mean()),
        "probability_elo_better": float((draws < 0).mean()),
        "delta_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
    }


def run_win_probability_elo_ablation(
    event_states_path: str | Path,
    game_dim_path: str | Path,
    *,
    train_season_labels: tuple[str, ...],
    test_season_labels: tuple[str, ...],
    artifact_root: str | Path,
    interval_seconds: int = 30,
    bootstrap_repetitions: int = 5000,
    seed: int = 7,
) -> dict:
    event_columns = [
        "event_id", "game_id", "season_label", "season_type", "actionId", "period",
        "seconds_remaining_period", "regulation_seconds_remaining", "seconds_elapsed_game",
        "is_overtime", "home_score_diff_after", "home_win", "is_terminal_event",
    ]
    events = pd.read_parquet(event_states_path, columns=event_columns)
    games = pd.read_parquet(game_dim_path)
    elo = build_pregame_elo(games)
    sampled = sample_game_states(events, interval_seconds=interval_seconds).merge(
        elo[["game_id", "pregame_home_elo", "pregame_away_elo", "pregame_elo_diff"]],
        on="game_id", validate="many_to_one",
    )
    train = sampled.loc[
        sampled["season_label"].isin(train_season_labels) & ~sampled["is_terminal_event"]
    ].copy()
    test = sampled.loc[
        sampled["season_label"].isin(test_season_labels) & ~sampled["is_terminal_event"]
    ].copy()
    if train.empty or test.empty:
        raise ValueError("Training and test season selections must both contain nonterminal states.")
    if set(train["game_id"]) & set(test["game_id"]):
        raise ValueError("A game appears in both training and test data.")
    y_train = train["home_win"].astype(int).to_numpy()
    y_test = test["home_win"].astype(int).to_numpy()
    models = {
        "state_only": _fit(make_features(train), y_train),
        "state_plus_elo": _fit(make_elo_features(train), y_train),
    }
    test["probability_state_only"] = models["state_only"].predict_proba(make_features(test))[:, 1]
    test["probability_state_plus_elo"] = models["state_plus_elo"].predict_proba(make_elo_features(test))[:, 1]

    variant_metrics: dict[str, dict] = {}
    checkpoint_rows: list[dict] = []
    for variant in models:
        probability_column = f"probability_{variant}"
        variant_metrics[variant] = _metrics(y_test, test[probability_column].to_numpy())
        for checkpoint_name, remaining in CHECKPOINTS.items():
            checkpoint = _checkpoint_rows(test, remaining)
            checkpoint_rows.append(
                {
                    "variant": variant,
                    "checkpoint": checkpoint_name,
                    "regulation_seconds_remaining": remaining,
                    **_metrics(
                        checkpoint["home_win"].astype(int).to_numpy(),
                        checkpoint[probability_column].to_numpy(),
                    ),
                }
            )
    paired = _paired_game_bootstrap(test, repetitions=bootstrap_repetitions, seed=seed)
    state_brier = variant_metrics["state_only"]["brier"]
    elo_brier = variant_metrics["state_plus_elo"]["brier"]
    paired["relative_brier_improvement"] = float((state_brier - elo_brier) / state_brier)

    run_id = f"wp_elo_ablation_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability_ablation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for variant, model in models.items():
        joblib.dump(model, output / f"{variant}.joblib")
    test.to_parquet(output / "test_predictions.parquet", index=False)
    pd.DataFrame(checkpoint_rows).to_parquet(output / "checkpoint_metrics.parquet", index=False)
    config = {
        "train_season_labels": list(train_season_labels),
        "test_season_labels": list(test_season_labels),
        "interval_seconds": interval_seconds,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "elo": {"initial_rating": 1500.0, "k_factor": 20.0, "home_advantage": 60.0, "offseason_regression": 0.25},
        "features": {"state_only": list(make_features(train).columns), "state_plus_elo": list(make_elo_features(train).columns)},
        "event_states_sha256": sha256_file(event_states_path),
        "game_dim_sha256": sha256_file(game_dim_path),
        "source_code_sha256": sha256_file(Path(__file__)),
        "baseline_source_code_sha256": sha256_file(Path(__file__).with_name("win_probability.py")),
    }
    metrics = {
        "train_games": int(train["game_id"].nunique()), "test_games": int(test["game_id"].nunique()),
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "nonterminal_variants": variant_metrics, "paired_game_bootstrap": paired,
        "checkpoints": checkpoint_rows,
    }
    run = {
        "run_id": run_id, "model_family": "win_probability_logistic_elo_ablation",
        "estimand": "probability_home_team_wins_given_post_action_state_and_time_safe_pregame_elo",
        "status": "research_diagnostic_single_outer_fold", "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "metrics": metrics, "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
