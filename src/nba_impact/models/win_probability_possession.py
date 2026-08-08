"""Leakage-safe possession-start win-probability ablation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.win_probability import CHECKPOINTS, _checkpoint_rows, _metrics
from nba_impact.models.win_probability_ablation import _fit, build_pregame_elo
from nba_impact.models.win_probability_lineup import (
    _paired_bootstrap,
    build_pregame_team_context,
    make_rolling_context_features,
)


def build_possession_start_states(
    possessions: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Construct state using only completed prior possessions and current control."""
    required = {
        "possession_id",
        "game_id",
        "possession_number",
        "season_label",
        "season_type",
        "period",
        "start_order_number",
        "start_seconds_elapsed",
        "offense_is_home",
        "points",
        "home_points",
        "away_points",
    }
    missing = required - set(possessions.columns)
    if missing:
        raise ValueError(
            f"Possessions are missing start-state columns: {sorted(missing)}"
        )
    ordered = possessions.sort_values(
        ["game_id", "possession_number", "start_order_number"], kind="stable"
    ).copy()
    if ordered.duplicated("possession_id").any():
        raise ValueError("Possession IDs must be unique.")
    if ordered["points"].lt(0).any():
        raise ValueError("Possession points must be nonnegative.")

    home_points = ordered["home_points"].astype(float).to_numpy()
    away_points = ordered["away_points"].astype(float).to_numpy()
    ordered["home_score_before"] = (
        pd.Series(home_points, index=ordered.index).groupby(ordered["game_id"]).cumsum()
        - home_points
    )
    ordered["away_score_before"] = (
        pd.Series(away_points, index=ordered.index).groupby(ordered["game_id"]).cumsum()
        - away_points
    )
    ordered["home_score_diff_before"] = (
        ordered["home_score_before"] - ordered["away_score_before"]
    )
    ordered["home_score_diff_after"] = ordered["home_score_diff_before"]
    ordered["seconds_elapsed_game"] = ordered["start_seconds_elapsed"].astype(float)
    ordered["is_overtime"] = ordered["period"].astype(int).gt(4)
    regulation = ~ordered["is_overtime"]
    ordered["regulation_seconds_remaining"] = np.where(
        regulation, np.maximum(2880.0 - ordered["seconds_elapsed_game"], 0.0), 0.0
    )
    period_start = np.where(
        regulation,
        (ordered["period"].astype(int) - 1) * 720.0,
        2880.0 + (ordered["period"].astype(int) - 5) * 300.0,
    )
    period_length = np.where(regulation, 720.0, 300.0)
    ordered["seconds_remaining_period"] = np.clip(
        period_length - (ordered["seconds_elapsed_game"] - period_start),
        0.0,
        period_length,
    )
    ordered["actionId"] = ordered["start_order_number"].astype(int)
    ordered["is_terminal_event"] = False

    dimension = games[["game_id", "home_win", "home_score", "away_score"]].copy()
    states = ordered.merge(dimension, on="game_id", validate="many_to_one")
    terminal_score = states.groupby("game_id", as_index=False).agg(
        reconstructed_home=("home_score_before", "last"),
        reconstructed_away=("away_score_before", "last"),
        final_home_points=("home_points", "last"),
        final_away_points=("away_points", "last"),
        home_score=("home_score", "first"),
        away_score=("away_score", "first"),
    )
    terminal_score["reconstructed_home"] += terminal_score["final_home_points"]
    terminal_score["reconstructed_away"] += terminal_score["final_away_points"]
    mismatch = terminal_score["reconstructed_home"].ne(
        terminal_score["home_score"]
    ) | terminal_score["reconstructed_away"].ne(terminal_score["away_score"])
    if mismatch.any():
        raise ValueError(f"{int(mismatch.sum())} games fail prefix-score conservation.")
    return states


def make_possession_features(
    states: pd.DataFrame, *, time_interactions: bool
) -> pd.DataFrame:
    """Add possession known at state creation; never use the possession outcome."""
    features = make_rolling_context_features(states)
    possession = np.where(states["offense_is_home"].astype(bool), 1.0, -1.0)
    features["home_possession"] = possession
    if time_interactions:
        effective_remaining = np.where(
            states["is_overtime"].astype(bool),
            states["seconds_remaining_period"].astype(float),
            states["regulation_seconds_remaining"].astype(float),
        )
        elapsed_fraction = np.minimum(
            states["seconds_elapsed_game"].astype(float) / 2880.0, 1.0
        )
        features["possession_time_pressure"] = possession / np.sqrt(
            effective_remaining / 60.0 + 1.0
        )
        features["possession_late_interaction"] = possession * elapsed_fraction
    return features


def _possession_effects(model, states: pd.DataFrame) -> list[dict]:
    rows = []
    for name, upper, max_margin in (
        ("all", np.inf, np.inf),
        ("last_6m", 360.0, np.inf),
        ("last_2m", 120.0, np.inf),
        ("close_last_2m", 120.0, 3.0),
        ("close_last_1m", 60.0, 3.0),
        ("last_10s", 10.0, np.inf),
        ("close_last_10s", 10.0, 3.0),
        ("tied_last_10s", 10.0, 0.0),
    ):
        remaining = np.where(
            states["is_overtime"].astype(bool),
            states["seconds_remaining_period"].astype(float),
            states["regulation_seconds_remaining"].astype(float),
        )
        subset = states.loc[
            (remaining <= upper) & states["home_score_diff_before"].abs().le(max_margin)
        ].copy()
        if subset.empty:
            continue
        home = subset.copy()
        away = subset.copy()
        home["offense_is_home"] = True
        away["offense_is_home"] = False
        swing = (
            model.predict_proba(make_possession_features(home, time_interactions=True))[
                :, 1
            ]
            - model.predict_proba(
                make_possession_features(away, time_interactions=True)
            )[:, 1]
        )
        rows.append(
            {
                "window": name,
                "rows": int(len(subset)),
                "mean_home_possession_swing": float(np.mean(swing)),
                "median_home_possession_swing": float(np.median(swing)),
                "p95_absolute_swing": float(np.quantile(np.abs(swing), 0.95)),
            }
        )
    return rows


def run_win_probability_possession_ablation(
    possessions_path: str | Path,
    game_dim_path: str | Path,
    *,
    artifact_root: str | Path,
    train_season: str = "2024-25",
    test_season: str = "2025-26",
    bootstrap_repetitions: int = 5000,
    seed: int = 7,
) -> dict:
    possessions = pd.read_parquet(possessions_path)
    games = pd.read_parquet(game_dim_path)
    states = build_possession_start_states(possessions, games)
    elo = build_pregame_elo(games)
    context = build_pregame_team_context(games)
    states = states.merge(
        elo[["game_id", "pregame_elo_diff"]], on="game_id", validate="many_to_one"
    ).merge(context, on="game_id", validate="many_to_one")
    train = states.loc[states["season_label"].eq(train_season)].copy()
    test = states.loc[states["season_label"].eq(test_season)].copy()
    if train.empty or test.empty:
        raise ValueError(
            "Both chronological seasons must contain possession-start states."
        )
    y_train = train["home_win"].astype(int).to_numpy()
    y_test = test["home_win"].astype(int).to_numpy()
    feature_sets = {
        "context": make_rolling_context_features,
        "context_plus_possession": lambda frame: make_possession_features(
            frame, time_interactions=False
        ),
        "context_plus_possession_time": lambda frame: make_possession_features(
            frame, time_interactions=True
        ),
    }
    models = {
        name: _fit(builder(train), y_train) for name, builder in feature_sets.items()
    }
    for name, model in models.items():
        test[f"probability_{name}"] = model.predict_proba(feature_sets[name](test))[
            :, 1
        ]
    metrics = {
        name: _metrics(y_test, test[f"probability_{name}"].to_numpy())
        for name in models
    }
    paired = {
        "possession_vs_context": _paired_bootstrap(
            test,
            "probability_context",
            "probability_context_plus_possession",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
        "possession_time_vs_context": _paired_bootstrap(
            test,
            "probability_context",
            "probability_context_plus_possession_time",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
        "possession_time_vs_constant_possession": _paired_bootstrap(
            test,
            "probability_context_plus_possession",
            "probability_context_plus_possession_time",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
    }
    checkpoints = []
    for checkpoint, remaining in CHECKPOINTS.items():
        rows = _checkpoint_rows(test, remaining)
        outcome = rows["home_win"].astype(int).to_numpy()
        checkpoints.append(
            {
                "checkpoint": checkpoint,
                "regulation_seconds_remaining": remaining,
                **{
                    name: _metrics(outcome, rows[f"probability_{name}"].to_numpy())
                    for name in models
                },
            }
        )
    close_late = test.loc[
        test["regulation_seconds_remaining"].le(120)
        & test["home_score_diff_before"].abs().le(3)
        & ~test["is_overtime"]
    ]
    close_late_metrics = {
        name: _metrics(
            close_late["home_win"].astype(int).to_numpy(),
            close_late[f"probability_{name}"].to_numpy(),
        )
        for name in models
    }

    run_id = f"wp_possession_start_v2_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability_possession" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")
    test.to_parquet(output / "test_predictions.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "win_probability_logistic_possession_start_ablation",
        "estimand": "home_win_probability_at_start_of_observed_team_control",
        "status": "research_candidate_single_outer_fold",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_season_label": train_season,
            "test_season_label": test_season,
            "state_timing": "score from completed prior possessions; control and clock at current possession start",
            "outcome_exclusion": "current and future possession points are never features",
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
            "features": {
                name: list(builder(train).columns)
                for name, builder in feature_sets.items()
            },
            "source_hashes": {
                "possessions": sha256_file(possessions_path),
                "game_dim": sha256_file(game_dim_path),
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "metrics": {
            "train_games": int(train["game_id"].nunique()),
            "test_games": int(test["game_id"].nunique()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "variants": metrics,
            "paired_game_bootstrap": paired,
            "checkpoints": checkpoints,
            "close_last_2m_margin_le_3": close_late_metrics,
            "possession_effects": _possession_effects(
                models["context_plus_possession_time"], test
            ),
        },
        "caveats": [
            "Possession starts are observed event states, not pre-event forecasts.",
            "Only CDN-covered, lineup-reconciled games enter this fold.",
            "This artifact reports one chronological outer fold; promotion uses both frozen folds.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
