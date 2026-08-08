"""Leakage-safe state-only NBA win-probability baseline."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic


FEATURE_COLUMNS = (
    "home_score_diff",
    "effective_seconds_remaining",
    "elapsed_fraction",
    "is_overtime",
    "score_time_pressure",
    "score_late_interaction",
)

CHECKPOINTS = {
    "game_start": 2880.0,
    "halftime": 1440.0,
    "fourth_quarter_start": 720.0,
    "fourth_quarter_6m": 360.0,
    "fourth_quarter_2m": 120.0,
    "fourth_quarter_1m": 60.0,
}


def sample_game_states(events: pd.DataFrame, interval_seconds: int = 30) -> pd.DataFrame:
    if interval_seconds < 1:
        raise ValueError("interval_seconds must be positive")
    sampled = events.copy()
    sampled["time_bucket"] = np.floor(sampled["seconds_elapsed_game"] / interval_seconds).astype(int)
    sampled = sampled.sort_values(["game_id", "seconds_elapsed_game", "actionId"], kind="stable")
    sampled = sampled.groupby(["game_id", "time_bucket"], as_index=False, sort=False).tail(1)
    starters = events.sort_values(["game_id", "seconds_elapsed_game", "actionId"], kind="stable").groupby(
        "game_id", as_index=False, sort=False
    ).head(1)
    terminals = events.loc[events["is_terminal_event"]]
    sampled = pd.concat([sampled, starters, terminals], ignore_index=True)
    sampled = sampled.drop_duplicates("event_id", keep="last")
    return sampled.sort_values(["game_id", "seconds_elapsed_game", "actionId"], kind="stable").reset_index(drop=True)


def make_features(states: pd.DataFrame) -> pd.DataFrame:
    score = pd.to_numeric(states["home_score_diff_after"], errors="raise").astype(float)
    regulation_remaining = pd.to_numeric(states["regulation_seconds_remaining"], errors="raise").astype(float)
    period_remaining = pd.to_numeric(states["seconds_remaining_period"], errors="raise").astype(float)
    overtime = states["is_overtime"].astype(bool)
    effective_remaining = np.where(overtime, period_remaining, regulation_remaining)
    elapsed_fraction = np.minimum(pd.to_numeric(states["seconds_elapsed_game"], errors="raise") / 2880.0, 1.0)
    minutes_plus_one = effective_remaining / 60.0 + 1.0
    return pd.DataFrame(
        {
            "home_score_diff": score,
            "effective_seconds_remaining": effective_remaining,
            "elapsed_fraction": elapsed_fraction,
            "is_overtime": overtime.astype(float),
            "score_time_pressure": score / np.sqrt(minutes_plus_one),
            "score_late_interaction": score * elapsed_fraction,
        },
        index=states.index,
    )


def _calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float, bool]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    if float(np.std(logit)) < 1e-10:
        observed = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
        return float(np.log(observed / (1.0 - observed))), 0.0, False
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, y)
    return float(calibration.intercept_[0]), float(calibration.coef_[0, 0]), True


def _metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    intercept, slope, calibration_identified = _calibration(y, probability)
    return {
        "rows": int(len(y)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "auc": float(roc_auc_score(y, probability)),
        "accuracy_0_5": float(accuracy_score(y, probability >= 0.5)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_identified": calibration_identified,
        "probability_sd": float(np.std(probability)),
        "mean_predicted_home_win": float(np.mean(probability)),
        "actual_home_win_rate": float(np.mean(y)),
    }


def _predict(model: Pipeline, states: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(make_features(states))[:, 1]
    terminal = states["is_terminal_event"].to_numpy(dtype=bool)
    terminal_outcome = states["home_score_diff_after"].to_numpy(dtype=float) > 0
    probability[terminal] = terminal_outcome[terminal].astype(float)
    return probability


def _checkpoint_rows(states: pd.DataFrame, checkpoint: float) -> pd.DataFrame:
    regulation = states.loc[~states["is_overtime"].astype(bool)].copy()
    regulation["checkpoint_distance"] = (regulation["regulation_seconds_remaining"] - checkpoint).abs()
    return regulation.sort_values(
        ["game_id", "checkpoint_distance", "seconds_elapsed_game", "actionId"], kind="stable"
    ).groupby("game_id", as_index=False, sort=False).head(1)


def run_win_probability(
    event_states_path: str | Path,
    *,
    train_season_labels: tuple[str, ...],
    test_season_labels: tuple[str, ...],
    artifact_root: str | Path,
    interval_seconds: int = 30,
) -> dict:
    columns = [
        "event_id",
        "game_id",
        "season_label",
        "season_type",
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
    events = pd.read_parquet(event_states_path, columns=columns)
    sampled = sample_game_states(events, interval_seconds=interval_seconds)
    train = sampled.loc[sampled["season_label"].isin(train_season_labels)].copy()
    test = sampled.loc[sampled["season_label"].isin(test_season_labels)].copy()
    if train.empty or test.empty:
        raise ValueError("Training and test season selections must both contain states.")
    if set(train["game_id"]) & set(test["game_id"]):
        raise ValueError("A game appears in both training and test data.")

    y_train = train["home_win"].astype(int).to_numpy()
    y_test = test["home_win"].astype(int).to_numpy()
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)),
        ]
    )
    model.fit(make_features(train), y_train)
    test_probability = _predict(model, test)
    constant_probability = float(np.mean(y_train))
    overall = _metrics(y_test, test_probability)
    overall["constant_home_rate_brier"] = float(
        brier_score_loss(y_test, np.full(len(y_test), constant_probability))
    )
    overall["brier_skill_vs_constant"] = float(
        1.0 - overall["brier"] / overall["constant_home_rate_brier"]
    )

    checkpoint_rows: list[dict] = []
    test_with_probability = test.copy()
    test_with_probability["home_win_probability"] = test_probability
    for name, remaining in CHECKPOINTS.items():
        checkpoint = _checkpoint_rows(test_with_probability, remaining)
        metrics = _metrics(
            checkpoint["home_win"].astype(int).to_numpy(),
            checkpoint["home_win_probability"].to_numpy(),
        )
        checkpoint_rows.append({"checkpoint": name, "regulation_seconds_remaining": remaining, **metrics})
    final = test_with_probability.loc[test_with_probability["is_terminal_event"]]
    final_metrics = _metrics(
        final["home_win"].astype(int).to_numpy(), final["home_win_probability"].to_numpy()
    )
    checkpoint_rows.append({"checkpoint": "final", "regulation_seconds_remaining": 0.0, **final_metrics})
    checkpoints = pd.DataFrame(checkpoint_rows)

    run_id = f"wp_state_v0_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability" / run_id
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output / "model.joblib")
    test_with_probability.to_parquet(output / "test_predictions.parquet", index=False)
    checkpoints.to_parquet(output / "checkpoint_metrics.parquet", index=False)
    source_hash = sha256_file(event_states_path)
    metrics = {
        "overall_sampled_states": overall,
        "checkpoints": checkpoints.to_dict(orient="records"),
        "train_games": int(train["game_id"].nunique()),
        "test_games": int(test["game_id"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "feature_columns": list(FEATURE_COLUMNS),
        "sampling_interval_seconds": interval_seconds,
    }
    run = {
        "run_id": run_id,
        "model_family": "win_probability_state_logistic",
        "estimand": "probability_home_team_wins_given_post_action_score_and_clock_state",
        "status": "research_baseline_unverified",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_season_labels": list(train_season_labels),
            "test_season_labels": list(test_season_labels),
            "interval_seconds": interval_seconds,
            "features": list(FEATURE_COLUMNS),
            "event_states_path": str(Path(event_states_path).resolve()),
            "event_states_sha256": source_hash,
        },
        "metrics": metrics,
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
