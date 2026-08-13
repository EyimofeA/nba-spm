"""Cost-bounded feasibility test for observed-unit shot-defense signal."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic


def chronological_game_split(
    panel: pd.DataFrame, *, train_fraction: float
) -> tuple[pd.Series, dict]:
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0).")
    games = panel[["game_id", "game_date"]].drop_duplicates().sort_values(
        ["game_date", "game_id"], kind="stable"
    )
    split_index = int(len(games) * train_fraction)
    if split_index <= 0 or split_index >= len(games):
        raise ValueError("Chronological split must contain train and test games.")
    cutoff_date = games.iloc[split_index]["game_date"]
    train_games = set(
        games.loc[games["game_date"].lt(cutoff_date), "game_id"].astype(str)
    )
    if not train_games:
        raise ValueError("Chronological split has no dates strictly before the test period.")
    train = panel["game_id"].astype(str).isin(train_games)
    test_games = set(panel.loc[~train, "game_id"].astype(str))
    split = {
        "train_games": int(len(train_games)),
        "test_games": int(len(test_games)),
        "train_rows": int(train.sum()),
        "test_rows": int((~train).sum()),
        "train_date_max": str(games.loc[games["game_id"].astype(str).isin(train_games), "game_date"].max()),
        "test_date_min": str(games.loc[games["game_id"].astype(str).isin(test_games), "game_date"].min()),
    }
    return train, split


def _fit_stage(
    panel: pd.DataFrame,
    train: pd.Series,
    *,
    stage: str,
    include_defense_team: bool,
    alpha: float,
    max_iter: int,
    tolerance: float,
    random_state: int,
) -> dict:
    if stage == "zone":
        target = "shot_zone"
        categorical = ["shooter_id", "offense_team_id"]
        numeric = [
            "offense_is_home", "period", "regulation_seconds_remaining",
            "offense_score_diff_before",
        ]
    elif stage == "make":
        target = "shot_made"
        categorical = ["shooter_id", "offense_team_id", "shot_zone"]
        numeric = [
            "offense_is_home", "period", "regulation_seconds_remaining",
            "offense_score_diff_before", "shot_distance_feet", "shot_angle_radians",
        ]
    else:
        raise ValueError(f"Unknown shot-defense stage {stage!r}.")
    if include_defense_team:
        categorical.append("defense_team_id")
    features = panel[categorical + numeric].copy()
    for column in categorical:
        features[column] = features[column].astype(str)
    transform = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=tolerance,
        average=True,
        random_state=random_state,
    )
    pipeline = make_pipeline(transform, model)
    started = time.monotonic()
    pipeline.fit(features.loc[train], panel.loc[train, target])
    elapsed = time.monotonic() - started
    probabilities = pipeline.predict_proba(features.loc[~train])
    classes = pipeline[-1].classes_
    metrics = {
        "stage": stage,
        "model": "plus_defense_team" if include_defense_team else "offense_only",
        "log_loss": float(log_loss(panel.loc[~train, target], probabilities, labels=classes)),
        "fit_seconds": float(elapsed),
        "iterations": int(pipeline[-1].n_iter_),
        "converged": bool(pipeline[-1].n_iter_ < max_iter),
    }
    if stage == "make":
        positive = np.flatnonzero(classes == 1)
        if len(positive) != 1:
            raise ValueError("Shot-make model did not learn the expected binary classes.")
        metrics["brier"] = float(
            brier_score_loss(panel.loc[~train, target], probabilities[:, positive[0]])
        )
    return metrics


def run_shot_defense_team_pilot(
    panel_path: str | Path,
    *,
    artifact_root: str | Path,
    season: int = 2024,
    train_fraction: float = 0.70,
    alpha: float = 1e-5,
    max_iter: int = 300,
    tolerance: float = 1e-5,
    practical_relative_gain: float = 0.005,
    random_state: int = 20260813,
) -> dict:
    """Test defense-team signal before any player or lineup coefficient model."""
    source_path = Path(panel_path)
    panel = pd.read_parquet(source_path)
    panel = panel.loc[pd.to_numeric(panel["season_start"], errors="coerce").eq(season)].copy()
    if panel.empty:
        raise ValueError(f"Shot-defense panel has no season {season} rows.")
    panel = panel.sort_values(["game_date", "game_id", "orderNumber"], kind="stable").reset_index(drop=True)
    defense_columns = [f"defense_player_{index}" for index in range(1, 6)]
    panel["defense_lineup_id"] = panel[defense_columns].astype(str).agg("-".join, axis=1)
    train, split = chronological_game_split(panel, train_fraction=train_fraction)
    seen_lineups = set(panel.loc[train, "defense_lineup_id"])
    test = panel.loc[~train]
    lineup_coverage = {
        "train_unique_lineups": int(panel.loc[train, "defense_lineup_id"].nunique()),
        "test_unique_lineups": int(test["defense_lineup_id"].nunique()),
        "test_shot_seen_lineup_rate": float(test["defense_lineup_id"].isin(seen_lineups).mean()),
        "test_unique_seen_lineup_rate": float(
            len(set(test["defense_lineup_id"]) & seen_lineups)
            / max(test["defense_lineup_id"].nunique(), 1)
        ),
    }
    results = [
        _fit_stage(
            panel, train, stage=stage, include_defense_team=include_defense_team,
            alpha=alpha, max_iter=max_iter, tolerance=tolerance,
            random_state=random_state,
        )
        for stage in ("zone", "make")
        for include_defense_team in (False, True)
    ]
    lookup = {(row["stage"], row["model"]): row for row in results}
    offense_only = sum(lookup[(stage, "offense_only")]["log_loss"] for stage in ("zone", "make"))
    defense_team = sum(lookup[(stage, "plus_defense_team")]["log_loss"] for stage in ("zone", "make"))
    relative_gain = (offense_only - defense_team) / offense_only
    make_brier_change = (
        lookup[("make", "plus_defense_team")]["brier"]
        - lookup[("make", "offense_only")]["brier"]
    )
    converged = all(row["converged"] for row in results)
    passed_practical_gate = bool(
        converged
        and relative_gain >= practical_relative_gain
        and make_brier_change <= 0
    )
    config = {
        "season": season,
        "train_fraction": train_fraction,
        "alpha": alpha,
        "max_iter": max_iter,
        "tolerance": tolerance,
        "practical_relative_gain": practical_relative_gain,
        "random_state": random_state,
        "panel_sha256": sha256_file(source_path),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"shot_defense_team_pilot_v1_{identity}"
    output = Path(artifact_root) / "models" / "shot_defense_team_pilot" / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(results).to_csv(output / "stage_metrics.csv", index=False)
    run = {
        "run_id": run_id,
        "model_family": "two_stage_ridge_logistic_shot_defense_pilot",
        "estimand": "Observed defensive-unit association with zone mix and conversion conditional on an FGA.",
        "status": "research_challenger" if passed_practical_gate else "research_null",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand_id": "observed_lineup_shot_defense_v1",
        "evidence_status": "reused_feasibility_only",
        "config": config,
        "split": split,
        "lineup_coverage": lineup_coverage,
        "stage_metrics": results,
        "metrics": {
            "offense_only_combined_log_loss": float(offense_only),
            "plus_defense_team_combined_log_loss": float(defense_team),
            "relative_combined_log_loss_gain": float(relative_gain),
            "make_brier_change": float(make_brier_change),
            "passed_practical_gate": passed_practical_gate,
        },
        "summary": {
            "offense_only_combined_log_loss": float(offense_only),
            "plus_defense_team_combined_log_loss": float(defense_team),
            "relative_combined_log_loss_gain": float(relative_gain),
            "make_brier_change": float(make_brier_change),
            "all_models_converged": converged,
            "passed_practical_gate": passed_practical_gate,
        },
        "artifact_path": str(output.resolve()),
        "forbidden_interpretation": "Individual defender impact, primary-defender credit, causal defense, or AIO promotion.",
        "decision_rule": (
            "Stop before lineup/player coefficients when defense-team signal misses the "
            "0.5% relative combined-log-loss gate or worsens make Brier."
        ),
    }
    write_json_atomic(run, output / "run.json")
    return run
