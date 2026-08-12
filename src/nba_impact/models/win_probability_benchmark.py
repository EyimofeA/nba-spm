"""Same-play external benchmarking for the NBA win-probability model."""
from __future__ import annotations

import json
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from nba_impact.data.espn_win_probability import extract_espn_win_probability, read_gzip_json
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.win_probability import CHECKPOINTS, _checkpoint_rows, _metrics
from nba_impact.models.win_probability_ablation import build_pregame_elo, make_elo_features


def match_espn_to_local_states(
    espn: pd.DataFrame,
    local: pd.DataFrame,
    *,
    clock_tolerance_seconds: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    """Join ESPN plays to the nearest exact-score local post-action state."""
    if clock_tolerance_seconds < 0:
        raise ValueError("clock_tolerance_seconds must be nonnegative")
    score_keys = ["game_id", "period", "home_score_after", "away_score_after"]
    left = espn.reset_index(drop=True).copy()
    left["_espn_row_id"] = np.arange(len(left), dtype=np.int64)
    candidates = left.merge(local, on=score_keys, how="inner", suffixes=("_espn", "_local"))
    candidates["clock_delta_seconds"] = (
        candidates["seconds_remaining_period_espn"] - candidates["seconds_remaining_period"]
    ).abs()
    candidates = candidates.loc[candidates["clock_delta_seconds"] <= clock_tolerance_seconds]
    candidates = candidates.sort_values(
        ["_espn_row_id", "clock_delta_seconds", "actionId"], kind="stable"
    ).drop_duplicates("_espn_row_id", keep="first")
    candidates = candidates.drop(columns="_espn_row_id").reset_index(drop=True)
    matched_ids = candidates["espn_play_id"].nunique()
    coverage = {
        "espn_rows": int(len(left)),
        "matched_rows": int(len(candidates)),
        "unmatched_rows": int(len(left) - len(candidates)),
        "match_rate": float(len(candidates) / len(left)) if len(left) else 0.0,
        "unique_matched_play_ids": int(matched_ids),
        "exact_clock_match_rate_among_matched": float(
            np.isclose(candidates["clock_delta_seconds"], 0.0).mean()
        )
        if len(candidates)
        else 0.0,
        "max_clock_delta_seconds": float(candidates["clock_delta_seconds"].max())
        if len(candidates)
        else None,
    }
    return candidates, coverage


def _paired_game_bootstrap(
    predictions: pd.DataFrame, *, repetitions: int, seed: int
) -> dict[str, float | list[float]]:
    rows = predictions.assign(
        espn_loss=lambda frame: (frame["home_win"].astype(float) - frame["espn_home_win_probability"]) ** 2,
        local_loss=lambda frame: (frame["home_win"].astype(float) - frame["local_home_win_probability"]) ** 2,
    )
    by_game = rows.groupby("game_id", as_index=False).agg(
        espn_brier=("espn_loss", "mean"), local_brier=("local_loss", "mean")
    )
    deltas = (by_game["local_brier"] - by_game["espn_brier"]).to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        draws[index] = rng.choice(deltas, size=len(deltas), replace=True).mean()
    return {
        "games": int(len(deltas)),
        "mean_game_brier_delta_local_minus_espn": float(deltas.mean()),
        "probability_local_better": float((draws < 0).mean()),
        "delta_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
    }


def run_espn_win_probability_benchmark(
    event_states_path: str | Path,
    game_dim_path: str | Path,
    espn_index_path: str | Path,
    model_run_path: str | Path,
    *,
    artifact_root: str | Path,
    clock_tolerance_seconds: float = 1.0,
    bootstrap_repetitions: int = 5000,
    seed: int = 7,
) -> dict:
    """Compare the local Elo WP model and ESPN on identical matched play states."""
    model_run_path = Path(model_run_path)
    run_json_path = model_run_path / "run.json" if model_run_path.is_dir() else model_run_path
    with run_json_path.open(encoding="utf-8") as handle:
        model_run = json.load(handle)
    model_dir = run_json_path.parent
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", InconsistentVersionWarning)
        model = joblib.load(model_dir / "state_plus_elo.joblib")
    if any(isinstance(item.message, InconsistentVersionWarning) for item in caught):
        raise RuntimeError(
            "The saved scikit-learn model version does not match this runtime. "
            "Run the benchmark in the locked project environment (uv run)."
        )
    test_seasons = tuple(model_run["config"]["test_season_labels"])

    index = pd.read_parquet(espn_index_path)
    index = index.loc[index["season_label"].isin(test_seasons) & index["summary_path"].notna()].copy()
    if index.empty:
        raise ValueError("The ESPN index has no cached summaries for the model test seasons.")
    frames = []
    for row in index.itertuples(index=False):
        frames.append(
            extract_espn_win_probability(
                read_gzip_json(row.summary_path),
                game_id=str(row.game_id),
                season_label=str(row.season_label),
            )
        )
    espn = pd.concat(frames, ignore_index=True)
    if espn.duplicated(["game_id", "espn_play_id"]).any():
        raise ValueError("ESPN game/play IDs are not unique.")
    if not espn["espn_home_win_probability"].between(0.0, 1.0).all():
        raise ValueError("ESPN probabilities must be in [0, 1].")

    event_columns = [
        "event_id", "game_id", "season_label", "actionId", "period",
        "seconds_remaining_period", "regulation_seconds_remaining", "seconds_elapsed_game",
        "is_overtime", "home_score_after", "away_score_after", "home_score_diff_after",
        "home_win", "is_terminal_event",
    ]
    local = pd.read_parquet(event_states_path, columns=event_columns)
    local = local.loc[local["season_label"].isin(test_seasons)].copy()
    games = pd.read_parquet(game_dim_path)
    elo = build_pregame_elo(games)
    local = local.merge(elo[["game_id", "pregame_elo_diff"]], on="game_id", validate="many_to_one")
    matched, coverage = match_espn_to_local_states(
        espn, local, clock_tolerance_seconds=clock_tolerance_seconds
    )
    matched = matched.loc[~matched["is_terminal_event"].astype(bool)].copy()
    if matched.empty:
        raise ValueError("No nonterminal ESPN plays matched local states.")
    matched["local_home_win_probability"] = model.predict_proba(make_elo_features(matched))[:, 1]
    outcome = matched["home_win"].astype(int).to_numpy()
    metrics = {
        "espn": _metrics(outcome, matched["espn_home_win_probability"].to_numpy()),
        "local_state_plus_elo": _metrics(outcome, matched["local_home_win_probability"].to_numpy()),
    }
    metrics["relative_brier_difference_local_vs_espn"] = float(
        (metrics["local_state_plus_elo"]["brier"] - metrics["espn"]["brier"])
        / metrics["espn"]["brier"]
    )
    paired = _paired_game_bootstrap(matched, repetitions=bootstrap_repetitions, seed=seed)
    checkpoints = []
    for checkpoint_name, remaining in CHECKPOINTS.items():
        checkpoint = _checkpoint_rows(matched, remaining)
        checkpoint_outcome = checkpoint["home_win"].astype(int).to_numpy()
        espn_metrics = _metrics(
            checkpoint_outcome, checkpoint["espn_home_win_probability"].to_numpy()
        )
        local_metrics = _metrics(
            checkpoint_outcome, checkpoint["local_home_win_probability"].to_numpy()
        )
        checkpoints.append(
            {
                "checkpoint": checkpoint_name,
                "regulation_seconds_remaining": remaining,
                "espn": espn_metrics,
                "local_state_plus_elo": local_metrics,
                "brier_delta_local_minus_espn": float(local_metrics["brier"] - espn_metrics["brier"]),
                "paired_game_bootstrap": _paired_game_bootstrap(
                    checkpoint, repetitions=bootstrap_repetitions, seed=seed
                ),
            }
        )

    run_id = f"wp_espn_benchmark_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability_benchmark" / run_id
    output.mkdir(parents=True, exist_ok=False)
    matched.to_parquet(output / "matched_predictions.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "win_probability_external_benchmark",
        "estimand": "probability_home_team_wins_after_an_espn_play",
        "status": "research_external_benchmark_single_outer_season",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_run_id": model_run["run_id"],
            "test_season_labels": list(test_seasons),
            "clock_tolerance_seconds": clock_tolerance_seconds,
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
            "event_states_sha256": sha256_file(event_states_path),
            "game_dim_sha256": sha256_file(game_dim_path),
            "espn_index_sha256": sha256_file(espn_index_path),
            "model_run_sha256": sha256_file(run_json_path),
            "source_code_sha256": sha256_file(Path(__file__)),
        },
        "coverage": {
            **coverage,
            "indexed_games": int(len(index)),
            "matched_nonterminal_rows": int(len(matched)),
            "matched_games": int(matched["game_id"].nunique()),
        },
        "metrics": metrics,
        "paired_game_bootstrap": paired,
        "checkpoints": checkpoints,
        "caveats": [
            "ESPN's feature set and training data are proprietary and not observable here.",
            "Results are play-weighted overall; the paired bootstrap weights games equally.",
            "Only exact period/score matches within the configured clock tolerance are scored.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
