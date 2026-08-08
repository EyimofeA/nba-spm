"""Leakage-safe prior-season starter-strength ablation for win probability."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.espn_win_probability import extract_espn_win_probability, read_gzip_json
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_possessions,
    load_legacy_possessions,
    ratings_table,
)
from nba_impact.models.win_probability import CHECKPOINTS, _checkpoint_rows, _metrics, sample_game_states
from nba_impact.models.win_probability_ablation import (
    _fit,
    build_pregame_elo,
    make_elo_features,
)
from nba_impact.models.win_probability_benchmark import match_espn_to_local_states


def _fit_rating_panel(frame: pd.DataFrame, seasons: tuple[int, ...], prior_season_end: int) -> pd.DataFrame:
    config = RapmConfig(seasons=seasons, game_types=("regular",))
    design = build_design(frame, include_home=True)
    beta, _ = fit_coefficients(design, config)
    panel = ratings_table(design, beta)
    panel["prior_season_end"] = int(prior_season_end)
    panel["rating_window_start"] = int(min(seasons))
    panel["rating_window_end"] = int(max(seasons))
    return panel


def build_time_safe_prior_ratings(
    legacy_cache: str | Path,
    current_possessions_path: str | Path,
    current_segments_path: str | Path,
) -> pd.DataFrame:
    """Fit ratings through 2023-24 for train and through 2024-25 for test."""
    legacy = load_legacy_possessions(
        legacy_cache, (2022, 2023, 2024), game_types=("regular",)
    )
    current = load_current_possessions(
        current_possessions_path,
        current_segments_path,
        lineup_policy="start",
        game_types=("regular",),
    )
    through_2024 = _fit_rating_panel(legacy, (2022, 2023, 2024), 2024)
    through_2025_frame = pd.concat(
        [legacy.loc[legacy["season"].isin((2023, 2024))], current.loc[current["season"] == 2025]],
        ignore_index=True,
    )
    through_2025 = _fit_rating_panel(through_2025_frame, (2023, 2024, 2025), 2025)
    ratings = pd.concat([through_2024, through_2025], ignore_index=True)
    if ratings.duplicated(["prior_season_end", "player_id"]).any():
        raise ValueError("Prior rating panels are not unique by season and player.")
    if (ratings["rating_window_end"] > ratings["prior_season_end"]).any():
        raise ValueError("A prior rating window extends beyond its forecast season boundary.")
    return ratings


def build_starter_strength(player_games: pd.DataFrame, prior_ratings: pd.DataFrame) -> pd.DataFrame:
    """Sum prior RAPM for the ten official starters known at tipoff."""
    starters = player_games.loc[player_games["starter"].astype(bool)].copy()
    counts = starters.groupby(["game_id", "team_side"])["player_id"].size()
    if not counts.eq(5).all():
        raise ValueError("Every team-game must contain exactly five starters.")
    starters["prior_season_end"] = starters["season_start"].astype(int)
    rating_columns = [
        "prior_season_end", "player_id", "offense_per_100", "defense_per_100", "net_per_100"
    ]
    starters = starters.merge(
        prior_ratings[rating_columns],
        on=["prior_season_end", "player_id"],
        how="left",
        validate="many_to_one",
    )
    starters["rating_known"] = starters["net_per_100"].notna()
    for column in ("offense_per_100", "defense_per_100", "net_per_100"):
        starters[column] = starters[column].fillna(0.0)
    team = starters.groupby(
        ["game_id", "season_label", "team_side"], as_index=False
    ).agg(
        starter_offense_per_100=("offense_per_100", "sum"),
        starter_defense_per_100=("defense_per_100", "sum"),
        starter_net_per_100=("net_per_100", "sum"),
        known_starters=("rating_known", "sum"),
    )
    wide = team.pivot(
        index=["game_id", "season_label"],
        columns="team_side",
        values=[
            "starter_offense_per_100", "starter_defense_per_100", "starter_net_per_100",
            "known_starters",
        ],
    )
    wide.columns = ["_".join(column) for column in wide.columns]
    wide = wide.reset_index()
    wide["pregame_starter_net_diff"] = (
        wide["starter_net_per_100_home"] - wide["starter_net_per_100_away"]
    )
    wide["starter_rating_coverage"] = (
        wide["known_starters_home"] + wide["known_starters_away"]
    ) / 10.0
    return wide


def make_lineup_features(states: pd.DataFrame) -> pd.DataFrame:
    features = make_elo_features(states)
    regulation_remaining = pd.to_numeric(states["regulation_seconds_remaining"], errors="raise")
    period_remaining = pd.to_numeric(states["seconds_remaining_period"], errors="raise")
    effective_remaining = np.where(states["is_overtime"].astype(bool), period_remaining, regulation_remaining)
    strength = pd.to_numeric(states["pregame_starter_net_diff"], errors="raise").astype(float)
    features["pregame_starter_net_diff"] = strength
    features["pregame_starter_remaining"] = strength * np.sqrt(
        np.maximum(effective_remaining, 0.0) / 2880.0
    )
    return features


def _paired_bootstrap(
    predictions: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    *,
    repetitions: int,
    seed: int,
) -> dict:
    rows = predictions.assign(
        baseline_loss=lambda frame: (frame["home_win"].astype(float) - frame[baseline_column]) ** 2,
        candidate_loss=lambda frame: (frame["home_win"].astype(float) - frame[candidate_column]) ** 2,
    )
    deltas = rows.groupby("game_id").apply(
        lambda group: float(group["candidate_loss"].mean() - group["baseline_loss"].mean()),
        include_groups=False,
    ).to_numpy()
    rng = np.random.default_rng(seed)
    draws = np.array(
        [rng.choice(deltas, size=len(deltas), replace=True).mean() for _ in range(repetitions)]
    )
    return {
        "games": int(len(deltas)),
        "mean_game_brier_delta_candidate_minus_baseline": float(deltas.mean()),
        "probability_candidate_better": float((draws < 0).mean()),
        "delta_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
    }


def _load_espn_plays(index_path: str | Path, season_label: str) -> pd.DataFrame:
    index = pd.read_parquet(index_path)
    index = index.loc[index["season_label"].eq(season_label) & index["summary_path"].notna()]
    frames = [
        extract_espn_win_probability(
            read_gzip_json(row.summary_path), game_id=str(row.game_id), season_label=season_label
        )
        for row in index.itertuples(index=False)
    ]
    return pd.concat(frames, ignore_index=True)


def run_win_probability_lineup_ablation(
    event_states_path: str | Path,
    game_dim_path: str | Path,
    player_games_path: str | Path,
    current_possessions_path: str | Path,
    current_segments_path: str | Path,
    legacy_cache: str | Path,
    espn_index_path: str | Path,
    *,
    artifact_root: str | Path,
    interval_seconds: int = 30,
    bootstrap_repetitions: int = 5000,
    seed: int = 7,
) -> dict:
    train_season, test_season = "2024-25", "2025-26"
    prior_ratings = build_time_safe_prior_ratings(
        legacy_cache, current_possessions_path, current_segments_path
    )
    player_games = pd.read_parquet(player_games_path)
    strengths = build_starter_strength(player_games, prior_ratings)
    games = pd.read_parquet(game_dim_path)
    elo = build_pregame_elo(games)
    event_columns = [
        "event_id", "game_id", "season_label", "actionId", "period",
        "seconds_remaining_period", "regulation_seconds_remaining", "seconds_elapsed_game",
        "is_overtime", "home_score_after", "away_score_after", "home_score_diff_after",
        "home_win", "is_terminal_event",
    ]
    events = pd.read_parquet(event_states_path, columns=event_columns)
    states = sample_game_states(events, interval_seconds=interval_seconds).merge(
        elo[["game_id", "pregame_elo_diff"]], on="game_id", validate="many_to_one"
    ).merge(
        strengths[["game_id", "pregame_starter_net_diff", "starter_rating_coverage"]],
        on="game_id",
        validate="many_to_one",
    )
    train = states.loc[states["season_label"].eq(train_season) & ~states["is_terminal_event"]].copy()
    test = states.loc[states["season_label"].eq(test_season) & ~states["is_terminal_event"]].copy()
    y_train = train["home_win"].astype(int).to_numpy()
    y_test = test["home_win"].astype(int).to_numpy()
    models = {
        "elo": _fit(make_elo_features(train), y_train),
        "elo_plus_starters": _fit(make_lineup_features(train), y_train),
    }
    test["probability_elo"] = models["elo"].predict_proba(make_elo_features(test))[:, 1]
    test["probability_elo_plus_starters"] = models["elo_plus_starters"].predict_proba(
        make_lineup_features(test)
    )[:, 1]
    variants = {
        "elo": _metrics(y_test, test["probability_elo"].to_numpy()),
        "elo_plus_starters": _metrics(y_test, test["probability_elo_plus_starters"].to_numpy()),
    }
    paired = _paired_bootstrap(
        test,
        "probability_elo",
        "probability_elo_plus_starters",
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    checkpoints = []
    for checkpoint_name, remaining in CHECKPOINTS.items():
        checkpoint = _checkpoint_rows(test, remaining)
        outcome = checkpoint["home_win"].astype(int).to_numpy()
        baseline = _metrics(outcome, checkpoint["probability_elo"].to_numpy())
        candidate = _metrics(outcome, checkpoint["probability_elo_plus_starters"].to_numpy())
        checkpoints.append(
            {
                "checkpoint": checkpoint_name,
                "regulation_seconds_remaining": remaining,
                "elo": baseline,
                "elo_plus_starters": candidate,
                "paired_game_bootstrap": _paired_bootstrap(
                    checkpoint,
                    "probability_elo",
                    "probability_elo_plus_starters",
                    repetitions=bootstrap_repetitions,
                    seed=seed,
                ),
            }
        )

    test_events = events.loc[events["season_label"].eq(test_season)].merge(
        elo[["game_id", "pregame_elo_diff"]], on="game_id", validate="many_to_one"
    ).merge(
        strengths[["game_id", "pregame_starter_net_diff", "starter_rating_coverage"]],
        on="game_id",
        validate="many_to_one",
    )
    espn = _load_espn_plays(espn_index_path, test_season)
    external, external_coverage = match_espn_to_local_states(espn, test_events)
    external = external.loc[~external["is_terminal_event"]].copy()
    external["probability_elo"] = models["elo"].predict_proba(make_elo_features(external))[:, 1]
    external["probability_elo_plus_starters"] = models["elo_plus_starters"].predict_proba(
        make_lineup_features(external)
    )[:, 1]
    external_start = _checkpoint_rows(external, CHECKPOINTS["game_start"])
    external_outcome = external_start["home_win"].astype(int).to_numpy()
    external_metrics = {
        "espn": _metrics(external_outcome, external_start["espn_home_win_probability"].to_numpy()),
        "elo": _metrics(external_outcome, external_start["probability_elo"].to_numpy()),
        "elo_plus_starters": _metrics(
            external_outcome, external_start["probability_elo_plus_starters"].to_numpy()
        ),
    }
    external_paired = {
        "starters_vs_elo": _paired_bootstrap(
            external_start,
            "probability_elo",
            "probability_elo_plus_starters",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
        "starters_vs_espn": _paired_bootstrap(
            external_start,
            "espn_home_win_probability",
            "probability_elo_plus_starters",
            repetitions=bootstrap_repetitions,
            seed=seed,
        ),
    }

    run_id = f"wp_lineup_ablation_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability_lineup" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")
    prior_ratings.to_parquet(output / "prior_ratings.parquet", index=False)
    strengths.to_parquet(output / "pregame_starter_strength.parquet", index=False)
    test.to_parquet(output / "test_predictions.parquet", index=False)
    external_start.to_parquet(output / "espn_game_start_predictions.parquet", index=False)
    coverage_by_season = (
        strengths.groupby("season_label")["starter_rating_coverage"]
        .agg(["count", "mean", "min"])
        .reset_index()
        .to_dict("records")
    )
    run = {
        "run_id": run_id,
        "model_family": "win_probability_logistic_prior_starter_rapm_ablation",
        "estimand": "tipoff_and_in_game_home_win_probability_with_official_starters",
        "status": "research_candidate_inconclusive_single_outer_fold",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_season_label": train_season,
            "test_season_label": test_season,
            "rating_windows": {"2024-25": [2022, 2023, 2024], "2025-26": [2023, 2024, 2025]},
            "starter_definition": "official_boxscore_starters_known_at_tipoff",
            "missing_player_rating": 0.0,
            "interval_seconds": interval_seconds,
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
            "source_hashes": {
                "event_states": sha256_file(event_states_path),
                "game_dim": sha256_file(game_dim_path),
                "player_games": sha256_file(player_games_path),
                "current_possessions": sha256_file(current_possessions_path),
                "current_segments": sha256_file(current_segments_path),
                "espn_index": sha256_file(espn_index_path),
                "source_code": sha256_file(Path(__file__)),
                "rapm_source_code": sha256_file(Path(__file__).with_name("rapm.py")),
                **{
                    f"legacy_possessions_{season}": sha256_file(
                        Path(legacy_cache) / f"matchups_{season}.parquet"
                    )
                    for season in (2022, 2023, 2024)
                },
            },
        },
        "metrics": {
            "train_games": int(train["game_id"].nunique()),
            "test_games": int(test["game_id"].nunique()),
            "variants": variants,
            "paired_game_bootstrap": paired,
            "checkpoints": checkpoints,
            "espn_game_start": external_metrics,
            "espn_game_start_paired": external_paired,
            "espn_match_coverage": external_coverage,
            "starter_rating_coverage": coverage_by_season,
        },
        "caveats": [
            "Official starters are a tipoff-time feature, not an earlier-day forecast feature.",
            "Unrated starters are assigned the centered RAPM mean of zero.",
            "Only one chronological outer test season is currently available.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
