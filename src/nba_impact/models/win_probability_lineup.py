"""Leakage-safe prior-season starter-strength ablation for win probability."""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.espn_win_probability import (
    extract_espn_win_probability,
    read_gzip_json,
)
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_possessions,
    load_legacy_possessions,
    ratings_table,
)
from nba_impact.models.win_probability import (
    CHECKPOINTS,
    _checkpoint_rows,
    _metrics,
    sample_game_states,
)
from nba_impact.models.win_probability_ablation import (
    _fit,
    build_pregame_elo,
    make_elo_features,
)
from nba_impact.models.win_probability_benchmark import match_espn_to_local_states


def _fit_rating_panel(
    frame: pd.DataFrame, seasons: tuple[int, ...], prior_season_end: int
) -> pd.DataFrame:
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
    """Fit three-year ratings available before each rich-event season."""
    legacy = load_legacy_possessions(
        legacy_cache, (2021, 2022, 2023, 2024), game_types=("regular",)
    )
    current = load_current_possessions(
        current_possessions_path,
        current_segments_path,
        lineup_policy="start",
        game_types=("regular",),
    )
    through_2023 = _fit_rating_panel(legacy, (2021, 2022, 2023), 2023)
    through_2024 = _fit_rating_panel(legacy, (2022, 2023, 2024), 2024)
    through_2025_frame = pd.concat(
        [
            legacy.loc[legacy["season"].isin((2023, 2024))],
            current.loc[current["season"] == 2025],
        ],
        ignore_index=True,
    )
    through_2025 = _fit_rating_panel(through_2025_frame, (2023, 2024, 2025), 2025)
    ratings = pd.concat([through_2023, through_2024, through_2025], ignore_index=True)
    if ratings.duplicated(["prior_season_end", "player_id"]).any():
        raise ValueError("Prior rating panels are not unique by season and player.")
    if (ratings["rating_window_end"] > ratings["prior_season_end"]).any():
        raise ValueError(
            "A prior rating window extends beyond its forecast season boundary."
        )
    return ratings


def build_starter_strength(
    player_games: pd.DataFrame, prior_ratings: pd.DataFrame
) -> pd.DataFrame:
    """Sum prior RAPM for the ten official starters known at tipoff."""
    starters = player_games.loc[player_games["starter"].astype(bool)].copy()
    counts = starters.groupby(["game_id", "team_side"])["player_id"].size()
    if not counts.eq(5).all():
        raise ValueError("Every team-game must contain exactly five starters.")
    starters["prior_season_end"] = starters["season_start"].astype(int)
    rating_columns = [
        "prior_season_end",
        "player_id",
        "offense_per_100",
        "defense_per_100",
        "net_per_100",
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
            "starter_offense_per_100",
            "starter_defense_per_100",
            "starter_net_per_100",
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
    regulation_remaining = pd.to_numeric(
        states["regulation_seconds_remaining"], errors="raise"
    )
    period_remaining = pd.to_numeric(states["seconds_remaining_period"], errors="raise")
    effective_remaining = np.where(
        states["is_overtime"].astype(bool), period_remaining, regulation_remaining
    )
    strength = pd.to_numeric(states["pregame_starter_net_diff"], errors="raise").astype(
        float
    )
    features["pregame_starter_net_diff"] = strength
    features["pregame_starter_remaining"] = strength * np.sqrt(
        np.maximum(effective_remaining, 0.0) / 2880.0
    )
    return features


def build_pregame_team_context(
    games: pd.DataFrame,
    *,
    update_weight: float = 0.10,
    offseason_retention: float = 0.75,
) -> pd.DataFrame:
    """Calculate rolling margin and rest before each date, then batch-update results."""
    ordered = games.sort_values(["season_start", "game_date", "game_id"], kind="stable")
    margin_rating: defaultdict[int, float] = defaultdict(float)
    last_game: dict[int, pd.Timestamp] = {}
    active_season: int | None = None
    rows = []
    for (season_start, game_date), date_games in ordered.groupby(
        ["season_start", "game_date"], sort=True
    ):
        season_start = int(season_start)
        game_date = pd.Timestamp(game_date)
        if active_season is not None and season_start != active_season:
            for team_id in margin_rating:
                margin_rating[team_id] *= offseason_retention
        active_season = season_start
        for game in date_games.itertuples(index=False):
            home_id, away_id = int(game.home_team_id), int(game.away_team_id)
            home_rest = (
                min((game_date - last_game[home_id]).days, 7)
                if home_id in last_game
                else 7
            )
            away_rest = (
                min((game_date - last_game[away_id]).days, 7)
                if away_id in last_game
                else 7
            )
            rows.append(
                {
                    "game_id": str(game.game_id),
                    "pregame_rolling_margin_diff": margin_rating[home_id]
                    - margin_rating[away_id],
                    "pregame_rest_advantage_days": home_rest - away_rest,
                }
            )
        for game in date_games.itertuples(index=False):
            home_id, away_id = int(game.home_team_id), int(game.away_team_id)
            margin = float(game.home_margin)
            margin_rating[home_id] = (1.0 - update_weight) * margin_rating[
                home_id
            ] + update_weight * margin
            margin_rating[away_id] = (1.0 - update_weight) * margin_rating[
                away_id
            ] - update_weight * margin
            last_game[home_id] = game_date
            last_game[away_id] = game_date
    return pd.DataFrame(rows)


def _add_team_context_features(
    features: pd.DataFrame, states: pd.DataFrame
) -> pd.DataFrame:
    regulation_remaining = pd.to_numeric(
        states["regulation_seconds_remaining"], errors="raise"
    )
    period_remaining = pd.to_numeric(states["seconds_remaining_period"], errors="raise")
    effective_remaining = np.where(
        states["is_overtime"].astype(bool), period_remaining, regulation_remaining
    )
    remaining_scale = np.sqrt(np.maximum(effective_remaining, 0.0) / 2880.0)
    margin = pd.to_numeric(
        states["pregame_rolling_margin_diff"], errors="raise"
    ).astype(float)
    rest = pd.to_numeric(states["pregame_rest_advantage_days"], errors="raise").astype(
        float
    )
    features["pregame_rolling_margin_diff"] = margin
    features["pregame_rolling_margin_remaining"] = margin * remaining_scale
    features["pregame_rest_advantage_days"] = rest
    features["pregame_rest_remaining"] = rest * remaining_scale
    return features


def make_rolling_context_features(states: pd.DataFrame) -> pd.DataFrame:
    """State, Elo, rolling team margin, and rest without player ratings."""
    return _add_team_context_features(make_elo_features(states), states)


def make_team_context_features(states: pd.DataFrame) -> pd.DataFrame:
    """Legacy combined candidate retaining starter RAPM for direct ablation."""
    return _add_team_context_features(make_lineup_features(states), states)


def _paired_bootstrap(
    predictions: pd.DataFrame,
    baseline_column: str,
    candidate_column: str,
    *,
    repetitions: int,
    seed: int,
) -> dict:
    rows = predictions.assign(
        baseline_loss=lambda frame: (
            frame["home_win"].astype(float) - frame[baseline_column]
        )
        ** 2,
        candidate_loss=lambda frame: (
            frame["home_win"].astype(float) - frame[candidate_column]
        )
        ** 2,
    )
    deltas = (
        rows.groupby("game_id")
        .apply(
            lambda group: float(
                group["candidate_loss"].mean() - group["baseline_loss"].mean()
            ),
            include_groups=False,
        )
        .to_numpy()
    )
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            rng.choice(deltas, size=len(deltas), replace=True).mean()
            for _ in range(repetitions)
        ]
    )
    return {
        "games": int(len(deltas)),
        "mean_game_brier_delta_candidate_minus_baseline": float(deltas.mean()),
        "probability_candidate_better": float((draws < 0).mean()),
        "delta_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
    }


def _load_espn_plays(index_path: str | Path, season_label: str) -> pd.DataFrame:
    index = pd.read_parquet(index_path)
    index = index.loc[
        index["season_label"].eq(season_label) & index["summary_path"].notna()
    ]
    frames = [
        extract_espn_win_probability(
            read_gzip_json(row.summary_path),
            game_id=str(row.game_id),
            season_label=season_label,
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
    train_season: str = "2024-25",
    test_season: str = "2025-26",
    include_espn: bool = True,
    interval_seconds: int = 30,
    bootstrap_repetitions: int = 5000,
    seed: int = 7,
) -> dict:
    prior_ratings = build_time_safe_prior_ratings(
        legacy_cache, current_possessions_path, current_segments_path
    )
    player_games = pd.read_parquet(player_games_path)
    strengths = build_starter_strength(player_games, prior_ratings)
    games = pd.read_parquet(game_dim_path)
    elo = build_pregame_elo(games)
    team_context = build_pregame_team_context(games)
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
        "home_score_after",
        "away_score_after",
        "home_score_diff_after",
        "home_win",
        "is_terminal_event",
    ]
    events = pd.read_parquet(event_states_path, columns=event_columns)
    states = (
        sample_game_states(events, interval_seconds=interval_seconds)
        .merge(
            elo[["game_id", "pregame_elo_diff"]], on="game_id", validate="many_to_one"
        )
        .merge(
            strengths[
                ["game_id", "pregame_starter_net_diff", "starter_rating_coverage"]
            ],
            on="game_id",
            validate="many_to_one",
        )
        .merge(team_context, on="game_id", validate="many_to_one")
    )
    train = states.loc[
        states["season_label"].eq(train_season) & ~states["is_terminal_event"]
    ].copy()
    test = states.loc[
        states["season_label"].eq(test_season) & ~states["is_terminal_event"]
    ].copy()
    y_train = train["home_win"].astype(int).to_numpy()
    y_test = test["home_win"].astype(int).to_numpy()
    models = {
        "elo": _fit(make_elo_features(train), y_train),
        "elo_plus_starters": _fit(make_lineup_features(train), y_train),
        "elo_plus_team_context": _fit(make_rolling_context_features(train), y_train),
        "elo_plus_starters_team_context": _fit(
            make_team_context_features(train), y_train
        ),
    }
    test["probability_elo"] = models["elo"].predict_proba(make_elo_features(test))[:, 1]
    test["probability_elo_plus_starters"] = models["elo_plus_starters"].predict_proba(
        make_lineup_features(test)
    )[:, 1]
    test["probability_elo_plus_team_context"] = models[
        "elo_plus_team_context"
    ].predict_proba(make_rolling_context_features(test))[:, 1]
    test["probability_elo_plus_starters_team_context"] = models[
        "elo_plus_starters_team_context"
    ].predict_proba(make_team_context_features(test))[:, 1]
    variants = {
        "elo": _metrics(y_test, test["probability_elo"].to_numpy()),
        "elo_plus_starters": _metrics(
            y_test, test["probability_elo_plus_starters"].to_numpy()
        ),
        "elo_plus_team_context": _metrics(
            y_test, test["probability_elo_plus_team_context"].to_numpy()
        ),
        "elo_plus_starters_team_context": _metrics(
            y_test, test["probability_elo_plus_starters_team_context"].to_numpy()
        ),
    }
    paired = _paired_bootstrap(
        test,
        "probability_elo",
        "probability_elo_plus_starters",
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    context_paired = _paired_bootstrap(
        test,
        "probability_elo_plus_starters",
        "probability_elo_plus_starters_team_context",
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    starter_free_context_paired = _paired_bootstrap(
        test,
        "probability_elo_plus_starters_team_context",
        "probability_elo_plus_team_context",
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    starter_free_context_vs_elo = _paired_bootstrap(
        test,
        "probability_elo",
        "probability_elo_plus_team_context",
        repetitions=bootstrap_repetitions,
        seed=seed,
    )
    checkpoints = []
    for checkpoint_name, remaining in CHECKPOINTS.items():
        checkpoint = _checkpoint_rows(test, remaining)
        outcome = checkpoint["home_win"].astype(int).to_numpy()
        baseline = _metrics(outcome, checkpoint["probability_elo"].to_numpy())
        candidate = _metrics(
            outcome, checkpoint["probability_elo_plus_starters"].to_numpy()
        )
        starter_free_context = _metrics(
            outcome, checkpoint["probability_elo_plus_team_context"].to_numpy()
        )
        checkpoints.append(
            {
                "checkpoint": checkpoint_name,
                "regulation_seconds_remaining": remaining,
                "elo": baseline,
                "elo_plus_starters": candidate,
                "elo_plus_team_context": starter_free_context,
                "paired_game_bootstrap": _paired_bootstrap(
                    checkpoint,
                    "probability_elo",
                    "probability_elo_plus_starters",
                    repetitions=bootstrap_repetitions,
                    seed=seed,
                ),
            }
        )

    external_start = pd.DataFrame()
    external_metrics = None
    external_paired = None
    external_coverage = None
    if include_espn:
        test_events = (
            events.loc[events["season_label"].eq(test_season)]
            .merge(
                elo[["game_id", "pregame_elo_diff"]],
                on="game_id",
                validate="many_to_one",
            )
            .merge(
                strengths[
                    ["game_id", "pregame_starter_net_diff", "starter_rating_coverage"]
                ],
                on="game_id",
                validate="many_to_one",
            )
            .merge(team_context, on="game_id", validate="many_to_one")
        )
        espn = _load_espn_plays(espn_index_path, test_season)
        external, external_coverage = match_espn_to_local_states(espn, test_events)
        external = external.loc[~external["is_terminal_event"]].copy()
        external["probability_elo"] = models["elo"].predict_proba(
            make_elo_features(external)
        )[:, 1]
        external["probability_elo_plus_starters"] = models[
            "elo_plus_starters"
        ].predict_proba(make_lineup_features(external))[:, 1]
        external["probability_elo_plus_team_context"] = models[
            "elo_plus_team_context"
        ].predict_proba(make_rolling_context_features(external))[:, 1]
        external["probability_elo_plus_starters_team_context"] = models[
            "elo_plus_starters_team_context"
        ].predict_proba(make_team_context_features(external))[:, 1]
        external_start = _checkpoint_rows(external, CHECKPOINTS["game_start"])
        external_outcome = external_start["home_win"].astype(int).to_numpy()
        external_metrics = {
            "espn": _metrics(
                external_outcome, external_start["espn_home_win_probability"].to_numpy()
            ),
            "elo": _metrics(
                external_outcome, external_start["probability_elo"].to_numpy()
            ),
            "elo_plus_starters": _metrics(
                external_outcome,
                external_start["probability_elo_plus_starters"].to_numpy(),
            ),
            "elo_plus_team_context": _metrics(
                external_outcome,
                external_start["probability_elo_plus_team_context"].to_numpy(),
            ),
            "elo_plus_starters_team_context": _metrics(
                external_outcome,
                external_start["probability_elo_plus_starters_team_context"].to_numpy(),
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
            "team_context_vs_starters": _paired_bootstrap(
                external_start,
                "probability_elo_plus_starters",
                "probability_elo_plus_starters_team_context",
                repetitions=bootstrap_repetitions,
                seed=seed,
            ),
            "starter_free_context_vs_starter_context": _paired_bootstrap(
                external_start,
                "probability_elo_plus_starters_team_context",
                "probability_elo_plus_team_context",
                repetitions=bootstrap_repetitions,
                seed=seed,
            ),
            "starter_free_context_vs_espn": _paired_bootstrap(
                external_start,
                "espn_home_win_probability",
                "probability_elo_plus_team_context",
                repetitions=bootstrap_repetitions,
                seed=seed,
            ),
            "team_context_vs_espn": _paired_bootstrap(
                external_start,
                "espn_home_win_probability",
                "probability_elo_plus_starters_team_context",
                repetitions=bootstrap_repetitions,
                seed=seed,
            ),
        }

    run_id = f"wp_pregame_ablation_v3_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "win_probability_lineup" / run_id
    output.mkdir(parents=True, exist_ok=False)
    for name, model in models.items():
        joblib.dump(model, output / f"{name}.joblib")
    prior_ratings.to_parquet(output / "prior_ratings.parquet", index=False)
    strengths.to_parquet(output / "pregame_starter_strength.parquet", index=False)
    team_context.to_parquet(output / "pregame_team_context.parquet", index=False)
    test.to_parquet(output / "test_predictions.parquet", index=False)
    if include_espn:
        external_start.to_parquet(
            output / "espn_game_start_predictions.parquet", index=False
        )
    coverage_by_season = (
        strengths.groupby("season_label")["starter_rating_coverage"]
        .agg(["count", "mean", "min"])
        .reset_index()
        .to_dict("records")
    )
    run = {
        "run_id": run_id,
        "model_family": "win_probability_logistic_pregame_context_ablation",
        "estimand": "tipoff_and_in_game_home_win_probability_with_official_starters",
        "status": "research_candidate_single_outer_fold",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "train_season_label": train_season,
            "test_season_label": test_season,
            "rating_windows": {
                str(row.prior_season_end): [
                    int(row.rating_window_start),
                    int(row.rating_window_end),
                ]
                for row in prior_ratings[
                    ["prior_season_end", "rating_window_start", "rating_window_end"]
                ]
                .drop_duplicates()
                .itertuples(index=False)
            },
            "starter_definition": "official_boxscore_starters_known_at_tipoff",
            "missing_player_rating": 0.0,
            "team_context": {
                "rolling_margin_update_weight": 0.10,
                "offseason_retention": 0.75,
                "rest_cap_days": 7,
                "same_date_results_update_after_all_pregame_features": True,
            },
            "interval_seconds": interval_seconds,
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
            "source_hashes": {
                "event_states": sha256_file(event_states_path),
                "game_dim": sha256_file(game_dim_path),
                "player_games": sha256_file(player_games_path),
                "current_possessions": sha256_file(current_possessions_path),
                "current_segments": sha256_file(current_segments_path),
                "source_code": sha256_file(Path(__file__)),
                "rapm_source_code": sha256_file(Path(__file__).with_name("rapm.py")),
                **{
                    f"legacy_possessions_{season}": sha256_file(
                        Path(legacy_cache) / f"matchups_{season}.parquet"
                    )
                    for season in (2021, 2022, 2023, 2024)
                },
                **(
                    {"espn_index": sha256_file(espn_index_path)} if include_espn else {}
                ),
            },
        },
        "metrics": {
            "train_games": int(train["game_id"].nunique()),
            "test_games": int(test["game_id"].nunique()),
            "variants": variants,
            "paired_game_bootstrap": paired,
            "team_context_paired_game_bootstrap": context_paired,
            "starter_free_context_vs_starter_context": starter_free_context_paired,
            "starter_free_context_vs_elo": starter_free_context_vs_elo,
            "checkpoints": checkpoints,
            "espn_game_start": external_metrics,
            "espn_game_start_paired": external_paired,
            "espn_match_coverage": external_coverage,
            "starter_rating_coverage": coverage_by_season,
        },
        "caveats": [
            "Official starters are a tipoff-time feature, not an earlier-day forecast feature.",
            "Unrated starters are assigned the centered RAPM mean of zero.",
            "This artifact reports one chronological outer fold; promotion uses both frozen folds.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
