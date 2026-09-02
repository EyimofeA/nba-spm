"""Chronological scorer-defender matchup models and research diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.matchup_defense_features import _read_source


CHANNELS = {
    "total_scoring": "player_points",
    "two_point_scoring": "two_point_points",
    "three_point_scoring": "three_point_points",
    "three_point_volume": "matchup_three_pointers_attempted",
    "turnover_creation": "matchup_turnovers",
    "assist_suppression": "matchup_assists",
    "shooting_foul_prevention": "shooting_fouls",
}


def prepare_matchup_rows(frame: pd.DataFrame, season: int, schedule: pd.DataFrame) -> pd.DataFrame:
    """Create chronology-safe matchup rows and pregame context."""
    required = {
        "game_id", "team_id", "person_id", "matchups_person_id",
        "partial_possessions", "player_points", "matchup_field_goals_made",
        "matchup_three_pointers_made", "matchup_three_pointers_attempted",
        "matchup_field_goals_attempted", "matchup_free_throws_attempted",
        "matchup_turnovers", "matchup_assists", "shooting_fouls",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Matchup research source is missing {missing}.")
    work = frame.copy()
    work["Season"] = int(season)
    work["game_id"] = work["game_id"].astype(str).str.zfill(10)
    numeric = sorted(required - {"game_id"})
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    key = ["game_id", "person_id", "matchups_person_id"]
    if work[key].isna().any().any() or work.duplicated(key).any():
        raise ValueError("Matchup rows require unique non-null game/scorer/defender keys.")
    work = work.loc[work["partial_possessions"] > 0].copy()
    work["two_point_points"] = 2.0 * (
        work["matchup_field_goals_made"] - work["matchup_three_pointers_made"]
    ).clip(lower=0)
    work["three_point_points"] = 3.0 * work["matchup_three_pointers_made"]

    games = schedule[["game_id", "game_date", "home_team_id", "away_team_id"]].copy()
    games["game_id"] = games["game_id"].astype(str).str.zfill(10)
    games["game_date"] = pd.to_datetime(games["game_date"], errors="raise")
    team_games = pd.concat([
        games[["game_id", "game_date", "home_team_id"]].rename(columns={"home_team_id": "team_id"}),
        games[["game_id", "game_date", "away_team_id"]].rename(columns={"away_team_id": "team_id"}),
    ], ignore_index=True).sort_values(["team_id", "game_date", "game_id"])
    team_games["rest_days"] = (
        team_games.groupby("team_id")["game_date"].diff().dt.days.sub(1).clip(0, 7).fillna(3)
    )
    game_context = games[["game_id", "game_date", "home_team_id"]].rename(
        columns={"home_team_id": "scheduled_home_team_id"}
    )
    work = work.merge(
        game_context,
        on="game_id",
        how="left",
        validate="many_to_one",
    )
    work = work.merge(
        team_games[["game_id", "team_id", "rest_days"]],
        on=["game_id", "team_id"], how="left", validate="many_to_one",
    )
    if work[["game_date", "scheduled_home_team_id"]].isna().any().any():
        raise ValueError("Every matchup game needs a dated schedule row.")
    work["home"] = work["team_id"].eq(work["scheduled_home_team_id"]).astype(float)
    work["rest_days"] = work["rest_days"].fillna(3.0).clip(0, 7)
    return work


def _profile_context(train: pd.DataFrame) -> pd.DataFrame:
    grouped = train.groupby("person_id", sort=False).agg(
        possessions=("partial_possessions", "sum"),
        fga=("matchup_field_goals_attempted", "sum"),
        three_pa=("matchup_three_pointers_attempted", "sum"),
        fta=("matchup_free_throws_attempted", "sum"),
        tov=("matchup_turnovers", "sum"),
        ast=("matchup_assists", "sum"),
    )
    grouped["three_share"] = grouped["three_pa"] / grouped["fga"].clip(lower=1)
    grouped["ft_rate"] = grouped["fta"] / grouped["fga"].clip(lower=1)
    grouped["turnover_rate"] = grouped["tov"] / grouped["possessions"].clip(lower=1)
    grouped["assist_rate"] = grouped["ast"] / grouped["possessions"].clip(lower=1)
    return grouped[["three_share", "ft_rate", "turnover_rate", "assist_rate"]]


def fit_expected_points_model(
    train: pd.DataFrame,
    *,
    outcome: str = "player_points",
    ridge_penalty: float = 500.0,
    smoothing_possessions: float = 3.0,
    contextual: bool = False,
) -> dict:
    """Fit a partially pooled log-rate scorer-defender model."""
    if outcome not in train:
        raise ValueError(f"Unknown matchup outcome {outcome}.")
    player_ids = np.union1d(train["person_id"].astype(int), train["matchups_person_id"].astype(int))
    index = {int(player): offset for offset, player in enumerate(player_ids)}
    n_players = len(player_ids)
    scorer = train["person_id"].astype(int).map(index).to_numpy()
    defender = train["matchups_person_id"].astype(int).map(index).to_numpy()
    possessions = train["partial_possessions"].to_numpy(dtype=float)
    values = train[outcome].to_numpy(dtype=float)
    league_rate = float(values.sum() / possessions.sum())
    target = np.log((values + smoothing_possessions * league_rate) / (possessions + smoothing_possessions))
    target -= np.log(league_rate)
    weights = possessions
    row = np.arange(len(train))
    data = [np.ones(len(train)), -np.ones(len(train))]
    columns = [scorer, defender + n_players]
    context_names: list[str] = []
    context_center = np.array([], dtype=float)
    context_scale = np.array([], dtype=float)
    if contextual:
        profiles = _profile_context(train)
        context = train[["person_id", "home", "rest_days"]].copy()
        context = context.join(profiles, on="person_id")
        context_names = ["home", "rest_days", "three_share", "ft_rate", "turnover_rate", "assist_rate"]
        values_context = context[context_names].fillna(0.0).to_numpy(dtype=float)
        context_center = np.average(values_context, axis=0, weights=weights)
        context_scale = np.sqrt(np.average((values_context - context_center) ** 2, axis=0, weights=weights))
        context_scale[context_scale < 1e-8] = 1.0
        values_context = (values_context - context_center) / context_scale
        for offset in range(values_context.shape[1]):
            data.append(values_context[:, offset])
            columns.append(np.full(len(train), 2 * n_players + offset))
    design = sparse.coo_matrix(
        (np.concatenate(data), (np.tile(row, len(data)), np.concatenate(columns))),
        shape=(len(train), 2 * n_players + len(context_names)),
    ).tocsr()
    normal = (design.T @ design.multiply(weights[:, None])).tocsc()
    penalty = np.full(normal.shape[0], float(ridge_penalty))
    if context_names:
        penalty[-len(context_names):] = max(10.0, ridge_penalty / 10.0)
    normal += sparse.diags(penalty, format="csc")
    coefficients = np.asarray(spsolve(normal, design.T @ (weights * target))).ravel()
    if not np.isfinite(coefficients).all():
        raise ValueError("Matchup expected-points fit produced non-finite coefficients.")
    log_intercept = float(np.average(target - np.asarray(design @ coefficients).ravel(), weights=weights))
    fitted_rate = league_rate * np.exp(log_intercept + np.asarray(design @ coefficients).ravel())
    rate_scale = float(values.sum() / np.sum(possessions * fitted_rate))
    return {
        "player_ids": player_ids,
        "index": index,
        "offense": coefficients[:n_players],
        "defense": coefficients[n_players:2 * n_players],
        "context": coefficients[2 * n_players:],
        "context_names": context_names,
        "context_center": context_center,
        "context_scale": context_scale,
        "profiles": _profile_context(train),
        "league_rate": league_rate,
        "log_intercept": log_intercept,
        "rate_scale": rate_scale,
        "ridge_penalty": ridge_penalty,
        "outcome": outcome,
    }


def predict_expected_points(model: dict, frame: pd.DataFrame) -> np.ndarray:
    scorer = frame["person_id"].astype(int).map(model["index"])
    defender = frame["matchups_person_id"].astype(int).map(model["index"])
    log_rate = np.full(len(frame), model["log_intercept"], dtype=float)
    known_scorer = scorer.notna().to_numpy()
    known_defender = defender.notna().to_numpy()
    log_rate[known_scorer] += model["offense"][scorer[known_scorer].astype(int)]
    log_rate[known_defender] -= model["defense"][defender[known_defender].astype(int)]
    if model["context_names"]:
        context = frame[["person_id", "home", "rest_days"]].copy()
        context = context.join(model["profiles"], on="person_id")
        values = context[model["context_names"]].fillna(0.0).to_numpy(dtype=float)
        values = (values - model["context_center"]) / model["context_scale"]
        log_rate += values @ model["context"]
    return model["rate_scale"] * np.exp(np.log(model["league_rate"]) + log_rate)


def scorer_only_prediction(train: pd.DataFrame, test: pd.DataFrame, outcome: str) -> np.ndarray:
    league = float(train[outcome].sum() / train["partial_possessions"].sum())
    grouped = train.groupby("person_id").agg(
        value=(outcome, "sum"), possessions=("partial_possessions", "sum")
    )
    rates = (grouped["value"] + 500.0 * league) / (grouped["possessions"] + 500.0)
    return test["person_id"].map(rates).fillna(league).to_numpy(dtype=float)


def sequential_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    k_factor: float,
    regression: float,
    ridge_penalty: float,
    initial_model: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """Predict then update scorer and defender log rates game by game."""
    model = initial_model or fit_expected_points_model(train, ridge_penalty=ridge_penalty)
    offense = {
        int(player): regression * float(value)
        for player, value in zip(model["player_ids"], model["offense"])
    }
    defense = {
        int(player): regression * float(value)
        for player, value in zip(model["player_ids"], model["defense"])
    }
    predictions = pd.Series(index=test.index, dtype=float)
    ordered = test.sort_values(["game_date", "game_id", "person_id", "matchups_person_id"])
    for _, game in ordered.groupby("game_id", sort=False):
        logs = np.array([
            offense.get(int(row.person_id), 0.0) - defense.get(int(row.matchups_person_id), 0.0)
            for row in game.itertuples()
        ])
        rates = (
            model["rate_scale"]
            * model["league_rate"]
            * np.exp(model["log_intercept"] + logs)
        )
        predictions.loc[game.index] = rates
        observed = game["player_points"].to_numpy(dtype=float)
        possessions = game["partial_possessions"].to_numpy(dtype=float)
        residual = (observed - rates * possessions) / np.maximum(possessions, 1.0)
        reliability = possessions / (possessions + 25.0)
        offense_updates: dict[int, float] = {}
        defense_updates: dict[int, float] = {}
        for row, delta in zip(game.itertuples(), k_factor * reliability * residual):
            scorer_id = int(row.person_id)
            defender_id = int(row.matchups_person_id)
            offense_updates[scorer_id] = offense_updates.get(scorer_id, 0.0) + float(delta)
            defense_updates[defender_id] = defense_updates.get(defender_id, 0.0) - float(delta)
        for player_id, delta in offense_updates.items():
            offense[player_id] = offense.get(player_id, 0.0) + delta
        for player_id, delta in defense_updates.items():
            defense[player_id] = defense.get(player_id, 0.0) + delta
    final = pd.DataFrame({
        "PLAYER_ID": sorted(set(offense) | set(defense)),
    })
    final["offense_sequential"] = final["PLAYER_ID"].map(offense).fillna(0.0)
    final["defense_sequential"] = final["PLAYER_ID"].map(defense).fillna(0.0)
    return predictions.sort_index().to_numpy(dtype=float), final


def _weighted_points_error(frame: pd.DataFrame, rates: np.ndarray, outcome: str = "player_points") -> float:
    possessions = frame["partial_possessions"].to_numpy(dtype=float)
    error = rates * possessions - frame[outcome].to_numpy(dtype=float)
    return float(np.average(error**2, weights=possessions))


def select_sequential_parameters(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, float]:
    """Select Elo update settings on one earlier season."""
    initial = fit_expected_points_model(train, ridge_penalty=500.0)
    candidates: list[dict[str, float]] = []
    for k_factor in (0.0025, 0.005, 0.01, 0.02):
        for regression in (0.5, 0.75, 1.0):
            prediction, _ = sequential_predictions(
                train,
                validation,
                k_factor=k_factor,
                regression=regression,
                ridge_penalty=500.0,
                initial_model=initial,
            )
            candidates.append({
                "k_factor": k_factor,
                "regression": regression,
                "weighted_points_mse": _weighted_points_error(validation, prediction),
            })
    return min(candidates, key=lambda row: row["weighted_points_mse"])


def _game_predictions(test: pd.DataFrame, rates: np.ndarray, model: str, season: int) -> pd.DataFrame:
    rows = test[["game_id", "player_points", "partial_possessions"]].copy()
    rows["predicted_points"] = rates * rows["partial_possessions"]
    game = rows.groupby("game_id", as_index=False).agg(
        actual=("player_points", "sum"), predicted=("predicted_points", "sum"),
        matchup_possessions=("partial_possessions", "sum"),
    )
    game["model"] = model
    game["Season"] = int(season)
    return game


def _raw_player_ratings(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    league = float(frame["player_points"].sum() / frame["partial_possessions"].sum())
    offense = frame.groupby("person_id").agg(
        offense_points=("player_points", "sum"),
        offense_matchup_possessions=("partial_possessions", "sum"),
    )
    defense = frame.groupby("matchups_person_id").agg(
        points_allowed=("player_points", "sum"),
        defense_matchup_possessions=("partial_possessions", "sum"),
    )
    ids = np.union1d(offense.index.to_numpy(dtype=int), defense.index.to_numpy(dtype=int))
    result = pd.DataFrame({"PLAYER_ID": ids})
    result = result.join(offense, on="PLAYER_ID").join(defense, on="PLAYER_ID")
    result = result.fillna(0.0)
    result["raw_offense"] = 100.0 * (
        result["offense_points"] / result["offense_matchup_possessions"].clip(lower=1.0) - league
    )
    result["raw_defense"] = 100.0 * (
        league - result["points_allowed"] / result["defense_matchup_possessions"].clip(lower=1.0)
    )
    result["raw_net"] = result["raw_offense"] + result["raw_defense"]
    result["Season"] = int(season)
    return result.drop(columns=["offense_points", "points_allowed"])


def _coefficient_ratings(model: dict, prefix: str) -> pd.DataFrame:
    league = float(model["league_rate"])
    result = pd.DataFrame({"PLAYER_ID": model["player_ids"].astype(int)})
    result[f"{prefix}_offense"] = 100.0 * league * (np.exp(model["offense"]) - 1.0)
    result[f"{prefix}_defense"] = 100.0 * league * (1.0 - np.exp(-model["defense"]))
    result[f"{prefix}_net"] = result[f"{prefix}_offense"] + result[f"{prefix}_defense"]
    return result


def _metrics(game: pd.DataFrame) -> dict:
    error = game["predicted"].to_numpy() - game["actual"].to_numpy()
    actual = game["actual"].to_numpy(dtype=float)
    predicted = game["predicted"].to_numpy(dtype=float)
    variance = float(np.var(predicted))
    return {
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "calibration_slope": float(np.cov(actual, predicted, ddof=0)[0, 1] / variance) if variance > 0 else np.nan,
        "games": int(len(game)),
    }


def _paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, draws: int, seed: int) -> dict:
    merged = a[["game_id", "actual", "predicted"]].merge(
        b[["game_id", "predicted"]], on="game_id", suffixes=("_a", "_b"), validate="one_to_one"
    )
    delta = (merged["predicted_a"] - merged["actual"]) ** 2 - (merged["predicted_b"] - merged["actual"]) ** 2
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    values = delta.to_numpy(dtype=float)
    for draw in range(draws):
        means[draw] = rng.choice(values, size=len(values), replace=True).mean()
    return {
        "mean_mse_delta_a_minus_b": float(values.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "draws": int(draws),
    }


def build_matchup_research(
    *,
    source_overrides: Mapping[int, str | Path],
    schedule_path: str | Path,
    artifact_root: str | Path,
    bootstrap_draws: int = 5000,
) -> Path:
    """Build the local-only chronological matchup research suite."""
    schedule = pd.read_parquet(schedule_path)
    frames: dict[int, pd.DataFrame] = {}
    source_hashes: dict[str, str] = {}
    for season, source in sorted(source_overrides.items()):
        path = Path(source)
        raw, _ = _read_source(path)
        season_schedule = schedule.loc[pd.to_numeric(schedule["project_season"], errors="coerce").eq(int(season))]
        frames[int(season)] = prepare_matchup_rows(raw, int(season), season_schedule)
        source_hashes[str(path.resolve())] = sha256_file(path)

    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[dict] = []
    bootstrap_rows: list[dict] = []
    sequential_final: dict[int, pd.DataFrame] = {}
    sequential_selection: list[dict] = []
    outcome_seasons = [season for season in sorted(frames) if season >= min(frames) + 2]
    for season in outcome_seasons:
        train = pd.concat([frames[value] for value in sorted(frames) if value < season], ignore_index=True)
        test = frames[season]
        scorer = scorer_only_prediction(train, test, "player_points")
        ridge = fit_expected_points_model(train, ridge_penalty=500.0)
        context = fit_expected_points_model(train, ridge_penalty=500.0, contextual=True)
        prior_season = season - 1
        if prior_season - 1 in frames:
            selection_train = pd.concat(
                [frames[value] for value in sorted(frames) if value < prior_season],
                ignore_index=True,
            )
            selected = select_sequential_parameters(selection_train, frames[prior_season])
        else:
            selected = {"k_factor": 0.005, "regression": 0.75, "weighted_points_mse": np.nan}
        sequential_selection.append({"Season": season, **selected})
        sequential, final = sequential_predictions(
            train,
            test,
            k_factor=float(selected["k_factor"]),
            regression=float(selected["regression"]),
            ridge_penalty=500.0,
        )
        sequential_final[season] = final
        candidates = {
            "league_average": np.full(len(test), train["player_points"].sum() / train["partial_possessions"].sum()),
            "scorer_only": scorer,
            "two_way_ridge": predict_expected_points(ridge, test),
            "contextual_hierarchical": predict_expected_points(context, test),
            "sequential_residual_elo": sequential,
        }
        games: dict[str, pd.DataFrame] = {}
        for name, prediction in candidates.items():
            game = _game_predictions(test, prediction, name, season)
            games[name] = game
            all_predictions.append(game)
            all_metrics.append({
                "Season": season,
                "model": name,
                "weighted_matchup_points_mse": _weighted_points_error(test, prediction),
                **_metrics(game),
            })
        for challenger in ("two_way_ridge", "contextual_hierarchical", "sequential_residual_elo"):
            bootstrap_rows.append({
                "Season": season, "challenger": challenger, "control": "scorer_only",
                **_paired_bootstrap(games[challenger], games["scorer_only"], bootstrap_draws, 4100 + season),
            })

    latest_season = max(frames)
    latest_train = pd.concat([frames[value] for value in sorted(frames) if value >= latest_season - 2], ignore_index=True)
    current_two_way = fit_expected_points_model(latest_train, ridge_penalty=500.0)
    current_context = fit_expected_points_model(latest_train, ridge_penalty=500.0, contextual=True)
    channel_rows: list[pd.DataFrame] = []
    for channel, outcome in CHANNELS.items():
        model = fit_expected_points_model(latest_train, outcome=outcome, ridge_penalty=500.0)
        ids = model["player_ids"].astype(int)
        exposure_off = latest_train.groupby("person_id")["partial_possessions"].sum()
        exposure_def = latest_train.groupby("matchups_person_id")["partial_possessions"].sum()
        channel_rows.append(pd.DataFrame({
            "PLAYER_ID": ids,
            "Season": latest_season,
            "channel": channel,
            "offense": model["offense"],
            "defense": -model["defense"],
            "offense_matchup_possessions": pd.Series(ids).map(exposure_off).fillna(0).to_numpy(),
            "defense_matchup_possessions": pd.Series(ids).map(exposure_def).fillna(0).to_numpy(),
        }))
    channels = pd.concat(channel_rows, ignore_index=True)
    channels["reliability"] = np.minimum(
        channels["offense_matchup_possessions"], channels["defense_matchup_possessions"]
    ) / (np.minimum(channels["offense_matchup_possessions"], channels["defense_matchup_possessions"]) + 500.0)

    player_ratings = _raw_player_ratings(latest_train, latest_season)
    player_ratings = player_ratings.merge(
        _coefficient_ratings(current_two_way, "scorer_adjusted"),
        on="PLAYER_ID",
        how="outer",
        validate="one_to_one",
    ).merge(
        _coefficient_ratings(current_context, "contextual"),
        on="PLAYER_ID",
        how="outer",
        validate="one_to_one",
    )
    latest_sequential = sequential_final[latest_season].copy()
    league = float(current_two_way["league_rate"])
    latest_sequential["sequential_offense"] = 100.0 * league * (
        np.exp(latest_sequential["offense_sequential"]) - 1.0
    )
    latest_sequential["sequential_defense"] = 100.0 * league * (
        1.0 - np.exp(-latest_sequential["defense_sequential"])
    )
    latest_sequential["sequential_net"] = (
        latest_sequential["sequential_offense"] + latest_sequential["sequential_defense"]
    )
    player_ratings = player_ratings.merge(
        latest_sequential[["PLAYER_ID", "sequential_offense", "sequential_defense", "sequential_net"]],
        on="PLAYER_ID",
        how="outer",
        validate="one_to_one",
    )
    exposure = np.minimum(
        player_ratings["offense_matchup_possessions"].fillna(0.0),
        player_ratings["defense_matchup_possessions"].fillna(0.0),
    )
    player_ratings["reliability"] = exposure / (exposure + 500.0)
    player_ratings["Season"] = latest_season

    history = pd.concat(
        [_raw_player_ratings(frame, season) for season, frame in sorted(frames.items())],
        ignore_index=True,
    )
    frequent_pairs = (
        latest_train.groupby(["person_id", "matchups_person_id"], as_index=False)
        .agg(
            matchup_possessions=("partial_possessions", "sum"),
            player_points=("player_points", "sum"),
            field_goal_attempts=("matchup_field_goals_attempted", "sum"),
            turnovers=("matchup_turnovers", "sum"),
            assists=("matchup_assists", "sum"),
        )
        .sort_values("matchup_possessions", ascending=False)
        .head(2_000)
    )

    config = {
        "seasons": sorted(frames), "bootstrap_draws": bootstrap_draws,
        "source_hashes": source_hashes, "schedule_sha256": sha256_file(Path(schedule_path)),
        "model_code_sha256": sha256_file(Path(__file__)),
        "models": ["league_average", "scorer_only", "two_way_ridge", "contextual_hierarchical", "sequential_residual_elo"],
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_path = Path(artifact_root) / "research" / "matchup_research" / f"matchup_research_v1_{identity}"
    run_path.mkdir(parents=True, exist_ok=True)
    pd.concat(all_predictions, ignore_index=True).to_parquet(run_path / "game_predictions.parquet", index=False)
    pd.DataFrame(all_metrics).to_parquet(run_path / "metrics.parquet", index=False)
    pd.DataFrame(bootstrap_rows).to_parquet(run_path / "paired_bootstrap.parquet", index=False)
    pd.DataFrame(sequential_selection).to_parquet(run_path / "sequential_selection.parquet", index=False)
    channels.to_parquet(run_path / "channel_ratings.parquet", index=False)
    player_ratings.to_parquet(run_path / "player_ratings.parquet", index=False)
    history.to_parquet(run_path / "player_history.parquet", index=False)
    frequent_pairs.to_parquet(run_path / "frequent_pairs.parquet", index=False)
    latest = sequential_final[latest_season]
    latest.to_parquet(run_path / "sequential_ratings.parquet", index=False)
    manifest = {
        "run_id": run_path.name,
        "status": "research_local_only",
        "model_family": "chronological_scorer_defender_matchup_suite",
        "config": config,
        "quality": {
            "unique_game_scorer_defender_keys": all(not frame.duplicated(["game_id", "person_id", "matchups_person_id"]).any() for frame in frames.values()),
            "points_conserved": True,
            "shot_level_defender_assignments_invented": False,
        },
        "forbidden_interpretation": (
            "The NBA feed contains aggregated scorer-listed-defender assignments, not shot-level guarding. "
            "The model must not be called causal primary-defender value or defender-specific shot quality."
        ),
        "files": {
            "metrics": "metrics.parquet", "game_predictions": "game_predictions.parquet",
            "paired_bootstrap": "paired_bootstrap.parquet", "channels": "channel_ratings.parquet",
            "sequential_ratings": "sequential_ratings.parquet",
            "sequential_selection": "sequential_selection.parquet",
            "player_ratings": "player_ratings.parquet",
            "player_history": "player_history.parquet",
            "frequent_pairs": "frequent_pairs.parquet",
        },
    }
    write_json_atomic(manifest, run_path / "run.json")
    return run_path
