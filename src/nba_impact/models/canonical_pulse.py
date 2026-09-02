"""PULSE fit and chronological validation on canonical lineup stints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import (
    StintRapmDesign,
    build_stint_design,
    fit_stint_center_path,
    load_canonical_stints,
    stint_ratings,
)


def _identity_errors(ratings: pd.DataFrame) -> dict[str, float]:
    return {
        "prior_side_identity": float(
            (ratings["pulse_prior_offense"] + ratings["pulse_prior_defense"]
             - ratings["pulse_prior_net"]).abs().max()
        ),
        "update_side_identity": float(
            (ratings["lineup_update_offense"] + ratings["lineup_update_defense"]
             - ratings["lineup_update_net"]).abs().max()
        ),
        "pulse_sum_identity": float(
            (ratings["pulse_prior_net"] + ratings["lineup_update_net"]
             - ratings["pulse_net"]).abs().max()
        ),
        "pulse_side_identity": float(
            (ratings["pulse_offense"] + ratings["pulse_defense"]
             - ratings["pulse_net"]).abs().max()
        ),
        "rapm_side_identity": float(
            (ratings["rapm_offense"] + ratings["rapm_defense"]
             - ratings["rapm_net"]).abs().max()
        ),
    }


def _load_names(
    historical_player_sheets: Path,
    gabriel_player_sheets: Path,
    player_games_path: Path,
) -> pd.DataFrame:
    historical = []
    for season in range(1997, 2014):
        frame = pd.read_csv(
            historical_player_sheets / f"{season}.csv",
            usecols=["PLAYER_ID", "PLAYER_NAME"],
        )
        frame["Season"] = season
        historical.append(frame)
    old = (
        pd.concat(historical, ignore_index=True)
        .dropna(subset=["PLAYER_ID", "PLAYER_NAME"])
        .sort_values(["Season", "PLAYER_ID"], kind="stable")
        .drop_duplicates("PLAYER_ID", keep="last")
    )[["PLAYER_ID", "PLAYER_NAME"]]
    gabriel = []
    for season in range(2012, 2027):
        frame = pd.read_parquet(
            gabriel_player_sheets / f"{season}.parquet",
            columns=["PLAYER_ID", "PLAYER_NAME"],
        )
        frame["Season"] = season
        gabriel.append(frame)
    middle = (
        pd.concat(gabriel, ignore_index=True)
        .dropna(subset=["PLAYER_ID", "PLAYER_NAME"])
        .sort_values(["Season", "PLAYER_ID"], kind="stable")
        .drop_duplicates("PLAYER_ID", keep="last")
    )[["PLAYER_ID", "PLAYER_NAME"]]
    current = pd.read_parquet(
        player_games_path, columns=["player_id", "player_name", "game_date"]
    ).dropna(subset=["player_id", "player_name"])
    current = (
        current.sort_values(["game_date", "player_id"], kind="stable")
        .drop_duplicates("player_id", keep="last")
        .rename(columns={"player_id": "PLAYER_ID", "player_name": "PLAYER_NAME"})
    )[["PLAYER_ID", "PLAYER_NAME"]]
    for frame in (old, middle, current):
        frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
    return pd.concat([old, middle, current], ignore_index=True).drop_duplicates(
        "PLAYER_ID", keep="last"
    )


def fit_box_prior(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    training_before: int | None,
    offense_alpha: float,
    defense_alpha: float,
) -> tuple[pd.DataFrame, dict]:
    labels = targets.loc[
        targets["horizon"].eq(9) & targets["target_variant"].eq("normal")
    ].copy()
    if training_before is not None:
        labels = labels.loc[labels["Window_End"].lt(training_before)]
    panel = features.merge(labels, on=["PLAYER_ID", "Window_End"], validate="one_to_one")
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    models = {
        "offense": _fit(panel, BOX_PIPM_STYLE_FEATURES, "target_offense", offense_alpha),
        "defense": _fit(panel, BOX_PIPM_STYLE_FEATURES, "target_defense", defense_alpha),
    }
    prediction_rows = (
        features
        if training_before is None
        else features.loc[features["Window_End"].eq(training_before)]
    )
    priors = prediction_rows[["PLAYER_ID", "Window_End"]].copy()
    for side in ("offense", "defense"):
        priors[f"prior_{side}_per_100"] = models[side].predict(
            prediction_rows.loc[:, BOX_PIPM_STYLE_FEATURES]
        )
    priors["prior_net_per_100"] = (
        priors["prior_offense_per_100"] + priors["prior_defense_per_100"]
    )
    return priors, {
        "training_rows": int(len(panel)),
        "training_start": int(panel["Window_End"].min()),
        "training_end": int(panel["Window_End"].max()),
        "prediction_rows": int(len(priors)),
    }


def stint_prior_center(
    design: StintRapmDesign,
    priors: pd.DataFrame,
    season: int,
) -> tuple[np.ndarray, dict]:
    window = priors.loc[priors["Window_End"].eq(season)].set_index("PLAYER_ID")
    index = pd.Index(design.players)
    offense = window["prior_offense_per_100"].reindex(index)
    defense = window["prior_defense_per_100"].reindex(index)
    has_prior = offense.notna() & defense.notna()
    offense_coefficients = offense.fillna(0).to_numpy(float) / 100
    defense_coefficients = -defense.fillna(0).to_numpy(float) / 100
    offense_coefficients -= np.average(
        offense_coefficients, weights=design.off_possessions
    )
    defense_coefficients -= np.average(
        defense_coefficients, weights=design.def_possessions
    )
    center = np.concatenate([offense_coefficients, defense_coefficients, [0.0]])
    return center, {
        "Season": season,
        "players": len(index),
        "players_with_prior": int(has_prior.sum()),
        "offense_possession_coverage": float(
            np.average(has_prior.to_numpy(float), weights=design.off_possessions)
        ),
        "defense_possession_coverage": float(
            np.average(has_prior.to_numpy(float), weights=design.def_possessions)
        ),
    }


def fit_pulse_season(
    season: int,
    priors: pd.DataFrame,
    stint_root: Path,
    config: dict,
) -> tuple[pd.DataFrame, dict, StintRapmDesign, dict[str, tuple[np.ndarray, float]]]:
    design = build_stint_design(load_canonical_stints(stint_root, (season,)))
    update = config["lineup_update"]
    rapm_config = RapmConfig(
        (season,),
        lambda_off=float(update["lambda_off"]),
        lambda_def=float(update["lambda_def"]),
        lambda_home=float(update["lambda_home"]),
        data_scope="canonical_score_conserving_lineup_stints",
    )
    zero = np.zeros(design.X.shape[1])
    zero_beta, zero_intercept = fit_stint_center_path(
        design, rapm_config, zero, center_scales=(0.0,)
    )[0.0]
    center, coverage = stint_prior_center(design, priors, season)
    pulse_beta, pulse_intercept = fit_stint_center_path(
        design,
        rapm_config,
        center,
        center_scales=(float(update["prior_scale"]),),
    )[float(update["prior_scale"])]
    prior = stint_ratings(design, center).rename(
        columns={
            "offense": "pulse_prior_offense",
            "defense": "pulse_prior_defense",
            "net": "pulse_prior_net",
        }
    )
    pulse = stint_ratings(design, pulse_beta).rename(
        columns={"offense": "pulse_offense", "defense": "pulse_defense", "net": "pulse_net"}
    )
    rapm = stint_ratings(design, zero_beta).rename(
        columns={"offense": "rapm_offense", "defense": "rapm_defense", "net": "rapm_net"}
    )
    ratings = prior.merge(
        pulse.drop(columns=["Poss_Off", "Poss_Def"]), on="PLAYER_ID", validate="one_to_one"
    ).merge(
        rapm.drop(columns=["Poss_Off", "Poss_Def"]), on="PLAYER_ID", validate="one_to_one"
    )
    for side in ("offense", "defense", "net"):
        ratings[f"lineup_update_{side}"] = ratings[f"pulse_{side}"] - ratings[f"pulse_prior_{side}"]
    ratings["Season"] = season
    return ratings, coverage, design, {
        "prior": (center, zero_intercept),
        "pulse": (pulse_beta, pulse_intercept),
        "rapm": (zero_beta, zero_intercept),
    }


def predict_next_season_games(
    source: StintRapmDesign,
    target: StintRapmDesign,
    beta: np.ndarray,
    intercept: float,
    changed_players: set[int] | None = None,
) -> pd.DataFrame:
    source_lookup = {int(player): index for index, player in enumerate(source.players)}
    target_beta = np.zeros(target.X.shape[1])
    n_source = len(source.players)
    n_target = len(target.players)
    known = np.zeros(n_target, dtype=bool)
    source_exposure = np.zeros(n_target, dtype=float)
    for target_index, player in enumerate(target.players):
        source_index = source_lookup.get(int(player))
        if source_index is None:
            continue
        known[target_index] = True
        source_exposure[target_index] = min(
            source.off_possessions[source_index], source.def_possessions[source_index]
        )
        target_beta[target_index] = beta[source_index]
        target_beta[n_target + target_index] = beta[n_source + source_index]
    target_beta[-1] = beta[-1]
    rate = np.asarray(target.X @ target_beta).ravel() + intercept
    sign = np.where(target.home_offense, 1.0, -1.0)
    lineup_columns = target.X[:, : 2 * n_target]
    player_indices = lineup_columns.indices % n_target
    row_starts = lineup_columns.indptr[:-1]
    row_exposure = np.add.reduceat(source_exposure[player_indices], row_starts) / 10
    changed = np.array(
        [int(player) in (changed_players or set()) for player in target.players],
        dtype=float,
    )
    row_changed_share = np.add.reduceat(changed[player_indices], row_starts) / 10
    rows = pd.DataFrame(
        {
            "game_id": target.game_ids,
            "actual": target.points * sign,
            "predicted": rate * target.possessions * sign,
            "possessions": target.possessions,
            "exposure_weighted": row_exposure * target.possessions,
            "changed_weighted": row_changed_share * target.possessions,
        }
    )
    games = rows.groupby("game_id", as_index=False).agg(
        actual_margin=("actual", "sum"),
        predicted_margin=("predicted", "sum"),
        possessions=("possessions", "sum"),
        exposure_weighted=("exposure_weighted", "sum"),
        changed_weighted=("changed_weighted", "sum"),
    )
    games["mean_prior_exposure"] = games["exposure_weighted"] / games["possessions"]
    games["team_changer_share"] = games["changed_weighted"] / games["possessions"]
    games = games.drop(columns=["exposure_weighted", "changed_weighted"])
    games.attrs["known_player_rate"] = float(known.mean())
    return games


def game_metrics(games: pd.DataFrame) -> dict:
    error = games["actual_margin"] - games["predicted_margin"]
    variance = float(games["predicted_margin"].var(ddof=0))
    correlation = float(games[["actual_margin", "predicted_margin"]].corr().iloc[0, 1])
    slope = (
        float(np.cov(games["actual_margin"], games["predicted_margin"], ddof=0)[0, 1] / variance)
        if variance > 0 else float("nan")
    )
    return {
        "games": int(len(games)),
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": correlation,
        "calibration_slope": slope,
    }


def build_canonical_pulse(
    config_path: Path,
    features_path: Path,
    targets_path: Path,
    stint_root: Path,
    artifact_root: Path,
    *,
    historical_player_sheets: Path,
    gabriel_player_sheets: Path,
    player_games_path: Path,
) -> dict:
    config = yaml.safe_load(config_path.read_text())
    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(targets_path)
    hashes = {
        "config": sha256_file(config_path),
        "features": sha256_file(features_path),
        "targets": sha256_file(targets_path),
        "stint_manifest": sha256_file(stint_root / "manifest.json"),
        "builder": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = artifact_root / "models/pulse" / f"pulse_canonical_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)

    final_priors, final_quality = fit_box_prior(
        features,
        targets,
        training_before=None,
        offense_alpha=float(config["prior"]["offense_alpha"]),
        defense_alpha=float(config["prior"]["defense_alpha"]),
    )
    final_priors.to_parquet(output / "final_priors.parquet", index=False)
    final_rows = []
    coverage_rows = []
    for season in range(1997, 2027):
        ratings_path = checkpoint / f"ratings_{season}.parquet"
        coverage_path = checkpoint / f"coverage_{season}.json"
        if not ratings_path.exists():
            ratings, coverage, _, _ = fit_pulse_season(season, final_priors, stint_root, config)
            ratings.to_parquet(ratings_path, index=False)
            write_json_atomic(coverage, coverage_path)
        final_rows.append(pd.read_parquet(ratings_path))
        coverage_rows.append(json.loads(coverage_path.read_text()))

    validation_games = []
    validation_folds = []
    validation_priors = []
    team_by_season = {}
    for season in range(2014, 2027):
        sheet = pd.read_parquet(
            gabriel_player_sheets / f"{season}.parquet",
            columns=["PLAYER_ID", "TEAM_ID"],
        ).dropna()
        team_by_season[season] = {
            int(player): int(team)
            for player, team in zip(sheet["PLAYER_ID"], sheet["TEAM_ID"], strict=True)
        }
    for rating_season in range(2014, 2026):
        priors, quality = fit_box_prior(
            features,
            targets,
            training_before=rating_season,
            offense_alpha=float(config["prior"]["offense_alpha"]),
            defense_alpha=float(config["prior"]["defense_alpha"]),
        )
        priors["rating_season"] = rating_season
        validation_priors.append(priors)
        _, coverage, source_design, fits = fit_pulse_season(
            rating_season, priors, stint_root, config
        )
        target_design = build_stint_design(
            load_canonical_stints(stint_root, (rating_season + 1,))
        )
        changed_players = {
            player
            for player, next_team in team_by_season[rating_season + 1].items()
            if player in team_by_season[rating_season]
            and team_by_season[rating_season][player] != next_team
        }
        for candidate, (beta, intercept) in fits.items():
            games = predict_next_season_games(
                source_design, target_design, beta, intercept, changed_players
            )
            games["candidate"] = candidate
            games["rating_season"] = rating_season
            games["outcome_season"] = rating_season + 1
            games["squared_error"] = (
                games["actual_margin"] - games["predicted_margin"]
            ) ** 2
            validation_games.append(games)
            validation_folds.append(
                {
                    "candidate": candidate,
                    "rating_season": rating_season,
                    "outcome_season": rating_season + 1,
                    "training_start": quality["training_start"],
                    "training_end": quality["training_end"],
                    "prior_offense_coverage": coverage["offense_possession_coverage"],
                    "prior_defense_coverage": coverage["defense_possession_coverage"],
                    "known_player_rate": games.attrs["known_player_rate"],
                    **game_metrics(games),
                }
            )
        print(f"PULSE validation {rating_season}->{rating_season + 1}: complete", flush=True)

    ratings = pd.concat(final_rows, ignore_index=True)
    names = _load_names(historical_player_sheets, gabriel_player_sheets, player_games_path)
    ratings = ratings.merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    ratings["evidence_status"] = "final_descriptive_refit"
    ratings["estimand_id"] = config["estimand_id"]
    if ratings["PLAYER_NAME"].isna().any():
        raise ValueError("Canonical PULSE contains unnamed players.")
    errors = _identity_errors(ratings)
    if max(errors.values()) > 1e-9:
        raise ValueError(f"PULSE identity failure: {errors}")
    games = pd.concat(validation_games, ignore_index=True)
    folds = pd.DataFrame(validation_folds)
    summary = folds.groupby("candidate", as_index=False).agg(
        folds=("outcome_season", "nunique"),
        equal_season_mse=("mse", "mean"),
        mean_correlation=("correlation", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    rng = np.random.default_rng(20260902)
    wide = games.loc[games["candidate"].isin(["pulse", "rapm"])].pivot(
        index=["outcome_season", "game_id"], columns="candidate", values="squared_error"
    )
    if wide.isna().any().any():
        raise ValueError("PULSE and RAPM did not score identical games.")
    season_values = [
        group["pulse"].to_numpy() - group["rapm"].to_numpy()
        for _, group in wide.groupby(level="outcome_season")
    ]
    draws = np.array(
        [
            np.mean([
                values[rng.integers(0, len(values), len(values))].mean()
                for values in season_values
            ])
            for _ in range(5000)
        ]
    )
    bootstrap = {
        "pulse_minus_rapm_mse": float(np.mean([values.mean() for values in season_values])),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "probability_pulse_better": float(np.mean(draws < 0)),
        "draws": 5000,
    }
    context = games.loc[games["candidate"].isin(["pulse", "rapm"])].pivot(
        index=["outcome_season", "game_id"], columns="candidate", values="squared_error"
    ).reset_index()
    game_context = games.loc[games["candidate"].eq("pulse"), [
        "outcome_season", "game_id", "mean_prior_exposure", "team_changer_share"
    ]]
    context = context.merge(game_context, on=["outcome_season", "game_id"], validate="one_to_one")
    context["source_era"] = np.where(
        context["outcome_season"].lt(2018), "pre_tracking", "tracking"
    )
    context["exposure_group"] = np.where(
        context["mean_prior_exposure"].lt(2000), "low_exposure", "established"
    )
    context["team_change_group"] = np.where(
        context["team_changer_share"].gt(0.2), "high_team_change", "low_team_change"
    )
    subgroup_rows = []
    for dimension in ("source_era", "exposure_group", "team_change_group"):
        for value, group in context.groupby(dimension, sort=True):
            season_delta = group.groupby("outcome_season").apply(
                lambda part: float((part["pulse"] - part["rapm"]).mean()),
                include_groups=False,
            )
            subgroup_rows.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "games": int(len(group)),
                    "seasons": int(group["outcome_season"].nunique()),
                    "equal_season_mse_delta_pulse_minus_rapm": float(season_delta.mean()),
                }
            )
    subgroups = pd.DataFrame(subgroup_rows)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    pd.concat(validation_priors, ignore_index=True).to_parquet(
        output / "validation_priors.parquet", index=False
    )
    games.to_parquet(output / "validation_games.parquet", index=False)
    folds.to_parquet(output / "validation_folds.parquet", index=False)
    summary.to_parquet(output / "validation_summary.parquet", index=False)
    subgroups.to_parquet(output / "validation_subgroups.parquet", index=False)
    pd.DataFrame(coverage_rows).to_parquet(output / "coverage.parquet", index=False)
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "release_candidate",
        "source_hashes": hashes,
        "config": config,
        "final_prior": final_quality,
        "identity_errors": errors,
        "validation": {
            "rating_seasons": [2014, 2025],
            "outcome_seasons": [2015, 2026],
            "summary": summary.to_dict("records"),
            "paired_bootstrap": bootstrap,
            "subgroups": subgroups.to_dict("records"),
        },
        "artifacts": {
            "ratings": "ratings.parquet",
            "final_priors": "final_priors.parquet",
            "validation_priors": "validation_priors.parquet",
            "validation_games": "validation_games.parquet",
            "validation_folds": "validation_folds.parquet",
            "validation_summary": "validation_summary.parquet",
            "validation_subgroups": "validation_subgroups.parquet",
            "coverage": "coverage.parquet",
        },
    }
    write_json_atomic(run, output / "run.json")
    return run
