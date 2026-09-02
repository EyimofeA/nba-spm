"""Frozen CourtSignal PULSE prior and one-season lineup update."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit
from nba_impact.models.prior_informed_rapm import build_prior_center
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficient_center_path,
    load_unified_terminal_possessions,
)


def _rating_table(design, beta: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    n = len(design.players)
    X = design.X[mask]
    result = pd.DataFrame(
        {
            "PLAYER_ID": design.players.astype(int),
            "offense": 100.0 * beta[:n],
            "defense": -100.0 * beta[n : 2 * n],
            "Poss_Off": np.asarray(X[:, :n].sum(axis=0)).ravel(),
            "Poss_Def": np.asarray(X[:, n : 2 * n].sum(axis=0)).ravel(),
        }
    )
    result["net"] = result["offense"] + result["defense"]
    return result


def _fit_final_prior(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict]:
    target = targets.loc[
        targets["horizon"].eq(9) & targets["target_variant"].eq("normal")
    ].copy()
    panel = features.merge(
        target,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    models = {}
    for side in ("offense", "defense"):
        alpha = float(config["prior"][f"{side}_alpha"])
        models[side] = _fit(
            panel,
            BOX_PIPM_STYLE_FEATURES,
            f"target_{side}",
            alpha,
        )
    prior = features[["PLAYER_ID", "Window_End"]].copy()
    for side in ("offense", "defense"):
        prior[f"prior_{side}_per_100"] = models[side].predict(
            features.loc[:, BOX_PIPM_STYLE_FEATURES]
        )
    prior["prior_net_per_100"] = (
        prior["prior_offense_per_100"] + prior["prior_defense_per_100"]
    )
    return prior, {
        "training_rows": len(panel),
        "training_start": int(panel["Window_End"].min()),
        "training_end": int(panel["Window_End"].max()),
        "prediction_rows": len(prior),
    }


def _fit_season(
    season: int,
    priors: pd.DataFrame,
    config: dict,
    *,
    possession_cache: Path,
    silver_possessions: Path,
    silver_lineups: Path,
    legacy_aliases: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    frame = load_unified_terminal_possessions(
        possession_cache,
        silver_possessions,
        silver_lineups,
        (season,),
        transition_season=2024,
        game_types=("regular",),
    )
    if season < 2024:
        aliases = legacy_aliases.loc[legacy_aliases["season"].eq(season)]
        alias_map = dict(
            zip(
                aliases["source_player_id"].astype(int),
                aliases["canonical_player_id"].astype(int),
                strict=True,
            )
        )
        for column in (*AWAY_PLAYER_COLUMNS, *HOME_PLAYER_COLUMNS):
            frame[column] = frame[column].replace(alias_map)
    design = build_design(frame, include_home=True)
    mask = np.ones(len(design.y), dtype=bool)
    rapm = config["lineup_update"]
    rapm_config = RapmConfig(
        seasons=(season,),
        lambda_off=float(rapm["lambda_off"]),
        lambda_def=float(rapm["lambda_def"]),
        lambda_home=float(rapm["lambda_home"]),
        data_scope="pulse_single_season_terminal_lineup",
    )
    zero = np.zeros(design.X.shape[1], dtype=float)
    zero_beta, _ = fit_coefficient_center_path(
        design, rapm_config, zero, center_scales=(0.0,), row_mask=mask
    )[0.0]
    center, coverage = build_prior_center(
        design,
        priors,
        prior_window_end=season,
        train_mask=mask,
        test_mask=mask,
    )
    posterior, _ = fit_coefficient_center_path(
        design,
        rapm_config,
        center,
        center_scales=(float(rapm["prior_scale"]),),
        row_mask=mask,
    )[float(rapm["prior_scale"])]
    centered_prior = _rating_table(design, center, mask).rename(
        columns={
            "offense": "pulse_prior_offense",
            "defense": "pulse_prior_defense",
            "net": "pulse_prior_net",
        }
    )
    pulse = _rating_table(design, posterior, mask).rename(
        columns={
            "offense": "pulse_offense",
            "defense": "pulse_defense",
            "net": "pulse_net",
        }
    )
    rapm_rating = _rating_table(design, zero_beta, mask).rename(
        columns={
            "offense": "rapm_offense",
            "defense": "rapm_defense",
            "net": "rapm_net",
        }
    )
    result = centered_prior.merge(
        pulse.drop(columns=["Poss_Off", "Poss_Def"]),
        on="PLAYER_ID",
        validate="one_to_one",
    ).merge(
        rapm_rating.drop(columns=["Poss_Off", "Poss_Def"]),
        on="PLAYER_ID",
        validate="one_to_one",
    )
    for component in ("offense", "defense", "net"):
        result[f"lineup_update_{component}"] = (
            result[f"pulse_{component}"] - result[f"pulse_prior_{component}"]
        )
    result["Season"] = int(season)
    result["evidence_status"] = (
        "final_mapping_backcast" if season < 2006 else "final_descriptive_refit"
    )
    result["estimand_id"] = config["estimand_id"]
    ordered = [
        "PLAYER_ID",
        "Season",
        "Poss_Off",
        "Poss_Def",
        "pulse_prior_offense",
        "pulse_prior_defense",
        "pulse_prior_net",
        "lineup_update_offense",
        "lineup_update_defense",
        "lineup_update_net",
        "pulse_offense",
        "pulse_defense",
        "pulse_net",
        "rapm_offense",
        "rapm_defense",
        "rapm_net",
        "evidence_status",
        "estimand_id",
    ]
    return result[ordered], coverage


def _identity_errors(ratings: pd.DataFrame) -> dict[str, float]:
    return {
        "prior_side_identity": float(
            (
                ratings["pulse_prior_offense"]
                + ratings["pulse_prior_defense"]
                - ratings["pulse_prior_net"]
            )
            .abs()
            .max()
        ),
        "update_side_identity": float(
            (
                ratings["lineup_update_offense"]
                + ratings["lineup_update_defense"]
                - ratings["lineup_update_net"]
            )
            .abs()
            .max()
        ),
        "pulse_sum_identity": float(
            (
                ratings["pulse_prior_net"]
                + ratings["lineup_update_net"]
                - ratings["pulse_net"]
            )
            .abs()
            .max()
        ),
        "pulse_side_identity": float(
            (ratings["pulse_offense"] + ratings["pulse_defense"] - ratings["pulse_net"])
            .abs()
            .max()
        ),
        "rapm_side_identity": float(
            (ratings["rapm_offense"] + ratings["rapm_defense"] - ratings["rapm_net"])
            .abs()
            .max()
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
    current["PLAYER_ID"] = pd.to_numeric(current["PLAYER_ID"], errors="raise").astype(
        int
    )
    old["PLAYER_ID"] = pd.to_numeric(old["PLAYER_ID"], errors="raise").astype(int)
    middle["PLAYER_ID"] = pd.to_numeric(middle["PLAYER_ID"], errors="raise").astype(int)
    names = pd.concat([old, middle, current], ignore_index=True)
    names = names.drop_duplicates("PLAYER_ID", keep="last")
    if names["PLAYER_ID"].duplicated().any():
        raise ValueError("PULSE player names are not unique.")
    return names


def build_pulse_release(
    config_path: str | Path,
    *,
    features_path: str | Path,
    targets_path: str | Path,
    validation_run: str | Path,
    historical_player_sheets: str | Path,
    gabriel_player_sheets: str | Path,
    legacy_player_aliases: str | Path,
    player_games_path: str | Path,
    possession_cache: str | Path,
    silver_possessions: str | Path,
    silver_lineups: str | Path,
    artifact_root: str | Path,
) -> dict:
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text())
    features_path = Path(features_path)
    targets_path = Path(targets_path)
    validation_run = Path(validation_run)
    sources = {
        "config": config_path,
        "features": features_path,
        "targets": targets_path,
        "validation_priors": validation_run / "priors.parquet",
        "validation_ratings": validation_run / "ratings.parquet",
        "validation_games": validation_run / "game_predictions.parquet",
        "player_games": Path(player_games_path),
        "legacy_player_aliases": Path(legacy_player_aliases),
        "silver_possessions": Path(silver_possessions),
        "silver_lineups": Path(silver_lineups),
        "builder": Path(__file__),
        **{
            f"historical_names_{season}": Path(historical_player_sheets)
            / f"{season}.csv"
            for season in range(1997, 2014)
        },
        **{
            f"gabriel_names_{season}": Path(gabriel_player_sheets) / f"{season}.parquet"
            for season in range(2012, 2027)
        },
    }
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[
        :10
    ]
    output = Path(artifact_root) / "models" / "pulse" / f"pulse_v1_{identity}"
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())
    output.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(targets_path)
    legacy_aliases = pd.read_csv(legacy_player_aliases)
    priors, prior_quality = _fit_final_prior(features, targets, config)
    season_rows = []
    coverage_rows = []
    checkpoint = output / "checkpoints"
    checkpoint.mkdir(exist_ok=True)
    for season in range(1997, 2027):
        path = checkpoint / f"ratings_{season}.parquet"
        quality_path = checkpoint / f"coverage_{season}.json"
        if path.exists() and quality_path.exists():
            season_rows.append(pd.read_parquet(path))
            coverage_rows.append(json.loads(quality_path.read_text()))
            continue
        ratings, coverage = _fit_season(
            season,
            priors,
            config,
            possession_cache=Path(possession_cache),
            silver_possessions=Path(silver_possessions),
            silver_lineups=Path(silver_lineups),
            legacy_aliases=legacy_aliases,
        )
        ratings.to_parquet(path, index=False)
        coverage_row = {"Season": season, **coverage}
        write_json_atomic(coverage_row, quality_path)
        season_rows.append(ratings)
        coverage_rows.append(coverage_row)
        print(f"PULSE {season}: complete", flush=True)
    ratings = pd.concat(season_rows, ignore_index=True)
    names = _load_names(
        Path(historical_player_sheets),
        Path(gabriel_player_sheets),
        Path(player_games_path),
    )
    ratings = ratings.merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    if ratings["PLAYER_NAME"].isna().any():
        raise ValueError("PULSE release contains unnamed players.")
    identity_errors = _identity_errors(ratings)
    if max(identity_errors.values()) > 1e-9:
        raise ValueError(f"PULSE component identity failed: {identity_errors}")

    validation_priors = pd.read_parquet(validation_run / "priors.parquet")
    validation_priors = validation_priors.loc[
        validation_priors["candidate"].eq("box15_9y_normal")
    ]
    validation_ratings = pd.read_parquet(validation_run / "ratings.parquet")
    validation_ratings = validation_ratings.loc[
        validation_ratings["candidate"].isin(
            ["box15_9y_normal", "box15_9y_normal_aio", "zero_prior_rapm"]
        )
    ]
    validation_games = pd.read_parquet(validation_run / "game_predictions.parquet")
    validation_games = validation_games.loc[
        validation_games["candidate"].isin(
            ["box15_9y_normal", "box15_9y_normal_aio", "zero_prior_rapm"]
        )
    ]
    ratings.to_parquet(output / "ratings.parquet", index=False)
    priors.to_parquet(output / "uncentered_final_priors.parquet", index=False)
    pd.DataFrame(coverage_rows).to_parquet(output / "coverage.parquet", index=False)
    validation_priors.to_parquet(output / "validation_priors.parquet", index=False)
    validation_ratings.to_parquet(output / "validation_ratings.parquet", index=False)
    validation_games.to_parquet(output / "validation_games.parquet", index=False)
    run = {
        "run_id": output.name,
        "model_id": config["model_id"],
        "status": config["status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand_id": config["estimand_id"],
        "config": config,
        "source_hashes": hashes,
        "quality": {
            "rating_rows": len(ratings),
            "seasons": sorted(ratings["Season"].astype(int).unique().tolist()),
            "players": int(ratings["PLAYER_ID"].nunique()),
            "duplicate_keys": int(ratings.duplicated(["PLAYER_ID", "Season"]).sum()),
            "missing_names": int(ratings["PLAYER_NAME"].isna().sum()),
            "minimum_prior_off_possession_coverage": float(
                pd.DataFrame(coverage_rows)["train_off_possession_coverage"].min()
            ),
            "minimum_prior_def_possession_coverage": float(
                pd.DataFrame(coverage_rows)["train_def_possession_coverage"].min()
            ),
            "identity_errors": identity_errors,
            "prior_fit": prior_quality,
        },
        "relative_paths": {
            "ratings": "ratings.parquet",
            "coverage": "coverage.parquet",
            "validation_priors": "validation_priors.parquet",
            "validation_ratings": "validation_ratings.parquet",
            "validation_games": "validation_games.parquet",
        },
        "forbidden_interpretation": config["interpretation"]["forbidden"],
    }
    write_json_atomic(run, output / "run.json")
    return run
