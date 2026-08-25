"""Decomposed annual ratings from SPM-centered one-season normal RAPM."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.prior_informed_rapm import PRIOR_COLUMNS, build_prior_center
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficient_center_path,
    load_current_player_names,
    load_current_possessions,
    load_legacy_possessions,
    load_unified_terminal_possessions,
    ratings_table,
)


RATING_COMPONENTS = ("offense", "defense", "net")


def fit_annual_aio_season(
    frame: pd.DataFrame,
    priors: pd.DataFrame,
    config: RapmConfig,
    *,
    season: int,
) -> tuple[pd.DataFrame, dict]:
    """Fit zero-center and full-SPM-center RAPM on one complete season."""
    design = build_design(frame, include_home=config.include_home)
    row_mask = np.ones(len(design.y), dtype=bool)
    center, coverage = build_prior_center(
        design,
        priors,
        prior_window_end=season,
        train_mask=row_mask,
        test_mask=row_mask,
    )
    fitted = fit_coefficient_center_path(
        design,
        config,
        center,
        center_scales=(0.0, 1.0),
        row_mask=row_mask,
    )
    normal_beta, normal_intercept = fitted[0.0]
    aio_beta, aio_intercept = fitted[1.0]

    normal = ratings_table(design, normal_beta).rename(
        columns={
            "player_id": "PLAYER_ID",
            "offense_per_100": "normal_rapm_offense",
            "defense_per_100": "normal_rapm_defense",
            "net_per_100": "normal_rapm_net",
            "off_possessions": "Poss_Off",
            "def_possessions": "Poss_Def",
        }
    )
    normal = normal.drop(columns=["uncertainty_status"])
    aio = ratings_table(design, aio_beta).rename(
        columns={
            "player_id": "PLAYER_ID",
            "offense_per_100": "aio_offense",
            "defense_per_100": "aio_defense",
            "net_per_100": "aio_net",
        }
    )
    aio = aio[["PLAYER_ID", "aio_offense", "aio_defense", "aio_net"]]
    ratings = normal.merge(aio, on="PLAYER_ID", validate="one_to_one")

    n_players = len(design.players)
    centered = pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "spm_center_offense": center[:n_players] * 100.0,
            "spm_center_defense": -center[n_players : 2 * n_players] * 100.0,
        }
    )
    centered["spm_center_net"] = (
        centered["spm_center_offense"] + centered["spm_center_defense"]
    )
    ratings = ratings.merge(centered, on="PLAYER_ID", validate="one_to_one")

    raw = priors.loc[priors["Window_End"].eq(season), ["PLAYER_ID", *PRIOR_COLUMNS]].copy()
    raw = raw.rename(
        columns={
            "prior_offense_per_100": "spm_raw_offense",
            "prior_defense_per_100": "spm_raw_defense",
            "prior_net_per_100": "spm_raw_net",
        }
    )
    ratings = ratings.merge(raw, on="PLAYER_ID", how="left", validate="one_to_one")
    ratings["prior_available"] = ratings["spm_raw_offense"].notna()
    for component in RATING_COMPONENTS:
        ratings[f"rapm_update_{component}"] = (
            ratings[f"aio_{component}"] - ratings[f"spm_center_{component}"]
        )
    ratings["Season"] = season
    ratings["aio_net_rank"] = ratings["aio_net"].rank(
        method="min", ascending=False
    ).astype(int)

    quality = {
        "season": season,
        "players": len(ratings),
        "games": int(frame["gameid"].nunique()),
        "possession_rows": len(frame),
        "players_with_prior": int(ratings["prior_available"].sum()),
        "player_prior_coverage": float(ratings["prior_available"].mean()),
        "lineup_slot_prior_coverage": coverage["test_lineup_slot_coverage"],
        "normal_intercept_per_possession": float(normal_intercept),
        "aio_intercept_per_possession": float(aio_intercept),
        "max_component_identity_error": float(
            max(
                np.abs(ratings["aio_net"] - ratings["aio_offense"] - ratings["aio_defense"]).max(),
                np.abs(
                    ratings["aio_net"]
                    - ratings["spm_center_net"]
                    - ratings["rapm_update_net"]
                ).max(),
            )
        ),
    }
    return ratings.sort_values("aio_net", ascending=False), quality


def build_annual_aio_ratings(
    cache_dir: str | Path,
    priors_path: str | Path,
    names_path: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2017, 2025)),
    lambda_off: float = 3000.0,
    lambda_def: float = 3000.0,
    lambda_home: float = 300.0,
) -> dict:
    """Build a versioned, decomposed annual AIO rating panel."""
    if not seasons or tuple(sorted(set(seasons))) != seasons:
        raise ValueError("seasons must be unique and increasing.")
    priors = pd.read_parquet(priors_path)
    prior_manifest_path = Path(priors_path).parent / "run.json"
    prior_manifest = (
        json.loads(prior_manifest_path.read_text())
        if prior_manifest_path.exists()
        else {}
    )
    prior_training_rule = prior_manifest.get("config", {}).get(
        "training_rule", "unknown"
    )
    required = {"PLAYER_ID", "Window_End", *PRIOR_COLUMNS}
    if missing := sorted(required - set(priors.columns)):
        raise ValueError(f"Annual AIO priors are missing {missing}.")
    if priors.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Annual AIO prior keys must be unique.")
    names = pd.read_csv(names_path)[["PLAYER_ID", "PLAYER_NAME"]]
    if names["PLAYER_ID"].duplicated().any():
        raise ValueError("Annual AIO player-name IDs must be unique.")

    rating_frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    source_hashes: dict[str, str] = {}
    for season in seasons:
        frame = load_legacy_possessions(cache_dir, (season,), game_types=("regular",))
        config = RapmConfig(
            seasons=(season,),
            lambda_off=lambda_off,
            lambda_def=lambda_def,
            lambda_home=lambda_home,
            game_types=("regular",),
            data_scope="legacy_annual_aio_ratings",
        )
        ratings, quality = fit_annual_aio_season(
            frame, priors, config, season=season
        )
        ratings = ratings.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")
        rating_frames.append(ratings)
        quality_rows.append(quality)
        for path in frame.attrs.get("source_paths", []):
            source_hashes[str(path)] = sha256_file(path)

    panel = pd.concat(rating_frames, ignore_index=True)
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual AIO rating keys must be unique.")
    numeric = [
        column
        for column in panel.columns
        if column not in {"PLAYER_NAME"} and panel[column].dtype.kind in "fiu"
    ]
    if not np.isfinite(panel[numeric].to_numpy()).all():
        raise ValueError("Annual AIO numeric outputs must be finite.")

    run_id = f"annual_aio_ratings_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "annual_aio_ratings" / run_id
    output.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(output / "ratings.parquet", index=False)
    pd.DataFrame(quality_rows).to_parquet(output / "season_quality.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "annual_spm_centered_normal_rapm",
        "estimand": "single_regular_season_lineup_adjusted_points_per_100",
        "status": "research_leaderboard",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seasons": list(seasons),
            "lambda_off": lambda_off,
            "lambda_def": lambda_def,
            "lambda_home": lambda_home,
            "prior_center_scale": 1.0,
            "prior_run_id": prior_manifest.get("run_id"),
            "prior_training_rule": prior_training_rule,
            "lineup_policy": "legacy possession terminal lineup",
            "source_hashes": {
                "priors": sha256_file(priors_path),
                "names": sha256_file(names_path),
                "source_code": sha256_file(Path(__file__)),
                "possessions": source_hashes,
            },
        },
        "quality": {
            "rows": len(panel),
            "players": int(panel["PLAYER_ID"].nunique()),
            "duplicate_keys": 0,
            "missing_names": int(panel["PLAYER_NAME"].isna().sum()),
            "minimum_lineup_slot_prior_coverage": float(
                min(row["lineup_slot_prior_coverage"] for row in quality_rows)
            ),
            "maximum_component_identity_error": float(
                max(row["max_component_identity_error"] for row in quality_rows)
            ),
        },
        "metrics": {"season_quality": quality_rows},
        "ratings_path": str((output / "ratings.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "These ratings use complete-season features and possessions; they are descriptive, not preseason forecasts.",
            f"The SPM prior training rule is: {prior_training_rule}.",
            "The feature family and 2022-2024 seasons are already inspected and are not untouched promotion evidence.",
            "Legacy possessions stop after 2024 and the 2024 regular-season cache has 1,229 games.",
            "Uncertainty is not estimated in this version.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run


def build_current_annual_aio_ratings(
    possessions_path: str | Path,
    segments_path: str | Path,
    priors_path: str | Path,
    legacy_names_path: str | Path,
    player_games_path: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = (2024, 2025, 2026),
    lambda_off: float = 3000.0,
    lambda_def: float = 3000.0,
    lambda_home: float = 300.0,
) -> dict:
    """Build research AIO ratings on canonical current possession inputs.

    This adapter keeps canonical current possessions separate from the legacy
    cache. It is deliberately research-only until the unified timeline's
    source transition receives a separate release review.
    """
    requested = tuple(sorted({int(season) for season in seasons}))
    if not requested:
        raise ValueError("At least one current AIO season is required.")
    priors = pd.read_parquet(priors_path)
    required = {"PLAYER_ID", "Window_End", *PRIOR_COLUMNS}
    if missing := sorted(required - set(priors.columns)):
        raise ValueError(f"Current annual AIO priors are missing {missing}.")
    if priors.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Current annual AIO prior keys must be unique.")

    frame = load_current_possessions(
        possessions_path,
        segments_path,
        lineup_policy="terminal",
        game_types=("regular",),
    )
    available = {int(value) for value in frame["season"].unique()}
    missing_seasons = sorted(set(requested) - available)
    if missing_seasons:
        raise ValueError(
            f"Requested canonical current AIO seasons are unavailable: {missing_seasons}."
        )
    names = load_current_player_names(legacy_names_path, player_games_path)
    name_column = "player_name" if "player_name" in names.columns else "PLAYER_NAME"
    names = names.rename(columns={name_column: "PLAYER_NAME"})[["PLAYER_ID", "PLAYER_NAME"]]
    if names["PLAYER_ID"].duplicated().any():
        raise ValueError("Current AIO player-name IDs must be unique.")

    rating_frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    for season in requested:
        season_frame = frame.loc[frame["season"].eq(season)].copy()
        config = RapmConfig(
            seasons=(season,),
            lambda_off=lambda_off,
            lambda_def=lambda_def,
            lambda_home=lambda_home,
            game_types=("regular",),
            data_scope="canonical_current_annual_aio_terminal_lineups",
        )
        ratings, quality = fit_annual_aio_season(
            season_frame, priors, config, season=season
        )
        ratings = ratings.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")
        rating_frames.append(ratings)
        quality_rows.append(quality)

    panel = pd.concat(rating_frames, ignore_index=True)
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Current annual AIO rating keys must be unique.")
    numeric = [column for column in panel.columns if panel[column].dtype.kind in "fiu"]
    if not np.isfinite(panel[numeric].to_numpy()).all():
        raise ValueError("Current annual AIO numeric outputs must be finite.")
    component_error = np.abs(panel["aio_net"] - panel["aio_offense"] - panel["aio_defense"])
    if float(component_error.max()) > 1e-10:
        raise AssertionError("Current annual AIO does not satisfy net = offense + defense.")

    prior_manifest_path = Path(priors_path).parent / "run.json"
    prior_manifest = (
        json.loads(prior_manifest_path.read_text()) if prior_manifest_path.exists() else {}
    )
    run_id = f"current_annual_aio_ratings_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "current_annual_aio_ratings" / run_id
    output.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(output / "ratings.parquet", index=False)
    pd.DataFrame(quality_rows).to_parquet(output / "season_quality.parquet", index=False)
    source_hashes = {
        "possessions": sha256_file(possessions_path),
        "possession_lineup_segments": sha256_file(segments_path),
        "player_games": sha256_file(player_games_path),
        "priors": sha256_file(priors_path),
    }
    legacy_names = Path(legacy_names_path)
    if legacy_names.exists():
        source_hashes["legacy_player_names"] = sha256_file(legacy_names)
    run = {
        "run_id": run_id,
        "model_family": "canonical_current_annual_spm_centered_normal_rapm",
        "estimand": "single_regular_season_lineup_adjusted_points_per_100",
        "status": "research_current_aio_not_for_public_release",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seasons": list(requested),
            "lambda_off": lambda_off,
            "lambda_def": lambda_def,
            "lambda_home": lambda_home,
            "prior_center_scale": 1.0,
            "prior_run_id": prior_manifest.get("run_id"),
            "prior_training_rule": prior_manifest.get("config", {}).get("training_rule"),
            "lineup_policy": "terminal",
            "game_types": ["regular"],
            "source_hashes": source_hashes,
        },
        "quality": {
            "rows": int(len(panel)),
            "players": int(panel["PLAYER_ID"].nunique()),
            "duplicate_keys": 0,
            "missing_names": int(panel["PLAYER_NAME"].isna().sum()),
            "minimum_lineup_slot_prior_coverage": float(
                min(row["lineup_slot_prior_coverage"] for row in quality_rows)
            ),
            "maximum_component_identity_error": float(component_error.max()),
        },
        "metrics": {"season_quality": quality_rows},
        "ratings_path": str((output / "ratings.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "Research-only current AIO: do not expose it as public ratings until SPM completeness and validation pass.",
            "The prior is retrospective leave-one-season-out SPM, not a forecast.",
            "Games that fail canonical lineup-quality gates are excluded.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run


def build_unified_annual_aio_ratings(
    cache_dir: str | Path,
    possessions_path: str | Path,
    segments_path: str | Path,
    priors_path: str | Path,
    legacy_names_path: str | Path,
    player_games_path: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2014, 2027)),
    transition_season: int = 2024,
    lambda_off: float = 3000.0,
    lambda_def: float = 3000.0,
    lambda_home: float = 300.0,
) -> dict:
    """Fit the complete annual AIO timeline on one provenance-preserving adapter."""
    requested = tuple(sorted({int(season) for season in seasons}))
    if not requested:
        raise ValueError("At least one unified AIO season is required.")
    priors = pd.read_parquet(priors_path)
    required = {"PLAYER_ID", "Window_End", *PRIOR_COLUMNS}
    if missing := sorted(required - set(priors.columns)):
        raise ValueError(f"Unified annual AIO priors are missing {missing}.")
    if priors.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Unified annual AIO prior keys must be unique.")

    frame = load_unified_terminal_possessions(
        cache_dir,
        possessions_path,
        segments_path,
        requested,
        transition_season=transition_season,
        game_types=("regular",),
    )
    names = load_current_player_names(legacy_names_path, player_games_path)
    name_column = "player_name" if "player_name" in names.columns else "PLAYER_NAME"
    names = names.rename(columns={name_column: "PLAYER_NAME"})[["PLAYER_ID", "PLAYER_NAME"]]

    rating_frames: list[pd.DataFrame] = []
    quality_rows: list[dict] = []
    for season in requested:
        season_frame = frame.loc[frame["season"].eq(season)].copy()
        config = RapmConfig(
            seasons=(season,),
            lambda_off=lambda_off,
            lambda_def=lambda_def,
            lambda_home=lambda_home,
            game_types=("regular",),
            data_scope="unified_annual_aio_terminal_lineups",
        )
        ratings, quality = fit_annual_aio_season(season_frame, priors, config, season=season)
        ratings = ratings.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")
        quality["rapm_input_source"] = frame.loc[
            frame["season"].eq(season), "rapm_input_source"
        ].iloc[0]
        rating_frames.append(ratings)
        quality_rows.append(quality)

    panel = pd.concat(rating_frames, ignore_index=True)
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Unified annual AIO rating keys must be unique.")
    numeric = [column for column in panel.columns if panel[column].dtype.kind in "fiu"]
    if not np.isfinite(panel[numeric].to_numpy()).all():
        raise ValueError("Unified annual AIO numeric outputs must be finite.")
    component_error = np.abs(panel["aio_net"] - panel["aio_offense"] - panel["aio_defense"])
    if float(component_error.max()) > 1e-10:
        raise AssertionError("Unified annual AIO does not satisfy net = offense + defense.")

    prior_manifest_path = Path(priors_path).parent / "run.json"
    prior_manifest = json.loads(prior_manifest_path.read_text()) if prior_manifest_path.exists() else {}
    run_id = f"unified_annual_aio_ratings_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "unified_annual_aio_ratings" / run_id
    output.mkdir(parents=True, exist_ok=False)
    panel.to_parquet(output / "ratings.parquet", index=False)
    pd.DataFrame(quality_rows).to_parquet(output / "season_quality.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "unified_annual_spm_centered_normal_rapm",
        "estimand": "single_regular_season_lineup_adjusted_points_per_100",
        "status": "research_unified_timeline_not_for_public_release",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seasons": list(requested),
            "transition_season": transition_season,
            "lambda_off": lambda_off,
            "lambda_def": lambda_def,
            "lambda_home": lambda_home,
            "prior_center_scale": 1.0,
            "prior_run_id": prior_manifest.get("run_id"),
            "prior_training_rule": prior_manifest.get("config", {}).get("training_rule"),
            "lineup_policy": "terminal",
            "game_types": ["regular"],
            "source_by_season": frame.attrs["source_by_season"],
            "source_hashes": {
                "cache_dir": str(Path(cache_dir).resolve()),
                "possessions": sha256_file(possessions_path),
                "possession_lineup_segments": sha256_file(segments_path),
                "player_games": sha256_file(player_games_path),
                "priors": sha256_file(priors_path),
            },
        },
        "quality": {
            "rows": int(len(panel)),
            "players": int(panel["PLAYER_ID"].nunique()),
            "duplicate_keys": 0,
            "missing_names": int(panel["PLAYER_NAME"].isna().sum()),
            "minimum_lineup_slot_prior_coverage": float(min(row["lineup_slot_prior_coverage"] for row in quality_rows)),
            "maximum_component_identity_error": float(component_error.max()),
        },
        "metrics": {"season_quality": quality_rows},
        "ratings_path": str((output / "ratings.parquet").resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "The source transition is explicit: legacy terminal cache before 2024 and canonical event terminal lineups from 2024 onward.",
            "The 2024 target overlap is an engineering compatibility check, not a proof that sources are interchangeable.",
            "The prior is retrospective leave-one-season-out SPM, not a forecast.",
            "This unified research artifact is not a public-release endorsement; the legacy-to-canonical source transition needs separate review.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
