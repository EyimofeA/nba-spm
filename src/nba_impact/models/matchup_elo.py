"""Regularized Elo-scale offense and defense ratings from matchup assignments.

The NBA matchup feed gives an offensive player, a listed defender, assigned
partial possessions, and points.  It does *not* give event-level guarding
responsibility.  This module therefore estimates descriptive matchup rates,
not primary-defender or causal defensive value.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.matchup_defense_features import RAW_COLUMNS, _read_source


BASE_ELO = 1500.0
ELO_PER_LOG_RATE = 400.0 / np.log(10.0)


def _validate_matchups(frame: pd.DataFrame) -> pd.DataFrame:
    """Return valid positive-exposure matchup rows or fail closed."""
    required = {
        "game_id",
        "person_id",
        "matchups_person_id",
        "partial_possessions",
        "player_points",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Matchup source is missing columns: {missing}.")
    work = frame.copy()
    for column in set(RAW_COLUMNS).intersection(work.columns):
        if column != "game_id":
            work[column] = pd.to_numeric(work[column], errors="coerce")
    identity = ["game_id", "person_id", "matchups_person_id"]
    if work[identity + ["partial_possessions", "player_points"]].isna().any().any():
        raise ValueError(
            "Matchup source has null matchup identity, exposure, or points."
        )
    if work.duplicated(identity).any():
        raise ValueError("Matchup source has duplicate game/scorer/defender rows.")
    if (work["partial_possessions"] < 0).any() or (work["player_points"] < 0).any():
        raise ValueError("Matchup source has negative exposure or points.")
    return work.loc[
        work["partial_possessions"] > 0,
        identity + ["partial_possessions", "player_points"],
    ].copy()


def fit_matchup_elo(
    frame: pd.DataFrame,
    season: int,
    *,
    ridge_penalty: float = 500.0,
    smoothing_possessions: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """Fit one season of Elo-scale scorer and defender matchup ratings.

    The fitted log-rate equation is ``log(PPP_ij / league_PPP) = o_i - d_j``.
    ``o_i`` is offensive matchup strength and ``d_j`` is defensive matchup
    strength.  Both are exposure-weight centered after the ridge fit.  Scores
    are then shown on an Elo scale where a 400-point difference implies a
    tenfold *modelled* points-rate ratio.  That scaling is a display convention;
    this is not sequential game-win Elo.
    """
    work = _validate_matchups(frame)
    work["source_season"] = int(season)
    work["source_weight"] = 1.0
    return _fit_matchup_elo_weighted(
        work,
        rating_season=season,
        ridge_penalty=ridge_penalty,
        smoothing_possessions=smoothing_possessions,
    )


def _fit_matchup_elo_weighted(
    work: pd.DataFrame,
    *,
    rating_season: int,
    ridge_penalty: float,
    smoothing_possessions: float,
) -> tuple[pd.DataFrame, dict]:
    """Fit the two-way log-rate model with predeclared source-season weights."""
    if ridge_penalty <= 0 or smoothing_possessions < 0:
        raise ValueError(
            "ridge_penalty must be positive and smoothing_possessions nonnegative."
        )
    required = {"source_season", "source_weight"}
    if missing := sorted(required - set(work.columns)):
        raise ValueError(f"Weighted matchup Elo is missing {missing}.")
    work = work.copy()
    work["source_season"] = pd.to_numeric(work["source_season"], errors="raise").astype(
        int
    )
    work["source_weight"] = pd.to_numeric(work["source_weight"], errors="raise")
    if (work["source_weight"] <= 0).any() or not np.isfinite(
        work["source_weight"]
    ).all():
        raise ValueError(
            "Weighted matchup Elo source weights must be finite and positive."
        )
    scorer_ids = np.sort(work["person_id"].astype(int).unique())
    defender_ids = np.sort(work["matchups_person_id"].astype(int).unique())
    player_ids = np.union1d(scorer_ids, defender_ids)
    player_index = {int(player): index for index, player in enumerate(player_ids)}
    n_players = len(player_ids)
    scorer_index = work["person_id"].astype(int).map(player_index).to_numpy()
    defender_index = work["matchups_person_id"].astype(int).map(player_index).to_numpy()
    possessions = work["partial_possessions"].to_numpy(dtype=float)
    weights = possessions * work["source_weight"].to_numpy(dtype=float)
    points = work["player_points"].to_numpy(dtype=float)
    season_totals = work.groupby("source_season", sort=True).agg(
        points=("player_points", "sum"), possessions=("partial_possessions", "sum")
    )
    season_ppp = season_totals["points"] / season_totals["possessions"]
    if not np.isfinite(season_ppp.to_numpy()).all() or (season_ppp <= 0).any():
        raise ValueError("Matchup source has no positive league scoring rate.")
    row_league_ppp = work["source_season"].map(season_ppp).to_numpy(dtype=float)
    league_ppp = float(np.average(row_league_ppp, weights=weights))

    # A small prior prevents zero-point, tiny matchup rows from dominating the
    # log transform.  Exposure remains the fitting weight.
    target = np.log(
        (points + smoothing_possessions * row_league_ppp)
        / (possessions + smoothing_possessions)
    ) - np.log(row_league_ppp)
    weighted_mean = float(np.average(target, weights=weights))
    centered_target = target - weighted_mean
    row_index = np.arange(len(work))
    design = sparse.coo_matrix(
        (
            np.concatenate([np.ones(len(work)), -np.ones(len(work))]),
            (
                np.concatenate([row_index, row_index]),
                np.concatenate([scorer_index, defender_index + n_players]),
            ),
        ),
        shape=(len(work), 2 * n_players),
    ).tocsr()
    weighted_design = design.multiply(weights[:, None])
    normal = (design.T @ weighted_design).tocsc()
    normal += sparse.eye(2 * n_players, format="csc") * ridge_penalty
    rhs = design.T @ (weights * centered_target)
    coefficients = np.asarray(spsolve(normal, rhs)).ravel()
    if not np.isfinite(coefficients).all():
        raise ValueError("Matchup Elo fit produced non-finite coefficients.")

    scorer_exposure = np.bincount(scorer_index, weights=weights, minlength=n_players)
    defender_exposure = np.bincount(
        defender_index, weights=weights, minlength=n_players
    )
    offense = coefficients[:n_players]
    defense = coefficients[n_players:]
    offense_center = float(np.average(offense, weights=scorer_exposure))
    defense_center = float(np.average(defense, weights=defender_exposure))
    offense = offense - offense_center
    defense = defense - defense_center
    intercept_log_ppp = (
        np.log(league_ppp) + weighted_mean + offense_center - defense_center
    )

    output = pd.DataFrame(
        {
            "PLAYER_ID": player_ids.astype(int),
            "Season": int(rating_season),
            "offense_matchup_possessions": scorer_exposure,
            "defense_matchup_possessions": defender_exposure,
            "offense_elo": BASE_ELO + ELO_PER_LOG_RATE * offense,
            "defense_elo": BASE_ELO + ELO_PER_LOG_RATE * defense,
            "net_elo": ELO_PER_LOG_RATE * (offense + defense),
        }
    )
    output["expected_points_per_matchup_possession"] = np.exp(
        intercept_log_ppp + offense - defense
    )
    if output[["offense_elo", "defense_elo", "net_elo"]].isna().any().any():
        raise ValueError("Matchup Elo output contains null scores.")
    audit = {
        "season": int(rating_season),
        "source_rows": int(len(work)),
        "positive_exposure_rows": int(len(work)),
        "games": int(work["game_id"].nunique()),
        "players": int(n_players),
        "league_points_per_matchup_possession": league_ppp,
        "intercept_points_per_matchup_possession": float(np.exp(intercept_log_ppp)),
        "ridge_penalty": ridge_penalty,
        "smoothing_possessions": smoothing_possessions,
        "weighted_offense_log_rate_mean": float(
            np.average(offense, weights=scorer_exposure)
        ),
        "weighted_defense_log_rate_mean": float(
            np.average(defense, weights=defender_exposure)
        ),
        "source_seasons": [int(value) for value in season_totals.index],
        "source_season_league_ppp": {
            str(int(key)): float(value) for key, value in season_ppp.items()
        },
        "effective_matchup_possessions": float(weights.sum()),
    }
    return output, audit


def fit_time_decayed_matchup_elo(
    source_frames: Mapping[int, pd.DataFrame],
    rating_season: int,
    *,
    window_seasons: int = 3,
    time_decay: float = 0.70,
    ridge_penalty: float = 500.0,
    smoothing_possessions: float = 3.0,
) -> tuple[pd.DataFrame, dict]:
    """Fit a trailing fixed-width matchup model, with exponentially decayed years.

    The output for ``rating_season`` uses only the inclusive interval
    ``rating_season-window_seasons+1`` through ``rating_season``.  Each source
    season has its own league scoring baseline before the scorer/defender fit,
    so a league-wide scoring shift does not masquerade as player change.
    """
    if window_seasons < 2:
        raise ValueError("Time-decayed matchup Elo needs at least two source seasons.")
    if not 0 < time_decay <= 1:
        raise ValueError("Time-decayed matchup Elo decay must be in (0, 1].")
    required = tuple(range(rating_season - window_seasons + 1, rating_season + 1))
    missing = [season for season in required if season not in source_frames]
    if missing:
        raise ValueError(
            f"Time-decayed matchup Elo for {rating_season} is missing required seasons {missing}."
        )
    weighted_frames = []
    for source_season in required:
        work = _validate_matchups(source_frames[source_season])
        work["source_season"] = source_season
        work["source_weight"] = time_decay ** (rating_season - source_season)
        weighted_frames.append(work)
    ratings, audit = _fit_matchup_elo_weighted(
        pd.concat(weighted_frames, ignore_index=True),
        rating_season=rating_season,
        ridge_penalty=ridge_penalty,
        smoothing_possessions=smoothing_possessions,
    )
    ratings["window_start_season"] = required[0]
    ratings["window_end_season"] = required[-1]
    ratings["time_decay"] = time_decay
    audit["window_start_season"] = required[0]
    audit["window_end_season"] = required[-1]
    audit["window_seasons"] = window_seasons
    audit["time_decay"] = time_decay
    return ratings, audit


def build_matchup_elo(
    *,
    source_overrides: Mapping[int, str | Path],
    artifact_root: str | Path,
    ridge_penalty: float = 500.0,
    smoothing_possessions: float = 3.0,
) -> dict:
    """Fit annual matchup Elo-scale ratings from pinned raw matchup sources."""
    if not source_overrides:
        raise ValueError("At least one season-to-source mapping is required.")
    outputs: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    season_quality: dict[str, dict] = {}
    for season, source in sorted(source_overrides.items()):
        source_path = Path(source)
        frame, manifest = _read_source(source_path)
        if int(manifest.get("season", season)) != int(season):
            raise ValueError(
                f"{source_path}: manifest season does not match requested {season}."
            )
        ratings, quality = fit_matchup_elo(
            frame,
            int(season),
            ridge_penalty=ridge_penalty,
            smoothing_possessions=smoothing_possessions,
        )
        outputs.append(ratings)
        season_quality[str(season)] = quality
        source_hashes[str(source_path.resolve())] = sha256_file(source_path)
    panel = pd.concat(outputs, ignore_index=True)
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Matchup Elo panel has duplicate player-season rows.")
    config = {
        "seasons": sorted(int(season) for season in source_overrides),
        "source_hashes": source_hashes,
        "ridge_penalty": ridge_penalty,
        "smoothing_possessions": smoothing_possessions,
        "elo_base": BASE_ELO,
        "elo_per_log_rate": ELO_PER_LOG_RATE,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[
        :10
    ]
    output_dir = (
        Path(artifact_root) / "models" / "matchup_elo" / f"matchup_elo_v1_{identity}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    ratings_path = output_dir / "ratings.parquet"
    panel.to_parquet(ratings_path, index=False)
    run = {
        "run_id": output_dir.name,
        "model_family": "matchup_elo_scale_two_way_log_rate_ridge",
        "status": "research_descriptive",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": (
            "Regular-season, season-level descriptive scorer and listed-defender matchup "
            "rate association. It is not primary-defender, causal defense, or RAPM."
        ),
        "equation": "log(PPP_scorer,defender / league_PPP) = offense_scorer - defense_defender",
        "config": config,
        "quality": {
            "rows": int(len(panel)),
            "players": int(panel["PLAYER_ID"].nunique()),
            "duplicate_player_seasons": int(
                panel.duplicated(["PLAYER_ID", "Season"]).sum()
            ),
            "season_quality": season_quality,
        },
        "ratings_path": str(ratings_path.resolve()),
        "artifact_path": str(output_dir.resolve()),
        "caveat": (
            "NBA matchup assignments are aggregated scorer-defender exposure, not exact event-level "
            "guarding assignments. Scores are Elo-scale log-rate parameters, not sequential game Elo."
        ),
    }
    write_json_atomic(run, output_dir / "run.json")
    return run


def build_time_decayed_matchup_elo(
    *,
    source_overrides: Mapping[int, str | Path],
    artifact_root: str | Path,
    window_seasons: int = 3,
    time_decay: float = 0.70,
    ridge_penalty: float = 500.0,
    smoothing_possessions: float = 3.0,
) -> dict:
    """Build trailing multi-year, time-decayed matchup ratings.

    This is a stability challenger to the annual descriptive model. It does not
    establish a new causal defensive estimand and is not an AIO feature.
    """
    if not source_overrides:
        raise ValueError("At least one season-to-source mapping is required.")
    frames: dict[int, pd.DataFrame] = {}
    source_hashes: dict[str, str] = {}
    for season, source in sorted(source_overrides.items()):
        source_path = Path(source)
        frame, manifest = _read_source(source_path)
        if int(manifest.get("season", season)) != int(season):
            raise ValueError(
                f"{source_path}: manifest season does not match requested {season}."
            )
        frames[int(season)] = frame
        source_hashes[str(source_path.resolve())] = sha256_file(source_path)
    source_seasons = tuple(sorted(frames))
    eligible_ends = tuple(
        season
        for season in source_seasons
        if all(
            candidate in frames
            for candidate in range(season - window_seasons + 1, season + 1)
        )
    )
    if not eligible_ends:
        raise ValueError("No complete consecutive matchup windows are available.")
    outputs = []
    quality: dict[str, dict] = {}
    for rating_season in eligible_ends:
        ratings, audit = fit_time_decayed_matchup_elo(
            frames,
            rating_season,
            window_seasons=window_seasons,
            time_decay=time_decay,
            ridge_penalty=ridge_penalty,
            smoothing_possessions=smoothing_possessions,
        )
        outputs.append(ratings)
        quality[str(rating_season)] = audit
    panel = pd.concat(outputs, ignore_index=True)
    if panel.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError(
            "Time-decayed matchup Elo panel has duplicate player-season rows."
        )
    config = {
        "source_seasons": list(source_seasons),
        "rating_seasons": list(eligible_ends),
        "source_hashes": source_hashes,
        "window_seasons": window_seasons,
        "time_decay": time_decay,
        "ridge_penalty": ridge_penalty,
        "smoothing_possessions": smoothing_possessions,
        "elo_base": BASE_ELO,
        "elo_per_log_rate": ELO_PER_LOG_RATE,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[
        :10
    ]
    output_dir = (
        Path(artifact_root)
        / "models"
        / "matchup_elo"
        / f"matchup_elo_time_decay_v1_{identity}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    ratings_path = output_dir / "ratings.parquet"
    panel.to_parquet(ratings_path, index=False)
    run = {
        "run_id": output_dir.name,
        "model_family": "time_decayed_three_season_two_way_matchup_log_rate_ridge",
        "status": "research_descriptive",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": (
            "Trailing multi-season descriptive scorer and listed-defender matchup rate association. "
            "It is not primary-defender, causal defense, or RAPM."
        ),
        "equation": (
            "log(PPP_scorer,defender,season / league_PPP_season) = offense_scorer - defense_defender; "
            "row weight = partial_possessions * time_decay^(rating_season-source_season)"
        ),
        "config": config,
        "quality": {
            "rows": int(len(panel)),
            "players": int(panel["PLAYER_ID"].nunique()),
            "duplicate_player_seasons": int(
                panel.duplicated(["PLAYER_ID", "Season"]).sum()
            ),
            "season_quality": quality,
        },
        "ratings_path": str(ratings_path.resolve()),
        "artifact_path": str(output_dir.resolve()),
        "caveat": (
            "The source only supplies aggregated scorer-listed-defender assignments. "
            "Time decay stabilizes the rate fit; it does not solve help, scheme, assignment, or causal-attribution bias."
        ),
    }
    write_json_atomic(run, output_dir / "run.json")
    return run
