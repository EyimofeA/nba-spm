"""Transparent annual player-skill features from public tracking aggregates."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


PLAYER_SKILL_FEATURES = (
    "shot_difficulty_expected_points_per_attempt",
    "shot_difficulty_expected_points_per_attempt_relative",
    "shot_making_points_above_expected_p100_eb",
    "tight_shot_attempt_share_eb",
    "pass_creation_points_per_potential_assist_eb",
    "high_value_assist_share_eb",
    "bad_pass_turnovers_per_100_passes_eb",
    "screen_assist_points_p100_eb",
    "deflections_p100_eb",
    "charges_drawn_p100_eb",
    "defensive_boxouts_p100_eb",
    "loose_balls_recovered_p100_eb",
)
PLAYER_SKILL_MODEL_FEATURES = tuple(
    feature
    for feature in PLAYER_SKILL_FEATURES
    if feature != "shot_difficulty_expected_points_per_attempt"
)


def _numeric(frame: pd.DataFrame, columns: set[str], source: str) -> pd.DataFrame:
    if missing := sorted(columns - set(frame.columns)):
        raise ValueError(f"{source} is missing {missing}.")
    output = frame.copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _shrunken_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    season: pd.Series,
    *,
    prior_exposure: float,
    scale: float = 1.0,
) -> pd.Series:
    """Return an exposure-weighted rate shrunk to its same-season league rate."""
    work = pd.DataFrame(
        {"numerator": numerator, "denominator": denominator, "Season": season}
    )
    totals = work.groupby("Season", dropna=False).agg(
        numerator=("numerator", "sum"), denominator=("denominator", "sum")
    )
    centers = totals["numerator"] / totals["denominator"].where(
        totals["denominator"].gt(0)
    )
    center = work["Season"].map(centers)
    result = scale * (numerator.fillna(0) + prior_exposure * center) / (
        denominator.fillna(0).clip(lower=0) + prior_exposure
    )
    return result.where(numerator.notna() & denominator.notna())


def _prepare_shooting(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "PLAYER_ID", "year", "shot_type", "2FGM", "2FGA", "3PM", "3PA", "FGM", "FGA"
    }
    frame = _numeric(source, required - {"shot_type"}, "Shooting source").rename(
        columns={"year": "Season"}
    )
    frame["shot_type"] = frame["shot_type"].astype(str).str.lower()
    frame = frame.dropna(subset=["PLAYER_ID", "Season"])
    frame[["PLAYER_ID", "Season"]] = frame[["PLAYER_ID", "Season"]].astype(int)
    if frame.duplicated(["PLAYER_ID", "Season", "shot_type"]).any():
        raise ValueError("Shooting source has duplicate player-season-shot-type rows.")

    for points, makes, attempts in ((2.0, "2FGM", "2FGA"), (3.0, "3PM", "3PA")):
        group = frame.groupby(["Season", "shot_type"], dropna=False)
        total_makes = group[makes].transform("sum")
        total_attempts = group[attempts].transform("sum")
        other_attempts = total_attempts - frame[attempts].fillna(0)
        expected_accuracy = (total_makes - frame[makes].fillna(0)) / other_attempts.where(
            other_attempts.gt(0)
        )
        frame[f"expected_{attempts}"] = frame[attempts].fillna(0) * expected_accuracy
        frame[f"actual_points_{attempts}"] = points * frame[makes].fillna(0)
        frame[f"expected_points_{attempts}"] = points * frame[f"expected_{attempts}"]

    frame["tight_fga"] = frame["FGA"].where(
        frame["shot_type"].isin({"tight", "very_tight"}), 0.0
    )
    annual = frame.groupby(["PLAYER_ID", "Season"], as_index=False).agg(
        shot_attempts=("FGA", "sum"),
        tight_fga=("tight_fga", "sum"),
        actual_points_2=("actual_points_2FGA", "sum"),
        expected_points_2=("expected_points_2FGA", "sum"),
        actual_points_3=("actual_points_3PA", "sum"),
        expected_points_3=("expected_points_3PA", "sum"),
    )
    annual["expected_points"] = annual["expected_points_2"] + annual["expected_points_3"]
    annual["points_above_expected"] = (
        annual["actual_points_2"] + annual["actual_points_3"] - annual["expected_points"]
    )
    annual["shot_difficulty_expected_points_per_attempt"] = annual[
        "expected_points"
    ] / annual["shot_attempts"].where(annual["shot_attempts"].gt(0))
    league = annual.groupby("Season").agg(
        expected_points=("expected_points", "sum"),
        shot_attempts=("shot_attempts", "sum"),
    )
    league_expected_points = league["expected_points"] / league["shot_attempts"].where(
        league["shot_attempts"].gt(0)
    )
    annual["shot_difficulty_expected_points_per_attempt_relative"] = (
        annual["shot_difficulty_expected_points_per_attempt"]
        - annual["Season"].map(league_expected_points)
    )
    annual["tight_shot_attempt_share_eb"] = _shrunken_ratio(
        annual["tight_fga"], annual["shot_attempts"], annual["Season"],
        prior_exposure=100.0,
    )
    return annual


def _prepare_passing(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "nba_id", "year", "OffPoss", "POTENTIAL_AST", "AST_PTS_CREATED",
        "High Value Assist %", "BadPassTurnovers", "Passes",
    }
    frame = _numeric(source, required, "Passing source").rename(
        columns={"nba_id": "PLAYER_ID", "year": "Season"}
    )
    frame = frame.dropna(subset=["PLAYER_ID", "Season"])
    frame[["PLAYER_ID", "Season"]] = frame[["PLAYER_ID", "Season"]].astype(int)
    if frame.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Passing source has duplicate player-season rows.")
    high_value = frame["High Value Assist %"].where(
        frame["High Value Assist %"].le(1.0), frame["High Value Assist %"] / 100.0
    )
    frame["high_value_assist_count"] = high_value * frame["POTENTIAL_AST"]
    frame["pass_creation_points_per_potential_assist_eb"] = _shrunken_ratio(
        frame["AST_PTS_CREATED"], frame["POTENTIAL_AST"], frame["Season"],
        prior_exposure=100.0,
    )
    frame["high_value_assist_share_eb"] = _shrunken_ratio(
        frame["high_value_assist_count"], frame["POTENTIAL_AST"], frame["Season"],
        prior_exposure=100.0,
    )
    frame["bad_pass_turnovers_per_100_passes_eb"] = _shrunken_ratio(
        frame["BadPassTurnovers"], frame["Passes"], frame["Season"],
        prior_exposure=250.0, scale=100.0,
    )
    return frame


def _prepare_hustle(source: pd.DataFrame) -> pd.DataFrame:
    required = {
        "PLAYER_ID", "year", "POSS", "SCREEN_AST_PTS", "DEFLECTIONS",
        "CHARGES_DRAWN", "DEF_BOXOUTS", "LOOSE_BALLS_RECOVERED",
    }
    frame = _numeric(source, required, "Hustle source").rename(
        columns={"year": "Season"}
    )
    frame = frame.dropna(subset=["PLAYER_ID", "Season"])
    frame[["PLAYER_ID", "Season"]] = frame[["PLAYER_ID", "Season"]].astype(int)
    if frame.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Hustle source has duplicate player-season rows.")
    for output, numerator in {
        "screen_assist_points_p100_eb": "SCREEN_AST_PTS",
        "deflections_p100_eb": "DEFLECTIONS",
        "charges_drawn_p100_eb": "CHARGES_DRAWN",
        "defensive_boxouts_p100_eb": "DEF_BOXOUTS",
        "loose_balls_recovered_p100_eb": "LOOSE_BALLS_RECOVERED",
    }.items():
        frame[output] = _shrunken_ratio(
            frame[numerator], frame["POSS"], frame["Season"],
            prior_exposure=500.0, scale=100.0,
        )
    return frame


def compute_player_skill_features(
    shooting: pd.DataFrame,
    passing: pd.DataFrame,
    hustle: pd.DataFrame,
    shotzone: pd.DataFrame,
) -> pd.DataFrame:
    """Build annual skill features while preserving missing source coverage."""
    shots = _prepare_shooting(shooting)
    passes = _prepare_passing(passing)
    work = _prepare_hustle(hustle)
    zone_required = {"EntityId", "year", "OffPoss"}
    zones = _numeric(shotzone, zone_required, "Shot-zone source").rename(
        columns={"EntityId": "PLAYER_ID", "year": "Season", "OffPoss": "shot_off_poss"}
    )
    zones = zones.dropna(subset=["PLAYER_ID", "Season"])
    zones[["PLAYER_ID", "Season"]] = zones[["PLAYER_ID", "Season"]].astype(int)
    if zones.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Shot-zone source has duplicate player-season rows.")

    keys = pd.concat(
        [
            shots[["PLAYER_ID", "Season"]],
            passes[["PLAYER_ID", "Season"]],
            work[["PLAYER_ID", "Season"]],
            zones[["PLAYER_ID", "Season"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    output = keys.merge(
        shots, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one"
    ).merge(
        zones[["PLAYER_ID", "Season", "shot_off_poss"]],
        on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
    ).merge(
        passes, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
        suffixes=("", "_passing"),
    ).merge(
        work, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one",
        suffixes=("", "_hustle"),
    )
    raw_poe_p100 = 100.0 * output["points_above_expected"] / output[
        "shot_off_poss"
    ].where(output["shot_off_poss"].gt(0))
    reliability = output["shot_attempts"] / (output["shot_attempts"] + 200.0)
    output["shot_making_points_above_expected_p100_eb"] = reliability * raw_poe_p100
    output["has_shooting_tracking"] = output["shot_attempts"].notna()
    output["has_passing_tracking"] = output["POTENTIAL_AST"].notna()
    output["has_hustle_tracking"] = output["POSS"].notna()
    selected = [
        "PLAYER_ID", "Season", *PLAYER_SKILL_FEATURES,
        "has_shooting_tracking", "has_passing_tracking", "has_hustle_tracking",
    ]
    result = output[selected].sort_values(["Season", "PLAYER_ID"]).reset_index(drop=True)
    if result.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Player-skill output keys are not unique.")
    numeric = result[list(PLAYER_SKILL_FEATURES)]
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Player-skill output contains infinite values.")
    return result


def profile_player_skill_features(features: pd.DataFrame) -> pd.DataFrame:
    """Summarize annual coverage and robust year-over-year location drift."""
    rows = []
    for feature in PLAYER_SKILL_FEATURES:
        pooled = features[feature].dropna()
        pooled_iqr = float(pooled.quantile(0.75) - pooled.quantile(0.25))
        prior_median = None
        for season, group in features.groupby("Season", sort=True):
            values = group[feature].dropna()
            median = float(values.median()) if not values.empty else float("nan")
            shift = (
                (median - prior_median) / pooled_iqr
                if prior_median is not None and pooled_iqr > 0 and np.isfinite(median)
                else float("nan")
            )
            rows.append(
                {
                    "Season": int(season), "feature": feature,
                    "rows": int(len(group)), "observed_rows": int(len(values)),
                    "missing_fraction": float(group[feature].isna().mean()),
                    "p10": float(values.quantile(0.10)) if not values.empty else float("nan"),
                    "median": median,
                    "p90": float(values.quantile(0.90)) if not values.empty else float("nan"),
                    "year_over_year_median_shift_iqr": shift,
                }
            )
            if np.isfinite(median):
                prior_median = median
    return pd.DataFrame(rows)


def build_player_skill_features(
    shooting_path: str | Path,
    passing_path: str | Path,
    hustle_path: str | Path,
    shotzone_path: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2014, 2025)),
) -> dict:
    sources = {
        "shooting": Path(shooting_path), "passing": Path(passing_path),
        "hustle": Path(hustle_path), "shotzone": Path(shotzone_path),
    }
    frames = {name: pd.read_csv(path, low_memory=False) for name, path in sources.items()}
    for name, frame in frames.items():
        frames[name] = frame.loc[pd.to_numeric(frame["year"], errors="coerce").isin(seasons)]
    features = compute_player_skill_features(
        frames["shooting"], frames["passing"], frames["hustle"], frames["shotzone"]
    )
    config = {
        "seasons": list(seasons),
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
        "builder_sha256": sha256_file(Path(__file__)),
        "shooting_prior_attempts": 200.0,
        "ratio_prior_opportunities": 100.0,
        "pass_prior_attempts": 250.0,
        "hustle_prior_possessions": 500.0,
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"player_skill_features_v1_{identity}"
    output = Path(artifact_root) / "features" / "player_skill" / run_id
    output.mkdir(parents=True, exist_ok=False)
    path = output / "features.parquet"
    features.to_parquet(path, index=False)
    profile = profile_player_skill_features(features)
    profile_path = output / "season_profile.parquet"
    profile.to_parquet(profile_path, index=False)
    coverage = {
        column: int(features[column].sum())
        for column in ("has_shooting_tracking", "has_passing_tracking", "has_hustle_tracking")
    }
    missing = {
        feature: float(features[feature].isna().mean()) for feature in PLAYER_SKILL_FEATURES
    }
    run = {
        "run_id": run_id,
        "dataset": "annual_player_skill_features_v1",
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "rows": int(len(features)),
            "players": int(features["PLAYER_ID"].nunique()),
            "seasons": sorted(features["Season"].unique().astype(int).tolist()),
            "duplicate_keys": 0,
            "infinite_values": 0,
            "coverage_rows": coverage,
            "missing_fraction": missing,
            "max_absolute_year_over_year_median_shift_iqr": float(
                profile["year_over_year_median_shift_iqr"].abs().max()
            ),
        },
        "feature_names": list(PLAYER_SKILL_FEATURES),
        "model_candidate_feature_names": list(PLAYER_SKILL_MODEL_FEATURES),
        "features_path": str(path.resolve()),
        "season_profile_path": str(profile_path.resolve()),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "Features are descriptive tracking skills and roles, not causal player credit.",
            "Shot expectations use leave-one-player-out season and defender-distance bucket averages.",
            "Hustle coverage begins in 2018; missing seasons remain missing rather than becoming zero.",
            "Source coverage varies by season; inspect season_profile.parquet before model use.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
