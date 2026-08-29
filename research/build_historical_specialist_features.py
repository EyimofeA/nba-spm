#!/usr/bin/env python3
"""Build annual and five-year shooting/OREB specialist feature panels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.historical_factor_features import (
    REBOUND_FEATURES,
    SHOT_FEATURES,
    build_rebound_responsibility_features,
    build_shot_context_features,
)
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.matchup_defense_features import _read_source
from nba_impact.models.historical_factor_targets import load_gabriel_events


ROOT = Path(__file__).resolve().parents[1]
EVENT_ROOT = ROOT / "data/lake/bronze/gabriel_merged_playbyplay/old_data"
ASSIGNMENT_ROOT = ROOT / "data/lake/bronze/gabriel_matchup_fga/scraped_data"
PLAYER_SHEETS = ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
MATCHUP_CONFIG = ROOT / "configs/models/matchup_elo_v1_sources.json"
BASE_ANNUAL = (
    ROOT
    / "artifacts/research/complete_feature_coverage"
    / "semantically_complete_spm_features_v1_8be676bd0f/annual_features.parquet"
)
OUTPUT_ROOT = ROOT / "artifacts/research/historical_specialist_features"
SEASONS = tuple(range(2014, 2027))
WINDOW_ENDS = tuple(range(2018, 2027))
ASSIGNMENT_SEASONS = tuple(range(2018, 2027))
ZERO_CENTERED = {
    "shotmaking_above_expected_p100_eb",
    "rim_shotmaking_above_expected_p100_eb",
    "defender_shotmaking_points_saved_p100_eb",
    "rim_deterrence_vs_scorer_p100_eb",
    "rebound_conversion_above_expected_eb",
    "height_x_defensive_boxouts",
    "height_x_dreb_contested_share",
}
SHOT_DEFENSE_FEATURES = {
    "defender_expected_points_conceded_per_shot",
    "defender_shotmaking_points_saved_p100_eb",
    "defender_rim_expected_points_conceded_per_shot",
    "rim_deterrence_vs_scorer_p100_eb",
    "has_assigned_shot_defense",
}
BOXOUT_FEATURES = {
    "defensive_boxouts_p100_specialist",
    "offensive_boxouts_p100_specialist",
    "boxout_team_rebound_conversion_eb",
    "boxout_player_rebound_conversion_eb",
    "height_x_defensive_boxouts",
    "height_x_offensive_boxouts",
    "has_boxout_tracking",
}


def _weighted_feature(group: pd.DataFrame, feature: str) -> float:
    if feature in SHOT_DEFENSE_FEATURES:
        weight = group["shot_context_def_exposure"]
    elif feature in SHOT_FEATURES:
        weight = group["shot_context_off_exposure"]
    else:
        weight = group["rebound_context_exposure"]
    if feature in BOXOUT_FEATURES:
        weight = weight * group["has_boxout_tracking"]
    valid = pd.to_numeric(group[feature], errors="coerce").notna() & weight.gt(0)
    if not valid.any():
        return 0.0 if feature in ZERO_CENTERED or feature.startswith("has_") else np.nan
    return float(np.average(group.loc[valid, feature], weights=weight.loc[valid]))


def _pool_five_year(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = (*SHOT_FEATURES, *REBOUND_FEATURES)
    for window_end in WINDOW_ENDS:
        window = annual.loc[annual["Season"].between(window_end - 4, window_end)].copy()
        for player_id, group in window.groupby("PLAYER_ID", sort=False):
            record = {
                "PLAYER_ID": int(player_id),
                "Window_Start": window_end - 4,
                "Window_End": window_end,
                "shot_assignment_exposure_fraction": float(
                    group["shot_context_def_exposure"].sum()
                    / max(1.0, group["DefPoss"].sum())
                ),
                "boxout_source_exposure_fraction": float(
                    np.average(
                        group["has_boxout_tracking"],
                        weights=group["rebound_context_exposure"].clip(lower=0),
                    )
                    if group["rebound_context_exposure"].sum() > 0
                    else 0.0
                ),
            }
            record.update({feature: _weighted_feature(group, feature) for feature in features})
            rows.append(record)
    output = pd.DataFrame(rows)
    for feature in (*SHOT_FEATURES, *REBOUND_FEATURES):
        median = output.groupby("Window_End")[feature].transform("median")
        output[feature] = output[feature].fillna(median).fillna(0.0)
    return output


def main() -> None:
    if max(SEASONS) >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    matchup_sources = {
        int(season): ROOT / path
        for season, path in json.loads(MATCHUP_CONFIG.read_text()).items()
    }
    fallback_2026 = (
        ROOT
        / "data/lake/bronze/shufinskiy_nba_data/revision=e829d46"
        / "matchups/season=2026/matchups_2025.tar.xz"
    )
    if not matchup_sources[2026].exists() and fallback_2026.exists():
        matchup_sources[2026] = fallback_2026
    config = {
        "experiment_id": "historical_specialist_features_v1",
        "seasons": list(SEASONS),
        "five_year_window_ends": list(WINDOW_ENDS),
        "shot_assignment_seasons": list(ASSIGNMENT_SEASONS),
        "shot_quality_prior_attempts": 100.0,
        "shotmaking_prior_attempts": 500.0,
        "rim_deterrence_prior_possessions": 500.0,
        "rebound_conversion_prior_chances": 100.0,
        "boxout_prior_events": 50.0,
        "source_revisions": {
            "merged_playbyplay": "6e077a0f62153e72db300ba1f0a45b30584fd3d2",
            "matchup_fga": "d539693f4db9a39788873b644c35423673fe6efe",
        },
        "source_hashes": {
            "builder": sha256_file(Path(__file__)),
            "feature_code": sha256_file(
                ROOT / "src/nba_impact/data/historical_factor_features.py"
            ),
            "base_annual": sha256_file(BASE_ANNUAL),
            "matchup_config": sha256_file(MATCHUP_CONFIG),
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"historical_specialist_features_v1_{identity}"
    if (output / "run.json").exists():
        print(output)
        return

    base = pd.read_parquet(BASE_ANNUAL).rename(columns={"Window_End": "Season"})
    annual_rows = []
    quality_rows = []
    for season in SEASONS:
        events, event_quality = load_gabriel_events(EVENT_ROOT, season)
        assignment_path = ASSIGNMENT_ROOT / f"{season}_dfgtotal.csv"
        assignments = (
            pd.read_csv(assignment_path, low_memory=False)
            if season in ASSIGNMENT_SEASONS
            else None
        )
        matchup_pairs = None
        if season in ASSIGNMENT_SEASONS:
            matchup_pairs, _ = _read_source(matchup_sources[season])
        shot, shot_quality = build_shot_context_features(
            events, assignments, matchup_pairs, season
        )
        sheet = pd.read_parquet(PLAYER_SHEETS / f"{season}.parquet")
        rebound, rebound_quality = build_rebound_responsibility_features(sheet, season)
        keys = base.loc[base["Season"].eq(season), ["PLAYER_ID", "Season", "OffPoss", "DefPoss"]]
        combined = (
            keys.merge(shot, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one")
            .merge(rebound, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one")
        )
        for exposure in (
            "shot_context_off_exposure",
            "shot_context_def_exposure",
            "rebound_context_exposure",
        ):
            combined[exposure] = combined[exposure].fillna(0.0)
        for flag in ("has_assigned_shot_defense", "has_boxout_tracking"):
            combined[flag] = combined[flag].fillna(0.0)
        for feature in (*SHOT_FEATURES, *REBOUND_FEATURES):
            if feature in ZERO_CENTERED:
                combined[feature] = combined[feature].fillna(0.0)
            else:
                combined[feature] = combined[feature].fillna(combined[feature].median()).fillna(0.0)
        annual_rows.append(combined)
        quality_rows.append(
            {
                "Season": season,
                **event_quality,
                **{f"shot_{key}": value for key, value in shot_quality.items()},
                **{f"rebound_{key}": value for key, value in rebound_quality.items()},
            }
        )
    annual = pd.concat(annual_rows, ignore_index=True)
    five_year = _pool_five_year(annual)
    quality = pd.DataFrame(quality_rows)
    if annual[list((*SHOT_FEATURES, *REBOUND_FEATURES))].isna().any().any():
        raise ValueError("Annual specialist features contain null values.")
    if five_year[list((*SHOT_FEATURES, *REBOUND_FEATURES))].isna().any().any():
        raise ValueError("Five-year specialist features contain null values.")

    output.mkdir(parents=True, exist_ok=False)
    annual.to_parquet(output / "annual_features.parquet", index=False)
    five_year.to_parquet(output / "five_year_features.parquet", index=False)
    quality.to_parquet(output / "source_quality.parquet", index=False)
    write_json_atomic(
        {
            "run_id": output.name,
            "status": "research_features",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "quality": {
                "annual_rows": int(len(annual)),
                "five_year_rows": int(len(five_year)),
                "annual_seasons": list(SEASONS),
                "five_year_window_ends": list(WINDOW_ENDS),
                "season_2027_loaded": False,
            },
            "limitations": {
                "shot_clock": "Unavailable in the pinned event source; game clock is not used as a substitute.",
                "defender_distance": "Unavailable in the pinned assignment source.",
                "shot_assignment": "NBA video tags can name multiple defenders; each tag receives equal fractional weight.",
                "boxout_responsibility": "Observed player box-out totals exist, but no event-level unique box-out assignment exists.",
            },
            "paths": {
                "annual_features": "annual_features.parquet",
                "five_year_features": "five_year_features.parquet",
                "source_quality": "source_quality.parquet",
            },
        },
        output / "run.json",
    )
    print(output)


if __name__ == "__main__":
    main()
