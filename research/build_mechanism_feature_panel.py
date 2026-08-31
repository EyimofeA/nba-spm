#!/usr/bin/env python3
"""Build annual and rolling five-year mechanism features."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.mechanism_features import (
    DEFENSE_MECHANISM_FEATURES,
    MECHANISM_FEATURES,
    OFFENSE_MECHANISM_FEATURES,
    compute_mechanism_features,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "mechanism_feature_panel_v1"
SOURCE_RUN = ROOT / (
    "artifacts/research/complete_feature_coverage/"
    "semantically_complete_spm_features_v1_fdee01ec4e"
)


def _rolling_panel(
    annual_source: pd.DataFrame,
    annual_features: pd.DataFrame,
    canonical_keys: pd.DataFrame,
) -> pd.DataFrame:
    annual = annual_source[["PLAYER_ID", "Window_End", "OffPoss", "DefPoss"]].merge(
        annual_features,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    rows = []
    for window_end, keys in canonical_keys.groupby("Window_End", sort=True):
        window = annual.loc[annual["Window_End"].between(window_end - 4, window_end)]
        for player_id, group in window.groupby("PLAYER_ID", sort=False):
            row = {"PLAYER_ID": int(player_id), "Window_End": int(window_end)}
            for side, features, exposure in (
                ("offense", OFFENSE_MECHANISM_FEATURES, "OffPoss"),
                ("defense", DEFENSE_MECHANISM_FEATURES, "DefPoss"),
            ):
                del side
                weights = group[exposure].to_numpy(dtype=float).clip(min=0.0)
                for feature in features:
                    values = group[feature].to_numpy(dtype=float)
                    row[feature] = (
                        float(np.average(values, weights=weights))
                        if weights.sum() > 0
                        else 0.0
                    )
            rows.append(row)
    rolling = pd.DataFrame(rows)
    output = canonical_keys.merge(
        rolling,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    output[list(MECHANISM_FEATURES)] = output[list(MECHANISM_FEATURES)].fillna(0.0)
    return output


def main() -> None:
    annual_path = SOURCE_RUN / "annual_features.parquet"
    five_year_path = SOURCE_RUN / "five_year_features.parquet"
    annual_source = pd.read_parquet(annual_path)
    annual = compute_mechanism_features(annual_source)
    canonical_keys = pd.read_parquet(
        five_year_path, columns=["PLAYER_ID", "Window_End"]
    )
    five_year = _rolling_panel(annual_source, annual, canonical_keys)

    if annual[list(MECHANISM_FEATURES)].isna().any().any():
        raise ValueError("Annual mechanism panel contains missing values.")
    if five_year[list(MECHANISM_FEATURES)].isna().any().any():
        raise ValueError("Five-year mechanism panel contains missing values.")
    if annual["Window_End"].max() > 2026 or five_year["Window_End"].max() > 2026:
        raise ValueError("An unavailable future season entered the mechanism panel.")

    ranges = []
    for panel_name, frame in (("annual", annual), ("five_year", five_year)):
        for feature in MECHANISM_FEATURES:
            values = frame[feature]
            ranges.append(
                {
                    "panel": panel_name,
                    "feature": feature,
                    "minimum": float(values.min()),
                    "p01": float(values.quantile(0.01)),
                    "median": float(values.median()),
                    "p99": float(values.quantile(0.99)),
                    "maximum": float(values.max()),
                    "nonzero_fraction": float(values.ne(0).mean()),
                }
            )
    range_audit = pd.DataFrame(ranges)
    registry = pd.DataFrame(
        [
            {
                "feature": feature,
                "side": "offense" if feature in OFFENSE_MECHANISM_FEATURES else "defense",
                "family": {
                    "pass_value_per_potential_assist_eb": "passing",
                    "load_adjusted_shot_quality_residual": "shot_quality",
                    "load_adjusted_creation_residual": "creation",
                    "spacing_creation_interaction": "spacing_creation",
                    "dreb_conversion_above_expected_eb": "rebound_responsibility",
                    "foul_adjusted_activity_residual": "defensive_activity",
                    "workload_adjusted_shot_suppression_residual": "shot_defense",
                    "rim_protection_workload_value": "rim_protection",
                }[feature],
                "information_cutoff": "same_season_only",
                "stabilization": (
                    "same_season_empirical_bayes"
                    if feature == "pass_value_per_potential_assist_eb"
                    else "same_season_leave_one_player_out_residual_and_exposure_shrinkage"
                    if "residual" in feature or "above_expected" in feature
                    else "stabilized_input_interaction"
                ),
                "rapm_target_used": False,
                "status": "research_candidate",
            }
            for feature in MECHANISM_FEATURES
        ]
    )

    source_hashes = {
        "annual_features": sha256_file(annual_path),
        "five_year_features": sha256_file(five_year_path),
        "source_manifest": sha256_file(SOURCE_RUN / "run.json"),
        "builder": sha256_file(Path(__file__)),
        "feature_code": sha256_file(
            ROOT / "src/nba_impact/data/mechanism_features.py"
        ),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "source_run_id": SOURCE_RUN.name,
        "annual_seasons": sorted(annual["Window_End"].astype(int).unique().tolist()),
        "five_year_window_ends": sorted(
            five_year["Window_End"].astype(int).unique().tolist()
        ),
        "offense_features": list(OFFENSE_MECHANISM_FEATURES),
        "defense_features": list(DEFENSE_MECHANISM_FEATURES),
        "source_hashes": source_hashes,
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / (
        "artifacts/research/mechanism_feature_panel/"
        f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "annual_features.parquet": annual,
        "five_year_features.parquet": five_year,
        "feature_registry.parquet": registry,
        "range_audit.parquet": range_audit,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_features_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "annual_rows": len(annual),
            "five_year_rows": len(five_year),
            "annual_missing_values": 0,
            "five_year_missing_values": 0,
            "duplicate_annual_keys": int(
                annual.duplicated(["PLAYER_ID", "Window_End"]).sum()
            ),
            "duplicate_five_year_keys": int(
                five_year.duplicated(["PLAYER_ID", "Window_End"]).sum()
            ),
            "rapm_target_used": False,
        },
        "files": {},
        "forbidden_interpretation": (
            "These engineered features are research candidates, not player impact, "
            "causal credit, or evidence of model improvement."
        ),
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(range_audit.to_string(index=False))


if __name__ == "__main__":
    main()
