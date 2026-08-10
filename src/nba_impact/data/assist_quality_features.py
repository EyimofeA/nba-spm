"""Annual free-throw-adjusted assist quality features."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

ASSIST_QUALITY_FEATURES = (
    "ft_assists_p100_eb",
    "adjusted_assist_points_p100_eb",
    "adjusted_potential_assists_p100_eb",
    "assist_points_per_potential_eb",
)


def _shrink_rate(
    value: pd.Series,
    exposure: pd.Series,
    season: pd.Series,
    *,
    strength: float,
) -> pd.Series:
    frame = pd.DataFrame({"value": value, "exposure": exposure, "Season": season})
    centers = {}
    for label, group in frame.groupby("Season"):
        valid = group["value"].notna() & group["exposure"].gt(0)
        centers[label] = (
            float(np.average(group.loc[valid, "value"], weights=group.loc[valid, "exposure"]))
            if valid.any() else np.nan
        )
    center = frame["Season"].map(centers)
    reliability = exposure.clip(lower=0) / (exposure.clip(lower=0) + strength)
    return reliability * value.fillna(center) + (1.0 - reliability) * center


def compute_assist_quality_features(
    source: pd.DataFrame,
    *,
    rate_prior_possessions: float = 500.0,
    efficiency_prior_opportunities: float = 100.0,
) -> pd.DataFrame:
    """Correct and stabilize the public assist-data calculations.

    The upstream `assist_ts_pct3` name is not retained because its unit is
    points per adjusted potential assist, not true-shooting percentage.
    """
    frame = source.copy().rename(columns={"year": "Season"})
    required = {
        "PLAYER_ID", "Season", "OffPoss", "AssistPoints", "FT_AST",
        "POTENTIAL_AST", "expected_teammate_ft_percentage3",
    }
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"Assist-quality source is missing {missing}.")
    for column in required - {"Season"}:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Season"] = pd.to_numeric(frame["Season"], errors="coerce")
    frame = frame.dropna(subset=["PLAYER_ID", "Season"]).copy()
    frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
    frame["Season"] = frame["Season"].astype(int)
    frame = frame.drop(columns=["index"], errors="ignore")
    duplicate_rows = frame.loc[
        frame.duplicated(["PLAYER_ID", "Season"], keep=False)
    ]
    if not duplicate_rows.empty:
        unique_payloads = duplicate_rows.drop_duplicates()
        if unique_payloads.duplicated(["PLAYER_ID", "Season"]).any():
            raise ValueError("Assist-quality source has conflicting player-season rows.")
        frame = frame.drop_duplicates(["PLAYER_ID", "Season"], keep="first")

    frame["adjusted_potential_assists"] = (
        frame["POTENTIAL_AST"].fillna(0) + 0.88 * frame["FT_AST"].fillna(0)
    )
    frame["adjusted_assist_points"] = (
        frame["AssistPoints"].fillna(0)
        + 0.88 * 2.0 * frame["FT_AST"].fillna(0)
        * frame["expected_teammate_ft_percentage3"].fillna(0)
    )
    possessions = frame["OffPoss"].where(frame["OffPoss"].gt(0))
    frame["ft_assists_p100"] = 100.0 * frame["FT_AST"] / possessions
    frame["adjusted_assist_points_p100"] = (
        100.0 * frame["adjusted_assist_points"] / possessions
    )
    frame["adjusted_potential_assists_p100"] = (
        100.0 * frame["adjusted_potential_assists"] / possessions
    )
    for name in (
        "ft_assists_p100", "adjusted_assist_points_p100",
        "adjusted_potential_assists_p100",
    ):
        frame[f"{name}_eb"] = _shrink_rate(
            frame[name], frame["OffPoss"], frame["Season"],
            strength=rate_prior_possessions,
        )

    league = frame.groupby("Season", as_index=False).agg(
        points=("adjusted_assist_points", "sum"),
        opportunities=("adjusted_potential_assists", "sum"),
    )
    league["league_efficiency"] = league["points"] / league["opportunities"].where(
        league["opportunities"].gt(0)
    )
    frame = frame.merge(
        league[["Season", "league_efficiency"]], on="Season", validate="many_to_one"
    )
    reliability = frame["adjusted_potential_assists"] / (
        frame["adjusted_potential_assists"] + efficiency_prior_opportunities
    )
    raw_efficiency = frame["adjusted_assist_points"] / frame[
        "adjusted_potential_assists"
    ].where(frame["adjusted_potential_assists"].gt(0))
    frame["assist_points_per_potential_eb"] = (
        reliability * raw_efficiency.fillna(frame["league_efficiency"])
        + (1.0 - reliability) * frame["league_efficiency"]
    )
    return frame[["PLAYER_ID", "Season", *ASSIST_QUALITY_FEATURES]].sort_values(
        ["Season", "PLAYER_ID"]
    ).reset_index(drop=True)


def build_assist_quality_features(
    source_path: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2014, 2025)),
) -> dict:
    source = pd.read_csv(source_path, low_memory=False)
    source = source.loc[pd.to_numeric(source["year"], errors="coerce").isin(seasons)]
    source_duplicate_keys = int(
        source.loc[
            source.duplicated(["PLAYER_ID", "year"], keep=False),
            ["PLAYER_ID", "year"],
        ].drop_duplicates().shape[0]
    )
    source_infinite = int(
        np.isinf(source.select_dtypes(include="number")).sum().sum()
    )
    features = compute_assist_quality_features(source)
    config = {
        "seasons": list(seasons), "rate_prior_possessions": 500.0,
        "efficiency_prior_opportunities": 100.0,
        "source_sha256": sha256_file(source_path),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"assist_quality_features_v1_{identity}"
    output = Path(artifact_root) / "features" / "assist_quality" / run_id
    output.mkdir(parents=True, exist_ok=False)
    path = output / "features.parquet"
    features.to_parquet(path, index=False)
    run = {
        "run_id": run_id, "dataset": "annual_assist_quality_features_v1",
        "status": "validated", "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "rows": len(features), "players": int(features["PLAYER_ID"].nunique()),
            "duplicate_keys": int(features.duplicated(["PLAYER_ID", "Season"]).sum()),
            "source_infinite_values": source_infinite,
            "source_duplicate_keys": source_duplicate_keys,
            "output_nonfinite_values": int(
                (~np.isfinite(features[list(ASSIST_QUALITY_FEATURES)])).sum().sum()
            ),
        },
        "feature_names": list(ASSIST_QUALITY_FEATURES),
        "model_candidate_features": [
            "ft_assists_p100_eb", "assist_points_per_potential_eb"
        ],
        "features_path": str(path.resolve()), "artifact_path": str(output.resolve()),
        "caveats": [
            "The upstream field called assist TS is points per adjusted potential assist, not true-shooting percentage.",
            "Expected teammate free-throw percentage is context dependent and is used only inside adjusted assist points.",
        ],
    }
    if run["quality"]["duplicate_keys"] or run["quality"]["output_nonfinite_values"]:
        raise ValueError("Assist-quality output failed key or finite-value checks.")
    write_json_atomic(run, output / "run.json")
    return run
