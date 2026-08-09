"""Basketball-domain feature engineering layered on the validated v1 table."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features import RATIO_SPECS, _aggregate_window, _load_source

TEMPORAL_FEATURES = (
    "PTS_p100", "AST_p100", "TOV_p100", "STL_p100", "BLK_p100",
    "DREB_p100", "PF_p100", "FTA_p100", "FG3A_p100", "drives_p100",
    "touches_p100", "potential_assists_p100", "true_shooting_pct",
    "at_rim_frequency", "arc3_frequency", "shot_quality_average",
    "rebound_contests_p100", "rebound_chances_p100", "recovered_blocks_p100",
)

RELATIVE_FEATURES = (
    "PTS_p100", "AST_p100", "TOV_p100", "STL_p100", "BLK_p100",
    "DREB_p100", "PF_p100", "FTA_p100", "FG3A_p100", "drives_p100",
    "potential_assists_p100", "true_shooting_pct", "at_rim_frequency",
    "arc3_frequency", "shot_quality_average",
)

EXTRA_COUNTS = (
    "PtsAssisted2s", "PtsAssisted3s", "PtsUnassisted2s", "PtsUnassisted3s",
    "PULL_UP_FGA", "CATCH_SHOOT_FGA", "DRIVE_PASSES", "DRIVES",
    "AST", "POTENTIAL_AST", "AST_PTS_CREATED", "TOUCHES", "PTS",
    "LiveBallTurnovers", "PASSES_MADE", "PFD", "FGA", "FG2A", "FG3A",
    "REB_CONTEST", "REB_CHANCES", "DREB_CONTEST", "DREB_UNCONTEST",
    "RecoveredBlocks", "BLK", "STL", "PF", "DefPoss", "OffPoss",
    "PAINT_TOUCHES", "POST_TOUCHES", "ELBOW_TOUCHES",
)

BOUNDED_FEATURES = (
    "self_created_point_share", "assisted_three_share", "pull_up_attempt_share",
    "potential_assist_conversion", "drive_pass_rate",
    "rebound_contest_share", "dreb_contested_share",
    "rim_and_three_frequency", "midrange_frequency",
)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def _bounded_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = _ratio(numerator, denominator)
    return ratio.where(numerator <= denominator)


def _pooled_counts(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    for column in EXTRA_COUNTS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.groupby("PLAYER_ID", as_index=False)[list(EXTRA_COUNTS)].sum(min_count=1)


def _prior_strength(denominator: str) -> float:
    if denominator == "TOUCHES":
        return 500.0
    if denominator in {"DRIVES", "TOV"}:
        return 150.0
    if "TOUCHES" in denominator:
        return 100.0
    return 100.0


def _stabilized_ratios(frames: list[pd.DataFrame], player_ids: pd.Series) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    needed = {column for values in RATIO_SPECS.values() for column in values}
    for column in needed:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    sums = frame.groupby("PLAYER_ID", as_index=True)[sorted(needed)].sum(min_count=1)
    output = pd.DataFrame({"PLAYER_ID": player_ids})
    for name, (numerator, denominator) in RATIO_SPECS.items():
        league_denominator = frame[denominator].sum()
        league_rate = (
            frame[numerator].sum() / league_denominator
            if league_denominator > 0
            else np.nan
        )
        strength = _prior_strength(denominator)
        stabilized = (sums[numerator].fillna(0) + strength * league_rate) / (
            sums[denominator].fillna(0) + strength
        )
        output[f"{name}_eb"] = output["PLAYER_ID"].map(stabilized)
    return output


def _entropy(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    shares = frame[columns].fillna(0).clip(lower=0)
    total = shares.sum(axis=1)
    shares = shares.div(total.replace(0, np.nan), axis=0)
    entropy = -(shares.where(shares > 0) * np.log(shares.where(shares > 0))).sum(axis=1)
    return entropy.where(total > 0)


def _engineer_window(
    base: pd.DataFrame,
    frames: list[pd.DataFrame],
    seasonal: list[pd.DataFrame],
) -> pd.DataFrame:
    output = base.copy()
    stabilized = _stabilized_ratios(frames, output["PLAYER_ID"])
    output = output.merge(stabilized, on="PLAYER_ID", validate="one_to_one")
    counts = _pooled_counts(frames).set_index("PLAYER_ID")
    mapped = {column: output["PLAYER_ID"].map(counts[column]) for column in EXTRA_COUNTS}

    engineered: dict[str, pd.Series] = {}

    assisted = mapped["PtsAssisted2s"] + mapped["PtsAssisted3s"]
    unassisted = mapped["PtsUnassisted2s"] + mapped["PtsUnassisted3s"]
    engineered["self_created_point_share"] = _ratio(unassisted, assisted + unassisted)
    engineered["assisted_three_share"] = _ratio(
        mapped["PtsAssisted3s"], mapped["PtsAssisted3s"] + mapped["PtsUnassisted3s"]
    )
    engineered["pull_up_attempt_share"] = _ratio(
        mapped["PULL_UP_FGA"], mapped["PULL_UP_FGA"] + mapped["CATCH_SHOOT_FGA"]
    )
    engineered["potential_assist_conversion"] = _bounded_ratio(
        mapped["AST"], mapped["POTENTIAL_AST"]
    )
    engineered["assist_points_per_touch"] = _ratio(mapped["AST_PTS_CREATED"], mapped["TOUCHES"])
    engineered["total_points_created_per_touch"] = _ratio(
        mapped["PTS"] + mapped["AST_PTS_CREATED"], mapped["TOUCHES"]
    )
    engineered["drive_pass_rate"] = _ratio(mapped["DRIVE_PASSES"], mapped["DRIVES"])
    creation_events = mapped["DRIVES"] + mapped["POTENTIAL_AST"] + mapped["PULL_UP_FGA"]
    engineered["creation_load_p100"] = 100 * _ratio(creation_events, mapped["OffPoss"])
    engineered["live_ball_turnovers_per_creation"] = _ratio(
        mapped["LiveBallTurnovers"], creation_events
    )
    engineered["foul_pressure_per_fga"] = _ratio(mapped["PFD"], mapped["FGA"])
    engineered["stocks_p100_def"] = 100 * _ratio(mapped["STL"] + mapped["BLK"], mapped["DefPoss"])
    engineered["defensive_activity_p100"] = 100 * _ratio(
        mapped["STL"] + mapped["BLK"] + mapped["REB_CONTEST"], mapped["DefPoss"]
    )
    engineered["rebound_contest_share"] = _ratio(mapped["REB_CONTEST"], mapped["REB_CHANCES"])
    engineered["dreb_contested_share"] = _ratio(
        mapped["DREB_CONTEST"], mapped["DREB_CONTEST"] + mapped["DREB_UNCONTEST"]
    )
    engineered["block_recovery_rate"] = _ratio(mapped["RecoveredBlocks"], mapped["BLK"])
    engineered["stocks_per_foul"] = _ratio(mapped["STL"] + mapped["BLK"], mapped["PF"])
    engineered["interior_role_load"] = 100 * _ratio(
        mapped["PAINT_TOUCHES"] + mapped["POST_TOUCHES"] + mapped["ELBOW_TOUCHES"],
        mapped["OffPoss"],
    )

    zone_frequencies = [
        "at_rim_frequency", "short_mid_frequency", "long_mid_frequency",
        "corner3_frequency", "arc3_frequency",
    ]
    engineered["shot_profile_entropy"] = _entropy(output, zone_frequencies)
    engineered["effective_shot_zones"] = np.exp(engineered["shot_profile_entropy"])
    engineered["rim_and_three_frequency"] = (
        output["at_rim_frequency"] + output["corner3_frequency"] + output["arc3_frequency"]
    )
    engineered["midrange_frequency"] = output["short_mid_frequency"] + output["long_mid_frequency"]
    engineered["expected_zone_points"] = (
        2 * output["at_rim_frequency"] * output["at_rim_accuracy_eb"]
        + 2 * output["short_mid_frequency"] * output["short_mid_accuracy_eb"]
        + 2 * output["long_mid_frequency"] * output["long_mid_accuracy_eb"]
        + 3 * output["corner3_frequency"] * output["corner3_accuracy_eb"]
        + 3 * output["arc3_frequency"] * output["arc3_accuracy_eb"]
    )

    for feature in RELATIVE_FEATURES:
        defensive_rates = {"STL_p100", "BLK_p100", "DREB_p100", "PF_p100"}
        weights = output["DefPoss"] if feature in defensive_rates else output["OffPoss"]
        valid = output[feature].notna() & weights.gt(0)
        center = (
            np.average(output.loc[valid, feature], weights=weights[valid])
            if valid.any()
            else np.nan
        )
        engineered[f"{feature}_relative"] = output[feature] - center

    seasonal_indexed = [frame.set_index("PLAYER_ID") for frame in seasonal]
    for feature in TEMPORAL_FEATURES:
        values = pd.concat([frame[feature] for frame in seasonal_indexed], axis=1)
        values.columns = ["old", "middle", "latest"]
        latest = output["PLAYER_ID"].map(values["latest"])
        trend = output["PLAYER_ID"].map((values["latest"] - values["old"]) / 2)
        volatility = output["PLAYER_ID"].map(values.std(axis=1, ddof=0))
        engineered[f"{feature}_latest"] = latest.fillna(output[feature])
        engineered[f"{feature}_trend"] = trend.fillna(0.0)
        engineered[f"{feature}_volatility"] = volatility.fillna(0.0)
    return pd.concat([output, pd.DataFrame(engineered, index=output.index)], axis=1)


def build_statistical_features_v2(
    source_dir: str | Path,
    base_features_path: str | Path,
    *,
    artifact_root: str | Path,
    window_ends: tuple[int, ...] = tuple(range(2016, 2025)),
) -> dict:
    source = Path(source_dir)
    base = pd.read_parquet(base_features_path)
    loaded = {}
    source_hashes = {}
    for season in range(min(window_ends) - 2, max(window_ends) + 1):
        path = source / f"{season}.csv"
        loaded[season] = _load_source(path, season)[0]
        source_hashes[str(season)] = sha256_file(path)
    outputs = []
    for end in window_ends:
        frames = [loaded[season] for season in range(end - 2, end + 1)]
        seasonal = [
            _aggregate_window([frame], season)
            for frame, season in zip(
                frames, range(end - 2, end + 1), strict=True
            )
        ]
        base_window = base.loc[base["Window_End"].eq(end)].copy()
        outputs.append(_engineer_window(base_window, frames, seasonal))
    features = pd.concat(outputs, ignore_index=True)
    if features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("V2 feature keys are not unique.")
    new_features = [column for column in features if column not in base.columns]
    raw_new_features = features[new_features].copy()
    missing_by_feature = raw_new_features.isna().mean().sort_values(ascending=False)
    for feature in new_features:
        window_median = features.groupby("Window_End")[feature].transform("median")
        features[feature] = features[feature].fillna(window_median).fillna(0.0)
    bound_violations = {
        column: int(((features[column] < 0) | (features[column] > 1)).sum())
        for column in BOUNDED_FEATURES
    }
    audit = pd.DataFrame(
        {
            "feature": new_features,
            "missing_fraction_before_neutral_fill": [
                float(raw_new_features[column].isna().mean()) for column in new_features
            ],
            "minimum": [float(features[column].min()) for column in new_features],
            "median": [float(features[column].median()) for column in new_features],
            "maximum": [float(features[column].max()) for column in new_features],
        }
    )
    config = {
        "base_features_sha256": sha256_file(base_features_path),
        "source_hashes": source_hashes,
        "builder_sha256": sha256_file(Path(__file__)),
        "window_ends": list(window_ends),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"statistical_features_v2_{identity}"
    output = Path(artifact_root) / "features" / "statistical_impact" / run_id
    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "features.parquet"
    audit_path = output / "audit.parquet"
    features.to_parquet(features_path, index=False)
    audit.to_parquet(audit_path, index=False)
    run = {
        "run_id": run_id,
        "dataset": "statistical_impact_features_v2",
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "rows": len(features),
            "players": features["PLAYER_ID"].nunique(),
            "total_features": len(features.columns) - 4,
            "new_features": len(new_features),
            "duplicate_keys": 0,
            "infinite_values": int(np.isinf(features.select_dtypes(include="number")).sum().sum()),
            "bounded_feature_violations": int(sum(bound_violations.values())),
            "max_new_feature_missing_fraction_before_neutral_fill": float(missing_by_feature.max()),
            "new_feature_missing_values_after_neutral_fill": int(
                features[new_features].isna().sum().sum()
            ),
        },
        "bounded_feature_violations": bound_violations,
        "new_feature_names": new_features,
        "features_path": str(features_path.resolve()),
        "audit_path": str(audit_path.resolve()),
        "artifact_path": str(output.resolve()),
    }
    if run["quality"]["infinite_values"]:
        raise ValueError("V2 features contain infinite values.")
    if run["quality"]["bounded_feature_violations"]:
        raise ValueError("V2 features contain bounded-feature violations.")
    write_json_atomic(run, output / "run.json")
    return run
