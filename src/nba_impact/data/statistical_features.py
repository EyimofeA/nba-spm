"""Build validated three-season box and tracking features for impact models."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

CORE_RATE_SPECS = {
    "PTS_p100": ("PTS", "OffPoss"),
    "AST_p100": ("AST", "OffPoss"),
    "TOV_p100": ("TOV", "OffPoss"),
    "STL_p100": ("STL", "DefPoss"),
    "BLK_p100": ("BLK", "DefPoss"),
    "OREB_p100": ("OREB", "OffPoss"),
    "DREB_p100": ("DREB", "DefPoss"),
    "PF_p100": ("PF", "DefPoss"),
    "PFD_p100": ("PFD", "OffPoss"),
    "FTA_p100": ("FTA", "OffPoss"),
    "FTM_p100": ("FTM", "OffPoss"),
    "FG2A_p100": ("FG2A", "OffPoss"),
    "FG2M_p100": ("FG2M", "OffPoss"),
    "FG3A_p100": ("FG3A", "OffPoss"),
    "FG3M_p100": ("FG3M", "OffPoss"),
}

TRACKING_RATE_SPECS = {
    "drives_p100": ("DRIVES", "OffPoss"),
    "drive_points_p100": ("DRIVE_PTS", "OffPoss"),
    "drive_assists_p100": ("DRIVE_AST", "OffPoss"),
    "drive_turnovers_p100": ("DRIVE_TOV", "OffPoss"),
    "drive_fta_p100": ("DRIVE_FTA", "OffPoss"),
    "touches_p100": ("TOUCHES", "OffPoss"),
    "front_court_touches_p100": ("FRONT_CT_TOUCHES", "OffPoss"),
    "paint_touches_p100": ("PAINT_TOUCHES", "OffPoss"),
    "post_touches_p100": ("POST_TOUCHES", "OffPoss"),
    "elbow_touches_p100": ("ELBOW_TOUCHES", "OffPoss"),
    "time_of_possession_p100": ("TIME_OF_POSS", "OffPoss"),
    "passes_made_p100": ("PASSES_MADE", "OffPoss"),
    "passes_received_p100": ("PASSES_RECEIVED", "OffPoss"),
    "potential_assists_p100": ("POTENTIAL_AST", "OffPoss"),
    "secondary_assists_p100": ("SECONDARY_AST", "OffPoss"),
    "assist_points_created_p100": ("AST_PTS_CREATED", "OffPoss"),
    "catch_shoot_fga_p100": ("CATCH_SHOOT_FGA", "OffPoss"),
    "pull_up_fga_p100": ("PULL_UP_FGA", "OffPoss"),
    "at_rim_fga_p100": ("AtRimFGA", "OffPoss"),
    "short_mid_fga_p100": ("ShortMidRangeFGA", "OffPoss"),
    "long_mid_fga_p100": ("LongMidRangeFGA", "OffPoss"),
    "corner3_fga_p100": ("Corner3FGA", "OffPoss"),
    "arc3_fga_p100": ("Arc3FGA", "OffPoss"),
    "open_fga_p100": ("open_FGA", "OffPoss"),
    "wide_open_fga_p100": ("wide_open_FGA", "OffPoss"),
    "tight_fga_p100": ("tight_FGA", "OffPoss"),
    "very_tight_fga_p100": ("very_tight_FGA", "OffPoss"),
    "live_ball_turnovers_p100": ("LiveBallTurnovers", "OffPoss"),
    "bad_pass_turnovers_p100": ("BadPassTurnovers", "OffPoss"),
    "lost_ball_turnovers_p100": ("LostBallTurnovers", "OffPoss"),
    "travels_p100": ("Travels", "OffPoss"),
    "offensive_fouls_p100": ("Offensive Fouls", "OffPoss"),
    "shooting_fouls_drawn_p100": ("ShootingFouls", "OffPoss"),
    "rebound_contests_p100": ("REB_CONTEST", "DefPoss"),
    "rebound_chances_p100": ("REB_CHANCES", "DefPoss"),
    "dreb_contests_p100": ("DREB_CONTEST", "DefPoss"),
    "dreb_chances_p100": ("DREB_CHANCES", "DefPoss"),
    "recovered_blocks_p100": ("RecoveredBlocks", "DefPoss"),
}

RATIO_SPECS = {
    "fg2_pct": ("FG2M", "FG2A"),
    "fg3_pct": ("FG3M", "FG3A"),
    "ft_pct": ("FTM", "FTA"),
    "at_rim_accuracy": ("AtRimFGM", "AtRimFGA"),
    "short_mid_accuracy": ("ShortMidRangeFGM", "ShortMidRangeFGA"),
    "long_mid_accuracy": ("LongMidRangeFGM", "LongMidRangeFGA"),
    "corner3_accuracy": ("Corner3FGM", "Corner3FGA"),
    "arc3_accuracy": ("Arc3FGM", "Arc3FGA"),
    "catch_shoot_accuracy": ("CATCH_SHOOT_FGM", "CATCH_SHOOT_FGA"),
    "catch_shoot_3_accuracy": ("CATCH_SHOOT_FG3M", "CATCH_SHOOT_FG3A"),
    "pull_up_accuracy": ("PULL_UP_FGM", "PULL_UP_FGA"),
    "pull_up_3_accuracy": ("PULL_UP_FG3M", "PULL_UP_FG3A"),
    "open_accuracy": ("open_FGM", "open_FGA"),
    "open_3_accuracy": ("open_FG3M", "open_FG3A"),
    "wide_open_accuracy": ("wide_open_FGM", "wide_open_FGA"),
    "wide_open_3_accuracy": ("wide_open_FG3M", "wide_open_FG3A"),
    "tight_accuracy": ("tight_FGM", "tight_FGA"),
    "tight_3_accuracy": ("tight_FG3M", "tight_FG3A"),
    "very_tight_accuracy": ("very_tight_FGM", "very_tight_FGA"),
    "very_tight_3_accuracy": ("very_tight_FG3M", "very_tight_FG3A"),
    "at_rim_frequency": ("AtRimFGA", "FGA"),
    "short_mid_frequency": ("ShortMidRangeFGA", "FGA"),
    "long_mid_frequency": ("LongMidRangeFGA", "FGA"),
    "corner3_frequency": ("Corner3FGA", "FGA"),
    "arc3_frequency": ("Arc3FGA", "FGA"),
    "drive_points_per_drive": ("DRIVE_PTS", "DRIVES"),
    "drive_assists_per_drive": ("DRIVE_AST", "DRIVES"),
    "drive_turnovers_per_drive": ("DRIVE_TOV", "DRIVES"),
    "drive_fta_per_drive": ("DRIVE_FTA", "DRIVES"),
    "passes_per_touch": ("PASSES_MADE", "TOUCHES"),
    "potential_assists_per_touch": ("POTENTIAL_AST", "TOUCHES"),
    "paint_points_per_touch": ("PAINT_TOUCH_PTS", "PAINT_TOUCHES"),
    "post_points_per_touch": ("POST_TOUCH_PTS", "POST_TOUCHES"),
    "elbow_points_per_touch": ("ELBOW_TOUCH_PTS", "ELBOW_TOUCHES"),
    "live_ball_turnover_share": ("LiveBallTurnovers", "TOV"),
    "bad_pass_turnover_share": ("BadPassTurnovers", "TOV"),
    "lost_ball_turnover_share": ("LostBallTurnovers", "TOV"),
}

NATURAL_WEIGHTED_AVERAGES = {
    "avg_seconds_per_touch": ("AVG_SEC_PER_TOUCH", "TOUCHES"),
    "avg_dribbles_per_touch": ("AVG_DRIB_PER_TOUCH", "TOUCHES"),
    "shot_quality_average": ("ShotQualityAvg", "FGA"),
    "OnOffRtg": ("OnOffRtg", "OffPoss"),
    "OnDefRtg": ("OnDefRtg", "DefPoss"),
}


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid_denominator = denominator.where(denominator > 0)
    return numerator / valid_denominator


def _required_source_columns() -> set[str]:
    columns = {"PLAYER_ID", "OffPoss", "DefPoss", "FGA"}
    for source, denominator in (
        list(CORE_RATE_SPECS.values())
        + list(TRACKING_RATE_SPECS.values())
        + list(RATIO_SPECS.values())
        + list(NATURAL_WEIGHTED_AVERAGES.values())
    ):
        columns.update((source, denominator))
    return columns


def _load_source(path: Path, season: int) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"PLAYER_ID", "OffPoss", "DefPoss"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{path} is missing required columns {missing}.")
    original_rows = len(frame)
    contract_columns = sorted(_required_source_columns() & set(frame.columns))
    frame = frame.drop_duplicates(subset=contract_columns).copy()
    removed = original_rows - len(frame)
    if frame.duplicated("PLAYER_ID").any():
        ids = sorted(frame.loc[frame.duplicated("PLAYER_ID", False), "PLAYER_ID"].unique())
        raise ValueError(f"{path} has conflicting PLAYER_ID rows: {ids[:10]}.")
    frame["source_season"] = season
    return frame, removed


def _aggregate_window(frames: list[pd.DataFrame], window_end: int) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    needed = _required_source_columns()
    for column in needed - {"PLAYER_ID"}:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    sums = frame.groupby("PLAYER_ID", as_index=False)[sorted(needed - {"PLAYER_ID"})].sum(
        min_count=1
    )
    output_columns: dict[str, pd.Series | int] = {
        "PLAYER_ID": sums["PLAYER_ID"],
        "OffPoss": sums["OffPoss"],
        "DefPoss": sums["DefPoss"],
    }
    for name, (source, denominator) in {**CORE_RATE_SPECS, **TRACKING_RATE_SPECS}.items():
        output_columns[name] = 100.0 * _safe_ratio(sums[source], sums[denominator])
    for name, (numerator, denominator) in RATIO_SPECS.items():
        output_columns[name] = _safe_ratio(sums[numerator], sums[denominator])
    usage_events = sums["FGA"] + 0.44 * sums["FTA"] + sums["TOV"]
    output_columns["usage_events_p100"] = 100.0 * _safe_ratio(
        usage_events, sums["OffPoss"]
    )
    true_shooting_denominator = 2.0 * (sums["FGA"] + 0.44 * sums["FTA"])
    output_columns["true_shooting_pct"] = _safe_ratio(
        sums["PTS"], true_shooting_denominator
    )

    for name, (source, weight) in NATURAL_WEIGHTED_AVERAGES.items():
        valid = frame[source].notna() & frame[weight].gt(0)
        numerator = (frame[source].where(valid) * frame[weight].where(valid)).groupby(
            frame["PLAYER_ID"]
        ).sum(min_count=1)
        denominator = frame[weight].where(valid).groupby(frame["PLAYER_ID"]).sum(min_count=1)
        weighted = numerator / denominator
        output_columns[name] = sums["PLAYER_ID"].map(weighted)
    output_columns["Window_End"] = int(window_end)
    return pd.DataFrame(output_columns)


def build_statistical_feature_windows(
    source_dir: str | Path,
    *,
    artifact_root: str | Path,
    window_ends: tuple[int, ...] = tuple(range(2016, 2025)),
    window_seasons: int = 3,
) -> dict:
    """Build content-addressed pooled features from complete source seasons."""
    if window_seasons < 1:
        raise ValueError("window_seasons must be positive.")
    source = Path(source_dir)
    required_seasons = sorted(
        {season for end in window_ends for season in range(end - window_seasons + 1, end + 1)}
    )
    loaded: dict[int, pd.DataFrame] = {}
    source_records = []
    duplicate_rows_removed = 0
    for season in required_seasons:
        path = source / f"{season}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing source season {path}.")
        frame, removed = _load_source(path, season)
        loaded[season] = frame
        duplicate_rows_removed += removed
        source_records.append(
            {"season": season, "path": str(path.resolve()), "sha256": sha256_file(path)}
        )

    windows = []
    audits = []
    for end in window_ends:
        seasons = tuple(range(end - window_seasons + 1, end + 1))
        window = _aggregate_window([loaded[season] for season in seasons], end)
        bounded = [
            column
            for column in window.columns
            if column.endswith(("_accuracy", "_frequency", "_share"))
        ]
        bound_violations = int(
            sum(((window[column] < 0) | (window[column] > 1)).sum() for column in bounded)
        )
        audits.append(
            {
                "window_end": end,
                "source_seasons": ",".join(map(str, seasons)),
                "rows": len(window),
                "players": window["PLAYER_ID"].nunique(),
                "duplicate_keys": int(window.duplicated(["PLAYER_ID", "Window_End"]).sum()),
                "bounded_ratio_violations": bound_violations,
                "feature_missing_fraction": float(
                    window.drop(columns=["PLAYER_ID", "Window_End", "OffPoss", "DefPoss"])
                    .isna()
                    .mean()
                    .mean()
                ),
            }
        )
        windows.append(window)
    features = pd.concat(windows, ignore_index=True)
    audit = pd.DataFrame(audits)
    if features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Built statistical features contain duplicate keys.")
    if int(audit["bounded_ratio_violations"].sum()) > 0:
        raise ValueError("Built statistical features contain bounded-ratio violations.")

    config = {
        "window_ends": list(window_ends),
        "window_seasons": window_seasons,
        "builder_sha256": sha256_file(Path(__file__)),
        "source_hashes": {str(record["season"]): record["sha256"] for record in source_records},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"statistical_features_v1_{identity}"
    output = Path(artifact_root) / "features" / "statistical_impact" / run_id
    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "features.parquet"
    audit_path = output / "audit.parquet"
    features.to_parquet(features_path, index=False)
    audit.to_parquet(audit_path, index=False)
    run = {
        "run_id": run_id,
        "dataset": "statistical_impact_features",
        "grain": (
            "player_single_season"
            if window_seasons == 1
            else f"player_{window_seasons}_season_window"
        ),
        "status": "validated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "rows": len(features),
            "players": features["PLAYER_ID"].nunique(),
            "features": len(features.columns) - 4,
            "duplicate_source_rows_collapsed_on_feature_contract": duplicate_rows_removed,
            "duplicate_keys": 0,
            "bounded_ratio_violations": 0,
        },
        "feature_groups": {
            "core_rates": list(CORE_RATE_SPECS),
            "tracking_rates": list(TRACKING_RATE_SPECS),
            "pooled_ratios": list(RATIO_SPECS),
            "natural_weighted_averages": list(NATURAL_WEIGHTED_AVERAGES),
        },
        "excluded_primary_inputs": ["AGE", "MIN", "GP", "position", "experience"],
        "reliability_only_columns": ["OffPoss", "DefPoss"],
        "artifact_path": str(output.resolve()),
        "features_path": str(features_path.resolve()),
        "audit_path": str(audit_path.resolve()),
        "sources": source_records,
    }
    write_json_atomic(run, output / "run.json")
    return run
