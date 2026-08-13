"""Build player-neutral context known at the start of each canonical possession."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import sha256_file, write_json_atomic


POSSESSION_START_CONTEXT_SCHEMA_VERSION = "possession_start_context_v1"
CAUSAL_FEATURE_COLUMNS = (
    "period",
    "is_overtime",
    "seconds_remaining_period_start",
    "regulation_seconds_remaining_start",
    "offense_score_diff_start",
    "offense_is_home",
    "previous_possession_points",
    "is_first_possession",
)


def _require(frame: pd.DataFrame, columns: set[str]) -> None:
    if missing := sorted(columns - set(frame.columns)):
        raise ValueError(f"Possession table is missing required columns: {missing}.")


def compute_possession_start_context(possessions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Derive context from information available before each possession begins.

    Scores are reconstructed from *prior canonical possessions*, not joined from
    an event at the current possession's start. This preserves a clean causal
    boundary even when action IDs differ across upstream feeds.
    """
    _require(
        possessions,
        {
            "possession_id",
            "game_id",
            "possession_number",
            "season_start",
            "season_end",
            "season_label",
            "season_type",
            "game_date",
            "period",
            "start_order_number",
            "start_seconds_elapsed",
            "offense_team_id",
            "defense_team_id",
            "home_team_id",
            "away_team_id",
            "offense_is_home",
            "points",
            "home_points",
            "away_points",
            "lineup_ready",
        },
    )
    frame = possessions.copy()
    frame["game_id"] = frame["game_id"].astype(str).str.zfill(10)
    if frame["possession_id"].duplicated().any():
        raise ValueError("Possession context requires unique possession IDs.")
    numeric = [
        "possession_number",
        "period",
        "start_order_number",
        "start_seconds_elapsed",
        "offense_team_id",
        "defense_team_id",
        "home_team_id",
        "away_team_id",
        "points",
        "home_points",
        "away_points",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Possession table contains non-finite context inputs.")
    if frame[["points", "home_points", "away_points"]].lt(0).any().any():
        raise ValueError("Possession outcomes cannot have negative points.")
    point_identity = np.abs(frame["points"] - frame["home_points"] - frame["away_points"])
    if float(point_identity.max()) > 1e-9:
        raise ValueError("Possession home/away points do not sum to possession points.")
    ordered = frame.sort_values(
        ["game_id", "possession_number", "start_order_number"], kind="stable"
    ).reset_index(drop=True)
    expected_number = ordered.groupby("game_id", sort=False).cumcount() + 1
    if not ordered["possession_number"].eq(expected_number).all():
        raise ValueError("Possession numbers must be contiguous and begin at one per game.")
    if ordered.duplicated(["game_id", "possession_number"]).any():
        raise ValueError("Possession numbers are not unique within games.")

    groups = ordered.groupby("game_id", sort=False)
    ordered["home_score_start"] = groups["home_points"].cumsum() - ordered["home_points"]
    ordered["away_score_start"] = groups["away_points"].cumsum() - ordered["away_points"]
    ordered["home_score_diff_start"] = (
        ordered["home_score_start"] - ordered["away_score_start"]
    )
    ordered["offense_score_diff_start"] = np.where(
        ordered["offense_is_home"].astype(bool),
        ordered["home_score_diff_start"],
        -ordered["home_score_diff_start"],
    )
    previous_points = groups["points"].shift(1)
    ordered["is_first_possession"] = previous_points.isna()
    ordered["previous_possession_points"] = previous_points.fillna(0.0)
    ordered["is_overtime"] = ordered["period"].gt(4)
    ordered["seconds_remaining_period_start"] = np.where(
        ordered["is_overtime"],
        300.0 - (ordered["start_seconds_elapsed"] - 2880.0 - (ordered["period"] - 5) * 300.0),
        720.0 - (ordered["start_seconds_elapsed"] - (ordered["period"] - 1) * 720.0),
    )
    ordered["regulation_seconds_remaining_start"] = np.maximum(
        2880.0 - ordered["start_seconds_elapsed"], 0.0
    )
    if ordered["seconds_remaining_period_start"].lt(-1e-9).any() or ordered[
        "seconds_remaining_period_start"
    ].gt(720.0 + 1e-9).any():
        raise ValueError("Possession start times fall outside their period bounds.")

    output_columns = [
        "possession_id",
        "game_id",
        "possession_number",
        "season_start",
        "season_end",
        "season_label",
        "season_type",
        "game_date",
        "offense_team_id",
        "defense_team_id",
        "home_team_id",
        "away_team_id",
        "lineup_ready",
        "points",
        *CAUSAL_FEATURE_COLUMNS,
    ]
    output = ordered[output_columns].copy()
    output["period"] = output["period"].astype(int)
    output["is_overtime"] = output["is_overtime"].astype(bool)
    output["is_first_possession"] = output["is_first_possession"].astype(bool)
    output["offense_is_home"] = output["offense_is_home"].astype(bool)
    duplicate_rows = int(output.duplicated("possession_id", keep=False).sum())
    nonfinite_values = int((~np.isfinite(output.select_dtypes(include="number"))).sum().sum())
    feature_leakage_columns = sorted(set(CAUSAL_FEATURE_COLUMNS) & {"points", "action_count", "end_action_number", "end_order_number"})
    quality = {
        "passed": duplicate_rows == 0 and nonfinite_values == 0 and not feature_leakage_columns,
        "input_rows": int(len(possessions)),
        "output_rows": int(len(output)),
        "games": int(output["game_id"].nunique()),
        "duplicate_possession_ids": duplicate_rows,
        "nonfinite_numeric_values": nonfinite_values,
        "maximum_point_identity_error": float(point_identity.max()),
        "causal_feature_columns": list(CAUSAL_FEATURE_COLUMNS),
        "forbidden_feature_columns_present": feature_leakage_columns,
        "lineup_ready_rate": float(output["lineup_ready"].mean()),
    }
    if not quality["passed"]:
        raise ValueError(f"Possession-start context quality gates failed: {quality}.")
    return output, quality


def build_possession_start_context(
    possessions_path: str | Path,
    destination: str | Path,
    manifest_dir: str | Path,
) -> dict:
    """Build and atomically write one causally bounded row per possession."""
    source_path = Path(possessions_path)
    output_path = Path(destination)
    context, quality = compute_possession_start_context(pd.read_parquet(source_path))
    source = {
        "path": str(source_path.resolve()),
        "bytes": source_path.stat().st_size,
        "sha256": sha256_file(source_path),
    }
    identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    context.to_parquet(partial, index=False)
    partial.replace(output_path)
    snapshot = {
        "snapshot_id": f"possession_start_context_{identity}",
        "dataset": POSSESSION_START_CONTEXT_SCHEMA_VERSION,
        "status": "research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": quality["passed"],
        "row_count": int(len(context)),
        "game_count": int(context["game_id"].nunique()),
        "grain": "one canonical possession with context known before it begins",
        "target": "points scored in the current possession",
        "causal_feature_columns": list(CAUSAL_FEATURE_COLUMNS),
        "forbidden_model_inputs": [
            "player IDs",
            "team IDs",
            "lineup IDs",
            "current-possession actions",
            "current-possession duration",
            "current-possession end state",
            "current possession points",
        ],
        "source_files": [source],
        "quality": quality,
        "path": str(output_path.resolve()),
        "sha256": sha256_file(output_path),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    write_json_atomic(snapshot, Path(manifest_dir) / f"{snapshot['snapshot_id']}.json")
    return snapshot
