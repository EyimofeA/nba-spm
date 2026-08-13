"""Build conservative player-season shooting role-context features.

The upstream tables have one row per player-season dribble bucket.  This
module retains source player IDs, derives only count-supported shares, and does
not declare the output to be a production AIO feature block.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import sha256_file, write_json_atomic


ROLE_CONTEXT_SCHEMA_VERSION = "annual_role_context_v1"
_BUCKETS = ("0", "1", "2", "3_6", "7+")
_GROUPS = {
    "zero_dribble": ("0",),
    "one_two_dribble": ("1", "2"),
    "three_plus_dribble": ("3_6", "7+"),
}


def _prepare_source(source: pd.DataFrame, *, label: str) -> pd.DataFrame:
    required = {"PLAYER_ID", "PLAYER", "dribbles", "year", "FGM", "FGA"}
    if missing := sorted(required - set(source.columns)):
        raise ValueError(f"{label} source is missing {missing}.")
    frame = source[list(required)].copy()
    frame = frame.rename(columns={"year": "Season"})
    for column in ("PLAYER_ID", "Season", "FGM", "FGA"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["dribbles"] = frame["dribbles"].astype(str).str.strip()
    if unknown := sorted(set(frame["dribbles"].dropna()) - set(_BUCKETS)):
        raise ValueError(f"{label} source has unknown dribble buckets: {unknown}.")
    frame = frame.dropna(subset=["PLAYER_ID", "Season", "dribbles", "FGM", "FGA"]).copy()
    if frame.empty:
        raise ValueError(f"{label} source has no usable rows.")
    if not np.all(np.equal(frame[["PLAYER_ID", "Season"]], np.floor(frame[["PLAYER_ID", "Season"]]))):
        raise ValueError(f"{label} source has non-integral player IDs or seasons.")
    if (frame[["FGM", "FGA"]] < 0).any().any() or frame["FGM"].gt(frame["FGA"]).any():
        raise ValueError(f"{label} source has invalid made/attempt counts.")
    frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
    frame["Season"] = frame["Season"].astype(int)
    duplicates = frame.duplicated(["PLAYER_ID", "Season", "dribbles"], keep=False)
    if duplicates.any():
        raise ValueError(f"{label} source has duplicate player-season-dribble keys.")
    return frame


def _aggregate_context(source: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    keys = ["PLAYER_ID", "Season"]
    totals = source.groupby(keys, as_index=False).agg(
        **{
            f"{prefix}_fga": ("FGA", "sum"),
            f"{prefix}_fgm": ("FGM", "sum"),
        }
    )
    for name, buckets in _GROUPS.items():
        subset = source.loc[source["dribbles"].isin(buckets)]
        counts = subset.groupby(keys, as_index=False).agg(
            **{
                f"{prefix}_{name}_fga": ("FGA", "sum"),
                f"{prefix}_{name}_fgm": ("FGM", "sum"),
            }
        )
        totals = totals.merge(counts, on=keys, how="left", validate="one_to_one")
        for suffix in ("fga", "fgm"):
            totals[f"{prefix}_{name}_{suffix}"] = totals[
                f"{prefix}_{name}_{suffix}"
            ].fillna(0.0)
        attempts = totals[f"{prefix}_{name}_fga"]
        totals[f"{prefix}_{name}_share"] = attempts / totals[f"{prefix}_fga"].where(
            totals[f"{prefix}_fga"].gt(0)
        )
        totals[f"{prefix}_{name}_fg_pct"] = totals[f"{prefix}_{name}_fgm"] / attempts.where(
            attempts.gt(0)
        )
    totals[f"{prefix}_fg_pct"] = totals[f"{prefix}_fgm"] / totals[f"{prefix}_fga"].where(
        totals[f"{prefix}_fga"].gt(0)
    )
    return totals


def compute_role_context_features(
    shooting_by_dribble: pd.DataFrame,
    jump_shot_by_dribble: pd.DataFrame,
    *,
    seasons: tuple[int, ...],
) -> pd.DataFrame:
    """Return one count-derived role row per source player-season.

    Source display fields (including team, age, games, minutes, and percentage
    columns) are intentionally not carried into this table.
    """
    shots = _prepare_source(shooting_by_dribble, label="shooting-by-dribble")
    jumps = _prepare_source(jump_shot_by_dribble, label="jump-shot-by-dribble")
    allowed = set(seasons)
    shots = shots.loc[shots["Season"].isin(allowed)].copy()
    jumps = jumps.loc[jumps["Season"].isin(allowed)].copy()
    if shots.empty or jumps.empty:
        raise ValueError("Requested seasons have no rows in one or both role-context sources.")
    shot_features = _aggregate_context(shots, prefix="all_shot")
    jump_features = _aggregate_context(jumps, prefix="jump_shot")
    output = shot_features.merge(
        jump_features, on=["PLAYER_ID", "Season"], how="outer", validate="one_to_one"
    ).sort_values(["Season", "PLAYER_ID"], kind="stable").reset_index(drop=True)
    numeric = output.columns.difference(["PLAYER_ID", "Season"])
    if output.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Role-context output has duplicate player-season keys.")
    if np.isinf(output[numeric].select_dtypes(include="number")).any().any():
        raise ValueError("Role-context output contains infinite values.")
    return output


def build_role_context_features(
    shooting_by_dribble_path: str | Path,
    jump_shot_by_dribble_path: str | Path,
    *,
    artifact_root: str | Path,
    seasons: tuple[int, ...] = tuple(range(2014, 2026)),
) -> dict:
    """Build a content-addressed research-only role-context artifact."""
    shooting_path = Path(shooting_by_dribble_path)
    jump_path = Path(jump_shot_by_dribble_path)
    features = compute_role_context_features(
        pd.read_csv(shooting_path, low_memory=False),
        pd.read_csv(jump_path, low_memory=False),
        seasons=seasons,
    )
    config = {
        "schema_version": ROLE_CONTEXT_SCHEMA_VERSION,
        "seasons": list(seasons),
        "source_sha256": {
            "shooting_by_dribble": sha256_file(shooting_path),
            "jump_shot_by_dribble": sha256_file(jump_path),
        },
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"role_context_features_v1_{identity}"
    output = Path(artifact_root) / "features" / "role_context" / run_id
    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "features.parquet"
    features.to_parquet(features_path, index=False)
    feature_names = [column for column in features.columns if column not in {"PLAYER_ID", "Season"}]
    run = {
        "run_id": run_id,
        "dataset": ROLE_CONTEXT_SCHEMA_VERSION,
        "status": "research_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "rows": int(len(features)),
            "players": int(features["PLAYER_ID"].nunique()),
            "duplicate_keys": int(features.duplicated(["PLAYER_ID", "Season"]).sum()),
            "infinite_values": int(np.isinf(features[feature_names]).sum().sum()),
        },
        "feature_names": feature_names,
        "features_path": str(features_path.resolve()),
        "artifact_path": str(output.resolve()),
        "forbidden_interpretation": "This research-only role context is not a player rating or a production AIO feature block.",
        "caveats": [
            "Source team, age, games, minutes, and supplied percentage fields are intentionally excluded.",
            "Jump-shot source display percentage fields are incomplete before 2025; this artifact derives shares and percentages from FGA and FGM counts only.",
            "Season 2026 is excluded by default because it is incomplete infrastructure data.",
        ],
    }
    if run["quality"]["duplicate_keys"] or run["quality"]["infinite_values"]:
        raise ValueError("Role-context artifact failed output integrity checks.")
    write_json_atomic(run, output / "run.json")
    return run
