#!/usr/bin/env python3
"""Build a target-free measurement atlas for completed SPM features."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from nba_impact.data.manifest import sha256_file, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "spm_feature_atlas_v1"
CONTRACT = ROOT / "research/experiments/spm_feature_atlas_v1.yml"
FEATURE_PANEL = ROOT / (
    "artifacts/research/complete_feature_coverage/"
    "semantically_complete_spm_features_v1_fdee01ec4e/five_year_features.parquet"
)
MECHANISM_PANEL = ROOT / (
    "artifacts/research/mechanism_feature_panel/"
    "mechanism_feature_panel_v1_9224606a01/five_year_features.parquet"
)
REGISTRY = ROOT / (
    "artifacts/research/exhaustive_spm_feature_registry/"
    "exhaustive_spm_feature_registry_v1_1ea059390e/feature_registry.parquet"
)


def _stability(panel: pd.DataFrame, feature: str) -> tuple[float, int, int]:
    rows = []
    pairs = 0
    for season in range(int(panel["Window_End"].min()), int(panel["Window_End"].max())):
        left = panel.loc[panel["Window_End"].eq(season), ["PLAYER_ID", feature]]
        right = panel.loc[
            panel["Window_End"].eq(season + 1), ["PLAYER_ID", feature]
        ]
        joined = left.merge(right, on="PLAYER_ID", suffixes=("_left", "_right"))
        if len(joined) < 3:
            continue
        correlation = joined[f"{feature}_left"].corr(joined[f"{feature}_right"])
        if np.isfinite(correlation):
            rows.append(float(correlation))
            pairs += len(joined)
    return (float(np.median(rows)) if rows else np.nan, len(rows), pairs)


def _source_shift(early: pd.Series, late: pd.Series) -> tuple[float, float]:
    pooled_sd = float(np.sqrt((early.var(ddof=1) + late.var(ddof=1)) / 2.0))
    smd = float((late.mean() - early.mean()) / pooled_sd) if pooled_sd > 0 else 0.0
    values = pd.concat([early, late], ignore_index=True).to_numpy(dtype=float)
    labels = np.concatenate([np.zeros(len(early)), np.ones(len(late))])
    auc = float(roc_auc_score(labels, values)) if np.unique(values).size > 1 else 0.5
    return smd, max(auc, 1.0 - auc)


def _redundancy(
    panel: pd.DataFrame, features: list[str], threshold: float
) -> pd.DataFrame:
    early = panel.loc[panel["Window_End"].le(2021), features].corr()
    late = panel.loc[panel["Window_End"].ge(2022), features].corr()
    rows = []
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            early_value = float(early.loc[left, right])
            late_value = float(late.loc[left, right])
            if max(abs(early_value), abs(late_value)) < threshold:
                continue
            rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "early_correlation": early_value,
                    "late_correlation": late_value,
                    "stable_redundancy": bool(
                        abs(early_value) >= threshold and abs(late_value) >= threshold
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("Experiment contract does not match the builder.")
    panel = pd.read_parquet(FEATURE_PANEL)
    mechanism = pd.read_parquet(MECHANISM_PANEL)
    registry = pd.read_parquet(REGISTRY)
    panel = panel.merge(mechanism, on=["PLAYER_ID", "Window_End"], how="left")
    expected_seasons = set(contract["inputs"]["seasons"])
    if set(panel["Window_End"].unique()) != expected_seasons:
        raise ValueError("Feature panel seasons differ from the frozen atlas contract.")

    excluded = {"PLAYER_ID", "Window_End", "OffPoss", "DefPoss"}
    features = sorted(
        feature
        for feature in registry["feature"].unique()
        if feature in panel.columns and feature not in excluded
    )
    if panel[features].isna().any().any() or not np.isfinite(panel[features]).all().all():
        raise ValueError("The completed atlas requires finite values for every field.")

    registry_by_feature = registry.drop_duplicates("feature").set_index("feature")
    early_mask = panel["Window_End"].le(2021)
    rows = []
    for feature in features:
        values = panel[feature].astype(float)
        stability, transitions, adjacent_pairs = _stability(panel, feature)
        smd, auc = _source_shift(values.loc[early_mask], values.loc[~early_mask])
        metadata = registry_by_feature.loc[feature]
        rows.append(
            {
                "feature": feature,
                "suggested_side": metadata["suggested_side"],
                "family": metadata["family"],
                "source_family": metadata["source_family"],
                "lane": metadata["lane"],
                "in_box15_control": bool(metadata["in_box15_control"]),
                "new_mechanism_feature": bool(metadata["new_mechanism_feature"]),
                "rows": len(values),
                "coverage": float(values.notna().mean()),
                "minimum": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "median": float(values.median()),
                "p99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
                "standard_deviation": float(values.std(ddof=1)),
                "adjacent_season_median_correlation": stability,
                "adjacent_transitions": transitions,
                "adjacent_player_pairs": adjacent_pairs,
                "early_late_smd": smd,
                "early_late_univariate_auc": auc,
                "source_shift_flag": bool(
                    auc >= float(contract["thresholds"]["source_shift_auc"])
                    or abs(smd)
                    >= float(contract["thresholds"]["source_shift_absolute_smd"])
                ),
            }
        )
    atlas = pd.DataFrame(rows).sort_values(
        ["source_shift_flag", "early_late_univariate_auc", "feature"],
        ascending=[False, False, True],
    )
    redundancy = _redundancy(
        panel,
        features,
        float(contract["thresholds"]["stable_redundancy_absolute_correlation"]),
    )
    stable_pairs = set(
        redundancy.loc[redundancy["stable_redundancy"], "feature_left"]
    ) | set(redundancy.loc[redundancy["stable_redundancy"], "feature_right"])
    atlas["stable_redundancy_member"] = atlas["feature"].isin(stable_pairs)
    atlas["screen_status"] = np.select(
        [
            atlas["lane"].isin(
                ["lineup_derived_circular", "descriptive_role", "predictive_only"]
            ),
            atlas["source_shift_flag"],
            atlas["stable_redundancy_member"],
        ],
        ["exclude_from_retrospective_spm", "audit_source_shift", "prune_within_fold"],
        default="eligible_for_fold_internal_screen",
    )

    source_paths = {
        "contract": CONTRACT,
        "feature_panel": FEATURE_PANEL,
        "mechanism_panel": MECHANISM_PANEL,
        "registry": REGISTRY,
        "builder": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "features": len(features),
        "seasons": sorted(expected_seasons),
        "thresholds": contract["thresholds"],
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / "artifacts/research/spm_feature_atlas" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "feature_atlas.parquet": atlas,
        "feature_atlas.csv": atlas,
        "redundancy_pairs.parquet": redundancy,
    }
    for name, frame in outputs.items():
        if name.endswith(".csv"):
            frame.to_csv(output / name, index=False)
        else:
            frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "target_free_measurement_audit_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "results": {
            "features": len(atlas),
            "source_shift_flags": int(atlas["source_shift_flag"].sum()),
            "stable_redundancy_pairs": int(redundancy["stable_redundancy"].sum()),
            "eligible_features": int(
                atlas["screen_status"].eq("eligible_for_fold_internal_screen").sum()
            ),
        },
        "quality": {
            "missing_values": int(panel[features].isna().sum().sum()),
            "season_2027_loaded": False,
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "rows": len(frame),
            "sha256": sha256_file(output / name),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(json.dumps(run["results"], indent=2))
    print("\nLargest source shifts")
    print(
        atlas.sort_values("early_late_univariate_auc", ascending=False)
        .head(20)[
            [
                "feature",
                "family",
                "adjacent_season_median_correlation",
                "early_late_smd",
                "early_late_univariate_auc",
                "screen_status",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
