#!/usr/bin/env python3
"""Inventory every current SPM, mechanism, and predictive-skill feature."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.mechanism_features import (
    DEFENSE_MECHANISM_FEATURES,
    OFFENSE_MECHANISM_FEATURES,
)
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "exhaustive_spm_feature_registry_v1"
WIDE_RUN = ROOT / (
    "artifacts/research/full_feature_2014_2026/features/statistical_impact/"
    "statistical_features_v2_cb03edaf32"
)
COMPLETE_RUN = ROOT / (
    "artifacts/research/complete_feature_coverage/"
    "semantically_complete_spm_features_v1_fdee01ec4e"
)
SKILL_RUN = ROOT / (
    "artifacts/models/predictive_player_skills/"
    "predictive_player_skills_2026_v1_a7eb0386fe"
)
MECHANISM_RUN = ROOT / (
    "artifacts/research/mechanism_feature_panel/"
    "mechanism_feature_panel_v1_9224606a01"
)


def _family(feature: str) -> str:
    name = feature.lower()
    rules = (
        (("oreb", "dreb", "rebound", "boxout"), "rebounding"),
        (("assist", "pass", "creation", "touch", "drive"), "creation_passing"),
        (("turnover", "tov", "travel", "security"), "turnovers"),
        (("rim", "paint", "post", "interior"), "rim_interior"),
        (("fg3", "three", "3pt", "arc3", "corner3", "spacing"), "three_spacing"),
        (("shot", "shoot", "true_shooting", "zts", "fg2", "accuracy"), "shooting"),
        (("deflect", "steal", "stl", "block", "charge", "loose"), "disruption"),
        (("foul", "pf", "pfd", "fta", "ftm"), "fouls_free_throws"),
        (("matchup", "dfg", "suppression"), "matchup_shot_defense"),
        (("role",), "roles"),
        (("latest", "trend", "volatility"), "temporal"),
    )
    for tokens, family in rules:
        if any(token in name for token in tokens):
            return family
    return "general_box_or_rate"


def _source_family(feature: str, wide_manifest: dict) -> str:
    mappings = (
        ("playtype", wide_manifest["playtype_feature_names"]),
        ("defensive_tracking", wide_manifest["defensive_tracking_feature_names"]),
        ("matchup_defense", wide_manifest["matchup_defense_feature_names"]),
        ("assist_quality", wide_manifest["assist_quality_feature_names"]),
        ("player_skill", wide_manifest["player_skill_feature_names"]),
    )
    for family, features in mappings:
        if feature in features:
            return family
    if feature in OFFENSE_MECHANISM_FEATURES or feature in DEFENSE_MECHANISM_FEATURES:
        return "same_season_mechanism"
    if feature.endswith(("_latest", "_trend", "_volatility")):
        return "derived_temporal"
    if feature.endswith("_eb") or "_eb_" in feature:
        return "same_season_empirical_bayes"
    return "gabriel_player_sheet_or_derived"


def _lane(feature: str, *, predictive_skill_only: bool) -> str:
    if feature in {"OnOffRtg", "OnDefRtg"}:
        return "lineup_derived_circular"
    if predictive_skill_only or feature.endswith(("_latest", "_trend", "_volatility")):
        return "predictive_only"
    if "role_" in feature:
        return "descriptive_role"
    return "retrospective_candidate"


def _suggested_side(feature: str, offense: set[str], defense: set[str]) -> str:
    in_offense = feature in offense
    in_defense = feature in defense
    if in_offense and in_defense:
        return "both"
    if in_offense:
        return "offense"
    if in_defense:
        return "defense"
    family = _family(feature)
    return (
        "defense"
        if family in {"disruption", "matchup_shot_defense"}
        else "offense"
        if family in {"creation_passing", "shooting", "three_spacing"}
        else "unspecified"
    )


def main() -> None:
    wide_path = WIDE_RUN / "features.parquet"
    wide = pd.read_parquet(wide_path)
    wide_manifest = json.loads((WIDE_RUN / "run.json").read_text())
    complete_manifest = json.loads((COMPLETE_RUN / "run.json").read_text())
    completion = pd.read_csv(COMPLETE_RUN / "completion_ledger.csv").set_index("feature")
    skills = pd.read_parquet(SKILL_RUN / "skill_definitions.parquet")
    mechanism_registry = pd.read_parquet(MECHANISM_RUN / "feature_registry.parquet")
    keys = {"PLAYER_ID", "Window_End", "OffPoss", "DefPoss"}
    wide_features = [column for column in wide.columns if column not in keys]
    current_offense = set(complete_manifest["feature_contract"]["offense"])
    current_defense = set(complete_manifest["feature_contract"]["defense"])
    offense = set(current_offense)
    defense = set(current_defense)
    offense.update(OFFENSE_MECHANISM_FEATURES)
    defense.update(DEFENSE_MECHANISM_FEATURES)
    universe = list(dict.fromkeys((*wide_features, *mechanism_registry["feature"], *skills["key"])))

    rows = []
    skill_keys = set(skills["key"])
    for feature in universe:
        in_wide = feature in wide.columns
        predictive_skill_only = feature in skill_keys and not in_wide
        lane = _lane(feature, predictive_skill_only=predictive_skill_only)
        coverage = float(wide[feature].notna().mean()) if in_wide else None
        completion_method = (
            str(completion.at[feature, "completion_method"])
            if feature in completion.index
            else "not_in_completed_contract"
        )
        rows.append(
            {
                "feature": feature,
                "suggested_side": _suggested_side(feature, offense, defense),
                "family": _family(feature),
                "source_family": (
                    "predictive_skill_estimate"
                    if predictive_skill_only
                    else _source_family(feature, wide_manifest)
                ),
                "lane": lane,
                "in_wide_295_panel": in_wide,
                "in_completed_contract": (
                    feature in current_offense or feature in current_defense
                ),
                "in_box15_control": feature in BOX_PIPM_STYLE_FEATURES,
                "new_mechanism_feature": feature in set(mechanism_registry["feature"]),
                "observed_nonmissing_fraction_in_wide_panel": coverage,
                "completion_method": completion_method,
                "target_or_future_information_used": False,
                "recommended_action": (
                    "exclude_from_clean_prior"
                    if lane == "lineup_derived_circular"
                    else "keep_in_predictive_lane"
                    if lane == "predictive_only"
                    else "retain_as_frozen_control"
                    if feature in BOX_PIPM_STYLE_FEATURES
                    else "family_screen_then_chronological_selection"
                ),
            }
        )
    registry = pd.DataFrame(rows).sort_values(
        ["lane", "family", "feature"], kind="stable"
    )
    source_paths = {
        "wide_features": wide_path,
        "wide_manifest": WIDE_RUN / "run.json",
        "complete_manifest": COMPLETE_RUN / "run.json",
        "completion_ledger": COMPLETE_RUN / "completion_ledger.csv",
        "skill_definitions": SKILL_RUN / "skill_definitions.parquet",
        "mechanism_registry": MECHANISM_RUN / "feature_registry.parquet",
        "builder": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "source_hashes": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / (
        "artifacts/research/exhaustive_spm_feature_registry/"
        f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    registry.to_parquet(output / "feature_registry.parquet", index=False)
    registry.to_csv(output / "feature_registry.csv", index=False)
    summary = (
        registry.groupby(["lane", "recommended_action"], as_index=False)
        .agg(features=("feature", "nunique"))
        .sort_values(["lane", "recommended_action"])
    )
    summary.to_parquet(output / "summary.parquet", index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_registry_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "unique_features": int(registry["feature"].nunique()),
            "duplicate_feature_rows": int(registry.duplicated("feature").sum()),
            "wide_panel_features": len(wide_features),
            "predictive_skill_definitions": int(len(skills)),
            "new_mechanism_features": int(len(mechanism_registry)),
        },
        "files": {},
        "forbidden_interpretation": (
            "Registry membership is not evidence that a feature improves impact prediction."
        ),
    }
    for name in ("feature_registry.parquet", "feature_registry.csv", "summary.parquet"):
        path = output / name
        run["files"][name] = {"path": name, "sha256": sha256_file(path)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
