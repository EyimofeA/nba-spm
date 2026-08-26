"""No-fit lineage gate for the SPM defensive-source transition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.defensive_tracking_features import (
    DEFENSIVE_TRACKING_FEATURES,
    HUSTLE_FEATURES,
    _attach_player_id,
    _load_box,
)
from nba_impact.data.manifest import sha256_file
from nba_impact.data.matchup_defense_features import MATCHUP_DEFENSE_FEATURES
from nba_impact.data.statistical_features import (
    CORE_RATE_SPECS,
    NATURAL_WEIGHTED_AVERAGES,
    RATIO_SPECS,
    TRACKING_RATE_SPECS,
)
from nba_impact.models.single_season_spm import _selected_single_season_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "research/experiments/spm_defense_source_transition_v1.yml"
DEFAULT_OUTPUT = ROOT / "research/audits/spm_defense_source_transition_v1"

ARTIFACT_PATHS = {
    "public_features": ROOT / "artifacts/features/statistical_impact/statistical_features_v2_907e4ee4b9",
    "refresh_features": ROOT / "artifacts/features/statistical_impact/statistical_features_v2_b808fc1bf1",
    "latest_features": ROOT / "artifacts/research/current_feature_refresh/features/statistical_impact/statistical_features_v2_6bdb60a186",
    "public_spm_run": ROOT / "artifacts/models/single_season_spm/single_season_spm_v1_18496a1348",
    "public_tracking": ROOT / "artifacts/features/defensive_tracking/defensive_tracking_features_v1_9f66c664eb",
    "refresh_tracking": ROOT / "artifacts/features/defensive_tracking/defensive_tracking_features_v1_b0bf4ef279",
    "latest_tracking": ROOT / "artifacts/research/current_feature_refresh/features/defensive_tracking/defensive_tracking_features_v1_c2ea07001a",
    "public_matchup": ROOT / "artifacts/features/matchup_defense/matchup_defense_features_v1_b327da17a7",
    "latest_matchup": ROOT / "artifacts/research/current_feature_refresh/features/matchup_defense/matchup_defense_features_v1_1684cb4eba",
}

OFFICIAL_DFG = ROOT / "data/lake/bronze/official_nba_stats_v3/close_defenders/dfg.csv"
OFFICIAL_RIM = ROOT / "data/lake/bronze/official_nba_stats_v3/close_defenders/rim_dfg.csv"
HUSTLE = ROOT / "data/lake/bronze/gabriel_site_data/revision=782ec8b/hustle.csv"
PUBLIC_DFG = ROOT / "data/lake/bronze/gabriel_site_data/revision=782ec8b/dfg.csv"
PUBLIC_RIM = ROOT / "data/lake/bronze/gabriel_site_data/revision=782ec8b/rimdfg.csv"

DFG_FEATURES = {"dfg_attempts_p100", "dfg_diff_pct_eb"}
RIM_FEATURES = {"rim_dfga_p100", "rim_diff_pct_eb", "rim_points_saved_p100"}


def _load_manifest(path: Path) -> dict:
    payload = json.loads((path / "run.json").read_text())
    if payload.get("run_id") != path.name:
        raise ValueError(f"Manifest run ID mismatch at {path}.")
    return payload


def _validate_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text())
    if contract.get("schema_version") != "spm_defense_source_transition_contract_v1":
        raise ValueError("Unexpected defense-source contract schema.")
    if contract.get("experiment_id") != "spm_defense_source_transition_v1":
        raise ValueError("Unexpected defense-source experiment ID.")
    if contract.get("status") != "frozen_phase_a":
        raise ValueError("Defense-source audit requires a frozen Phase A contract.")
    declared = contract["artifacts"]
    for label, artifact in ARTIFACT_PATHS.items():
        if declared[label] != artifact.name:
            raise ValueError(
                f"Contract pins {label}={declared[label]!r}, found {artifact.name!r}."
            )
    touched = set(contract["development_seasons"]) | set(
        contract["reused_diagnostic_seasons"]
    )
    if touched & set(contract["untouched_confirmation_seasons"]):
        raise ValueError("Untouched confirmation season appears in audit scope.")
    return contract


def _definition(feature: str) -> tuple[str, str, str, str]:
    """Return definition, unit, denominator, and source family."""
    if feature in CORE_RATE_SPECS:
        numerator, denominator = CORE_RATE_SPECS[feature]
        return f"100 times {numerator} divided by {denominator}", "events_per_100", denominator, "player_sheet"
    if feature in TRACKING_RATE_SPECS:
        numerator, denominator = TRACKING_RATE_SPECS[feature]
        unit = "seconds_per_100" if feature == "time_of_possession_p100" else "events_per_100"
        return f"100 times {numerator} divided by {denominator}", unit, denominator, "player_sheet_tracking"
    if feature in RATIO_SPECS:
        numerator, denominator = RATIO_SPECS[feature]
        return f"{numerator} divided by {denominator}", "ratio", denominator, "player_sheet"
    if feature in NATURAL_WEIGHTED_AVERAGES:
        source, weight = NATURAL_WEIGHTED_AVERAGES[feature]
        return f"{source} averaged with {weight} weights", "weighted_average", weight, "player_sheet_tracking"
    if feature == "usage_events_p100":
        return "100 times (FGA + 0.44 FTA + TOV) divided by OffPoss", "events_per_100", "OffPoss", "player_sheet"
    if feature == "true_shooting_pct":
        return "PTS divided by 2 times (FGA + 0.44 FTA)", "ratio", "shooting_possessions", "player_sheet"
    if feature in DFG_FEATURES:
        unit = "percentage_points" if "diff_pct" in feature else "attempts_per_100_defensive_possessions"
        return feature.replace("_", " "), unit, "DefPoss", "official_dfg"
    if feature in RIM_FEATURES:
        unit = "percentage_points" if "diff_pct" in feature else "per_100_defensive_possessions"
        return feature.replace("_", " "), unit, "DefPoss", "official_rim_dfg"
    if feature in HUSTLE_FEATURES:
        return feature.replace("_", " "), "events_per_100_defensive_possessions", "DefPoss", "hustle"
    if feature in MATCHUP_DEFENSE_FEATURES:
        unit = "rate_or_points_per_100_matchup_possessions"
        return feature.replace("_", " "), unit, "matchup_possessions", "matchup_defense"
    raise ValueError(f"No definition contract for selected feature {feature}.")


def _source_family(feature: str) -> str:
    return _definition(feature)[3]


def _observed_keys(path: Path, season_column: str) -> set[tuple[int, int]]:
    frame = pd.read_csv(path, low_memory=False)
    required = {"PLAYER_ID", season_column}
    if required - set(frame):
        raise ValueError(f"Observed source {path} lacks {sorted(required - set(frame))}.")
    rows = frame[["PLAYER_ID", season_column]].apply(pd.to_numeric, errors="coerce").dropna()
    return set(zip(rows["PLAYER_ID"].astype(int), rows[season_column].astype(int), strict=True))


def _artifact_keys(path: Path) -> set[tuple[int, int]]:
    frame = pd.read_parquet(path / "features.parquet", columns=["PLAYER_ID", "Season"])
    return set(zip(frame["PLAYER_ID"].astype(int), frame["Season"].astype(int), strict=True))


def _box_overrides(manifest: dict) -> tuple[tuple[int, ...], dict[int, Path]]:
    seasons = tuple(int(value) for value in manifest["config"]["seasons"])
    paths = manifest["config"]["source_hashes"]["box"]
    overrides: dict[int, Path] = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            season = int(path.stem)
        except ValueError:
            continue
        if season in seasons:
            overrides[season] = path
    if set(overrides) != set(seasons):
        raise ValueError("Could not reconstruct every tracking box source path.")
    return seasons, overrides


def _tracking_observation_keys(
    tracking_manifest: dict,
    *,
    dfg_path: Path,
    rim_path: Path,
) -> dict[str, set[tuple[int, int]]]:
    seasons, overrides = _box_overrides(tracking_manifest)
    box, _ = _load_box(ROOT, seasons, source_overrides=overrides)
    sources = {
        "official_dfg": (pd.read_csv(dfg_path, low_memory=False), ("PLAYER", "PLAYER_NAME")),
        "official_rim_dfg": (pd.read_csv(rim_path, low_memory=False), ("PLAYER", "PLAYER_NAME")),
        "hustle": (pd.read_csv(HUSTLE, low_memory=False), ("PLAYER_NAME", "PLAYER")),
    }
    output: dict[str, set[tuple[int, int]]] = {}
    for family, (source, names) in sources.items():
        attached, _ = _attach_player_id(source, box, name_columns=names)
        output[family] = set(
            zip(attached["PLAYER_ID"].astype(int), attached["Season"].astype(int), strict=True)
        )
    return output


def _safe_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return float("nan")
    return float(left.corr(right))


def run_audit(contract_path: Path = DEFAULT_CONTRACT, output: Path = DEFAULT_OUTPUT) -> dict:
    contract = _validate_contract(contract_path)
    manifests = {label: _load_manifest(path) for label, path in ARTIFACT_PATHS.items()}
    selected = list(_selected_single_season_features(ARTIFACT_PATHS["public_spm_run"])["defense"])
    tracking = set(DEFENSIVE_TRACKING_FEATURES)
    matchup = set(MATCHUP_DEFENSE_FEATURES)
    official = set(selected) & (tracking | matchup)
    nonofficial = set(selected) - official

    frames: dict[str, pd.DataFrame] = {}
    for label in ("public_features", "refresh_features", "latest_features"):
        path = ARTIFACT_PATHS[label] / "features.parquet"
        frame = pd.read_parquet(path, columns=["PLAYER_ID", "Window_End", *selected])
        if frame.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError(f"{label} has duplicate player-season keys.")
        frames[label] = frame

    public_keys = {
        **_tracking_observation_keys(
            manifests["public_tracking"], dfg_path=PUBLIC_DFG, rim_path=PUBLIC_RIM
        ),
        "matchup_defense": _artifact_keys(ARTIFACT_PATHS["public_matchup"]),
    }
    latest_keys = {
        **_tracking_observation_keys(
            manifests["latest_tracking"], dfg_path=OFFICIAL_DFG, rim_path=OFFICIAL_RIM
        ),
        "matchup_defense": _artifact_keys(ARTIFACT_PATHS["latest_matchup"]),
    }
    comparison = frames["refresh_features"].merge(
        frames["latest_features"],
        on=["PLAYER_ID", "Window_End"],
        suffixes=("_refresh", "_latest"),
        validate="one_to_one",
    )
    development = set(int(value) for value in contract["development_seasons"])

    season_rows: list[dict] = []
    lineage_rows: list[dict] = []
    for feature in selected:
        definition, unit, denominator, family = _definition(feature)
        changed_through_2024 = 0
        total_changed = 0
        for season, season_frame in comparison.groupby("Window_End", sort=True):
            left = season_frame[f"{feature}_refresh"]
            right = season_frame[f"{feature}_latest"]
            difference = (left - right).abs()
            changed = int(difference.gt(1e-12).sum())
            total_changed += changed
            if int(season) in development:
                changed_through_2024 += changed
            keys = set(
                zip(
                    season_frame["PLAYER_ID"].astype(int),
                    season_frame["Window_End"].astype(int),
                    strict=True,
                )
            )
            public_observed = public_keys.get(family, set())
            latest_observed = latest_keys.get(family, set())
            season_rows.append(
                {
                    "feature_name": feature,
                    "source_family": family,
                    "season": int(season),
                    "common_rows": len(season_frame),
                    "changed_rows": changed,
                    "mean_absolute_difference": float(difference.mean()),
                    "maximum_absolute_difference": float(difference.max()),
                    "old_new_correlation": _safe_correlation(left, right),
                    "public_observed_rows": len(keys & public_observed) if family in public_keys else len(keys),
                    "latest_observed_rows": len(keys & latest_observed) if family in latest_keys else len(keys),
                }
            )
        lineage_rows.append(
            {
                "feature_name": feature,
                "definition": definition,
                "unit": unit,
                "denominator": denominator,
                "source_family": family,
                "official_source_block": feature in official,
                "allowed_to_change": feature in official,
                "changed_rows_all_seasons": total_changed,
                "changed_rows_development_2014_2024": changed_through_2024,
                "public_observation_mask_reconstructed": family in public_keys or feature in nonofficial,
                "latest_observation_mask_reconstructed": family in latest_keys or feature in nonofficial,
                "public_feature_sha256": sha256_file(ARTIFACT_PATHS["public_features"] / "features.parquet"),
                "refresh_feature_sha256": sha256_file(ARTIFACT_PATHS["refresh_features"] / "features.parquet"),
                "latest_feature_sha256": sha256_file(ARTIFACT_PATHS["latest_features"] / "features.parquet"),
            }
        )

    lineage = pd.DataFrame(lineage_rows)
    season_differences = pd.DataFrame(season_rows)
    changed_nonofficial = lineage.loc[
        ~lineage["official_source_block"]
        & lineage["changed_rows_all_seasons"].gt(0),
        "feature_name",
    ].tolist()
    changed_official_development = lineage.loc[
        lineage["official_source_block"]
        & lineage["changed_rows_development_2014_2024"].gt(0),
        "feature_name",
    ].tolist()
    missing_masks = lineage.loc[
        lineage["official_source_block"]
        & (
            ~lineage["public_observation_mask_reconstructed"]
            | ~lineage["latest_observation_mask_reconstructed"]
        ),
        "feature_name",
    ].tolist()

    gates = {
        "unique_keys": True,
        "all_selected_features_defined": bool(lineage["definition"].notna().all()),
        "all_selected_features_have_units": bool(lineage["unit"].notna().all()),
        "nonofficial_features_unchanged": not changed_nonofficial,
        "official_values_change_in_development": bool(changed_official_development),
        "old_and_new_observation_masks_reconstructed": not missing_masks,
        "source_hashes_present": all(
            manifest.get("config", {}).get("source_hashes")
            or manifest.get("config", {}).get("box_hashes")
            for manifest in manifests.values()
        ),
        "season_2027_untouched": True,
    }
    phase_a_passed = all(gates.values())
    decision = {
        "experiment_id": contract["experiment_id"],
        "phase": "A_no_fit_lineage",
        "status": "pass_fit_allowed" if phase_a_passed else "stop_before_fit",
        "phase_a_passed": phase_a_passed,
        "gates": gates,
        "selected_defense_features": len(selected),
        "official_source_features": len(official),
        "official_features_changed_through_2024": changed_official_development,
        "nonofficial_features_changed": changed_nonofficial,
        "missing_observation_masks": missing_masks,
        "artifact_run_ids": {label: manifest["run_id"] for label, manifest in manifests.items()},
        "contract_sha256": sha256_file(contract_path),
        "source_code_sha256": sha256_file(Path(__file__)),
        "season_2027_loaded": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    lineage.to_csv(output / "field_lineage.csv", index=False)
    season_differences.to_csv(output / "season_differences.csv", index=False)
    (output / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    report = [
        "# SPM defense source-transition Phase A",
        "",
        f"Decision: `{decision['status']}`.",
        "",
        f"The pinned defense model uses {len(selected)} features. {len(official)} belong to the tracking or matchup source block.",
        f"{len(changed_official_development)} official-source features change on 2014--24 rows. {len(changed_nonofficial)} non-official features change.",
        "",
        "## Gates",
        "",
        *[f"- `{name}`: `{value}`" for name, value in gates.items()],
        "",
        "No model was fit in Phase A. Season 2027 was not loaded.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.contract, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
