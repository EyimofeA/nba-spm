#!/usr/bin/env python3
"""Build the SPM feature registry and low-cost falsification controls."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "spm_feature_registry_controls_v1"
CONTRACT = ROOT / "research/experiments/spm_feature_registry_controls_v1.yml"
FEATURE_RUN = ROOT / "artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_8be676bd0f"
LADDER_RUN = ROOT / "artifacts/research/final_box_feature_ladder/final_box_feature_ladder_v1_8bb26f12e7"
TARGETS = ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
RATING_SEASONS = tuple(range(2021, 2027))
SEEDS = tuple(range(1, 21))


def _units(feature: str) -> str:
    if feature.startswith("has_"):
        return "binary_indicator"
    if feature.endswith("_p100") or feature.endswith("_eb_p100"):
        return "events_per_100_possessions"
    if feature.endswith("_pct") or "diff_pct" in feature or feature == "zts_pct_points":
        return "percentage_or_percentage_points"
    if feature.endswith("_share") or feature.endswith("_frequency") or feature.endswith("_eb"):
        return "rate_or_stabilized_rate"
    if "per_touch" in feature or "per_drive" in feature:
        return "events_per_opportunity"
    if feature.endswith("_relative"):
        return "season_relative_value"
    return "engineered_index_or_rate"


def _registry() -> pd.DataFrame:
    manifest = json.loads((FEATURE_RUN / "run.json").read_text())
    ledger = pd.read_csv(FEATURE_RUN / "completion_ledger.csv").set_index("feature")
    low = pd.read_csv(FEATURE_RUN / "features_below_99pct_completion.csv")
    low = low.loc[low["panel"].eq("annual")].set_index("feature")
    candidates = pd.read_parquet(LADDER_RUN / "candidate_features.parquet")
    candidate_order = json.loads((LADDER_RUN / "run.json").read_text())["config"]["candidate_order"]
    order = {candidate: index for index, candidate in enumerate(candidate_order)}
    rows = []
    for side in ("offense", "defense"):
        for feature in manifest["feature_contract"][side]:
            feature_candidates = candidates.loc[
                candidates["side"].eq(side) & candidates["feature"].eq(feature), "candidate"
            ]
            first_candidate = min(feature_candidates, key=order.get)
            source_family = ledger.at[feature, "source_family"] if feature in ledger.index else "derived_completion"
            completion_method = ledger.at[feature, "completion_method"] if feature in ledger.index else "observed_or_derived"
            coverage = float(low.at[feature, "coverage_fraction"]) if feature in low.index else 1.0
            rows.append(
                {
                    "registry_version": "spm_feature_registry_v1",
                    "feature": feature,
                    "side": side,
                    "units": _units(feature),
                    "source_family": source_family,
                    "completion_method": completion_method,
                    "observed_annual_coverage": coverage,
                    "first_ladder_candidate": first_candidate,
                    "frozen_Box15": feature in BOX_PIPM_STYLE_FEATURES,
                    "source_availability_indicator": feature.startswith("has_"),
                    "same_window_label_sharing_risk": True,
                    "publication_status": "frozen_control" if feature in BOX_PIPM_STYLE_FEATURES else "researched_not_promoted",
                }
            )
    return pd.DataFrame(rows)


def _panel() -> pd.DataFrame:
    features = pd.read_parquet(FEATURE_RUN / "five_year_features.parquet")
    targets = pd.read_parquet(TARGETS)
    fields = ["PLAYER_ID", "Window_End", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]
    panel = features.merge(targets[fields], on=["PLAYER_ID", "Window_End"], how="inner", validate="one_to_one")
    panel = panel.loc[panel["Window_End"].between(2018, 2026)].copy()
    panel["sample_weight"] = np.sqrt(np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1))
    if panel["Window_End"].max() >= 2027:
        raise ValueError("Season 2027 entered feature controls.")
    return panel.sort_values(["Window_End", "PLAYER_ID"], kind="stable").reset_index(drop=True)


def _negative_controls(panel: pd.DataFrame) -> pd.DataFrame:
    alpha = pd.read_parquet(LADDER_RUN / "alpha_selection.parquet")
    alpha = alpha.loc[alpha["candidate"].eq("box_15")].set_index(["rating_season", "side"])
    rows = []
    for seed in SEEDS:
        work = panel.copy()
        work["negative_control_noise"] = np.random.default_rng(20260829 + seed).normal(size=len(work))
        for season in RATING_SEASONS:
            train = work.loc[work["Window_End"].lt(season)]
            test = work.loc[work["Window_End"].eq(season)]
            for side in ("offense", "defense"):
                target = f"target_{side}"
                selected_alpha = float(alpha.at[(season, side), "selected_alpha"])
                control = _fit(train, BOX_PIPM_STYLE_FEATURES, target, selected_alpha).predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
                noise_features = (*BOX_PIPM_STYLE_FEATURES, "negative_control_noise")
                challenger = _fit(train, noise_features, target, selected_alpha).predict(test.loc[:, noise_features])
                weight = test["sample_weight"].to_numpy(dtype=float)
                actual = test[target].to_numpy(dtype=float)
                control_mse = float(np.average((actual - control) ** 2, weights=weight))
                challenger_mse = float(np.average((actual - challenger) ** 2, weights=weight))
                rows.append(
                    {
                        "seed": seed,
                        "rating_season": season,
                        "component": side,
                        "control_mse": control_mse,
                        "negative_control_mse": challenger_mse,
                        "mse_delta": challenger_mse - control_mse,
                    }
                )
    return pd.DataFrame(rows)


def _label_sharing_audit() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target = pd.read_parquet(LADDER_RUN / "target_metrics.parquet")
    target = (
        target.loc[target["component"].eq("net")]
        .groupby("candidate", as_index=False)
        .agg(mean_target_rmse=("weighted_rmse", "mean"))
    )
    downstream = pd.read_parquet(LADDER_RUN / "summary.parquet")
    downstream = downstream.loc[downstream["candidate"].str.endswith("_aio")].copy()
    downstream["candidate"] = downstream["candidate"].str.replace(
        r"_aio$", "", regex=True
    )
    joined = target.merge(
        downstream[["candidate", "mean_margin_rmse", "mean_margin_correlation"]],
        on="candidate",
        how="inner",
        validate="one_to_one",
    )
    base_row = joined.loc[joined["candidate"].eq("box_15")].iloc[0]
    joined["target_rmse_delta_vs_Box15"] = joined["mean_target_rmse"] - float(base_row["mean_target_rmse"])
    joined["downstream_mse_delta_vs_Box15"] = joined["mean_margin_rmse"] ** 2 - float(base_row["mean_margin_rmse"]) ** 2
    candidates = joined.loc[~joined["candidate"].isin(["box_15", "completed_full_ridge_ceiling"])]
    relationship = {
        "pearson_target_delta_vs_downstream_delta": float(candidates["target_rmse_delta_vs_Box15"].corr(candidates["downstream_mse_delta_vs_Box15"], method="pearson")),
        "spearman_target_delta_vs_downstream_delta": float(candidates["target_rmse_delta_vs_Box15"].corr(candidates["downstream_mse_delta_vs_Box15"], method="spearman")),
        "candidates": len(candidates),
    }
    folds = pd.read_parquet(LADDER_RUN / "fold_metrics.parquet")
    folds = folds.loc[folds["candidate"].str.endswith("_aio") & ~folds["candidate"].eq("zero_prior_rapm")]
    winners = (
        folds.sort_values(["rating_season", "margin_rmse", "candidate"], kind="stable")
        .groupby("rating_season", as_index=False)
        .first()[["rating_season", "test_season", "candidate", "margin_rmse", "margin_correlation"]]
    )
    return joined, winners, relationship


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    registry = _registry()
    panel = _panel()
    controls = _negative_controls(panel)
    control_summary = (
        controls.groupby(["seed", "component"], as_index=False)
        .agg(mean_mse_delta=("mse_delta", "mean"), folds=("rating_season", "nunique"))
    )
    aggregate_controls = (
        control_summary.groupby("component", as_index=False)
        .agg(
            mean_mse_delta=("mean_mse_delta", "mean"),
            median_mse_delta=("mean_mse_delta", "median"),
            seeds_with_lower_mse=("mean_mse_delta", lambda values: int(np.sum(np.asarray(values) < 0))),
            seeds=("seed", "nunique"),
        )
    )
    sharing, winners, relationship = _label_sharing_audit()
    sources = {
        "contract": CONTRACT,
        "feature_manifest": FEATURE_RUN / "run.json",
        "completion_ledger": FEATURE_RUN / "completion_ledger.csv",
        "features": FEATURE_RUN / "five_year_features.parquet",
        "targets": TARGETS,
        "ladder_manifest": LADDER_RUN / "run.json",
        "runner": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "negative_control_seeds": list(SEEDS),
        "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in sources.items()},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/spm_feature_registry_controls" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "feature_registry.parquet": registry,
        "negative_control_folds.parquet": controls,
        "negative_control_summary.parquet": aggregate_controls,
        "label_sharing_audit.parquet": sharing,
        "fold_winners.parquet": winners,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_controls_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "registry_rows": len(registry),
            "registry_unique_side_features": int(registry[["side", "feature"]].drop_duplicates().shape[0]),
            "negative_control_seeds": len(SEEDS),
            "season_2027_loaded": False,
        },
        "results": {
            "negative_controls": aggregate_controls.to_dict(orient="records"),
            "label_sharing_relationship": relationship,
            "downstream_fold_winners": winners.to_dict(orient="records"),
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {"path": name, "sha256": sha256_file(output / name), "rows": len(frame)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(aggregate_controls.to_string(index=False))
    print(json.dumps(relationship, indent=2))
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
