"""Fixed-development defense feature and role challenger."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.annual_defense_ridge_nested import _ridge
from nba_impact.models.statistical_impact import _metrics
from nba_impact.models.statistical_model_comparison import _fit_model


def _select_variant(metrics: pd.DataFrame) -> str:
    summary = (
        metrics.groupby(["variant", "added_features"], as_index=False)
        .agg(mean_weighted_rmse=("weighted_rmse", "mean"))
        .sort_values(
            ["mean_weighted_rmse", "added_features", "variant"],
            ascending=[True, True, True], kind="stable",
        )
    )
    return str(summary.iloc[0]["variant"])


def run_defense_role_challenger(
    features_path: str | Path,
    targets_path: str | Path,
    frozen_spm_run: str | Path,
    contract_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    """Select on fixed older seasons, then score reused later diagnostics."""
    contract = json.loads(Path(contract_path).read_text())
    frozen_dir = Path(frozen_spm_run)
    frozen = json.loads((frozen_dir / "run.json").read_text())
    features = pd.read_parquet(features_path).rename(columns={"Window_End": "Season"})
    targets = pd.read_parquet(targets_path)
    baseline = tuple(frozen["models"]["defense"]["features"])
    variants = {
        name: tuple(dict.fromkeys((*baseline, *additions)))
        for name, additions in contract["candidate_additions"].items()
    }
    if missing := sorted(set().union(*map(set, variants.values())) - set(features.columns)):
        raise ValueError(f"Defense challenger features are missing: {missing}")
    panel = features.merge(
        targets, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    )
    panel = panel.loc[panel["Season"].ge(contract["training_start_season"])].copy()
    panel["sample_weight"] = np.sqrt(
        panel[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1)
    )
    alpha = float(contract["ridge_alpha"])
    selection_rows = []
    for validation_season in contract["selection_validation_seasons"]:
        train = panel.loc[panel["Season"].lt(validation_season)]
        validation = panel.loc[panel["Season"].eq(validation_season)]
        if min(len(train), len(validation)) == 0:
            raise ValueError(f"Empty defense selection fold {validation_season}.")
        for variant, feature_names in variants.items():
            model = _fit_model(_ridge(alpha), train, feature_names, "target_defense")
            prediction = model.predict(validation.loc[:, feature_names])
            selection_rows.append(
                {
                    "validation_season": validation_season,
                    "variant": variant,
                    "added_features": len(feature_names) - len(baseline),
                    "train_rows": len(train),
                    **_metrics(
                        validation["target_defense"].to_numpy(), prediction,
                        validation["sample_weight"].to_numpy(),
                    ),
                }
            )
    selection_metrics = pd.DataFrame(selection_rows)
    selected_variant = _select_variant(selection_metrics)

    diagnostic_rows = []
    predictions = []
    for test_season in contract["diagnostic_test_seasons"]:
        train = panel.loc[panel["Season"].lt(test_season)]
        test = panel.loc[panel["Season"].eq(test_season)].copy()
        if min(len(train), len(test)) == 0:
            raise ValueError(f"Empty defense diagnostic fold {test_season}.")
        output = test[
            ["PLAYER_ID", "Season", "target_defense", "Poss_Off", "Poss_Def", "sample_weight"]
        ].copy()
        for label, variant in (("baseline", "baseline"), ("challenger", selected_variant)):
            feature_names = variants[variant]
            model = _fit_model(_ridge(alpha), train, feature_names, "target_defense")
            prediction = model.predict(test.loc[:, feature_names])
            output[f"prediction_{label}"] = prediction
            diagnostic_rows.append(
                {
                    "test_season": test_season,
                    "model": label,
                    "source_variant": variant,
                    "feature_count": len(feature_names),
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "target_std": float(test["target_defense"].std()),
                    "prediction_std": float(np.std(prediction)),
                    **_metrics(
                        test["target_defense"].to_numpy(), prediction,
                        test["sample_weight"].to_numpy(),
                    ),
                }
            )
        predictions.append(output)
    diagnostic_metrics = pd.DataFrame(diagnostic_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    paired = diagnostic_metrics.pivot(
        index="test_season", columns="model", values=["weighted_rmse", "correlation"]
    )
    rmse_delta = paired["weighted_rmse"]["challenger"] - paired["weighted_rmse"]["baseline"]
    correlation_delta = paired["correlation"]["challenger"] - paired["correlation"]["baseline"]

    hashes = {
        "features": sha256_file(features_path),
        "targets": sha256_file(targets_path),
        "frozen_run": sha256_file(frozen_dir / "run.json"),
        "contract": sha256_file(contract_path),
        "builder": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"defense_role_challenger_v1_{identity}"
    output_dir = Path(artifact_root) / "models" / "defense_role_challenger" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    selection_metrics.to_parquet(output_dir / "selection_metrics.parquet", index=False)
    diagnostic_metrics.to_parquet(output_dir / "diagnostic_metrics.parquet", index=False)
    prediction_frame.to_parquet(output_dir / "diagnostic_predictions.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "annual_defense_fixed_development_feature_challenger",
        "estimand": contract["estimand"],
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {**contract, "source_hashes": hashes},
        "selection": {
            "selected_variant": selected_variant,
            "selected_added_features": len(variants[selected_variant]) - len(baseline),
        },
        "metrics": {
            "diagnostic_rmse_wins": int(rmse_delta.lt(0).sum()),
            "diagnostic_folds": int(len(rmse_delta)),
            "mean_challenger_minus_baseline_rmse": float(rmse_delta.mean()),
            "mean_challenger_minus_baseline_correlation": float(correlation_delta.mean()),
        },
        "quality": {
            "panel_rows": int(len(panel)),
            "panel_seasons": int(panel["Season"].nunique()),
            "duplicate_prediction_keys": int(
                prediction_frame.duplicated(["PLAYER_ID", "Season"]).sum()
            ),
        },
        "decision": {
            "promote": False,
            "basis": (
                "The feature family and 2022-24 seasons were previously inspected. "
                "This run can reject a weak design but cannot promote a model."
            ),
        },
        "artifact_path": str(output_dir.resolve()),
    }
    write_json_atomic(run, output_dir / "run.json")
    return run
