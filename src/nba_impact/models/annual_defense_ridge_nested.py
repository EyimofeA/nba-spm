"""Nested forward selection of annual defensive SPM ridge regularization."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_impact import _metrics
from nba_impact.models.statistical_model_comparison import _fit_model


def _ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _select_alpha(inner_metrics: pd.DataFrame) -> float:
    summary = (
        inner_metrics.groupby("alpha", as_index=False)
        .agg(mean_weighted_rmse=("weighted_rmse", "mean"))
        .sort_values(
            ["mean_weighted_rmse", "alpha"], ascending=[True, False], kind="stable"
        )
    )
    return float(summary.iloc[0]["alpha"])


def run_annual_defense_ridge_nested(
    features_path: str | Path,
    targets_path: str | Path,
    frozen_spm_run: str | Path,
    contract_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    """Select ridge alpha on past seasons and evaluate on forward outer seasons."""
    contract = json.loads(Path(contract_path).read_text())
    frozen_dir = Path(frozen_spm_run)
    frozen = json.loads((frozen_dir / "run.json").read_text())
    features = pd.read_parquet(features_path).rename(columns={"Window_End": "Season"})
    targets = pd.read_parquet(targets_path)
    feature_names = tuple(frozen["models"]["defense"]["features"])
    missing = sorted(set(feature_names) - set(features.columns))
    if missing:
        raise ValueError(f"Frozen defensive features are missing: {missing}")
    panel = features.merge(
        targets,
        on=["PLAYER_ID", "Season"],
        how="inner",
        validate="one_to_one",
    )
    panel = panel.loc[panel["Season"].ge(contract["training_start_season"])].copy()
    panel["sample_weight"] = np.sqrt(
        panel[["Poss_Off", "Poss_Def"]].min(axis=1).clip(lower=1)
    )

    candidates = tuple(float(value) for value in contract["ridge_alphas"])
    baseline_alpha = float(contract["baseline_alpha"])
    inner_rows = []
    outer_rows = []
    prediction_rows = []
    for outer_season in contract["outer_test_seasons"]:
        inner_seasons = (outer_season - 2, outer_season - 1)
        fold_inner_rows = []
        for inner_season in inner_seasons:
            train = panel.loc[panel["Season"].lt(inner_season)]
            validation = panel.loc[panel["Season"].eq(inner_season)]
            if min(len(train), len(validation)) == 0:
                raise ValueError(
                    f"Inner fold {inner_season} for outer {outer_season} is empty."
                )
            for alpha in candidates:
                model = _fit_model(
                    _ridge(alpha), train, feature_names, "target_defense"
                )
                prediction = model.predict(validation.loc[:, feature_names])
                row = {
                    "outer_test_season": outer_season,
                    "inner_validation_season": inner_season,
                    "alpha": alpha,
                    "train_seasons": int(train["Season"].nunique()),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    **_metrics(
                        validation["target_defense"].to_numpy(),
                        prediction,
                        validation["sample_weight"].to_numpy(),
                    ),
                }
                inner_rows.append(row)
                fold_inner_rows.append(row)
        selected_alpha = _select_alpha(pd.DataFrame(fold_inner_rows))
        outer_train = panel.loc[panel["Season"].lt(outer_season)]
        outer_test = panel.loc[panel["Season"].eq(outer_season)].copy()
        if min(len(outer_train), len(outer_test)) == 0:
            raise ValueError(f"Outer fold {outer_season} is empty.")
        fold_predictions = outer_test[
            [
                "PLAYER_ID",
                "Season",
                "target_defense",
                "Poss_Off",
                "Poss_Def",
                "sample_weight",
            ]
        ].copy()
        for variant, alpha in (
            ("nested_selected", selected_alpha),
            ("fixed_3000", baseline_alpha),
        ):
            model = _fit_model(
                _ridge(alpha), outer_train, feature_names, "target_defense"
            )
            prediction = model.predict(outer_test.loc[:, feature_names])
            fold_predictions[f"prediction_{variant}"] = prediction
            outer_rows.append(
                {
                    "test_season": outer_season,
                    "variant": variant,
                    "alpha": alpha,
                    "train_seasons": int(outer_train["Season"].nunique()),
                    "train_rows": len(outer_train),
                    "test_rows": len(outer_test),
                    "target_std": float(outer_test["target_defense"].std()),
                    "prediction_std": float(np.std(prediction)),
                    **_metrics(
                        outer_test["target_defense"].to_numpy(),
                        prediction,
                        outer_test["sample_weight"].to_numpy(),
                    ),
                }
            )
        prediction_rows.append(fold_predictions)

    inner_metrics = pd.DataFrame(inner_rows)
    outer_metrics = pd.DataFrame(outer_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    paired = outer_metrics.pivot(
        index="test_season", columns="variant", values=["weighted_rmse", "correlation"]
    )
    rmse_delta = paired["weighted_rmse"]["nested_selected"] - paired[
        "weighted_rmse"
    ]["fixed_3000"]
    correlation_delta = paired["correlation"]["nested_selected"] - paired[
        "correlation"
    ]["fixed_3000"]
    rmse_wins = int((rmse_delta < 0).sum())
    mean_rmse_delta = float(rmse_delta.mean())
    mean_correlation_delta = float(correlation_delta.mean())
    gate = contract["promotion_gate"]
    passed = (
        rmse_wins >= int(gate["minimum_outer_rmse_wins"])
        and (mean_rmse_delta < 0 if gate["require_lower_mean_outer_rmse"] else True)
        and (
            mean_correlation_delta >= 0
            if gate["require_noninferior_mean_outer_correlation"]
            else True
        )
    )
    selected_counts = Counter(
        outer_metrics.loc[
            outer_metrics["variant"].eq("nested_selected"), "alpha"
        ].tolist()
    )

    source_hashes = {
        "features": sha256_file(features_path),
        "targets": sha256_file(targets_path),
        "frozen_run": sha256_file(frozen_dir / "run.json"),
        "contract": sha256_file(contract_path),
        "builder": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"annual_defense_ridge_nested_v1_{identity}"
    output = Path(artifact_root) / "models" / "annual_defense_ridge_nested" / run_id
    output.mkdir(parents=True, exist_ok=False)
    inner_metrics.to_parquet(output / "inner_metrics.parquet", index=False)
    outer_metrics.to_parquet(output / "outer_metrics.parquet", index=False)
    predictions.to_parquet(output / "outer_predictions.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "annual_defensive_spm_nested_ridge",
        "estimand": contract["estimand"],
        "status": "promotion_gate_passed" if passed else "promotion_gate_failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {**contract, "source_hashes": source_hashes},
        "quality": {
            "panel_rows": len(panel),
            "panel_seasons": int(panel["Season"].nunique()),
            "outer_folds": int(outer_metrics["test_season"].nunique()),
            "inner_fold_candidate_rows": len(inner_metrics),
            "duplicate_prediction_keys": int(
                predictions.duplicated(["PLAYER_ID", "Season"]).sum()
            ),
        },
        "metrics": {
            "outer_rmse_wins": rmse_wins,
            "outer_folds": int(len(rmse_delta)),
            "mean_selected_minus_fixed_rmse": mean_rmse_delta,
            "mean_selected_minus_fixed_correlation": mean_correlation_delta,
            "selected_alpha_counts": {
                str(alpha): count for alpha, count in sorted(selected_counts.items())
            },
        },
        "decision": {
            "promote_nested_selection": passed,
            "basis": "Predeclared forward outer-fold gate; 2025 was not used.",
        },
        "artifact_path": str(output.resolve()),
        "caveats": [
            "The target is noisy one-season RAPM, not ground truth.",
            "This experiment changes regularization only; it does not test new features.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
