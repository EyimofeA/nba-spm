from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file
from nba_impact.models.statistical_direct_net import (
    run_statistical_direct_net_comparison,
)


def test_direct_net_comparison_reuses_identical_component_rows(tmp_path) -> None:
    rng = np.random.default_rng(19)
    feature_rows = []
    target_rows = []
    component_rows = []
    for window_end in range(2016, 2025):
        for player_id in range(1, 31):
            offense = rng.normal()
            defense = rng.normal()
            feature_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "creation": offense,
                    "disruption": defense,
                }
            )
            target_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "Off": offense,
                    "Def": -defense,
                    "Poss_Off": 900 + player_id,
                    "Poss_Def": 910 + player_id,
                }
            )
            if window_end in (2022, 2023, 2024):
                component_rows.append(
                    {
                        "PLAYER_ID": player_id,
                        "Window_End": window_end,
                        "prediction_ridge_offense": offense * 0.8,
                        "prediction_ridge_defense": defense * 0.8,
                        "prediction_histogram_gbm_offense": offense * 0.9,
                    }
                )
    features_path = tmp_path / "features.parquet"
    targets_path = tmp_path / "targets.csv"
    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    pd.DataFrame(target_rows).to_csv(targets_path, index=False)
    component_path = tmp_path / "component"
    component_path.mkdir()
    pd.DataFrame(component_rows).to_parquet(
        component_path / "fold_predictions.parquet", index=False
    )
    (component_path / "run.json").write_text(
        json.dumps(
            {
                "run_id": "component_test",
                "config": {
                    "features": ["creation", "disruption"],
                    "source_hashes": {
                        "features": sha256_file(features_path),
                        "targets": sha256_file(targets_path),
                    },
                },
            }
        )
    )

    run = run_statistical_direct_net_comparison(
        features_path,
        targets_path,
        component_path,
        artifact_root=tmp_path,
        histogram_grid=((0.05, 7, 1.0),),
    )
    output = tmp_path / "models" / "statistical_direct_net" / run["run_id"]
    metrics = pd.read_parquet(output / "fold_metrics.parquet")
    assert run["config"]["parent_component_run"] == "component_test"
    assert set(metrics["variant"]) == {
        "ridge_components",
        "histogram_offense_plus_ridge_defense",
        "direct_histogram_gbm",
    }
    assert len(metrics) == 9
    assert (output / "model_direct_histogram_gbm.joblib").exists()
