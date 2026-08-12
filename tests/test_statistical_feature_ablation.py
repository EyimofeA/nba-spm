from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.statistical_feature_ablation import (
    default_feature_groups,
    fit_optimized_statistical_aio,
    run_statistical_feature_ablation,
)


def test_feature_ablation_keeps_models_and_folds_fixed(tmp_path) -> None:
    rng = np.random.default_rng(23)
    feature_rows = []
    target_rows = []
    for window_end in range(2016, 2025):
        for player_id in range(1, 31):
            offense = rng.normal()
            defense = rng.normal()
            feature_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "PTS_p100": offense,
                    "drives_p100": offense + rng.normal(scale=0.2),
                    "rebound_chances_p100": defense,
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
    features_path = tmp_path / "features.parquet"
    targets_path = tmp_path / "targets.csv"
    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    pd.DataFrame(target_rows).to_csv(targets_path, index=False)
    run = run_statistical_feature_ablation(
        features_path,
        targets_path,
        artifact_root=tmp_path,
    )
    output = tmp_path / "models" / "statistical_feature_ablation" / run["run_id"]
    metrics = pd.read_parquet(output / "component_metrics.parquet")
    assert set(metrics["test_window_end"]) == {2022, 2023, 2024}
    assert set(metrics["target"]) == {"offense", "defense"}
    assert run["config"]["offense_model"]["family"] == "histogram_gbm"
    assert run["config"]["defense_model"]["family"] == "ridge"
    assert default_feature_groups(("PTS_p100", "drives_p100")) == {
        "core_box": ("PTS_p100",),
        "creation_role": ("drives_p100",),
    }
    fitted = fit_optimized_statistical_aio(
        features_path,
        targets_path,
        output,
        artifact_root=tmp_path,
    )
    fitted_output = tmp_path / "models" / "statistical_aio" / fitted["run_id"]
    assert fitted["status"] == "research_challenger"
    assert (fitted_output / "model_offense.joblib").exists()
    assert (fitted_output / "model_defense.joblib").exists()
