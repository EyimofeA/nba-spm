from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.statistical_model_comparison import (
    run_statistical_model_comparison,
)


def test_model_comparison_uses_identical_purged_folds(tmp_path) -> None:
    rng = np.random.default_rng(11)
    feature_rows = []
    target_rows = []
    for window_end in range(2016, 2025):
        for player_id in range(1, 36):
            offense = rng.normal()
            defense = rng.normal()
            feature_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "creation": offense + rng.normal(scale=0.15),
                    "disruption": defense + rng.normal(scale=0.15),
                }
            )
            target_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "Off": offense,
                    "Def": -defense,
                    "Poss_Off": 800 + player_id,
                    "Poss_Def": 810 + player_id,
                }
            )
    feature_path = tmp_path / "features.parquet"
    target_path = tmp_path / "targets.csv"
    pd.DataFrame(feature_rows).to_parquet(feature_path, index=False)
    pd.DataFrame(target_rows).to_csv(target_path, index=False)

    run = run_statistical_model_comparison(
        feature_path,
        target_path,
        artifact_root=tmp_path,
        features=("creation", "disruption"),
        ridge_alphas=(10.0,),
        elastic_grid=((0.01, 0.5),),
        histogram_grid=((0.05, 7, 1.0),),
    )
    output = tmp_path / "models" / "statistical_model_comparison" / run["run_id"]
    metrics = pd.read_parquet(output / "fold_metrics.parquet")
    assert run["status"] == "research_comparison"
    assert set(metrics["test_window_end"]) == {2022, 2023, 2024}
    assert set(metrics["train_max_window_end"]) == {2019, 2020, 2021}
    assert set(metrics["family"]) == {"ridge", "elastic_net", "histogram_gbm"}
    assert len(run["metrics"]["summary"]) == 9
    assert (output / "model_histogram_gbm_defense.joblib").exists()
