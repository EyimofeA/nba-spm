from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.statistical_impact import run_statistical_impact_baseline


def test_statistical_impact_uses_purged_chronological_folds(tmp_path) -> None:
    rng = np.random.default_rng(7)
    feature_rows = []
    target_rows = []
    for window_end in range(2016, 2025):
        for player_id in range(1, 31):
            creation = rng.normal() + player_id / 30
            defense = rng.normal() - player_id / 60
            feature_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "box_creation": creation,
                    "tracking_creation": creation + rng.normal(scale=0.2),
                    "tracking_defense": defense + rng.normal(scale=0.2),
                    "OnOffRtg": creation - defense,
                    "OnDefRtg": -defense,
                }
            )
            target_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "Off": 0.7 * creation + rng.normal(scale=0.3),
                    "Def": -(0.7 * defense + rng.normal(scale=0.3)),
                    "Poss_Off": 1000 + player_id,
                    "Poss_Def": 1000 + player_id,
                }
            )
    feature_path = tmp_path / "features.parquet"
    target_path = tmp_path / "targets.csv"
    pd.DataFrame(feature_rows).to_parquet(feature_path, index=False)
    pd.DataFrame(target_rows).to_csv(target_path, index=False)
    run = run_statistical_impact_baseline(
        feature_path,
        target_path,
        artifact_root=tmp_path,
        feature_sets={
            "box_rates": ("box_creation",),
            "advanced": (
                "box_creation",
                "tracking_creation",
                "tracking_defense",
            ),
            "advanced_plus_onoff": (
                "box_creation",
                "tracking_creation",
                "tracking_defense",
                "OnOffRtg",
                "OnDefRtg",
            ),
        },
        alpha_grid=(1.0, 10.0),
    )
    output = tmp_path / "models" / "statistical_impact" / run["run_id"]
    folds = pd.read_parquet(output / "fold_metrics.parquet")
    assert run["status"] == "research_baseline"
    assert set(folds["test_window_end"]) == {2022, 2023, 2024}
    assert set(folds["train_max_window_end"]) == {2019, 2020, 2021}
    assert len(run["metrics"]["summary"]) == 9
    assert (output / "model_advanced_offense.joblib").exists()
