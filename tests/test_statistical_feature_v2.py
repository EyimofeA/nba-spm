from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from nba_impact.models.statistical_feature_v2 import (
    candidate_feature_blocks,
    run_statistical_feature_v2_comparison,
)


def test_public_metrics_are_a_separate_offense_block() -> None:
    blocks = candidate_feature_blocks(
        (
            "fg3_pct_eb",
            "shooting_proficiency_2017_eb",
            "box_creation_2017_eb_p100",
            "behavioral_passer_score_v1",
        )
    )
    assert blocks["offense"]["stabilized_ratios"] == ("fg3_pct_eb",)
    assert blocks["offense"]["public_basketball_metrics"] == (
        "shooting_proficiency_2017_eb",
        "box_creation_2017_eb_p100",
        "behavioral_passer_score_v1",
    )


def test_v2_comparison_selects_signal_on_discovery_and_confirms(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "nba_impact.models.statistical_feature_v2._frozen_model",
        lambda target: Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("model", Ridge(alpha=1.0))]
        ),
    )
    rng = np.random.default_rng(20260809)
    feature_rows = []
    target_rows = []
    for window_end in range(2016, 2025):
        for player_id in range(50):
            offense_signal = rng.normal()
            defense_signal = rng.normal()
            feature_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "OffPoss": 1000.0,
                    "DefPoss": 1000.0,
                    "x_base": rng.normal(),
                    "PTS_p100_latest": offense_signal,
                    "BLK_p100_latest": defense_signal,
                }
            )
            target_rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Window_End": window_end,
                    "Off": 2.0 * offense_signal + rng.normal(scale=0.05),
                    "Def": -(1.5 * defense_signal + rng.normal(scale=0.05)),
                    "Poss_Off": 1000.0,
                    "Poss_Def": 1000.0,
                }
            )
    features_path = tmp_path / "features.parquet"
    targets_path = tmp_path / "targets.csv"
    pd.DataFrame(feature_rows).to_parquet(features_path, index=False)
    pd.DataFrame(target_rows).to_csv(targets_path, index=False)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "run.json").write_text(
        json.dumps(
            {
                "models": {
                    "offense": {"features": ["x_base"]},
                    "defense": {"features": ["x_base"]},
                }
            }
        )
    )
    run = run_statistical_feature_v2_comparison(
        features_path,
        targets_path,
        baseline,
        artifact_root=tmp_path,
    )
    assert "recent_level" in run["selected_blocks"]["offense"]
    assert "recent_level" in run["selected_blocks"]["defense"]
    assert run["confirmed"] is True
    assert run["status"] == "research_challenger"
    assert Path(run["models"]["offense"]["path"]).exists()
