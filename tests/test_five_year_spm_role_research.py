from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.five_year_spm_role_research import _summarize, add_zone_shotmaking


def test_zone_shotmaking_rewards_makes_above_same_zone_expectation() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "Window_End": [2024, 2024, 2024],
            "OffPoss": [1000.0, 1000.0, 1000.0],
            "at_rim_fga_p100": [10.0, 10.0, 10.0],
            "at_rim_accuracy": [0.8, 0.5, 0.5],
            "short_mid_fga_p100": [0.0, 0.0, 0.0],
            "short_mid_accuracy": [0.0, 0.0, 0.0],
            "long_mid_fga_p100": [0.0, 0.0, 0.0],
            "long_mid_accuracy": [0.0, 0.0, 0.0],
            "corner3_fga_p100": [0.0, 0.0, 0.0],
            "corner3_accuracy": [0.0, 0.0, 0.0],
            "arc3_fga_p100": [0.0, 0.0, 0.0],
            "arc3_accuracy": [0.0, 0.0, 0.0],
        }
    )
    result = add_zone_shotmaking(frame, prior_attempts=0.0)
    assert result.loc[0, "zone_shotmaking_p100_raw"] > 0
    assert result.loc[1, "zone_shotmaking_p100_raw"] < 0
    assert np.isfinite(result["zone_shotmaking_p100_eb"]).all()


def test_zone_shotmaking_does_not_reward_easy_zone_mix_by_itself() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 4],
            "Window_End": [2024] * 4,
            "OffPoss": [1000.0] * 4,
            "at_rim_fga_p100": [20.0, 20.0, 0.0, 0.0],
            "at_rim_accuracy": [0.7, 0.7, 0.0, 0.0],
            "short_mid_fga_p100": [0.0] * 4,
            "short_mid_accuracy": [0.0] * 4,
            "long_mid_fga_p100": [0.0, 0.0, 20.0, 20.0],
            "long_mid_accuracy": [0.0, 0.0, 0.4, 0.4],
            "corner3_fga_p100": [0.0] * 4,
            "corner3_accuracy": [0.0] * 4,
            "arc3_fga_p100": [0.0] * 4,
            "arc3_accuracy": [0.0] * 4,
        }
    )
    result = add_zone_shotmaking(frame, prior_attempts=0.0)
    assert np.allclose(result["zone_shotmaking_p100_raw"], 0.0)


def test_summary_keeps_candidate_name_after_baseline_join() -> None:
    rows = []
    for variant, pearson in (("baseline", 0.4), ("role_context", 0.5)):
        rows.append(
            {
                "variant": variant,
                "rating_season": 2022,
                "test_season": 2023,
                "side": "net",
                "test_rows": 100,
                "weighted_mae": 1.0,
                "weighted_rmse": 1.2,
                "weighted_pearson": pearson,
                "spearman": pearson,
            }
        )
    _, deltas = _summarize(pd.DataFrame(rows))
    assert deltas.loc[0, "variant"] == "role_context"
    assert np.isclose(deltas.loc[0, "weighted_pearson_delta"], 0.1)
