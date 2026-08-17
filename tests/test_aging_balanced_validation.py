from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.aging_balanced_validation import (
    build_transition_pairs,
    evaluate_aging_balanced_pairs,
)


def test_aging_adjustment_removes_known_directional_change() -> None:
    rows = []
    ages = []
    for season in range(2017, 2025):
        for player in range(80):
            age = 20 + player % 18 + (season - 2017)
            stable = (player - 40) / 20
            age_effect = -0.08 * (age - 27)
            target = stable + age_effect * (season - 2017)
            rows.append(
                {
                    "PLAYER_ID": player, "Season": season, "sample_weight": 1.0,
                    "target_offense": target, "target_defense": target / 2,
                    "target_net": 1.5 * target, "spm_offense": stable,
                    "spm_defense": stable / 2, "spm_net": 1.5 * stable,
                }
            )
            ages.append({"PLAYER_ID": player, "Season": season, "AGE": age})
    predictions = pd.DataFrame(rows)
    age_panel = pd.DataFrame(ages)
    pairs = build_transition_pairs(predictions, age_panel, direction="forward")
    metrics, scored = evaluate_aging_balanced_pairs(
        pairs, minimum_training_origins=3, aging_ridge_alpha=0.01
    )

    net = metrics.loc[metrics["component"].eq("net")].groupby("variant")[
        "weighted_rmse"
    ].mean()
    assert net["aging_adjusted"] < net["raw"]
    assert not scored.empty
    assert set(scored["direction"]) == {"forward"}


def test_reverse_pair_uses_previous_season() -> None:
    predictions = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1], "Season": [2023, 2024], "sample_weight": [1.0, 1.0],
            "target_offense": [2.0, 3.0], "target_defense": [1.0, 1.5],
            "target_net": [3.0, 4.5], "spm_offense": [2.0, 3.0],
            "spm_defense": [1.0, 1.5], "spm_net": [3.0, 4.5],
        }
    )
    ages = pd.DataFrame(
        {"PLAYER_ID": [1, 1], "Season": [2023, 2024], "AGE": [25.0, 26.0]}
    )
    reverse = build_transition_pairs(predictions, ages, direction="reverse")
    assert len(reverse) == 1
    assert reverse.iloc[0]["Season"] == 2024
    assert reverse.iloc[0]["adjacent_target_net"] == 3.0
