from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.behavior_roles import (
    BehaviorRoleConfig,
    ROLE_AFFINITY_FEATURES,
    ROLE_AXIS_FEATURES,
    ROLE_INPUT_FEATURES,
    compute_behavior_roles,
)


def test_behavior_roles_are_stable_and_exclude_forbidden_inputs() -> None:
    rng = np.random.default_rng(4)
    context_features = {
        "all_shot_zero_dribble_share", "all_shot_three_plus_dribble_share",
        "jump_shot_zero_dribble_share", "jump_shot_three_plus_dribble_share",
    }
    annual_rows = []
    context_rows = []
    for season in range(2014, 2022):
        for player in range(240):
            role = player % 8
            values = role * 1.5 + rng.normal(scale=0.08, size=len(ROLE_INPUT_FEATURES))
            base = {"PLAYER_ID": player, "Season": season}
            annual = dict(base)
            context = dict(base)
            for feature, value in zip(ROLE_INPUT_FEATURES, values, strict=True):
                (context if feature in context_features else annual)[feature] = value
            annual["AGE"] = 99
            annual["MIN"] = 4000
            annual_rows.append(annual)
            context_rows.append(context)
    result = compute_behavior_roles(
        pd.DataFrame(annual_rows), pd.DataFrame(context_rows),
        config=BehaviorRoleConfig(
            development_seasons=(2014, 2015, 2016, 2017, 2018),
            minimum_seed_ari=0.80, minimum_adjacent_same_role=0.80,
            minimum_adjacent_axis_cosine=0.80, maximum_cluster_share=0.20,
        ),
    )
    assignments = result["assignments"]
    assert result["metrics"]["passed"]
    assert result["metrics"]["adjacent_same_role_rate"] > 0.95
    assert set(ROLE_AXIS_FEATURES).issubset(assignments)
    assert set(ROLE_AFFINITY_FEATURES).issubset(assignments)
    all_affinities = [f"role_affinity_{index}" for index in range(8)]
    assert np.allclose(assignments[all_affinities].sum(axis=1), 1.0)
    assert "AGE" not in assignments and "MIN" not in assignments


def test_behavior_role_contract_contains_no_value_or_demographic_features() -> None:
    forbidden = ("age", "height", "position", "minutes", "games", "rapm", "spm", "accuracy")
    assert not [
        feature for feature in ROLE_INPUT_FEATURES
        if any(token in feature.lower() for token in forbidden)
    ]
