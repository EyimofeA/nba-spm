from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.dynamic_state_space import build_causal_state_space_filter


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 2, 2, 2],
            "Season": [2018, 2019, 2020, 2018, 2019, 2020],
            "target_offense": [1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
            "target_defense": [0.5, 0.0, -0.5, -0.5, 0.0, 0.5],
            "Poss_Off": [1000.0] * 6,
            "Poss_Def": [1000.0] * 6,
        }
    )
    targets["target_net"] = targets["target_offense"] + targets["target_defense"]
    variance = targets[["PLAYER_ID", "Season"]].copy()
    variance["observation_variance_offense"] = 0.25
    variance["observation_variance_defense"] = 0.25
    return targets, variance


def test_state_space_is_causal_and_preserves_component_identity() -> None:
    targets, variance = _inputs()
    first = build_causal_state_space_filter(targets, variance, phi=0.8, process_sd=0.5)
    changed = targets.copy()
    changed.loc[changed["Season"].eq(2020), "target_offense"] += 1000.0
    changed["target_net"] = changed["target_offense"] + changed["target_defense"]
    revised = build_causal_state_space_filter(changed, variance, phi=0.8, process_sd=0.5)
    pd.testing.assert_frame_equal(
        first.loc[first["Season"].lt(2020)].reset_index(drop=True),
        revised.loc[revised["Season"].lt(2020)].reset_index(drop=True),
    )
    assert np.allclose(
        first["filtered_net"], first["filtered_offense"] + first["filtered_defense"]
    )
    assert first[["filtered_variance_offense", "filtered_variance_defense"]].gt(0).all().all()


def test_state_space_rejects_missing_observation_variance() -> None:
    targets, variance = _inputs()
    with np.testing.assert_raises_regex(ValueError, "Every annual RAPM target"):
        build_causal_state_space_filter(targets, variance.iloc[:-1], phi=0.8, process_sd=0.5)
