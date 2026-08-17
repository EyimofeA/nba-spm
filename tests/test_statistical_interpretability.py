from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nba_impact.models.statistical_interpretability import (
    feature_mechanism,
    grouped_permutation_importance,
)


def test_grouped_permutation_finds_signal_and_is_deterministic() -> None:
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        {"PTS_p100": rng.normal(size=300), "noise": rng.normal(size=300)}
    )
    actual = 3.0 * frame["PTS_p100"].to_numpy() + rng.normal(scale=0.05, size=300)
    model = LinearRegression().fit(frame, actual)
    groups = {"shooting": ("PTS_p100",), "other": ("noise",)}

    first = grouped_permutation_importance(
        model, frame, actual, np.ones(300), groups, repeats=5, seed=7
    )
    second = grouped_permutation_importance(
        model, frame, actual, np.ones(300), groups, repeats=5, seed=7
    )

    pd.testing.assert_frame_equal(first, second)
    scores = first.set_index("group")["rmse_delta_mean"]
    assert scores["shooting"] > 2.0
    assert scores["shooting"] > scores["other"]


def test_feature_mechanisms_are_basketball_facing() -> None:
    assert feature_mechanism("bad_pass_turnovers_p100") == "ball_security"
    assert feature_mechanism("deflections_p100_eb") == "defensive_disruption"
    assert feature_mechanism("shot_quality_average") == "shooting_scoring_spacing"
    assert feature_mechanism("potential_assists_p100") == "creation_passing_role"
