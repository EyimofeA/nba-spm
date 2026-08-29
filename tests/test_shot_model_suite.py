import numpy as np
import pandas as pd

from nba_impact.models.expected_shot_quality import _feature_frame
from nba_impact.models.shot_model_suite import five_year_shooting_threat


def test_context_features_do_not_use_outcome_or_identity() -> None:
    frame = pd.DataFrame(
        {
            "shot_zone": ["rim"],
            "location_x": [0],
            "location_y": [10],
            "shot_distance_feet": [1],
            "period": [1],
            "regulation_seconds_remaining": [2800],
            "offense_score_diff_before": [0],
            "offense_is_home": [True],
            "shot_value": [2],
            "seconds_since_possession_start": [4.0],
            "is_transition": [False],
            "is_putback": [False],
            "is_second_chance": [False],
            "is_from_turnover": [False],
            "shot_finish": ["drive"],
        }
    )
    matrix, names = _feature_frame(frame, include_possession_context=True)
    assert matrix.shape == (1, len(names))
    assert not {"shot_made", "shooter_id", "assisted"}.intersection(names)


def test_five_year_threat_uses_only_current_and_four_prior_seasons() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1] * 6,
            "PLAYER_NAME": ["One"] * 6,
            "Season": range(2014, 2020),
            "FG3A": [100.0] * 6,
            "FG3M": [40.0] * 6,
            "OffPoss": [1000.0] * 6,
            "context_expected_3p_pct": [0.35] * 6,
            "league_3p_pct": [0.36] * 6,
        }
    )
    result = five_year_shooting_threat(annual)
    row = result.loc[result["Window_End"].eq(2019)].iloc[0]
    assert row["five_year_three_pa"] == 500.0
    assert np.isfinite(row["shooting_threat_p100"])
