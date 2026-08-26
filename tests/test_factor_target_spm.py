from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.factor_target_spm import (
    CONTEXT_FEATURES,
    FACTORS,
    INDIVIDUAL_FEATURES,
    SIDES,
    add_leave_one_out_teammate_context,
)


def test_every_factor_side_has_small_related_feature_contract() -> None:
    assert set(INDIVIDUAL_FEATURES) == {
        (factor, side) for factor in FACTORS for side in SIDES
    }
    assert set(CONTEXT_FEATURES) == set(INDIVIDUAL_FEATURES)
    assert all(2 <= len(values) <= 5 for values in INDIVIDUAL_FEATURES.values())
    assert all(2 <= len(values) <= 3 for values in CONTEXT_FEATURES.values())


def test_teammate_context_excludes_focal_player() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "Season": [2025, 2025, 2025],
            "TEAM_ID": [10, 10, 10],
            "OffPoss": [100.0, 200.0, 300.0],
            "DefPoss": [100.0, 200.0, 300.0],
            "crafted_spacing_stable_v1": [1.0, 2.0, 4.0],
            "box_creation_2017_eb_p100": [1.0, 2.0, 4.0],
            "at_rim_fga_p100": [1.0, 2.0, 4.0],
            "turnover_to_load_2017_eb": [1.0, 2.0, 4.0],
            "offensive_load_2017_eb_p100": [1.0, 2.0, 4.0],
            "OREB_p100": [1.0, 2.0, 4.0],
            "DREB_p100_relative": [1.0, 2.0, 4.0],
            "dreb_contests_p100": [1.0, 2.0, 4.0],
            "event_stops_p100": [1.0, 2.0, 4.0],
            "deflections_p100": [1.0, 2.0, 4.0],
            "rim_points_saved_p100": [1.0, 2.0, 4.0],
            "contested_shots_p100": [1.0, 2.0, 4.0],
        }
    )
    output = add_leave_one_out_teammate_context(frame)
    assert np.isclose(output.loc[0, "teammate_spacing"], (2 * 200 + 4 * 300) / 500)
    assert np.isclose(output.loc[2, "teammate_spacing"], (1 * 100 + 2 * 200) / 300)
