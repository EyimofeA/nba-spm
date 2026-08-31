from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.mechanism_features import (
    MECHANISM_FEATURES,
    compute_mechanism_features,
)


def _panel() -> pd.DataFrame:
    rows = []
    for season in (2024, 2025):
        for player in range(1, 9):
            rows.append(
                {
                    "PLAYER_ID": player,
                    "Window_End": season,
                    "OffPoss": 500.0 + 50 * player,
                    "DefPoss": 550.0 + 40 * player,
                    "assist_points_created_p100": 5.0 + player,
                    "potential_assists_p100": 4.0 + 0.5 * player,
                    "shot_quality_average_relative": 0.01 * player,
                    "offensive_load_2017_eb_p100": 10.0 + player,
                    "at_rim_frequency_eb": 0.2 + 0.01 * player,
                    "arc3_frequency_eb": 0.3 - 0.01 * player,
                    "box_creation_2017_eb_p100": 2.0 + 0.4 * player,
                    "touches_p100": 50.0 + player,
                    "crafted_spacing_stable_v1": -0.1 + 0.03 * player,
                    "contested_2pt_p100": 3.0 + 0.2 * player,
                    "contested_3pt_p100": 2.0 + 0.1 * player,
                    "deflections_p100": 1.0 + 0.1 * player,
                    "recovered_blocks_p100": 0.2 + 0.05 * player,
                    "PF_p100": 3.0 + 0.1 * player,
                    "has_hustle_tracking": 1.0,
                    "matchup_opponent_adjusted_points_saved_p100_eb": -0.5 + 0.1 * player,
                    "dfg_attempts_p100": 15.0 + player,
                    "rim_dfga_p100": 5.0 + 0.5 * player,
                    "has_matchup_tracking": 1.0,
                    "DREB_p100": 4.0 + 0.3 * player,
                    "dreb_chances_p100": 8.0 + 0.4 * player,
                    "dreb_contests_p100": 2.0 + 0.2 * player,
                    "rim_points_saved_p100": -0.2 + 0.05 * player,
                }
            )
    return pd.DataFrame(rows)


def test_mechanism_features_are_finite_unique_and_same_season() -> None:
    panel = _panel()
    result = compute_mechanism_features(panel)

    assert tuple(result.columns[2:]) == MECHANISM_FEATURES
    assert not result.duplicated(["PLAYER_ID", "Window_End"]).any()
    assert np.isfinite(result[list(MECHANISM_FEATURES)].to_numpy()).all()

    changed = panel.copy()
    changed.loc[changed["Window_End"].eq(2025), "shot_quality_average_relative"] += 99
    changed_result = compute_mechanism_features(changed)
    original_2024 = result.loc[result["Window_End"].eq(2024)].reset_index(drop=True)
    changed_2024 = changed_result.loc[
        changed_result["Window_End"].eq(2024)
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_2024, changed_2024)


def test_mechanism_features_reject_duplicate_keys() -> None:
    panel = _panel()
    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    try:
        compute_mechanism_features(duplicate)
    except ValueError as error:
        assert "duplicate player-seasons" in str(error)
    else:
        raise AssertionError("duplicate keys were accepted")
