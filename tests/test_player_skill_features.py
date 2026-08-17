from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.player_skill_features import (
    PLAYER_SKILL_FEATURES,
    compute_player_skill_features,
    profile_player_skill_features,
)


def test_player_skill_features_are_unique_finite_and_preserve_coverage() -> None:
    shooting = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 2, 2], "year": [2024] * 4,
            "shot_type": ["tight", "open", "tight", "open"],
            "2FGM": [5, 10, 4, 8], "2FGA": [10, 20, 10, 20],
            "3PM": [2, 4, 1, 5], "3PA": [5, 10, 5, 10],
            "FGM": [7, 14, 5, 13], "FGA": [15, 30, 15, 30],
        }
    )
    passing = pd.DataFrame(
        {
            "nba_id": [1, 2], "year": [2024, 2024], "OffPoss": [1000, 1000],
            "POTENTIAL_AST": [100, 80], "AST_PTS_CREATED": [130, 80],
            "High Value Assist %": [60.0, 0.5], "BadPassTurnovers": [10, 8],
            "Passes": [500, 400],
        }
    )
    hustle = pd.DataFrame(
        {
            "PLAYER_ID": [1], "year": [2024], "POSS": [1000],
            "SCREEN_AST_PTS": [20], "DEFLECTIONS": [30], "CHARGES_DRAWN": [5],
            "DEF_BOXOUTS": [40], "LOOSE_BALLS_RECOVERED": [10],
        }
    )
    shotzone = pd.DataFrame(
        {"EntityId": [1, 2], "year": [2024, 2024], "OffPoss": [1000, 900]}
    )

    result = compute_player_skill_features(shooting, passing, hustle, shotzone)

    assert len(result) == 2
    assert not result.duplicated(["PLAYER_ID", "Season"]).any()
    assert np.isfinite(result[list(PLAYER_SKILL_FEATURES)].dropna().to_numpy()).all()
    assert result.set_index("PLAYER_ID").loc[1, "has_hustle_tracking"]
    assert not result.set_index("PLAYER_ID").loc[2, "has_hustle_tracking"]
    assert result.set_index("PLAYER_ID").loc[1, "high_value_assist_share_eb"] > 0.5
    assert result.set_index("PLAYER_ID").loc[1, "shot_making_points_above_expected_p100_eb"] > 0
    weighted_relative = np.average(
        result["shot_difficulty_expected_points_per_attempt_relative"],
        weights=result["PLAYER_ID"].map({1: 45, 2: 45}),
    )
    assert abs(weighted_relative) < 1e-12
    profile = profile_player_skill_features(result)
    assert len(profile) == len(PLAYER_SKILL_FEATURES)
    assert set(profile["Season"]) == {2024}
    assert profile["missing_fraction"].between(0, 1).all()


def test_player_skill_features_reject_duplicate_shot_buckets() -> None:
    shooting = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1], "year": [2024, 2024], "shot_type": ["open", "open"],
            "2FGM": [1, 1], "2FGA": [2, 2], "3PM": [1, 1], "3PA": [2, 2],
            "FGM": [2, 2], "FGA": [4, 4],
        }
    )
    passing = pd.DataFrame(
        {
            "nba_id": [1], "year": [2024], "OffPoss": [10], "POTENTIAL_AST": [1],
            "AST_PTS_CREATED": [2], "High Value Assist %": [50],
            "BadPassTurnovers": [0], "Passes": [10],
        }
    )
    hustle = pd.DataFrame(
        {
            "PLAYER_ID": [1], "year": [2024], "POSS": [10], "SCREEN_AST_PTS": [0],
            "DEFLECTIONS": [0], "CHARGES_DRAWN": [0], "DEF_BOXOUTS": [0],
            "LOOSE_BALLS_RECOVERED": [0],
        }
    )
    shotzone = pd.DataFrame({"EntityId": [1], "year": [2024], "OffPoss": [10]})

    try:
        compute_player_skill_features(shooting, passing, hustle, shotzone)
    except ValueError as error:
        assert "duplicate player-season-shot-type" in str(error)
    else:
        raise AssertionError("duplicate shooting rows should fail")
