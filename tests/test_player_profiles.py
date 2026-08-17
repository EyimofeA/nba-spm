import pandas as pd

from nba_impact.api.player_profiles import (
    PROFILE_AXES,
    PROFILE_SIDES,
    build_player_skill_profiles,
)


def test_profiles_are_season_relative_and_good_direction_is_high() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 1, 2, 3],
            "Window_End": [2023, 2023, 2023, 2024, 2024, 2024],
            "true_shooting_pct_relative": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            "TOV_p100_relative": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
        }
    )
    result = build_player_skill_profiles(frame, [2023, 2024])
    first = result.set_index(["PLAYER_ID", "Season"])
    assert first.loc[(3, 2023), "shooting"] > first.loc[(1, 2023), "shooting"]
    assert first.loc[(3, 2023), "security"] > first.loc[(1, 2023), "security"]
    assert first.loc[(1, 2024), "shooting"] > first.loc[(3, 2024), "shooting"]
    assert first[["shooting", "security"]].max().max() <= 100


def test_spacing_is_a_separate_axis() -> None:
    frame = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "Window_End": [2024, 2024, 2024],
            "true_shooting_pct_relative": [0.1, 0.2, 0.3],
            "crafted_spacing_stable_v1": [5.0, -1.0, 2.0],
        }
    )
    result = build_player_skill_profiles(frame, [2024]).set_index("PLAYER_ID")
    assert result.loc[1, "spacing"] == 100.0
    assert result.loc[2, "spacing"] < result.loc[3, "spacing"] < result.loc[1, "spacing"]
    shooting_only = frame.drop(columns=["crafted_spacing_stable_v1"])
    comparison = result[["shooting"]].join(
        build_player_skill_profiles(shooting_only, [2024]).set_index("PLAYER_ID"),
        rsuffix="_without_spacing",
    )
    assert comparison["shooting"].equals(comparison["shooting_without_spacing"])


def test_profile_sides_cover_every_axis() -> None:
    sided = [axis for axes in PROFILE_SIDES.values() for axis in axes]
    assert set(sided) == set(PROFILE_AXES)
    assert "spacing" in PROFILE_SIDES["offense"]
    assert "spacing" not in PROFILE_SIDES["defense"]
