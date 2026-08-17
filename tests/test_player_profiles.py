import pandas as pd

from nba_impact.api.player_profiles import build_player_skill_profiles


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
