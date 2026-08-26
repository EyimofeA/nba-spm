import numpy as np
import pandas as pd

from research.run_spm_cheating_ladder import (
    _position_flags,
    _rolling_context,
    apply_team_reconciliation,
)


def test_position_flags_allow_hybrid_positions() -> None:
    assert _position_flags("G-F") == (1.0, 1.0, 0.0)
    assert _position_flags("F-C") == (0.0, 1.0, 1.0)
    assert _position_flags(None) == (0.0, 0.0, 0.0)


def test_team_reconciliation_matches_team_net_rating() -> None:
    ratings = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 4],
            "Season": [2024] * 4,
            "offense": [1.0, 0.0, -1.0, 2.0],
            "defense": [0.0, 1.0, 0.0, -1.0],
            "net": [1.0, 1.0, -1.0, 1.0],
        }
    )
    minutes = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 4],
            "Season": [2024] * 4,
            "team_id": [10] * 4,
            "minutes": [100.0] * 4,
        }
    )
    teams = pd.DataFrame(
        {"Season": [2024], "TEAM_ID": [10], "team_net_rating": [5.0]}
    )
    adjusted, constants = apply_team_reconciliation(ratings, minutes, teams)
    resulting_team_rating = 5.0 * np.average(adjusted["net"], weights=minutes["minutes"])
    assert np.isclose(resulting_team_rating, 5.0)
    assert np.isclose(constants["player_constant"].iloc[0], 0.5)
    assert np.allclose(adjusted["offense"] + adjusted["defense"], adjusted["net"])


def test_rolling_context_excludes_future_seasons() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 1, 1, 1],
            "Season": [2018, 2019, 2020, 2021, 2022, 2023],
            "AGE": [20, 21, 22, 23, 24, 25],
            "MIN": [100.0] * 6,
            "position": ["G"] * 6,
            "on_possessions": [100.0] * 6,
            "raw_onoff_net": [1.0, 1.0, 1.0, 1.0, 1.0, 999.0],
            "aupm": [2.0, 2.0, 2.0, 2.0, 2.0, 999.0],
        }
    )
    result = _rolling_context(annual, range(2022, 2023)).iloc[0]
    assert result["age_end"] == 24.0
    assert result["raw_onoff_net_5y"] == 1.0
    assert result["aupm_5y"] == 2.0
