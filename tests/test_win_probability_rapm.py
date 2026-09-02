import numpy as np
import pandas as pd

from nba_impact.models.win_probability_rapm import (
    build_conserved_wp_target,
    build_log_odds_wp_target,
)


def test_wp_target_conserves_to_game_result() -> None:
    frame = pd.DataFrame(
        {
            "possession_id": ["g:1", "g:2", "g:3"],
            "gameid": ["g", "g", "g"],
            "num": [1, 2, 3],
            "home_poss": [1, 0, 1],
            "home_win": [1, 1, 1],
            "probability_context": [0.5, 0.6, 0.7],
        }
    )
    target, games = build_conserved_wp_target(frame)
    np.testing.assert_allclose(target["home_wp_change"], [0.1, 0.1, 0.3])
    np.testing.assert_allclose(target["offense_wp_change"], [0.1, -0.1, 0.3])
    assert abs(games.iloc[0]["conservation_error"]) < 1e-12


def test_wp_target_rejects_invalid_probability() -> None:
    frame = pd.DataFrame(
        {
            "possession_id": ["g:1"],
            "gameid": ["g"],
            "num": [1],
            "home_poss": [1],
            "home_win": [0],
            "probability_context": [1.1],
        }
    )
    try:
        build_conserved_wp_target(frame)
    except ValueError as error:
        assert "[0, 1]" in str(error)
    else:
        raise AssertionError("Invalid probability was accepted.")


def test_log_odds_target_clips_terminal_state_and_telescopes() -> None:
    frame = pd.DataFrame(
        {
            "possession_id": ["g:1", "g:2", "g:3"],
            "gameid": ["g", "g", "g"],
            "num": [1, 2, 3],
            "home_poss": [1, 0, 1],
            "home_win": [1, 1, 1],
            "probability_context": [0.5, 0.6, 0.7],
        }
    )
    target, games = build_log_odds_wp_target(frame, epsilon=0.025)
    expected = np.diff([0.0, np.log(0.6 / 0.4), np.log(0.7 / 0.3), np.log(0.975 / 0.025)])
    np.testing.assert_allclose(target["home_log_odds_change"], expected)
    np.testing.assert_allclose(
        target["offense_log_odds_change"], expected * [1.0, -1.0, 1.0]
    )
    assert np.isfinite(target["home_log_odds_change"]).all()
    assert abs(games.iloc[0]["conservation_error"]) < 1e-12
