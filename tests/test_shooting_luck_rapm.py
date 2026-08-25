import numpy as np
import pandas as pd

from nba_impact.models.shooting_luck_rapm import (
    leave_game_out_empirical_bayes_rate,
    replace_shooter_with_dummy,
)


def test_leave_game_out_rate_excludes_current_game() -> None:
    frame = pd.DataFrame(
        {
            "season": [2026] * 4,
            "game_id": ["a", "a", "b", "b"],
            "player_id": [1] * 4,
            "made": [1.0, 1.0, 0.0, 0.0],
        }
    )
    rate = leave_game_out_empirical_bayes_rate(frame, prior_attempts=2)
    assert np.allclose(rate[:2], 0.25)
    assert np.allclose(rate[2:], 0.75)


def test_shooter_is_removed_only_from_offense() -> None:
    frame = pd.DataFrame(
        {
            "home_poss": [True],
            "shooter_id": [1],
            **{f"h{i}": [i] for i in range(1, 6)},
            **{f"a{i}": [i + 5] for i in range(1, 6)},
        }
    )
    output, missing = replace_shooter_with_dummy(frame)
    assert missing == 0
    assert output.loc[0, "h1"] == 0
    assert output.loc[0, "a1"] == 6
