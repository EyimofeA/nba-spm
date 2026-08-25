import numpy as np
import pandas as pd

from nba_impact.models.channel_rapm import (
    fit_possession_channels,
    fit_teammate_channels,
    lineup_sides,
    own_contribution_matrix,
)
from nba_impact.models.rapm import RapmConfig


def _frame(rows: int = 40) -> pd.DataFrame:
    output = []
    for index in range(rows):
        swap = index % 2
        output.append(
            {
                "possession_id": f"p{index}",
                "gameid": f"g{index // 10}",
                "season": 2026,
                "home_poss": swap,
                "pts": float(index % 4),
                **{f"a{slot}": slot for slot in range(1, 6)},
                **{f"h{slot}": slot + 10 for slot in range(1, 6)},
            }
        )
    return pd.DataFrame(output)


def test_lineup_sides_follow_possession_team() -> None:
    offense, defense = lineup_sides(_frame(2))
    np.testing.assert_array_equal(offense[0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(defense[0], [11, 12, 13, 14, 15])
    np.testing.assert_array_equal(offense[1], [11, 12, 13, 14, 15])


def test_own_contributions_remove_only_the_focal_players_event() -> None:
    frame = _frame(1)
    offense, _ = lineup_sides(frame)
    contributions = pd.DataFrame(
        [{"possession_id": "p0", "player_id": 3, "scoring": 2.0}]
    )
    own = own_contribution_matrix(
        frame, offense, contributions, targets=("scoring",)
    )
    np.testing.assert_array_equal(own[0, :, 0], [0, 0, 2, 0, 0])


def test_teammate_fit_has_one_focal_opportunity_per_player() -> None:
    frame = _frame()
    team = frame[["possession_id"]].assign(scoring=2.0, turnover=0.0)
    contributions = pd.DataFrame(
        [
            {
                "possession_id": possession,
                "player_id": 1 if home_poss == 0 else 11,
                "scoring": 2.0,
                "turnover": 0.0,
            }
            for possession, home_poss in zip(frame.possession_id, frame.home_poss)
        ]
    )
    fit = fit_teammate_channels(
        frame,
        focal_side="offense",
        team_targets=team,
        contributions=contributions,
        targets=("scoring", "turnover"),
        focal_penalty=10,
        nuisance_penalty=40,
        opponent_penalty=50,
        home_penalty=10,
        batch_size=7,
    )
    assert fit.coefficients.shape == (10, 2)
    assert fit.opportunities == len(frame)
    assert fit.exposures.sum() == len(frame) * 5
    np.testing.assert_allclose(
        np.average(fit.coefficients, axis=0, weights=fit.exposures), 0.0, atol=1e-12
    )


def test_multi_channel_fit_preserves_offense_plus_defense_identity() -> None:
    frame = _frame().assign(
        rim_points=lambda value: value["pts"],
        midrange_attempts=lambda value: value.index % 2,
    )
    ratings = fit_possession_channels(
        frame,
        targets=("rim_points", "midrange_attempts"),
        config=RapmConfig(seasons=(2026,), lambda_off=10, lambda_def=10),
    )
    for target in ("rim_points", "midrange_attempts"):
        np.testing.assert_allclose(
            ratings[f"{target}_offense"] + ratings[f"{target}_defense"],
            ratings[f"{target}_net"],
            atol=1e-12,
        )
