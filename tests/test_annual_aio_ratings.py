from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.annual_aio_ratings import fit_annual_aio_season
from nba_impact.models.rapm import RapmConfig


def test_annual_aio_ratings_preserve_decomposition() -> None:
    rng = np.random.default_rng(11)
    players = np.arange(1, 13)
    rows = []
    for possession in range(120):
        lineup = rng.choice(players, size=10, replace=False)
        rows.append(
            {
                "home_poss": bool(possession % 2),
                "pts": float(1 + (lineup[5:10].sum() - lineup[:5].sum()) * 0.002),
                **{f"a{i + 1}": int(value) for i, value in enumerate(lineup[:5])},
                **{f"h{i + 1}": int(value) for i, value in enumerate(lineup[5:])},
                "season": 2024,
                "date": "2023-11-01",
                "period": 1,
                "num": possession + 1,
                "gameid": f"00223000{possession // 20}",
            }
        )
    priors = pd.DataFrame(
        {
            "PLAYER_ID": players,
            "Window_End": 2024,
            "prior_offense_per_100": (players - players.mean()) * 0.2,
            "prior_defense_per_100": (players.mean() - players) * 0.1,
        }
    )
    priors["prior_net_per_100"] = (
        priors["prior_offense_per_100"] + priors["prior_defense_per_100"]
    )
    ratings, quality = fit_annual_aio_season(
        pd.DataFrame(rows),
        priors,
        RapmConfig(seasons=(2024,), lambda_off=50, lambda_def=50, lambda_home=10),
        season=2024,
    )
    assert len(ratings) == len(players)
    assert ratings["prior_available"].all()
    np.testing.assert_allclose(
        ratings["aio_net"], ratings["aio_offense"] + ratings["aio_defense"]
    )
    np.testing.assert_allclose(
        ratings["aio_net"],
        ratings["spm_center_net"] + ratings["rapm_update_net"],
    )
    assert quality["max_component_identity_error"] < 1e-12
