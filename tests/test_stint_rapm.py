import numpy as np
import pandas as pd

from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients
from nba_impact.models.stint_rapm import build_stint_design, fit_stint_center_path


def _lineup(prefix: str, start: int) -> dict[str, int]:
    return {f"{prefix}_player_{number}": start + number for number in range(1, 6)}


def test_weighted_stint_fit_matches_expanded_possessions():
    stints = pd.DataFrame(
        [
            {
                "season": 2026,
                "game_id": "1",
                "home_possessions": 2,
                "away_possessions": 1,
                "home_points": 3,
                "away_points": 0,
                **_lineup("home", 0),
                **_lineup("away", 10),
            },
            {
                "season": 2026,
                "game_id": "2",
                "home_possessions": 1,
                "away_possessions": 2,
                "home_points": 0,
                "away_points": 4,
                **_lineup("home", 20),
                **_lineup("away", 30),
            },
        ]
    )
    weighted = build_stint_design(stints)
    config = RapmConfig((2026,), lambda_off=10, lambda_def=20, lambda_home=5)
    weighted_beta, weighted_intercept = fit_stint_center_path(
        weighted,
        config,
        np.zeros(weighted.X.shape[1]),
        center_scales=(0.0,),
    )[0.0]

    rows = []
    for stint in stints.to_dict("records"):
        for home_offense, side, points in (
            (1, "home", [1.5, 1.5] if stint["home_points"] == 3 else [0]),
            (0, "away", [0] if stint["away_points"] == 0 else [2, 2]),
        ):
            for number, point in enumerate(points, start=1):
                rows.append(
                    {
                        "home_poss": home_offense,
                        "pts": point,
                        "season": 2026,
                        "date": "2026-01-01",
                        "period": 1,
                        "num": number,
                        "gameid": stint["game_id"],
                        **{f"h{i}": stint[f"home_player_{i}"] for i in range(1, 6)},
                        **{f"a{i}": stint[f"away_player_{i}"] for i in range(1, 6)},
                    }
                )
    expanded = build_design(pd.DataFrame(rows))
    expanded_beta, expanded_intercept = fit_coefficients(expanded, config)
    assert np.allclose(weighted_beta, expanded_beta)
    assert np.isclose(weighted_intercept, expanded_intercept)
