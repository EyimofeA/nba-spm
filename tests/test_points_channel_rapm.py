from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_impact.models.points_channel_rapm import (
    POINT_CHANNELS,
    build_points_channel_targets,
    fit_points_channel_rapm,
)
from nba_impact.models.rapm import RapmConfig


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(71)
    players = np.arange(1, 15)
    rows = []
    for season in (2022, 2023):
        for possession in range(240):
            lineup = rng.choice(players, size=10, replace=False)
            rows.append(
                {
                    "home_poss": bool(possession % 2),
                    "pts": float((possession + season) % 5),
                    **{f"a{i + 1}": int(value) for i, value in enumerate(lineup[:5])},
                    **{f"h{i + 1}": int(value) for i, value in enumerate(lineup[5:])},
                    "season": season,
                    "date": f"{season - 1}-11-01",
                    "period": 1,
                    "num": possession + 1,
                    "gameid": f"002{season}{possession // 24:04d}",
                }
            )
    return pd.DataFrame(rows)


def test_points_channel_targets_conserve_points() -> None:
    frame = _frame()
    targets = build_points_channel_targets(frame)
    np.testing.assert_allclose(targets.sum(axis=1), frame["pts"], atol=0.0)
    with pytest.raises(ValueError, match="integer possession points"):
        build_points_channel_targets(frame.assign(pts=1.5))


def test_points_channel_ratings_recompose_canonical_rapm() -> None:
    frame = _frame()
    result = fit_points_channel_rapm(
        frame,
        RapmConfig(
            seasons=(2022, 2023),
            lambda_off=50.0,
            lambda_def=80.0,
            lambda_home=10.0,
        ),
    )
    ratings = result.ratings
    for component in ("offense", "defense", "net"):
        np.testing.assert_allclose(
            ratings[component],
            ratings[[f"{channel}_{component}" for channel in POINT_CHANNELS]].sum(axis=1),
            atol=1e-12,
        )
    for channel in POINT_CHANNELS:
        np.testing.assert_allclose(
            ratings[f"{channel}_net"],
            ratings[f"{channel}_offense"] + ratings[f"{channel}_defense"],
            atol=1e-12,
        )
    assert result.quality["maximum_canonical_rapm_error"] < 1e-7
    assert result.quality["intercept_recomposition_error"] < 1e-9
