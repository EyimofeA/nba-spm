from __future__ import annotations

import numpy as np
import pandas as pd

from research.pulse_search_protocol import (
    fold_internal_affine,
    keep_calibrated_variant,
    scale_side_center,
)


def test_keep_calibrated_variant_requires_rmse_and_slope() -> None:
    assert keep_calibrated_variant(13.610, 0.871, 13.614)
    assert keep_calibrated_variant(13.614, 0.900, 13.614)
    assert not keep_calibrated_variant(13.598, 0.846, 13.614)
    assert not keep_calibrated_variant(13.620, 0.900, 13.614)


def test_scale_side_center_leaves_home_unchanged() -> None:
    center = np.array([1.0, 2.0, 3.0, 4.0, 0.5])
    scaled = scale_side_center(center, 2, offense_scale=0.5, defense_scale=2.0)
    assert scaled.tolist() == [0.5, 1.0, 6.0, 8.0, 0.5]


def test_fold_internal_affine_uses_only_earlier_seasons() -> None:
    games = pd.DataFrame(
        {
            "candidate": ["hgb_hustle_k1.0"] * 4,
            "outcome_season": [2015, 2015, 2016, 2016],
            "game_id": ["a", "b", "c", "d"],
            "official_margin": [10.0, -4.0, 8.0, -2.0],
            "predicted_margin": [5.0, -2.0, 40.0, 1.0],
        }
    )
    calibrated = fold_internal_affine(games, "hgb_hustle_k1.0")
    first = calibrated.loc[calibrated["outcome_season"].eq(2015)]
    assert (first["affine_slope"] == 1.0).all()
    assert (first["affine_intercept"] == 0.0).all()
    prior_pred = np.array([5.0, -2.0])
    prior_actual = np.array([10.0, -4.0])
    expected_slope = float(np.cov(prior_actual, prior_pred, ddof=0)[0, 1] / np.var(prior_pred))
    expected_intercept = float(prior_actual.mean() - expected_slope * prior_pred.mean())
    second = calibrated.loc[calibrated["outcome_season"].eq(2016)]
    assert np.isclose(second["affine_slope"].iloc[0], expected_slope)
    assert np.isclose(second["affine_intercept"].iloc[0], expected_intercept)
    future_only_slope = float(
        np.cov(
            np.array([10.0, -4.0, 8.0, -2.0]),
            np.array([5.0, -2.0, 40.0, 1.0]),
            ddof=0,
        )[0, 1]
        / np.var(np.array([5.0, -2.0, 40.0, 1.0]))
    )
    assert not np.isclose(second["affine_slope"].iloc[0], future_only_slope)
