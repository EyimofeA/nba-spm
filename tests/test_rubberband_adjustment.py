import numpy as np
import pandas as pd

from nba_impact.models.rubberband_adjustment import (
    RubberbandSpec,
    annotate_score_context,
    coefficient_table,
    deterministic_game_fold,
    fit_rubberband,
    predict_rubberband,
)


def test_score_context_uses_pre_possession_margin_and_actual_clock() -> None:
    frame = pd.DataFrame(
        {
            "gameid": ["1", "1", "1", "1"],
            "season": [2026] * 4,
            "period": [1, 1, 1, 4],
            "num": [1, 2, 3, 4],
            "pts": [2, 3, 0, 2],
            "home_poss": [1, 0, 1, 0],
            "start_seconds_elapsed": [10.0, 30.0, 400.0, 2800.0],
        }
    )
    result = annotate_score_context(frame)
    assert result["offense_margin_before"].tolist() == [0.0, -2.0, -1.0, 1.0]
    assert result["six_minute_bucket"].tolist() == [0, 0, 1, 7]


def test_rubberband_fit_recovers_time_specific_negative_slopes() -> None:
    rng = np.random.default_rng(7)
    rows = 8000
    bucket = rng.integers(0, 4, rows)
    margin = rng.integers(-20, 21, rows)
    slopes = np.asarray([-0.001, -0.002, -0.003, -0.004])
    frame = pd.DataFrame(
        {
            "gameid": np.asarray([f"g{x // 20}" for x in range(rows)]),
            "regulation": True,
            "six_minute_bucket": bucket * 2,
            "offense_margin_before": margin,
            "lineup_residual_points": slopes[bucket] * margin
            + rng.normal(0, 0.01, rows),
        }
    )
    spec = RubberbandSpec("quarter", 4, 20)
    fit = fit_rubberband(frame, spec, cluster_covariance=True)
    assert np.allclose(fit.coefficients[4:], slopes, atol=0.00005)
    assert np.allclose(
        predict_rubberband(fit, frame),
        fit.coefficients[bucket] + fit.coefficients[4 + bucket] * margin,
    )
    table = coefficient_table(fit)
    assert (table["slope_points_per_100_per_margin_point"] < 0).all()


def test_game_fold_is_stable() -> None:
    assert deterministic_game_fold("0022500001", 5) == deterministic_game_fold(
        "0022500001", 5
    )
