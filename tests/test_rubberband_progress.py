import numpy as np
import pandas as pd

from nba_impact.models.rubberband_adjustment import (
    RubberbandSpec,
    fit_rubberband,
)
from nba_impact.models.rubberband_progress import (
    annotate_possession_progress,
    possession_coefficient_table,
    slope_only_adjustment,
    use_possession_progress,
)


def test_progress_buckets_use_only_prior_possession_count() -> None:
    frame = pd.DataFrame(
        {
            "gameid": ["g"] * 53,
            "period": [1] * 53,
            "num": np.arange(1, 54),
        }
    )
    result = annotate_possession_progress(frame, bucket_size=25, buckets=8)
    assert result.loc[0, "regulation_possession_number"] == 1
    assert result.loc[24, "possession_progress_bucket"] == 0
    assert result.loc[25, "possession_progress_bucket"] == 1
    assert result.loc[50, "possession_progress_bucket"] == 2

    extended = pd.concat(
        [
            frame,
            pd.DataFrame(
                {"gameid": ["g"] * 20, "period": [2] * 20, "num": range(54, 74)}
            ),
        ],
        ignore_index=True,
    )
    extended_result = annotate_possession_progress(extended)
    assert np.array_equal(
        result["possession_progress_bucket"].to_numpy(),
        extended_result.loc[:52, "possession_progress_bucket"].to_numpy(),
    )


def test_slope_only_adjustment_excludes_bucket_intercepts() -> None:
    frame = pd.DataFrame(
        {
            "gameid": ["g"] * 100,
            "period": [1] * 100,
            "num": np.arange(100),
            "regulation": True,
            "offense_margin_before": np.tile([-10.0, 10.0], 50),
            "lineup_residual_points": np.tile([0.1, -0.1], 50),
        }
    )
    progress = use_possession_progress(annotate_possession_progress(frame))
    fit = fit_rubberband(progress, RubberbandSpec("progress", 8, 15))
    adjustment = slope_only_adjustment(fit, progress)
    assert np.isclose(adjustment.mean(), 0.0, atol=1e-12)
    assert adjustment[0] > 0
    assert adjustment[1] < 0
    table = possession_coefficient_table(fit, bucket_size=25)
    assert np.isnan(table.iloc[-1]["possessions_elapsed_end"])
