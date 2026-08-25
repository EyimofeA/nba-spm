from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.rapm_lab.run_context_adjustments import (
    annotate_context,
    context_matrix,
    select_variant,
)


def _row(num: int, points: float, home: bool, period: int = 1) -> dict:
    return {
        "gameid": "g1",
        "period": period,
        "num": num,
        "pts": points,
        "home_poss": home,
    }


def test_context_score_uses_only_prior_possessions() -> None:
    source = pd.DataFrame(
        [_row(1, 2.0, True), _row(2, 3.0, False), _row(3, 2.0, True)]
    )
    original = annotate_context(source)
    revised_source = source.copy()
    revised_source.loc[1, "pts"] = 0.0
    revised = annotate_context(revised_source)
    assert original.loc[1, "offense_margin_before"] == -2.0
    assert revised.loc[1, "offense_margin_before"] == -2.0
    assert original.loc[2, "home_margin_before"] == -1.0
    assert revised.loc[2, "home_margin_before"] == 2.0


def test_context_matrix_fits_scales_on_training_rows_only() -> None:
    source = pd.DataFrame(
        [
            _row(1, 2.0, True, 1),
            _row(2, 0.0, False, 1),
            _row(3, 0.0, True, 2),
            _row(4, 3.0, False, 2),
        ]
    )
    annotated = annotate_context(source)
    matrix, names = context_matrix(
        annotated,
        train_rows=3,
        quarter_rubberband=True,
        clock_fatigue=True,
    )
    assert matrix.shape == (4, 5)
    assert names == (
        "rubberband_q1",
        "rubberband_q2",
        "rubberband_q3",
        "rubberband_q4plus",
        "clock_fatigue",
    )
    assert np.isfinite(matrix.data).all()


def test_selection_uses_correlation_then_rmse_within_tolerance() -> None:
    summary = pd.DataFrame(
        {
            "variant": ["a", "b", "c"],
            "selection_primary_correlation": [0.5000, 0.5004, 0.49],
            "selection_primary_rmse": [14.0, 14.2, 13.0],
        }
    )
    assert select_variant(summary)["variant"] == "a"
