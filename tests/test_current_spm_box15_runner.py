from __future__ import annotations

import pandas as pd

from research.run_current_spm_box15 import _cutoffs


def test_cutoffs_are_mondays_inside_declared_window() -> None:
    cutoffs = _cutoffs(2026)
    assert len(cutoffs) > 10
    assert set(cutoffs.dayofweek) == {0}
    assert cutoffs.min() >= pd.Timestamp("2025-11-01")
    assert cutoffs.max() <= pd.Timestamp("2026-04-01")
