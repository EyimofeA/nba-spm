from __future__ import annotations

import pandas as pd
import pytest

from research.rapm_lab.run_reliability_weighted_conversion_rapm import build_target_frames


def test_reliability_hybrid_preserves_nonconversion_and_shrinks_conversion() -> None:
    base = pd.DataFrame({"possession_id": ["a", "b"], "pts": [5.0, 2.0]})
    ledger = pd.DataFrame(
        {
            "possession_id": ["a"],
            "season_end": [2025],
            "shooter_id": [1],
            "category": ["corner_3"],
            "actual_points": [3.0],
            "neutral_expected_points": [1.0],
            "skill_expected_points": [1.5],
        }
    )
    selected = pd.DataFrame({"category": ["corner_3"], "prior_attempts": [1.0]})
    frames, reliability = build_target_frames(base, ledger, selected, (0.5,))

    assert frames["neutral_residual_50pct"]["pts"].tolist() == pytest.approx([4.0, 2.0])
    assert frames["preseason_skill_expected"]["pts"].tolist() == pytest.approx([3.5, 2.0])
    assert frames["history_reliability_hybrid"]["pts"].tolist() == pytest.approx([4.25, 2.0])
    assert reliability.loc[0, "mean_reliability"] == pytest.approx(0.5)
