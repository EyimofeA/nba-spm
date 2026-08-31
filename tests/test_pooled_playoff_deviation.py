from __future__ import annotations

import pandas as pd
import pytest

from research.run_pooled_playoff_deviation import _paired_bootstrap


def test_paired_bootstrap_requires_identical_games() -> None:
    baseline = pd.DataFrame(
        {"gameid": ["a", "b"], "actual_margin": [5.0, -2.0], "predicted_margin": [4.0, 0.0]}
    )
    candidate = baseline.assign(predicted_margin=[5.0, -2.0])
    result = _paired_bootstrap(baseline, candidate, draws=100, seed=1)
    assert result["mse_delta"] < 0
    with pytest.raises(ValueError, match="identical games"):
        _paired_bootstrap(baseline, candidate.iloc[:1], draws=10)
