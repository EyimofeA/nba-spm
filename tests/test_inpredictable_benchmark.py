from __future__ import annotations

import numpy as np
import pytest

from nba_impact.models.inpredictable_benchmark import (
    neutralize_home_probability,
    parse_calculator_probability,
)


def test_parse_calculator_probability() -> None:
    assert parse_calculator_probability("<b>Win Probability: 57.4%</b>") == pytest.approx(0.574)


def test_neutralize_home_probability_is_symmetric() -> None:
    result = neutralize_home_probability(np.array([0.60]), np.array([0.40]))
    assert result[0] == pytest.approx(0.60)
