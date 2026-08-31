from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from research.run_aio_prior_calibration_precision import (
    fit_prior_affine,
    select_configuration,
)


def test_select_configuration_weights_seasons_equally() -> None:
    rows = []
    for season, count, errors in ((2022, 100, (1.0, 2.0)), (2023, 10, (4.0, 1.0))):
        for penalty, error in zip((1500.0, 3000.0), errors, strict=True):
            rows.extend(
                {
                    "test_season": season,
                    "offense_penalty": penalty,
                    "defense_penalty": 3000.0,
                    "squared_error": error,
                }
                for _ in range(count)
            )
    choice = select_configuration(
        pd.DataFrame(rows), ["offense_penalty", "defense_penalty"]
    )
    assert choice == {"offense_penalty": 3000.0, "defense_penalty": 3000.0}


def test_fit_prior_affine_recovers_side_specific_linear_map() -> None:
    history = pd.DataFrame(
        {
            "prior_offense_per_100": [-1.0, 0.0, 1.0],
            "prior_defense_per_100": [-2.0, 0.0, 2.0],
            "target_offense": [-1.0, 1.0, 3.0],
            "target_defense": [-2.0, 1.0, 4.0],
            "sample_weight": [1.0, 1.0, 1.0],
        }
    )
    current = pd.DataFrame(
        {
            "prior_offense_per_100": [2.0],
            "prior_defense_per_100": [4.0],
        }
    )
    calibrated, parameters = fit_prior_affine(history, current)
    assert calibrated["prior_offense_per_100"].iloc[0] == pytest.approx(5.0)
    assert calibrated["prior_defense_per_100"].iloc[0] == pytest.approx(7.0)
    assert parameters["prior_offense_slope"] == pytest.approx(2.0)
    assert parameters["prior_defense_slope"] == pytest.approx(1.5)
