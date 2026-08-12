from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_impact.models.precision_aware_prior import (
    PriorPrecision,
    calibrate_prior_precision,
    fit_precision_aware_center,
)
from nba_impact.models.rapm import RapmConfig, build_design


def test_calibration_removes_label_variance_and_does_not_make_width_negative() -> None:
    calibration = pd.DataFrame(
        {
            "label": [0.1] * 100,
            "prior": [0.0] * 100,
            "label_var": [0.02] * 100,
        }
    )
    result = calibrate_prior_precision(
        calibration, side="offense", label_column="label", prior_column="prior", label_variance_column="label_var"
    )
    assert result.tau_squared == 0.0
    assert result.status == "boundary_zero"


def test_precision_fit_requires_earlier_identified_side_precision() -> None:
    frame = pd.DataFrame(
        [
            {
                "home_poss": bool(i % 2), "pts": float(i % 3),
                **{f"a{j}": j for j in range(1, 6)}, **{f"h{j}": j + 5 for j in range(1, 6)},
                "season": 2024, "date": "2024-01-01", "period": 1, "num": i, "gameid": "g",
            }
            for i in range(30)
        ]
    )
    design = build_design(frame)
    precision = PriorPrecision("offense", 0.1, 0.2, 0.1, "identified")
    beta, _, penalty = fit_precision_aware_center(
        design, RapmConfig((2024,), lambda_home=5), np.zeros(design.X.shape[1]),
        sigma_squared=0.2, offense_precision=precision,
        defense_precision=PriorPrecision("defense", 0.2, 0.3, 0.1, "identified"),
    )
    assert np.isfinite(beta).all()
    assert penalty[0] == pytest.approx(2.0)
    assert penalty[len(design.players)] == pytest.approx(1.0)
