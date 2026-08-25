from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.forecast_dispersion_calibration import (
    _fit_variance_parameters,
    _gate_summary,
)
from nba_impact.models.predictive_backbone_combo import (
    _decide,
    _weighted_moments,
)


def test_weighted_moments_exact_values() -> None:
    prediction = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.5, 1.0, 4.0])
    weight = np.array([2.0, 1.0, 1.0])
    rmse, correlation = _weighted_moments(prediction, actual, weight)
    residual = prediction - actual
    expected_rmse = float(np.sqrt((weight * residual**2).sum() / weight.sum()))
    assert abs(rmse - expected_rmse) < 1e-12
    assert abs(correlation - 0.7713892158398701) < 1e-9


def test_weighted_moments_zero_variance_correlation_is_nan() -> None:
    rmse, correlation = _weighted_moments(np.ones(3), np.array([1.0, 2.0, 3.0]), np.ones(3))
    assert np.isnan(correlation)
    assert rmse > 0


def test_decision_rules_follow_preregistration() -> None:
    selection_win = {
        "combo_mean_50_50": 1.60,
        "predictive_spm_raw": 1.64,
        "state_space_filtered": 1.65,
    }
    confirmation_win = {
        "combo_mean_50_50": 1.76,
        "predictive_spm_raw": 1.80,
        "state_space_filtered": 1.79,
    }
    assert _decide(selection_win, confirmation_win) == "backbone_combo_promoted"

    confirmation_loss = {
        "combo_mean_50_50": 1.90,
        "predictive_spm_raw": 1.80,
        "state_space_filtered": 1.79,
    }
    assert _decide(selection_win, confirmation_loss) == "backbone_state_space_filtered_won_confirmation"

    selection_loss = {"combo_mean_50_50": 1.70, "predictive_spm_raw": 1.64, "state_space_filtered": 1.65}
    assert _decide(selection_loss, confirmation_win) == "backbone_state_space_filtered_combo_lost_selection"

    tie = {
        "combo_mean_50_50": 1.90,
        "predictive_spm_raw": 1.80,
        "state_space_filtered": 1.80,
    }
    assert _decide(selection_win, tie) == "backbone_state_space_filtered_confirmation_tie"
def test_variance_fit_recovers_exposure_model_and_clips_negative_slope() -> None:
    rng = np.random.default_rng(7)
    exposures = rng.uniform(800.0, 3000.0, size=500)
    noise = rng.normal(0.0, 0.05, size=500)
    residuals = np.sqrt(1.0 + 400.0 / exposures) * noise
    intercept, slope = _fit_variance_parameters(residuals, exposures, np.ones(500))
    assert 0.0005 < intercept < 0.01
    assert 0.2 < slope < 2.0

    decreasing = np.sqrt(np.maximum(2.0 - 600.0 / exposures, 0.05)) * noise
    clipped_intercept, clipped_slope = _fit_variance_parameters(decreasing, exposures, np.ones(500))
    assert clipped_intercept >= 0.0 and clipped_slope == 0.0


def test_gate_summary_flags_out_of_range_dispersion() -> None:
    inside = np.concatenate([np.full(34, 1.5), np.full(34, -1.5)])
    outside = np.concatenate([np.full(16, 3.0), np.full(16, -3.0)])
    frame = pd.DataFrame(
        {
            "forecast_offense": np.zeros(100),
            "panel_target_offense": np.concatenate([inside, outside]),
            "sd_offense": np.full(100, 2.2),
            "weight": np.ones(100),
        }
    )
    passing = _gate_summary(frame, "forecast_offense", "panel_target_offense", "sd_offense")
    assert passing["dispersion_pass"] and passing["coverage_pass"]

    narrow = frame.assign(sd_offense=np.full(100, 0.5))
    failing = _gate_summary(narrow, "forecast_offense", "panel_target_offense", "sd_offense")
    assert not failing["dispersion_pass"] and not failing["coverage_pass"]
