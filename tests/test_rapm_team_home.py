from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.rapm_lab.run_team_home_adjustments import (  # noqa: E402
    FitResult,
    PreparedFold,
    config_id,
    fit_model,
    select_candidate,
    team_effects,
)
from scipy.sparse import csr_matrix


def test_candidate_selection_uses_rmse_tolerance_then_more_shrinkage() -> None:
    summary = pd.DataFrame(
        {
            "variant": ["team_lambda_30", "team_lambda_300", "team_lambda_3000"],
            "team_home_lambda": [30.0, 300.0, 3000.0],
            "mean_rmse": [14.00, 14.005, 14.02],
        }
    )
    assert select_candidate(summary, tolerance=0.01)["team_home_lambda"] == 300.0


def test_team_effect_output_uses_net_home_scale() -> None:
    fit = FitResult(
        beta=np.asarray([0.0, 0.0, 0.01, 0.002, -0.002]),
        intercept=1.1,
        n_players=1,
        teams=np.asarray([10, 20]),
        team_home_lambda=300.0,
    )
    result = team_effects(
        fit,
        {10: "AAA", 20: "BBB"},
        evaluation_season=2026,
        variant=config_id(300.0),
    )
    np.testing.assert_allclose(result["home_net_advantage_per_100"], [2.4, 1.6])


def test_config_ids_are_stable() -> None:
    assert config_id(None) == "baseline_global"
    assert config_id(300.0) == "team_lambda_300"


def test_team_deviations_are_constrained_during_fit() -> None:
    # One offense player, one defense player, one global home column, and two
    # team columns. The raw team columns sum exactly to global home.
    X = csr_matrix(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, -1.0, 0.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, -1.0, 0.0, -1.0],
        ]
    )
    y = np.asarray([2.0, 0.0, 2.0, 0.0])
    intercept = float(y.mean())
    prepared = PreparedFold(
        X=X,
        y=y,
        evaluation=pd.DataFrame(),
        train_rows=4,
        n_players=1,
        teams=np.asarray([10, 20]),
        team_weights=np.asarray([2.0, 2.0]),
        xtx=(X.T @ X).tocsr(),
        rhs=np.asarray(X.T @ (y - intercept)).ravel(),
        intercept=intercept,
    )
    fit = fit_model(
        prepared,
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_global_home=300.0,
        team_home_lambda=30.0,
    )
    deviations = fit.beta[-2:]
    assert abs(np.average(deviations, weights=prepared.team_weights)) < 1e-10
