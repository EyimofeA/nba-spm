from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from nba_impact.models.predictive_current_aio import (
    build_season_statistics,
    build_spm_center,
    fit_from_season_statistics,
)
from nba_impact.models.rapm import RapmDesign


def _design() -> RapmDesign:
    players = np.asarray([1, 2], dtype=int)
    x = csr_matrix(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, -1.0],
                [1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 0.0, -1.0],
            ]
        )
    )
    return RapmDesign(
        X=x,
        y=np.asarray([2.0, 0.0, 3.0, 1.0]),
        players=players,
        game_ids=np.asarray(["a", "a", "b", "b"]),
        seasons=np.asarray([2020, 2020, 2021, 2021]),
        home_offense=np.asarray([True, False, True, False]),
        off_possessions=np.asarray([2, 2]),
        def_possessions=np.asarray([2, 2]),
    )


def test_season_statistics_recover_unweighted_fit_and_components() -> None:
    design = _design()
    stats = build_season_statistics(design)
    beta, intercept, off, deff = fit_from_season_statistics(
        stats,
        (2020, 2021),
        n_players=2,
        lambda_off=3.0,
        lambda_def=3.0,
        lambda_home=1.0,
        half_life=None,
    )
    assert np.isfinite(beta).all()
    assert np.isfinite(intercept)
    np.testing.assert_allclose(off, [2.0, 2.0])
    np.testing.assert_allclose(deff, [2.0, 2.0])
    assert abs(np.average(beta[:2], weights=off)) < 1e-12
    assert abs(np.average(beta[2:4], weights=deff)) < 1e-12


def test_spm_center_uses_positive_good_defense_and_reports_coverage() -> None:
    design = _design()
    predictions = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2],
            "Target_Season": [2022, 2022],
            "predicted_offense": [2.0, -2.0],
            "predicted_defense": [1.0, -1.0],
            "predicted_net": [3.0, -3.0],
        }
    )
    center, coverage = build_spm_center(
        design,
        predictions,
        target_season=2022,
        off_exposure=np.asarray([10.0, 10.0]),
        def_exposure=np.asarray([10.0, 10.0]),
        test_mask=np.ones(4, dtype=bool),
    )
    np.testing.assert_allclose(center[:2], [0.02, -0.02])
    np.testing.assert_allclose(center[2:4], [-0.01, 0.01])
    assert coverage["test_lineup_slot_coverage"] == 1.0


def test_time_decay_downweights_old_season() -> None:
    stats = build_season_statistics(_design())
    old_weight = 2.0 ** ((2020 - 2021) / 0.5)
    beta, _, off, _ = fit_from_season_statistics(
        stats,
        (2020, 2021),
        n_players=2,
        lambda_off=3.0,
        lambda_def=3.0,
        lambda_home=1.0,
        half_life=0.5,
    )
    assert np.isfinite(beta).all()
    np.testing.assert_allclose(off, [1.0 + old_weight, 1.0 + old_weight])
