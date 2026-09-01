from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from nba_impact.models.predictive_current_aio import (
    build_partitioned_weekly_cutoff_ledger,
    build_weekly_cutoff_ledger,
    build_season_statistics,
    build_spm_center,
    fit_from_season_statistics,
    _validate_spm_prior_lineage,
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


def test_spm_center_keeps_missing_prior_players_at_zero() -> None:
    design = _design()
    predictions = pd.DataFrame(
        {
            "PLAYER_ID": [1],
            "Target_Season": [2022],
            "predicted_offense": [2.0],
            "predicted_defense": [1.0],
            "predicted_net": [3.0],
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
    np.testing.assert_allclose(center[:4], 0.0)
    assert coverage["train_off_possession_coverage"] == 0.5


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


def test_spm_prior_lineage_requires_pinned_run_and_past_only_cutoffs(tmp_path) -> None:
    run = tmp_path / "frozen_spm_run"
    run.mkdir()
    path = run / "selected_predictions.parquet"
    valid = pd.DataFrame(
        {
            "method": ["raw", "raw"],
            "Target_Season": [2025, 2026],
            "training_target_end": [2024, 2025],
        }
    )
    _validate_spm_prior_lineage({"spm_prior_run": run.name}, path, valid)
    leaked = valid.copy()
    leaked.loc[1, "training_target_end"] = 2026
    with pytest.raises(ValueError, match="strictly before"):
        _validate_spm_prior_lineage({"spm_prior_run": run.name}, path, leaked)
    with pytest.raises(ValueError, match="Expected SPM prior run"):
        _validate_spm_prior_lineage({"spm_prior_run": "another_run"}, path, valid)


def test_weekly_cutoff_ledger_enforces_the_frozen_14_day_window() -> None:
    frame = pd.DataFrame(
        {
            "season": [2025, *([2026] * 4)],
            "date": [
                "2025-03-01",
                "2025-10-31",
                "2025-11-03",
                "2025-11-16",
                "2025-11-17",
            ],
            "gameid": ["prior", "seen", "a", "b", "c"],
        }
    )
    ledger = build_weekly_cutoff_ledger(frame, target_season=2026)
    shuffled = build_weekly_cutoff_ledger(
        frame.sample(frac=1.0, random_state=7), target_season=2026
    )

    first = ledger.iloc[0]
    assert first["cutoff_date"] == pd.Timestamp("2025-11-03")
    assert first["horizon_end_exclusive"] == pd.Timestamp("2025-11-17")
    assert first["observed_update_games"] == 1
    assert first["latest_observed_game_date"] < first["cutoff_date"]
    assert first["oracle_games"] == 2
    assert first["oracle_possession_rows"] == 2
    assert first["first_oracle_game_date"] >= first["cutoff_date"]
    assert first["last_oracle_game_date"] < first["horizon_end_exclusive"]
    assert ledger.iloc[-1]["cutoff_date"] == pd.Timestamp("2026-03-30")
    assert ledger["cutoff_date"].dt.dayofweek.eq(0).all()
    assert ledger["oracle_game_rowset_hash"].equals(
        shuffled["oracle_game_rowset_hash"]
    )

    conflicting = pd.concat(
        [
            frame,
            pd.DataFrame(
                {"season": [2025], "date": ["2025-03-02"], "gameid": ["prior"]}
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="exactly one date"):
        build_weekly_cutoff_ledger(conflicting, target_season=2026)


def test_partitioned_weekly_cutoff_ledger_scores_each_game_once() -> None:
    frame = pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "date": ["2025-11-03", "2025-11-04", "2025-11-09", "2025-11-10"],
            "gameid": ["a", "b", "c", "d"],
        }
    )
    ledger = build_partitioned_weekly_cutoff_ledger(frame, target_season=2026)
    assert ledger["oracle_games"].sum() == 4
    assert ledger["cutoff_date"].tolist() == [
        pd.Timestamp("2025-11-03"),
        pd.Timestamp("2025-11-10"),
    ]
    assert ledger["oracle_games"].tolist() == [3, 1]
