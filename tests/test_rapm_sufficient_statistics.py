from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models.rapm import RapmConfig, fit_coefficients
from nba_impact.models.rapm_sufficient_statistics import (
    bivariate_penalty_matrix,
    diagonal_penalty_matrix,
    score_stored_evaluation,
    season_adjusted_design,
    solve_stored_generalized_ridge,
    solve_stored_ridge,
    stored_homoskedastic_ridge_intervals,
    stored_training_diagnostics,
    stored_evaluation_predictions,
    store_lambda_research_matrices,
)


def _possessions() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    players = np.arange(1, 13)
    rows = []
    for season in (2022, 2023, 2024):
        for possession in range(80):
            lineup = rng.choice(players, size=10, replace=False)
            rows.append(
                {
                    "home_poss": bool(possession % 2),
                    "pts": float(rng.choice([0, 1, 2, 3], p=[0.45, 0.05, 0.45, 0.05])),
                    **{f"a{i + 1}": int(value) for i, value in enumerate(lineup[:5])},
                    **{f"h{i + 1}": int(value) for i, value in enumerate(lineup[5:])},
                    "season": season,
                    "date": f"{season - 1}-11-01",
                    "period": 1,
                    "num": possession + 1,
                    "gameid": f"002{season}{possession // 20:03d}",
                }
            )
    return pd.DataFrame(rows)


def test_stored_sufficient_statistics_reproduce_ridge_and_score_games(
    tmp_path: Path,
) -> None:
    frame = _possessions()
    train = frame.loc[frame["season"].isin((2022, 2023))].copy()
    evaluation = frame.loc[frame["season"].eq(2024)].copy()
    config = RapmConfig(
        seasons=(2022, 2023),
        lambda_off=50.0,
        lambda_def=80.0,
        lambda_home=5.0,
    )
    design, _ = season_adjusted_design(train)
    expected_beta, expected_intercept = fit_coefficients(design, config)

    manifest = store_lambda_research_matrices(
        train,
        tmp_path,
        evaluation_frame=evaluation,
        metadata={"test": True},
    )
    beta, intercept, players = solve_stored_ridge(
        tmp_path,
        lambda_off=config.lambda_off,
        lambda_def=config.lambda_def,
        lambda_home=config.lambda_home,
    )

    np.testing.assert_array_equal(players, design.players)
    np.testing.assert_allclose(beta, expected_beta, atol=1e-10)
    np.testing.assert_allclose(intercept, expected_intercept, atol=1e-10)
    assert manifest["evaluation"]["games"] == 4
    assert not (tmp_path / "train_design.npz").exists()
    metrics = score_stored_evaluation(tmp_path, beta, intercept)
    assert metrics["games"] == 4
    assert np.isfinite(list(metrics.values())).all()
    predictions = stored_evaluation_predictions(tmp_path, beta, intercept)
    assert len(predictions) == 4
    assert predictions["game_id"].is_unique


def test_generalized_diagonal_solver_matches_scalar_wrapper(tmp_path: Path) -> None:
    frame = _possessions()
    train = frame.loc[frame["season"].isin((2022, 2023))].copy()
    store_lambda_research_matrices(train, tmp_path)
    scalar_beta, scalar_intercept, players = solve_stored_ridge(
        tmp_path, lambda_off=50.0, lambda_def=80.0, lambda_home=5.0
    )
    penalty = diagonal_penalty_matrix(
        len(players), lambda_off=50.0, lambda_def=80.0, lambda_home=5.0
    )
    generalized = solve_stored_generalized_ridge(tmp_path, penalty)
    np.testing.assert_allclose(generalized.beta, scalar_beta, atol=1e-12)
    np.testing.assert_allclose(generalized.intercept, scalar_intercept, atol=1e-12)
    diagnostics = stored_training_diagnostics(
        tmp_path, generalized, penalty, probes=32, seed=19
    )
    assert 0 < diagnostics["effective_df"] < diagnostics["parameters"]
    assert diagnostics["gcv"] > 0


def test_stored_analytic_intervals_preserve_components(tmp_path: Path) -> None:
    frame = _possessions()
    store_lambda_research_matrices(frame, tmp_path)
    ratings, quality = stored_homoskedastic_ridge_intervals(
        tmp_path,
        lambda_off=50.0,
        lambda_def=80.0,
        lambda_home=5.0,
    )
    np.testing.assert_allclose(
        ratings["offense"] + ratings["defense"], ratings["net"], atol=1e-12
    )
    assert (ratings[["offense_se", "defense_se", "net_se"]] >= 0).all().all()
    assert (ratings["net_ci95_low"] <= ratings["net_ci80_low"]).all()
    assert (ratings["net_ci95_high"] >= ratings["net_ci80_high"]).all()
    assert quality["maximum_component_identity_error"] < 1e-12


def test_zero_correlation_bivariate_penalty_matches_diagonal() -> None:
    diagonal = diagonal_penalty_matrix(
        4, lambda_off=50.0, lambda_def=80.0, lambda_home=5.0
    )
    bivariate = bivariate_penalty_matrix(
        4,
        lambda_off=50.0,
        lambda_def=80.0,
        lambda_home=5.0,
        published_prior_correlation=0.0,
    )
    np.testing.assert_allclose(bivariate.toarray(), diagonal.toarray())


def test_bivariate_penalty_is_positive_definite() -> None:
    for correlation in (-0.75, -0.25, 0.25, 0.75):
        penalty = bivariate_penalty_matrix(
            3,
            lambda_off=np.asarray([100.0, 200.0, 300.0]),
            lambda_def=np.asarray([400.0, 500.0, 600.0]),
            lambda_home=5.0,
            published_prior_correlation=correlation,
        )
        assert np.linalg.eigvalsh(penalty.toarray()).min() > 0


def _lambda_grid_module():
    path = Path(__file__).parents[1] / "research" / "rapm_lab" / "run_lambda_grid.py"
    spec = importlib.util.spec_from_file_location("run_lambda_grid", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_selection_uses_correlation_band_then_mae() -> None:
    module = _lambda_grid_module()
    summaries = pd.DataFrame(
        {
            "selection_mean_correlation": [0.4000, 0.3997, 0.3990],
            "selection_mean_mae": [11.0, 10.5, 10.0],
            "lambda_off": [3000.0, 2000.0, 1000.0],
            "lambda_def": [3000.0, 2000.0, 1000.0],
            "lambda_home": [300.0, 300.0, 300.0],
        }
    )
    candidate = module.select_candidate(summaries, correlation_tolerance=0.0005)
    assert candidate["lambda_off"] == 2000.0


def test_paired_game_bootstrap_is_deterministic() -> None:
    module = _lambda_grid_module()
    predictions = pd.DataFrame(
        {
            "season": [2024] * 4 + [2025] * 4,
            "actual_margin": [1.0, 2.0, -1.0, -2.0] * 2,
            "baseline_prediction": [0.0] * 8,
            "candidate_prediction": [0.8, 1.8, -0.8, -1.8] * 2,
        }
    )
    first, first_summary = module.paired_bootstrap_mse_improvement(
        predictions, draws=50, seed=7
    )
    second, second_summary = module.paired_bootstrap_mse_improvement(
        predictions, draws=50, seed=7
    )
    pd.testing.assert_frame_equal(first, second)
    assert first_summary == second_summary
    assert first_summary["probability_mse_improvement"] == 1.0
