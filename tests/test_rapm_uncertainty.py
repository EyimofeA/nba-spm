from __future__ import annotations

import json

import numpy as np
import pandas as pd

from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients
from nba_impact.models.rapm_uncertainty import (
    RapmUncertaintyConfig,
    _draw_weights,
    fit_weighted_zero_prior,
    game_cluster_sandwich,
    run_rapm_uncertainty,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(41)
    rows = []
    for season in (2024, 2025):
        for game in range(4):
            for possession in range(12):
                away = [1, 2, 3, 4, 5]
                home = [6, 7, 8, 9, 10]
                # A cameo player makes missing-draw handling observable.
                if season == 2025 and game == 3:
                    home[0] = 11
                home_poss = bool(possession % 2)
                rows.append(
                    {
                        "home_poss": home_poss,
                        "pts": 1.05 + 0.03 * home_poss + rng.normal(0, 0.1),
                        **{f"a{i + 1}": value for i, value in enumerate(away)},
                        **{f"h{i + 1}": value for i, value in enumerate(home)},
                        "season": season,
                        "date": f"{season - 1}-11-01",
                        "period": 1,
                        "num": possession + 1,
                        "gameid": f"002{season}{game:04d}",
                    }
                )
    return pd.DataFrame(rows)


def test_weighted_one_fit_matches_frozen_point_estimator() -> None:
    design = build_design(_frame())
    config = RapmConfig(seasons=(2024, 2025), lambda_off=30, lambda_def=30, lambda_home=5)
    expected_beta, expected_intercept = fit_coefficients(design, config)
    beta, intercept, _, _, _ = fit_weighted_zero_prior(design, config)
    np.testing.assert_allclose(beta, expected_beta, atol=1e-10)
    assert intercept == expected_intercept


def test_draw_weights_are_game_whole_and_season_stratified() -> None:
    design = build_design(_frame())
    weights, counts = _draw_weights(design, 9, 4)
    assert counts == {"2024": 4, "2025": 4}
    for season in (2024, 2025):
        rows = np.flatnonzero(design.seasons == season)
        for game in np.unique(design.game_ids[rows]):
            game_weights = weights[rows[design.game_ids[rows] == game]]
            assert len(set(game_weights)) == 1


def test_sandwich_recentering_and_net_covariance_are_joint() -> None:
    design = build_design(_frame())
    config = RapmConfig(seasons=(2024, 2025), lambda_off=30, lambda_def=30, lambda_home=5)
    covariance, beta, _ = game_cluster_sandwich(design, config)
    n_players = len(design.players)
    assert np.allclose(covariance, covariance.T)
    np.testing.assert_allclose(
        np.average(beta[:n_players], weights=design.off_possessions), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.average(beta[n_players : 2 * n_players], weights=design.def_possessions), 0.0, atol=1e-12
    )
    net_variance = (
        np.diag(covariance)[:n_players]
        + np.diag(covariance)[n_players : 2 * n_players]
        - 2 * np.diag(covariance[:n_players, n_players : 2 * n_players])
    )
    assert np.isfinite(net_variance).all()


def test_bootstrap_resume_repairs_deleted_draw_and_preserves_component_identity(tmp_path) -> None:
    frame = _frame()
    config = RapmConfig(seasons=(2024, 2025), lambda_off=30, lambda_def=30, lambda_home=5)
    uncertainty = RapmUncertaintyConfig(draws=8, seed=77)
    first = run_rapm_uncertainty(frame, config, uncertainty, artifact_root=tmp_path)
    root = tmp_path / "models" / "rapm_uncertainty" / first["run_id"]
    missing = root / "bootstrap_draws" / "draw_0003.parquet"
    missing.unlink()
    second = run_rapm_uncertainty(frame, config, uncertainty, artifact_root=tmp_path)
    assert second["run_id"] == first["run_id"]
    assert missing.exists()
    ratings = pd.read_parquet(root / "ratings_uncertainty.parquet")
    np.testing.assert_allclose(
        ratings["net_estimate"], ratings["offense_estimate"] + ratings["defense_estimate"], atol=1e-10
    )
    assert (ratings["net_draw_coverage"] <= uncertainty.draws).all()
    metadata = json.loads((root / "bootstrap_draws" / "draw_0003.json").read_text())
    assert metadata["draw"] == 3
