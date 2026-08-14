from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_impact.models.precision_aware_prior import (
    PriorPrecision,
    build_prior_calibration_panel,
    calibrate_prior_precision,
    fit_precision_aware_center,
    run_precision_aware_prior_comparison,
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


def test_calibration_uses_heteroskedastic_label_variance_not_a_pooled_subtraction() -> None:
    rng = np.random.default_rng(42)
    label_variance = np.concatenate([np.full(100, 0.0001), np.full(100, 0.04)])
    latent = rng.normal(0.0, 0.02, size=len(label_variance))
    observed = latent + rng.normal(0.0, np.sqrt(label_variance))
    result = calibrate_prior_precision(
        pd.DataFrame({"label": observed, "prior": 0.0, "label_var": label_variance}),
        side="offense",
        label_column="label",
        prior_column="prior",
        label_variance_column="label_var",
    )
    assert result.status == "identified"
    assert result.tau_squared == pytest.approx(0.0004, abs=0.0004)
    assert result.calibration_rows == 200


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


def test_four_model_runner_uses_only_earlier_calibration_windows() -> None:
    rows = []
    for season in range(2017, 2023):
        for game in range(2):
            for possession in range(10):
                rows.append({
                    "home_poss": bool(possession % 2), "pts": float((game + possession) % 3),
                    **{f"a{j}": j for j in range(1, 6)}, **{f"h{j}": j + 5 for j in range(1, 6)},
                    "season": season, "date": f"{season-1}-11-01", "period": 1,
                    "num": possession, "gameid": f"{season}_{game}",
                })
    players = list(range(1, 21))
    priors = pd.DataFrame([
        {"PLAYER_ID": p, "Window_End": end, "prior_offense_per_100": p / 20, "prior_defense_per_100": p / 30, "prior_net_per_100": p / 12}
        for end in range(2018, 2022) for p in players
    ])
    calibration = pd.DataFrame([
        {"PLAYER_ID": p, "window_end": end, "offense_label": p / 20 + (p % 3) * .1, "offense_prior": p / 20,
         "offense_label_variance": .001, "defense_label": p / 30 + (p % 3) * .1, "defense_prior": p / 30,
         "defense_label_variance": .001}
        for end in range(2017, 2021) for p in players
    ])
    folds, calibrations, _ = run_precision_aware_prior_comparison(
        pd.DataFrame(rows), priors, calibration, RapmConfig(tuple(range(2017, 2023)), lambda_off=20, lambda_def=20),
        test_seasons=(2021, 2022), train_window=3, selection_seasons=(2021,), diagnostic_seasons=(2022,), bootstrap_repetitions=10,
    )
    assert set(folds["candidate"]) == {"zero_prior", "fixed_center_prior", "statistical_prior_only", "precision_aware_side_specific_prior"}
    assert (calibrations["calibration_latest_window"] < calibrations["prior_window_end"]).all()


def test_calibration_panel_keeps_labels_and_priors_in_coefficient_units(monkeypatch) -> None:
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
    n_players = len(design.players)

    def fake_sandwich(_design, _config):
        beta = np.concatenate([np.full(n_players, 0.02), np.full(n_players, -0.03), [0.0]])
        return np.eye(2 * n_players + 1) * 0.0004, beta, 0.0

    monkeypatch.setattr(
        "nba_impact.models.precision_aware_prior.game_cluster_sandwich", fake_sandwich
    )
    priors = pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "Window_End": 2024,
            "prior_offense_per_100": 2.5,
            "prior_defense_per_100": 1.5,
        }
    )
    panel = build_prior_calibration_panel(
        frame, priors, RapmConfig((2024,)), window_ends=(2024,), window_length=1
    )
    assert panel["offense_label"].eq(0.02).all()
    assert panel["defense_label"].eq(0.03).all()
    assert panel["offense_label_variance"].eq(0.0004).all()
    assert panel["offense_prior"].eq(0.025).all()
    assert panel["defense_prior"].eq(0.015).all()
