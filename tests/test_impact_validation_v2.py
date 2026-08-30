from __future__ import annotations

import numpy as np
import pandas as pd

import nba_impact.models.impact_validation_v2 as module
from nba_impact.models.impact_validation_v2 import (
    deterministic_game_folds,
    historical_box15_panel,
    paired_whole_game_mse_bootstrap,
    select_box_alpha_rolling_origin,
)


def test_game_folds_are_deterministic_exhaustive_and_disjoint() -> None:
    possessions = pd.DataFrame(
        {
            "gameid": ["g3", "g1", "g2", "g1", "g4", "g5"],
            "date": [
                "2021-01-03",
                "2021-01-01",
                "2021-01-02",
                "2021-01-01",
                "2021-01-04",
                "2021-01-05",
            ],
        }
    )
    first = deterministic_game_folds(possessions, folds=2)
    second = deterministic_game_folds(possessions.sample(frac=1, random_state=7), folds=2)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["game_id"]) == {"g1", "g2", "g3", "g4", "g5"}
    assert not first["game_id"].duplicated().any()
    assert set(first["fold"]) == {0, 1}


def test_historical_panel_forbids_rating_season_labels() -> None:
    features = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1],
            "Window_End": [2018, 2019, 2021],
            **{feature: [1.0, 2.0, 3.0] for feature in module.BOX_PIPM_STYLE_FEATURES},
        }
    )
    targets = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1],
            "Window_End": [2018, 2019, 2021],
            "target_offense": [0.0, 0.0, 99.0],
            "target_defense": [0.0, 0.0, 99.0],
            "target_net": [0.0, 0.0, 198.0],
            "Poss_Off": [100, 100, 100],
            "Poss_Def": [100, 100, 100],
        }
    )
    panel = historical_box15_panel(features, targets, rating_season=2021)
    assert panel["Window_End"].tolist() == [2018, 2019]
    assert panel["Window_End"].max() < 2021


def test_alpha_selection_never_trains_on_validation_or_later_windows(
    monkeypatch,
) -> None:
    train = pd.DataFrame(
        {
            "Window_End": [2018, 2019, 2020, 2021],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": [0.0, 0.0, 0.0, 0.0],
            "sample_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    calls: list[tuple[int, int]] = []

    class Model:
        def __init__(self, train_end: int):
            self.train_end = train_end

        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            validation_end = int(frame["feature"].iloc[0] + 2017)
            calls.append((self.train_end, validation_end))
            return np.zeros(len(frame))

    def fake_fit(frame, features, target, alpha):
        return Model(int(frame["Window_End"].max()))

    monkeypatch.setattr(module, "fit_box_model", fake_fit)
    select_box_alpha_rolling_origin(
        train, ("feature",), "target", alpha_grid=(10.0, 100.0)
    )
    assert calls
    assert all(train_end < validation_end for train_end, validation_end in calls)


def test_frozen_box15_contract_keeps_shared_fields_on_offensive_exposure() -> None:
    selected = {
        "offense": module.BOX_PIPM_STYLE_FEATURES,
        "defense": module.BOX_PIPM_STYLE_FEATURES,
    }
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1],
            "Window_End": [2020, 2021],
            "OffPoss": [100.0, 300.0],
            "DefPoss": [300.0, 100.0],
            **{
                feature: [0.0, 4.0]
                for feature in module.BOX_PIPM_STYLE_FEATURES
            },
        }
    )
    pooled = module.build_rolling_five_year_features(
        annual, None, selected, window_ends=(2021,)
    )
    assert pooled.loc[0, "STL_p100"] == 3.0
    assert pooled.loc[0, "DREB_p100"] == 3.0


def test_paired_bootstrap_resamples_games_and_reproduces() -> None:
    rows = []
    for game, actual in (("g1", 4.0), ("g2", -2.0), ("g3", 1.0)):
        rows.extend(
            [
                {
                    "season": 2021,
                    "game_id": game,
                    "candidate": "zero_prior_rapm",
                    "actual_margin": actual,
                    "predicted_margin": 0.0,
                },
                {
                    "season": 2021,
                    "game_id": game,
                    "candidate": "box15_aio",
                    "actual_margin": actual,
                    "predicted_margin": actual / 2,
                },
            ]
        )
    predictions = pd.DataFrame(rows)
    first_summary, first_draws = paired_whole_game_mse_bootstrap(
        predictions, draws=100, seed=7
    )
    second_summary, second_draws = paired_whole_game_mse_bootstrap(
        predictions.sample(frac=1, random_state=9), draws=100, seed=7
    )
    pd.testing.assert_frame_equal(first_summary, second_summary)
    pd.testing.assert_frame_equal(first_draws, second_draws)
    assert first_summary.loc[0, "games"] == 3
    assert first_summary.loc[0, "mean_mse_delta_candidate_minus_reference"] < 0
