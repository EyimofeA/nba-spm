from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.sparse_function_spm import (
    FEATURE_SPECS,
    fit_historical_predictions,
    selected_features,
    standardize_within_window,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    rows = []
    targets = []
    offense = selected_features()["offense"]
    defense = selected_features()["defense"]
    for season in range(2018, 2023):
        for player in range(30):
            values = {feature: rng.normal() for feature in (*offense, *defense)}
            rows.append({"PLAYER_ID": player, "Window_End": season, **values})
            targets.append(
                {
                    "PLAYER_ID": player,
                    "Window_End": season,
                    "target_offense": values[offense[0]] + 0.2 * values[offense[1]],
                    "target_defense": values[defense[0]] - 0.2 * values[defense[-1]],
                    "target_net": 0.0,
                    "Poss_Off": 1000 + player,
                    "Poss_Def": 990 + player,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(targets)


def test_contract_uses_one_feature_per_function() -> None:
    for side, specs in FEATURE_SPECS.items():
        assert len(specs) == len(selected_features()[side])
        assert len({function for function, _, _ in specs}) == len(specs)
        assert len(set(selected_features()[side])) == len(specs)


def test_within_window_standardization_has_zero_means() -> None:
    features, _ = _fixture()
    standardized = standardize_within_window(features)
    for feature in (*selected_features()["offense"], *selected_features()["defense"]):
        means = standardized.groupby("Window_End")[feature].mean()
        assert np.allclose(means, 0.0, atol=1e-12)


def test_historical_fit_uses_only_earlier_complete_windows() -> None:
    features, targets = _fixture()
    predictions, coefficients, models = fit_historical_predictions(
        standardize_within_window(features),
        targets,
        alpha=30.0,
        rating_seasons=(2021, 2022),
    )
    assert set(predictions["Window_End"]) == {2021, 2022}
    assert predictions.loc[
        predictions["Window_End"].eq(2021), "training_window_end"
    ].eq(2020).all()
    assert predictions.loc[
        predictions["Window_End"].eq(2022), "training_window_end"
    ].eq(2021).all()
    assert np.allclose(
        predictions["prediction_offense"] + predictions["prediction_defense"],
        predictions["prediction_net"],
    )
    assert set(coefficients["side"]) == {"offense", "defense"}
    assert set(models) == {"offense", "defense"}
