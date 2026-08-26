from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.predictive_trajectory_ablation import (
    METHODS,
    build_model_rows,
    predict_method,
    run_walk_forward,
)


def _rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    metadata = []
    for season in range(2019, 2023):
        for player in range(1, 31):
            age = 20 + player % 15
            offense = 0.1 * player
            defense = -0.05 * player
            age_effect = -0.04 * (age + 1 - 27) ** 2 / 25.0
            predictions.append(
                {
                    "PLAYER_ID": player,
                    "Target_Season": season,
                    "Window_End": season,
                    "raw_offense": offense,
                    "raw_defense": defense,
                    "raw_net": offense + defense,
                    "target_offense": offense + age_effect,
                    "target_defense": defense + age_effect,
                    "target_net": offense + defense + 2 * age_effect,
                    "sample_weight": 10.0,
                }
            )
            for value in (season - 1, season):
                metadata.append(
                    {
                        "PLAYER_ID": player,
                        "Season": value,
                        "PLAYER_NAME": f"P{player}",
                        "TEAM_ABBREVIATION": "A" if player % 2 else "B",
                        "AGE": age + value - (season - 1),
                        "MIN": 500.0 + player,
                        "GP": 50.0,
                    }
                )
    meta = pd.DataFrame(metadata).drop_duplicates(["PLAYER_ID", "Season"], keep="last")
    return pd.DataFrame(predictions), meta


def test_model_rows_use_prior_season_metadata() -> None:
    predictions, metadata = _rows()
    rows, quality = build_model_rows(
        predictions, metadata, scored_seasons=(2020, 2021, 2022)
    )
    assert quality["excluded_missing_origin_metadata"] == 0
    assert rows["forecast_age"].eq(rows["origin_age"] + 1).all()
    assert rows["Origin_Season"].eq(rows["Target_Season"] - 1).all()


def test_every_method_preserves_component_identity() -> None:
    predictions, metadata = _rows()
    rows, _ = build_model_rows(predictions, metadata, scored_seasons=(2020, 2021, 2022))
    train = rows.loc[rows["Target_Season"].lt(2022)]
    test = rows.loc[rows["Target_Season"].eq(2022)]
    for method in METHODS:
        output = predict_method(train, test, method, alpha=25.0)
        np.testing.assert_allclose(
            output["predicted_net"],
            output["predicted_offense"] + output["predicted_defense"],
        )


def test_walk_forward_never_uses_test_or_future_targets() -> None:
    predictions, metadata = _rows()
    rows, _ = build_model_rows(predictions, metadata, scored_seasons=(2020, 2021, 2022))
    scored, metrics = run_walk_forward(
        rows, scored_seasons=(2020, 2021, 2022), alpha=25.0
    )
    assert (scored["training_target_end"] < scored["Target_Season"]).all()
    assert set(metrics["target_season"]) == {2020, 2021, 2022}
    assert set(metrics["method"]) == set(METHODS)
