import numpy as np
import pandas as pd

from nba_impact.models.matchup_research import (
    fit_expected_points_model,
    predict_expected_points,
    sequential_predictions,
)


def _rows() -> pd.DataFrame:
    rows = []
    date = pd.Timestamp("2024-01-01")
    for game in range(20):
        for scorer, defender, points in ((1, 3, 8.0), (2, 3, 3.0), (1, 4, 5.0), (2, 4, 1.0)):
            rows.append({
                "game_id": f"{game:010d}", "game_date": date + pd.Timedelta(days=game),
                "person_id": scorer, "matchups_person_id": defender,
                "partial_possessions": 10.0, "player_points": points,
                "home": float(game % 2), "rest_days": 2.0,
                "matchup_field_goals_attempted": 4.0, "matchup_three_pointers_attempted": 1.0,
                "matchup_free_throws_attempted": 1.0, "matchup_turnovers": 1.0,
                "matchup_assists": 1.0,
            })
    return pd.DataFrame(rows)


def test_expected_points_orders_stronger_scorer_and_defender() -> None:
    frame = _rows()
    model = fit_expected_points_model(frame, ridge_penalty=10.0)
    assert model["offense"][model["index"][1]] > model["offense"][model["index"][2]]
    assert model["defense"][model["index"][4]] > model["defense"][model["index"][3]]
    prediction = predict_expected_points(model, frame)
    assert np.isfinite(prediction).all()


def test_sequential_model_predicts_before_updating() -> None:
    frame = _rows()
    prediction, ratings = sequential_predictions(
        frame.iloc[:40], frame.iloc[40:], k_factor=0.02, regression=0.999, ridge_penalty=10.0
    )
    assert len(prediction) == len(frame.iloc[40:])
    assert np.isfinite(prediction).all()
    assert ratings["PLAYER_ID"].is_unique
