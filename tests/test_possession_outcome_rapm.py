import numpy as np
import pandas as pd

from nba_impact.models.possession_outcome_rapm import (
    assign_events_to_possessions,
    build_factor_ledger,
    multinomial_metrics,
    points_class,
)


def test_factor_ledger_infers_historical_three_point_value() -> None:
    possession = {
        "game_id": "g",
        "season_type": "regular",
        "season_end": 2024,
        "possession_id": "p",
        "start_action_number": 1,
        "end_action_number": 2,
        "offense_team_id": 1,
        "offense_is_home": True,
        "points": 3,
        "possession_number": 1,
    }
    possessions = pd.DataFrame([possession])
    segments = pd.DataFrame(
        [{"possession_id": "p", "segment_number": 1, **{f"home_player_{i}": i for i in range(1, 6)}, **{f"away_player_{i}": i + 5 for i in range(1, 6)}}]
    )
    events = pd.DataFrame(
        [{
            "game_id": "g", "season_type": "regular", "season_end": 2024,
            "actionNumber": 1, "actionType": "Made Shot", "teamId": 1,
            "shotValue": np.nan, "description": "Player 25' 3PT Jump Shot",
        }]
    )
    ledger = build_factor_ledger(possessions, segments, events, seasons=(2024,))
    assert ledger.shots["efg_value"].tolist() == [1.5]
    assert ledger.quality["inferred_shot_values"] == 1


def test_true_shooting_uses_field_goals_plus_point_44_free_throws() -> None:
    possessions = pd.DataFrame(
        [{
            "game_id": "g", "season_type": "regular", "season_end": 2024,
            "possession_id": "p", "start_action_number": 1, "end_action_number": 2,
            "offense_team_id": 1, "offense_is_home": True, "points": 4,
            "possession_number": 1,
        }]
    )
    segments = pd.DataFrame(
        [{"possession_id": "p", "segment_number": 1,
          **{f"home_player_{i}": i for i in range(1, 6)},
          **{f"away_player_{i}": i + 5 for i in range(1, 6)}}]
    )
    events = pd.DataFrame(
        [
            {"game_id": "g", "season_type": "regular", "season_end": 2024,
             "actionNumber": 1, "actionType": "Made Shot", "teamId": 1,
             "shotValue": 3, "description": "3PT", "points_added": 3},
            {"game_id": "g", "season_type": "regular", "season_end": 2024,
             "actionNumber": 2, "actionType": "Free Throw", "teamId": 1,
             "shotValue": np.nan, "description": "Free Throw 1 of 1", "points_added": 1},
        ]
    )
    ledger = build_factor_ledger(possessions, segments, events, seasons=(2024,))
    row = ledger.ts_opportunities.iloc[0]
    assert row["ts_attempt_weight"] == 1.44
    assert row["ts_points"] == 4
    assert np.isclose(row["ts_value"], 4 / (2 * 1.44))


def test_event_mapping_respects_possession_end() -> None:
    possessions = pd.DataFrame(
        {
            "game_id": ["g", "g"],
            "possession_id": ["p1", "p2"],
            "start_action_number": [1, 5],
            "end_action_number": [4, 8],
            "offense_team_id": [1, 2],
        }
    )
    events = pd.DataFrame(
        {"game_id": ["g"] * 4, "actionNumber": [1, 4, 5, 9], "actionType": ["a"] * 4}
    )
    mapped = assign_events_to_possessions(possessions, events)
    assert mapped["possession_id"].tolist() == ["p1", "p1", "p2"]


def test_points_class_caps_long_possessions() -> None:
    np.testing.assert_array_equal(points_class(pd.Series([0, 1, 2, 3, 5])), [0, 1, 2, 3, 3])


def test_multinomial_metrics_probability_conservation() -> None:
    frame = pd.DataFrame(
        {"gameid": ["g"] * 4, "home_poss": [1, 0, 1, 0], "pts": [0.0, 1.0, 2.0, 3.0]}
    )
    probabilities = np.eye(4)
    metrics, _ = multinomial_metrics(frame, probabilities, three_plus_value=3.0)
    assert metrics["log_loss"] < 1e-10
    assert metrics["expected_points_rmse"] == 0.0
