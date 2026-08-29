import pandas as pd

from nba_impact.data.historical_factor_features import (
    build_rebound_responsibility_features,
    build_shot_context_features,
)


def test_shot_context_uses_fractional_multi_defender_assignments():
    events = pd.DataFrame(
        {
            "game_id": [1, 1, 2, 2],
            "actionNumber": [1, 2, 1, 2],
            "actionType": ["2pt", "3pt", "2pt", "3pt"],
            "person_id": [10, 10, 20, 20],
            "shotResult": ["Made", "Missed", "Missed", "Made"],
            "xLegacy": [0, 230, 0, 230],
            "yLegacy": [20, 40, 20, 40],
            "assisted": [True, False, True, False],
            "description": ["Layup", "Three", "Layup", "Three"],
            "qualifier": ["[]"] * 4,
            "previous_action": [""] * 4,
            "clock": ["PT10M00.00S"] * 4,
        }
    )
    assignments = pd.DataFrame(
        {
            "gi": [1, 1, 1, 2, 2],
            "ei": [1, 1, 2, 1, 2],
            "def_id": [30, 31, 30, 31, 31],
        }
    )
    matchups = pd.DataFrame(
        {
            "person_id": [10, 10, 20],
            "matchups_person_id": [30, 31, 31],
            "partial_possessions": [20.0, 20.0, 40.0],
        }
    )

    features, quality = build_shot_context_features(
        events, assignments, matchups, 2020
    )

    assert set(features["PLAYER_ID"]) == {10, 20, 30, 31}
    assert quality["multi_tagged_shots"] == 1
    assert quality["defender_distance_available"] is False
    assert features["rim_deterrence_vs_scorer_p100_eb"].notna().all()


def test_rebound_responsibility_builds_boxout_and_height_interactions():
    sheet = pd.DataFrame(
        {
            "PLAYER_ID": range(1, 21),
            "OffPoss": [1000.0] * 20,
            "DefPoss": [1000.0] * 20,
            "PTS": [500.0] * 20,
            "FGA": [400.0] * 20,
            "FTA": [100.0] * 20,
            "SelfOReb": [10.0] * 20,
            "PLAYER_HEIGHT_INCHES": range(70, 90),
            "DREB_CHANCES": [200.0] * 20,
            "DREB": range(100, 120),
            "DREB_CONTEST": [50.0] * 20,
            "DREB_CHANCE_DEFER": [20.0] * 20,
            "AVG_DREB_DIST": [7.0] * 20,
            "hustle_DEF_BOXOUTS": [30.0] * 20,
            "hustle_BOX_OUT_PLAYER_TEAM_REBS": [20.0] * 20,
            "hustle_BOX_OUT_PLAYER_REBS": [10.0] * 20,
        }
    )

    features, quality = build_rebound_responsibility_features(sheet, 2020)

    assert features["self_oreb_adjusted_ts"].notna().all()
    assert features["height_x_defensive_boxouts"].nunique() > 1
    assert quality["boxout_coverage"] == 1.0
