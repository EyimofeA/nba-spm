from __future__ import annotations

from nba_impact.data.official_matchups import flatten_official_matchups
from nba_impact.data.matchup_defense_features import RAW_COLUMNS


def test_flatten_official_matchups_maps_v3_statistics() -> None:
    statistics = {
        "matchupMinutes": "1:00", "matchupMinutesSort": 60.0,
        "partialPossessions": 5.0, "percentageDefenderTotalTime": 0.1,
        "percentageOffensiveTotalTime": 0.1, "percentageTotalTimeBothOn": 0.1,
        "switchesOn": 0, "playerPoints": 5, "teamPoints": 5,
        "matchupAssists": 1, "matchupPotentialAssists": 1, "matchupTurnovers": 1,
        "matchupBlocks": 0, "matchupFieldGoalsMade": 2,
        "matchupFieldGoalsAttempted": 3, "matchupFieldGoalsPercentage": 0.667,
        "matchupThreePointersMade": 0, "matchupThreePointersAttempted": 1,
        "matchupThreePointersPercentage": 0.0, "helpBlocks": 0,
        "helpFieldGoalsMade": 0, "helpFieldGoalsAttempted": 0,
        "helpFieldGoalsPercentage": 0.0, "matchupFreeThrowsMade": 1,
        "matchupFreeThrowsAttempted": 1, "shootingFouls": 0,
    }
    payload = {
        "boxScoreMatchups": {
            "gameId": "0022500001", "homeTeamId": 1, "awayTeamId": 2,
            "homeTeam": {"teamId": 1, "players": [{"personId": 10, "matchups": [{"personId": 20, "statistics": statistics}]}]},
            "awayTeam": {"teamId": 2, "players": []},
        }
    }
    frame = flatten_official_matchups(payload, "0022500001")
    assert len(frame) == 1
    assert set(RAW_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, "person_id"] == 10
    assert frame.loc[0, "matchups_person_id"] == 20
    assert frame.loc[0, "matchup_field_goals_attempted"] == 3
