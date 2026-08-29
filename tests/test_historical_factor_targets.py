import pandas as pd

from nba_impact.models.historical_factor_targets import (
    build_historical_factor_ledger,
    derive_game_dim,
)


def test_factor_ledger_uses_miss_lineup_and_positive_good_oreb_target():
    lineups = {
        "off_players_on": "1|2|3|4|5",
        "def_players_on": "6|7|8|9|10",
        "poc_ok": True,
        "date": 20131030,
        "playoffs": False,
        "season": 2014,
        "scoreHome": 0,
        "scoreAway": 0,
    }
    events = pd.DataFrame(
        [
            {**lineups, "game_id": "0021300001", "period": 1, "actionNumber": 1, "actionType": "2pt", "description": "MISS", "shotResult": "Missed", "person_id": 1, "teamId": 100, "xLegacy": 0, "yLegacy": 20},
            {**lineups, "game_id": "0021300001", "period": 1, "actionNumber": 2, "actionType": "rebound", "description": "Offensive rebound", "shotResult": None, "person_id": 2, "teamId": 100, "xLegacy": None, "yLegacy": None},
            {**lineups, "game_id": "0021300001", "period": 1, "actionNumber": 3, "actionType": "3pt", "description": "Made three", "shotResult": "Made", "person_id": 1, "teamId": 100, "xLegacy": 220, "yLegacy": 30, "scoreHome": 3},
            {**lineups, "game_id": "0021300001", "period": 1, "actionNumber": 4, "actionType": "freethrow", "description": "Made free throw", "shotResult": "Made", "person_id": 1, "teamId": 100, "xLegacy": None, "yLegacy": None, "scoreHome": 4},
        ]
    )
    games = pd.DataFrame(
        [{"game_id": "0021300001", "home_team_id": 100, "away_team_id": 200}]
    )
    ledger = build_historical_factor_ledger(events, games)
    assert ledger.opponent_oreb["offensive_rebound"].tolist() == [1.0]
    assert ledger.opponent_oreb["home_poss"].tolist() == [1]
    assert ledger.opponent_oreb[["h1", "h2", "h3", "h4", "h5"]].iloc[0].tolist() == [1, 2, 3, 4, 5]
    assert ledger.shooting["ts_attempt_weight"].tolist() == [1.0, 1.0, 0.44]
    assert ledger.shooting["ts_value"].tolist() == [0.0, 1.5, 1.0 / 0.88]


def test_game_dim_is_derived_from_observed_scoring():
    events = pd.DataFrame(
        {
            "game_id": ["0000000001"] * 4,
            "period": [1] * 4,
            "actionNumber": [1, 2, 3, 4],
            "actionType": ["2pt", "3pt", "3pt", "freethrow"],
            "shotResult": ["Made"] * 4,
            "teamId": [10, 20, 10, 20],
            "scoreHome": [2, 2, 5, 5],
            "scoreAway": [0, 3, 3, 4],
        }
    )

    games, quality = derive_game_dim(events)

    assert games.loc[0, "home_team_id"] == 10
    assert games.loc[0, "away_team_id"] == 20
    assert quality["games_without_score_derived_teams"] == 0
