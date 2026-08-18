from __future__ import annotations

import pandas as pd

from nba_impact.data.historical_v3_possessions import (
    infer_v3_possession_owners,
    validate_v3_owners_against_cdn,
)


def _events() -> pd.DataFrame:
    home, away = 1610612737, 1610612738
    return pd.DataFrame(
        [
            {"gameId": 21600001, "actionId": 1, "actionNumber": 1, "period": 1, "clock": "PT12M00.00S", "teamId": 0, "personId": 0, "actionType": "period", "subType": "start", "description": "Start"},
            {"gameId": 21600001, "actionId": 2, "actionNumber": 2, "period": 1, "clock": "PT12M00.00S", "teamId": home, "personId": 1, "actionType": "Jump Ball", "subType": None, "description": "Jump Ball"},
            {"gameId": 21600001, "actionId": 3, "actionNumber": 3, "period": 1, "clock": "PT11M40.00S", "teamId": home, "personId": 1, "actionType": "Made Shot", "subType": "Jump Shot", "description": "Player 18' Jump Shot"},
            {"gameId": 21600001, "actionId": 4, "actionNumber": 4, "period": 1, "clock": "PT11M20.00S", "teamId": away, "personId": 2, "actionType": "Missed Shot", "subType": "Jump Shot", "description": "MISS Player 3PT Jump Shot"},
            {"gameId": 21600001, "actionId": 5, "actionNumber": 5, "period": 1, "clock": "PT11M18.00S", "teamId": 0, "personId": home, "actionType": "Rebound", "subType": "Unknown", "description": "TEAM Rebound"},
            {"gameId": 21600001, "actionId": 6, "actionNumber": 6, "period": 1, "clock": "PT11M00.00S", "teamId": home, "personId": 1, "actionType": "Turnover", "subType": "Bad Pass", "description": "Turnover"},
            {"gameId": 21600001, "actionId": 7, "actionNumber": 7, "period": 1, "clock": "PT10M40.00S", "teamId": away, "personId": 2, "actionType": "Made Shot", "subType": "Jump Shot", "description": "Player 25' 3PT Jump Shot"},
            {"gameId": 21600001, "actionId": 8, "actionNumber": 8, "period": 1, "clock": "PT00M00.00S", "teamId": 0, "personId": 0, "actionType": "period", "subType": "end", "description": "End"},
        ]
    )


def test_v3_owner_state_machine_uses_team_rebound_and_action_points() -> None:
    result, issues = infer_v3_possession_owners(_events())
    home, away = 1610612737, 1610612738

    assert issues["invalid_two_team_games"] == 0
    assert result["possession"].tolist() == [0, home, home, away, home, home, away, home]
    assert result["points_added"].tolist() == [0, 0, 2, 0, 0, 0, 3, 0]
    assert result.loc[result["actionNumber"].eq(5), "owner_rule"].item() == "rebound_actor"


def test_v3_owner_validation_passes_exact_synthetic_reference() -> None:
    events = _events()
    home, away = 1610612737, 1610612738
    cdn = pd.DataFrame(
        {
            "gameId": [21600001] * 8,
            "actionNumber": list(range(1, 9)),
            "possession": [0, home, home, away, home, home, away, home],
        }
    )

    metrics = validate_v3_owners_against_cdn(events, cdn)

    assert metrics["passed"] is True
    assert metrics["core_action_agreement"] == 1.0
    assert metrics["exact_owner_sequence_game_rate"] == 1.0
