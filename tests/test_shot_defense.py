from __future__ import annotations

import pandas as pd
import pytest

from nba_impact.data.shot_defense import compute_shot_defense_events


def _inputs() -> tuple[pd.DataFrame, ...]:
    game_id = "0022300001"
    actions = [10, 20, 30]
    v3 = pd.DataFrame(
        {
            "gameId": [game_id] * 3,
            "actionId": [1, 2, 3],
            "actionNumber": actions,
            "period": [1, 1, 1],
            "clock": ["PT10M00.00S", "PT00M02.00S", "PT00M01.00S"],
            "teamId": [100, 100, 100],
            "personId": [1, 1, 1],
            "isFieldGoal": [1, 1, 1],
        }
    )
    shot_detail = pd.DataFrame(
        {
            "GAME_ID": [game_id] * 3,
            "GAME_EVENT_ID": actions,
            "PLAYER_ID": [1, 1, 1],
            "PERIOD": [1, 1, 1],
            "MINUTES_REMAINING": [10, 0, 0],
            "SECONDS_REMAINING": [0, 2, 1],
            "SHOT_TYPE": ["2PT Field Goal", "3PT Field Goal", "3PT Field Goal"],
            "SHOT_ZONE_BASIC": ["Restricted Area", "Above the Break 3", "Backcourt"],
            "SHOT_DISTANCE": [2, 40, 55],
            "LOC_X": [0, 20, 0],
            "LOC_Y": [20, 350, 700],
            "SHOT_ATTEMPTED_FLAG": [1, 1, 1],
            "SHOT_MADE_FLAG": [1, 0, 0],
        }
    )
    cdn = pd.DataFrame(
        {
            "gameId": [game_id] * 3,
            "actionNumber": actions,
            "orderNumber": [100, 200, 300],
            "period": [1, 1, 1],
            "clock": ["PT10M00.00S", "PT00M02.00S", "PT00M01.00S"],
            "possession": [100, 100, 100],
        }
    )
    states = pd.DataFrame(
        {
            "game_id": [game_id] * 3,
            "actionId": [1, 2, 3],
            "actionNumber": actions,
            "home_score_diff_before": [0, -2, -2],
            "seconds_remaining_period": [600.0, 2.0, 1.0],
            "regulation_seconds_remaining": [2760.0, 2162.0, 2161.0],
        }
    )
    segments = pd.DataFrame(
        {
            "game_id": [game_id],
            "start_order_number": [1],
            "end_order_number": [400],
            "ordinal_stint_id": [f"{game_id}_o001"],
            "offense_team_id": [100],
            **{f"home_player_{index}": [index] for index in range(1, 6)},
            **{f"away_player_{index}": [index + 5] for index in range(1, 6)},
        }
    )
    games = pd.DataFrame(
        {
            "game_id": [game_id],
            "season_start": [2023],
            "season_end": [2024],
            "season_label": ["2023-24"],
            "season_type": ["regular"],
            "game_date": ["2023-10-24"],
            "home_team_id": [100],
            "away_team_id": [200],
        }
    )
    return v3, shot_detail, cdn, states, segments, games


def test_shot_defense_panel_uses_exact_ordinal_lineup_and_excludes_heaves() -> None:
    panel, quality = compute_shot_defense_events(*_inputs())
    assert quality["passed"]
    assert quality["heave_rows_excluded"] == 1
    assert quality["backcourt_rows_excluded"] == 1
    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["shot_zone"] == "rim"
    assert row["shot_made"] == 1
    assert row["shooter_id"] == 1
    assert row[[f"offense_player_{index}" for index in range(1, 6)]].tolist() == [1, 2, 3, 4, 5]
    assert row[[f"defense_player_{index}" for index in range(1, 6)]].tolist() == [6, 7, 8, 9, 10]


def test_shot_defense_panel_rejects_incomplete_cdn_alignment() -> None:
    values = list(_inputs())
    values[2] = values[2].iloc[:1].copy()
    with pytest.raises(ValueError, match="failed quality gates"):
        compute_shot_defense_events(*values)
