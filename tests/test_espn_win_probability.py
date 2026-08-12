from __future__ import annotations

import pandas as pd
import pytest

from nba_impact.data.espn_win_probability import (
    extract_espn_win_probability,
    match_scoreboard_games,
    parse_espn_clock,
)
from nba_impact.models.win_probability_benchmark import match_espn_to_local_states


def test_parse_espn_clock_handles_last_minute_tenths() -> None:
    assert parse_espn_clock("11:48") == 708.0
    assert parse_espn_clock("5.1") == 5.1


def test_scoreboard_matching_normalizes_team_aliases_and_checks_score() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["0022400061"],
            "season_label": ["2024-25"],
            "game_date": pd.to_datetime(["2024-10-22"]),
            "home_team_tricode": ["BOS"],
            "away_team_tricode": ["WAS"],
            "home_score": [132],
            "away_score": [109],
        }
    )
    scoreboard = {
        "events": [
            {
                "id": "401704627",
                "competitions": [
                    {
                        "competitors": [
                            {"homeAway": "home", "score": "132", "team": {"abbreviation": "BOS"}},
                            {"homeAway": "away", "score": "109", "team": {"abbreviation": "WSH"}},
                        ]
                    }
                ],
            }
        ]
    }
    row = match_scoreboard_games(games, scoreboard)[0]
    assert row["status"] == "matched"
    assert row["espn_event_id"] == "401704627"


def test_extract_and_match_use_post_action_score_and_nearest_clock() -> None:
    payload = {
        "winprobability": [{"playId": "p1", "homeWinPercentage": 0.61}],
        "plays": [
            {
                "id": "p1",
                "sequenceNumber": "4",
                "period": {"number": 1},
                "clock": {"displayValue": "5.1"},
                "homeScore": 3,
                "awayScore": 2,
                "text": "made shot",
            }
        ],
    }
    espn = extract_espn_win_probability(payload, game_id="g1", season_label="2025-26")
    local = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "period": [1, 1],
            "home_score_after": [3, 3],
            "away_score_after": [2, 2],
            "seconds_remaining_period": [5.0, 7.0],
            "actionId": [10, 11],
        }
    )
    matched, coverage = match_espn_to_local_states(espn, local, clock_tolerance_seconds=1.0)
    assert coverage["match_rate"] == 1.0
    assert matched.loc[0, "actionId"] == 10
    assert matched.loc[0, "clock_delta_seconds"] == pytest.approx(0.1)
