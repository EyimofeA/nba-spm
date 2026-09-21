from __future__ import annotations

import pandas as pd

from nba_impact.data.possessions import LINEUP_COLUMNS
from research.run_pulse_search_h1_official_final import (
    apply_technical_points,
    minutes_value,
    segments_to_stints,
    segments_to_terminal,
    technical_free_throws,
    v3_terminal_scores,
)


def test_minutes_value_parses_clock_and_decimal() -> None:
    assert minutes_value("12:30") == 750
    assert minutes_value("12.5") == 750
    assert minutes_value("PT12M30S") == 750
    assert minutes_value(None) == 0


def test_technical_free_throws_keeps_made_technicals_only() -> None:
    events = pd.DataFrame(
        {
            "gameId": [22200001, 22200001, 22200001],
            "actionId": [10, 11, 12],
            "teamId": [1610612737, 1610612738, 1610612737],
            "actionType": ["Free Throw", "Free Throw", "Free Throw"],
            "subType": ["Technical", "Technical", "1 of 2"],
            "description": ["Garrett Technical Free Throw", "MISS Jones Technical Free Throw", "Bryant Free Throw 1 of 2"],
        }
    )
    tech = technical_free_throws(events)
    assert list(tech["event_order"]) == [10]


def test_apply_technical_points_subtracts_mapped_makes() -> None:
    possessions = pd.DataFrame({"possession_id": ["a", "b"], "points": [3, 1], "game_id": ["0022200001", "0022200001"]})
    assigned = pd.DataFrame({
        "possession_id": ["a", "a", "b"],
        "game_id": ["0022200001", "0022200001", "0022200001"],
        "event_order": [10, 11, 12],
    })
    tech = pd.DataFrame({
        "game_id": ["0022200001", "0022200001"],
        "event_order": [10, 99],
        "team_id": [1, 2],
        "points": [1, 1],
    })
    out = apply_technical_points(possessions, assigned, tech)
    assert list(out["technical_points"]) == [1, 0]
    assert list(out["points_excl_tech"]) == [2, 1]


def test_technical_free_throws_ignores_padded_non_ft_technicals() -> None:
    events = pd.DataFrame(
        {
            "gameId": [22200001],
            "actionId": [9],
            "teamId": [1610612737],
            "actionType": ["foul                                    "],
            "subType": ["Technical"],
            "description": ["Technical Foul"],
        }
    )
    assert technical_free_throws(events).empty


def test_v3_terminal_scores_use_last_event_not_max() -> None:
    events = pd.DataFrame(
        {
            "gameId": [22400001, 22400001, 22400001],
            "actionId": [1, 2, 3],
            "teamId": [1610612737, 1610612738, 1610612737],
            "location": ["h", "v", "h"],
            "scoreHome": [2, 100, 99],
            "scoreAway": [0, 98, 97],
        }
    )
    scores = v3_terminal_scores(events, 2025)
    assert int(scores["home_score"].iloc[0]) == 99
    assert int(scores["away_score"].iloc[0]) == 97
    assert int(scores["home_team_id"].iloc[0]) == 1610612737
    assert int(scores["away_team_id"].iloc[0]) == 1610612738


def test_v3_terminal_scores_skip_null_final_event() -> None:
    events = pd.DataFrame(
        {
            "gameId": [22400001, 22400001],
            "actionId": [1, 2],
            "teamId": [1610612737, 1610612738],
            "location": ["h", "v"],
            "scoreHome": [110, None],
            "scoreAway": [108, None],
        }
    )
    scores = v3_terminal_scores(events, 2025)
    assert int(scores["home_score"].iloc[0]) == 110
    assert int(scores["away_score"].iloc[0]) == 108


def test_segments_to_stints_counts_first_segment_possession_only() -> None:
    base = {column: [10 + i, 10 + i] for i, column in enumerate(LINEUP_COLUMNS, start=1)}
    segments = pd.DataFrame({
        "possession_id": ["p1", "p1"],
        "segment_number": [1, 2],
        "points": [2, 1],
        **base,
    })
    possessions = pd.DataFrame({
        "possession_id": ["p1"],
        "offense_is_home": [True],
        "season_end": [2025],
        "game_id": ["0022400001"],
        "technical_points": [1],
    })
    stints = segments_to_stints(possessions, segments)
    assert len(stints) == 1
    assert int(stints["home_possessions"].iloc[0]) == 1
    assert int(stints["away_possessions"].iloc[0]) == 0
    assert int(stints["home_points"].iloc[0]) == 3
    assert int(stints["home_technical_points_excluded"].iloc[0]) == 1
    assert int(stints["home_points_excl"].iloc[0]) == 2


def test_segments_to_terminal_uses_last_lineup_for_all_points() -> None:
    base_first = {column: [10 + i] for i, column in enumerate(LINEUP_COLUMNS, start=1)}
    base_last = {column: [20 + i] for i, column in enumerate(LINEUP_COLUMNS, start=1)}
    segments = pd.DataFrame({
        "possession_id": ["p1", "p1"],
        "segment_number": [1, 2],
        "points": [2, 1],
        **{column: base_first[column] + base_last[column] for column in LINEUP_COLUMNS},
    })
    possessions = pd.DataFrame({
        "possession_id": ["p1"],
        "offense_is_home": [True],
        "season_end": [2025],
        "game_id": ["0022400001"],
        "technical_points": [0],
        "points": [3],
    })
    terminal = segments_to_terminal(possessions, segments)
    assert len(terminal) == 1
    assert int(terminal["home_player_1"].iloc[0]) == 21
    assert int(terminal["home_points"].iloc[0]) == 3
    assert int(terminal["home_possessions"].iloc[0]) == 1


def test_segments_to_stints_survives_overlapping_game_id() -> None:
    base = {column: [10 + i, 10 + i] for i, column in enumerate(LINEUP_COLUMNS, start=1)}
    segments = pd.DataFrame({
        "possession_id": ["p1", "p1"],
        "game_id": ["0022400001", "0022400001"],
        "segment_number": [1, 2],
        "points": [2, 1],
        **base,
    })
    possessions = pd.DataFrame({
        "possession_id": ["p1"],
        "offense_is_home": [True],
        "season_end": [2025],
        "game_id": ["0022400001"],
        "technical_points": [0],
    })
    stints = segments_to_stints(possessions, segments)
    assert len(stints) == 1
    assert stints["game_id"].iloc[0] == "0022400001"
