from __future__ import annotations

import pandas as pd

from nba_impact.data.possessions import (
    attach_ordinal_lineups,
    build_ordinal_lineup_stints,
    collapse_cdn_possessions,
    reconcile_action_points,
)


def test_v3_score_correction_requires_clock_and_period_alignment() -> None:
    actions = pd.DataFrame(
        {
            "game_id": ["g", "g"], "orderNumber": [20, 10], "actionNumber": [2, 1],
            "period": [1, 1], "clock": ["PT11M00.00S", "PT12M00.00S"],
            "scoreHome": [1, 0], "scoreAway": [0, 0],
        }
    )
    v3 = pd.DataFrame(
        {
            "game_id": ["g"], "actionNumber": [2], "period": [1], "clock": ["PT11M00.00S"],
            "home_points_added": [2.0], "away_points_added": [0.0],
        }
    )
    result, stats = reconcile_action_points(actions, v3)
    assert result.sort_values("orderNumber")["home_points_added"].tolist() == [0.0, 2.0]
    assert stats["score_rows_corrected_by_v3"] == 1


def test_collapse_orders_by_order_number_and_keeps_retained_ball_points() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["g"] * 4, "orderNumber": [40, 10, 30, 20],
            "actionNumber": [1, 99, 2, 3], "period": [1] * 4,
            "seconds_elapsed_game": [40.0, 10.0, 30.0, 20.0],
            "possession": [20, 10, 10, 10], "points_added": [2.0, 3.0, 3.0, 1.0],
            "home_points_added": [0.0, 3.0, 3.0, 1.0], "away_points_added": [2.0, 0.0, 0.0, 0.0],
            "home_team_id": [10] * 4, "away_team_id": [20] * 4,
            "season_start": [2025] * 4, "season_end": [2026] * 4,
            "season_label": ["2025-26"] * 4, "season_type": ["regular"] * 4,
            "game_date": pd.to_datetime(["2026-01-01"] * 4), "ordinal_stint_id": ["s"] * 4,
            **{f"home_player_{i}": [i] * 4 for i in range(1, 6)},
            **{f"away_player_{i}": [i + 10] * 4 for i in range(1, 6)},
        }
    )
    possessions, segments = collapse_cdn_possessions(frame)
    assert possessions["points"].tolist() == [7.0, 2.0]
    assert possessions["start_action_number"].tolist() == [99, 1]
    assert segments["points"].sum() == 9.0


def test_ordinal_substitution_changes_lineup_only_when_pair_is_complete() -> None:
    actions = pd.DataFrame(
        {
            "game_id": ["g"] * 4, "orderNumber": [10, 20, 30, 40],
            "actionType": ["2pt", "substitution", "foul", "substitution"],
            "description": ["shot", "SUB out: P1", "foul", "SUB in: P6"],
            "personId": [1, 1, 11, 6], "teamId": [10, 10, 20, 10],
            "home_team_id": [10] * 4, "away_team_id": [20] * 4,
        }
    )
    players = pd.DataFrame(
        {
            "game_id": ["g"] * 10, "team_id": [10] * 5 + [20] * 5,
            "player_id": list(range(1, 6)) + list(range(11, 16)), "starter": [True] * 10,
        }
    )
    stints, quality = build_ordinal_lineup_stints(actions, players)
    assert quality.loc[0, "passed"]
    attached = attach_ordinal_lineups(actions, stints)
    assert attached.loc[attached["orderNumber"].eq(30), "home_player_1"].item() == 1
    assert attached.loc[attached["orderNumber"].eq(40), "home_player_5"].item() == 6
