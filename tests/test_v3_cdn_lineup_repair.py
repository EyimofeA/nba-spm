from __future__ import annotations

import pandas as pd

from nba_impact.data.v3_cdn_lineup_repair import (
    align_v3_substitutions_to_cdn,
    replay_aligned_lineups,
)


def test_alignment_requires_player_and_team_not_only_shared_clock() -> None:
    pairs = pd.DataFrame(
        [
            {
                "game_id": "002", "v3_action_id": 9, "v3_action_number": 12,
                "period": 1, "clock": "PT06M00.00S", "team_id": 10,
                "out_player_id": 1, "in_player_id": 6,
            }
        ]
    )
    cdn = pd.DataFrame(
        [
            {
                "game_id": "002", "orderNumber": 120000, "actionNumber": 12,
                "period": 1, "clock": "PT06M00.00S", "actionType": "substitution",
                "subType": "in", "personId": 7, "teamId": 10,
            },
            {
                "game_id": "002", "orderNumber": 130000, "actionNumber": 13,
                "period": 1, "clock": "PT06M00.00S", "actionType": "substitution",
                "subType": "in", "personId": 6, "teamId": 11,
            },
        ]
    )

    aligned, failures = align_v3_substitutions_to_cdn(pairs, cdn)

    assert aligned.empty
    assert failures.iloc[0]["reason"] == "cdn_incoming_candidates_0"


def test_alignment_maps_v3_action_to_cdn_ordinal_from_full_identity() -> None:
    pairs = pd.DataFrame(
        [
            {
                "game_id": "002", "v3_action_id": 9, "v3_action_number": 12,
                "period": 1, "clock": "PT06M00.00S", "team_id": 10,
                "out_player_id": 1, "in_player_id": 6,
            }
        ]
    )
    cdn = pd.DataFrame(
        [
            {
                "game_id": "002", "orderNumber": 120000, "actionNumber": 12,
                "period": 1, "clock": "PT06M00.00S", "actionType": "substitution",
                "subType": "in", "personId": 6, "teamId": 10,
            }
        ]
    )

    aligned, failures = align_v3_substitutions_to_cdn(pairs, cdn)

    assert failures.empty
    assert aligned.iloc[0]["v3_action_id"] == 9
    assert aligned.iloc[0]["cdn_order_number"] == 120000
    assert aligned.iloc[0]["alignment_key"].endswith("substitution_in")


def test_replay_emits_only_exact_ten_player_states() -> None:
    rows = []
    for order, period, clock in [
        (10000, 1, "PT12M00.00S"), (20000, 1, "PT06M00.00S"),
        (30000, 2, "PT12M00.00S"), (40000, 3, "PT12M00.00S"),
        (50000, 4, "PT12M00.00S"), (60000, 4, "PT00M00.00S"),
    ]:
        rows.append(
            {
                "game_id": "002", "orderNumber": order, "period": period, "clock": clock,
                "scoreHome": 0, "scoreAway": 0, "actionType": "period", "subType": "start",
                "personId": 0, "teamId": 0,
            }
        )
    actions = pd.DataFrame(rows)
    players = pd.DataFrame(
        [
            *[
                {"game_id": "002", "team_id": 10, "player_id": player, "starter": True,
                 "minutes_seconds": 360.0 if player == 1 else 2880.0}
                for player in range(1, 6)
            ],
            {"game_id": "002", "team_id": 10, "player_id": 6, "starter": False, "minutes_seconds": 2520.0},
            *[
                {"game_id": "002", "team_id": 20, "player_id": player, "starter": True,
                 "minutes_seconds": 2880.0}
                for player in range(11, 16)
            ],
        ]
    )
    aligned = pd.DataFrame(
        [
            {
                "game_id": "002", "v3_action_id": 2, "period": 1, "clock": "PT06M00.00S",
                "team_id": 10, "out_player_id": 1, "in_player_id": 6, "cdn_order_number": 20000,
            }
        ]
    )
    starts = {
        ("002", 10, 2): {2, 3, 4, 5, 6}, ("002", 20, 2): {11, 12, 13, 14, 15},
        ("002", 10, 3): {2, 3, 4, 5, 6}, ("002", 20, 3): {11, 12, 13, 14, 15},
        ("002", 10, 4): {2, 3, 4, 5, 6}, ("002", 20, 4): {11, 12, 13, 14, 15},
    }
    games = pd.DataFrame(
        [{"game_id": "002", "home_team_id": 10, "away_team_id": 20, "max_period": 4, "home_score": 0, "away_score": 0}]
    )

    stints, quality = replay_aligned_lineups(
        actions, players, aligned, starts, pd.DataFrame(), games, ("002",)
    )

    assert quality.iloc[0]["passed"], quality.to_dict("records")
    lineup_columns = [column for column in stints if column.endswith(tuple(str(index) for index in range(1, 6)))]
    for row in stints.itertuples(index=False):
        home = {getattr(row, f"home_player_{index}") for index in range(1, 6)}
        away = {getattr(row, f"away_player_{index}") for index in range(1, 6)}
        assert len(home) == len(away) == 5
        assert not home.intersection(away)
    assert lineup_columns
