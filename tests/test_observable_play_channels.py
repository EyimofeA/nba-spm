import pandas as pd

from nba_impact.data.observable_play_channels import (
    build_observable_channel_ledger,
    normalize_gabriel_events,
)


def _possessions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "possession_id": "p1",
                "game_id": "0000000001",
                "start_action_number": 1,
                "end_action_number": 9,
                "offense_team_id": 100,
                "home_team_id": 100,
                "away_team_id": 200,
                "home_poss": 1,
                "pts": 3.0,
                "gameid": "0000000001",
                "season": 2026,
                **{f"a{slot}": slot for slot in range(1, 6)},
                **{f"h{slot}": slot + 10 for slot in range(1, 6)},
            },
            {
                "possession_id": "p2",
                "game_id": "0000000001",
                "start_action_number": 10,
                "end_action_number": 20,
                "offense_team_id": 200,
                "home_team_id": 100,
                "away_team_id": 200,
                "home_poss": 0,
                "pts": 0.0,
                "gameid": "0000000001",
                "season": 2026,
                **{f"a{slot}": slot for slot in range(1, 6)},
                **{f"h{slot}": slot + 10 for slot in range(1, 6)},
            },
        ]
    )


def _events() -> pd.DataFrame:
    rows = [
        (2, "3pt", "Player 25' 3PT (3 PTS) (Mate 1 AST)", "['fastbreak']", "Made", 11, 12, None, None, 100),
        (11, "2pt", "MISS Player 2' Layup - blocked", "[]", "Missed", 1, None, None, 13, 200),
        (12, "rebound", "Mate REBOUND (Off:1 Def:0)", "[]", None, 2, None, None, None, 200),
        (13, "turnover", "Player Bad Pass Turnover", "[]", None, 1, None, 14, None, 200),
    ]
    return pd.DataFrame(
        [
            {
                "game_id": 1,
                "actionNumber": action,
                "actionType": kind,
                "description": description,
                "qualifier": qualifier,
                "shotResult": result,
                "person_id": person,
                "assister_id": assist,
                "stealPersonId": steal,
                "blockPersonId": block,
                "teamId": team,
            }
            for action, kind, description, qualifier, result, person, assist, steal, block, team in rows
        ]
    )


def test_observable_ledger_uses_possession_and_shot_denominators() -> None:
    events = normalize_gabriel_events(_events(), season=2026)
    ledger = build_observable_channel_ledger(_possessions(), events)
    first = ledger.possessions.set_index("possession_id").loc["p1"]
    assert first["transition_points"] == 3
    assert first["three_point_points"] == 3
    assert first["rim_assists"] == 0
    assert first["playtype_transition_points"] == 3
    assert (
        first[
            [
                "playtype_transition_points", "playtype_putback_points",
                "playtype_cut_points", "playtype_drive_points",
                "playtype_pullup_points", "playtype_post_points",
                "playtype_spotup_points", "playtype_other_points",
            ]
        ].sum()
        == 3
    )
    second = ledger.possessions.set_index("possession_id").loc["p2"]
    assert second["rim_points"] == 0
    assert second["midrange_attempts"] == 0
    assert len(ledger.shots) == 2
    assert ledger.shots["teammate_blocks"].sum() == 1
    assert len(ledger.rebounds) == 1
    assert ledger.rebounds["teammate_oreb"].sum() == 1


def test_normalization_removes_exact_team_mirrors() -> None:
    duplicated = pd.concat([_events(), _events()], ignore_index=True)
    normalized = normalize_gabriel_events(duplicated, season=2026)
    assert len(normalized) == len(_events())
    assert normalized["game_id"].eq("0000000001").all()
