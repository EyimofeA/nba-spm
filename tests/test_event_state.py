from __future__ import annotations

import pandas as pd

from nba_impact.data.event_state import build_event_states, parse_clock_seconds


def test_parse_clock_seconds() -> None:
    result = parse_clock_seconds(pd.Series(["PT12M00.00S", "PT00M02.30S", "PT00M00.00S"]))
    assert result.tolist() == [720.0, 2.3, 0.0]


def test_event_states_reconcile_terminal_score(tmp_path) -> None:
    root = tmp_path / "bronze"
    path = root / "nbastatsv3" / "season=2025" / "regular.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "gameId": 22500001,
                "actionId": action_id,
                "actionNumber": action_id,
                "clock": clock,
                "period": 1,
                "teamId": team_id,
                "teamTricode": tricode,
                "personId": 1,
                "playerName": "Player",
                "location": location,
                "description": description,
                "actionType": action_type,
                "subType": None,
                "scoreHome": home,
                "scoreAway": away,
                "shotValue": 2 if action_type == "Made Shot" else 0,
                "isFieldGoal": int(action_type == "Made Shot"),
            }
            for action_id, clock, team_id, tricode, location, description, action_type, home, away in [
                (1, "PT12M00.00S", 0, None, None, "Start", "period", 0, 0),
                (2, "PT11M00.00S", 10, "HOM", "h", "Made", "Made Shot", 2, 0),
                (3, "PT00M00.00S", 0, None, None, "End", "period", 2, 0),
            ]
        ]
    ).to_parquet(path, index=False)
    game_dim = tmp_path / "game_dim.parquet"
    pd.DataFrame(
        {
            "game_id": ["0022500001"],
            "season_start": [2025],
            "season_end": [2026],
            "season_label": ["2025-26"],
            "game_date": [pd.Timestamp("2025-10-21")],
            "home_team_id": [10],
            "away_team_id": [20],
            "home_score": [2],
            "away_score": [0],
            "home_win": [True],
        }
    ).to_parquet(game_dim, index=False)
    output = tmp_path / "event_states.parquet"
    snapshot = build_event_states(root, game_dim, output, tmp_path / "manifests")
    result = pd.read_parquet(output)
    assert snapshot["passed"]
    assert result["home_score_after"].tolist() == [0.0, 2.0, 2.0]
    assert result["points_added"].tolist() == [0.0, 2.0, 0.0]


def test_event_states_support_missing_shot_value_and_score_correction(tmp_path) -> None:
    root = tmp_path / "bronze"
    path = root / "nbastatsv3" / "season=2023" / "regular.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "gameId": 22300001,
                "actionId": action_id,
                "actionNumber": action_id,
                "clock": clock,
                "period": 1,
                "teamId": 10 if location == "h" else 0,
                "teamTricode": "HOM" if location == "h" else None,
                "personId": 1 if location == "h" else 0,
                "playerName": "Player" if location == "h" else None,
                "location": location,
                "description": description,
                "actionType": action_type,
                "subType": None,
                "scoreHome": home,
                "scoreAway": 0,
                "isFieldGoal": int(action_type == "Made Shot"),
            }
            for action_id, clock, location, description, action_type, home in [
                (1, "PT12M00.00S", None, "Start", "period", 0),
                (2, "PT11M00.00S", "h", "Made three", "Made Shot", 3),
                (3, "PT10M30.00S", None, "Score correction", "Instant Replay", 2),
                (4, "PT00M00.00S", None, "End", "period", 2),
            ]
        ]
    ).to_parquet(path, index=False)
    game_dim = tmp_path / "game_dim.parquet"
    pd.DataFrame(
        {
            "game_id": ["0022300001"],
            "season_start": [2023],
            "season_end": [2024],
            "season_label": ["2023-24"],
            "game_date": [pd.Timestamp("2023-10-24")],
            "home_team_id": [10],
            "away_team_id": [20],
            "home_score": [2],
            "away_score": [0],
            "home_win": [True],
        }
    ).to_parquet(game_dim, index=False)

    output = tmp_path / "event_states.parquet"
    snapshot = build_event_states(root, game_dim, output, tmp_path / "manifests")
    result = pd.read_parquet(output)

    assert snapshot["passed"]
    assert result["shotValue"].isna().all()
    assert result["home_score_after"].tolist() == [0.0, 3.0, 2.0, 2.0]
    assert result["home_points_added"].tolist() == [0.0, 3.0, -1.0, 0.0]


def test_event_states_replaces_stale_terminal_snapshot_with_game_final(tmp_path) -> None:
    root = tmp_path / "bronze"
    path = root / "nbastatsv3" / "season=2025" / "regular.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "gameId": 22500001,
                "actionId": action_id,
                "actionNumber": action_id,
                "clock": clock,
                "period": 4,
                "teamId": 10 if location == "h" else 0,
                "teamTricode": "HOM" if location == "h" else None,
                "personId": 1 if location == "h" else 0,
                "playerName": "Player" if location == "h" else None,
                "location": location,
                "description": description,
                "actionType": action_type,
                "subType": None,
                "scoreHome": home,
                "scoreAway": 0,
                "shotValue": 2 if action_type == "Made Shot" else 0,
                "isFieldGoal": int(action_type == "Made Shot"),
            }
            for action_id, clock, location, description, action_type, home in [
                (1, "PT00M01.00S", "h", "Made", "Made Shot", 2),
                (2, "PT00M00.00S", None, "End with stale score", "period", 0),
            ]
        ]
    ).to_parquet(path, index=False)
    game_dim = tmp_path / "game_dim.parquet"
    pd.DataFrame(
        {
            "game_id": ["0022500001"],
            "season_start": [2025],
            "season_end": [2026],
            "season_label": ["2025-26"],
            "game_date": [pd.Timestamp("2025-10-21")],
            "home_team_id": [10],
            "away_team_id": [20],
            "home_score": [2],
            "away_score": [0],
            "home_win": [True],
        }
    ).to_parquet(game_dim, index=False)

    output = tmp_path / "event_states.parquet"
    snapshot = build_event_states(root, game_dim, output, tmp_path / "manifests")
    result = pd.read_parquet(output)

    assert snapshot["passed"]
    assert snapshot["warnings"]["source_score_snapshot_disagreement_rows"] == 1
    assert result["home_score_after"].tolist() == [2.0, 2.0]
    assert result["home_points_added"].tolist() == [2.0, 0.0]
