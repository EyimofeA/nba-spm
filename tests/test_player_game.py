from __future__ import annotations

import pandas as pd

from nba_impact.data.player_game import build_player_games, minutes_to_seconds


def test_minutes_to_seconds_handles_nba_formats() -> None:
    assert minutes_to_seconds("24:30") == 1470
    assert minutes_to_seconds("PT12M03.5S") == 723.5
    assert minutes_to_seconds("") == 0


def test_build_player_games_uses_positions_as_starters(tmp_path) -> None:
    games = pd.DataFrame(
        {
            "game_id": ["0022500001"],
            "season_start": [2025],
            "season_end": [2026],
            "season_label": ["2025-26"],
            "season_type": ["regular"],
            "game_date": [pd.Timestamp("2025-10-21")],
            "home_team_id": [10],
            "home_team_tricode": ["HOM"],
            "away_team_id": [20],
            "away_team_tricode": ["AWY"],
            "max_period": [4],
        }
    )
    rows = []
    for team_id in (10, 20):
        for index in range(5):
            rows.append(
                {
                    "gameId": "0022500001",
                    "teamId": team_id,
                    "personId": team_id * 100 + index,
                    "firstName": "Player",
                    "familyName": str(index),
                    "position": "G" if index < 2 else "F",
                    "comment": "",
                    "minutes": "48:00",
                }
            )
    box = pd.DataFrame(rows)
    box = pd.concat([box, box.iloc[[0]]], ignore_index=True)
    espn = pd.DataFrame(
        {
            "game_id": ["0022500001"] * 10,
            "player_id": [row["personId"] for row in rows],
            "team": ["HOM"] * 5 + ["AWY"] * 5,
            "home": [1] * 5 + [0] * 5,
            "name": [f"Player {index % 5}" for index in range(10)],
            "starter": [1] * 10,
            "played": [1] * 10,
            "minutes_played": ["48:00"] * 10,
            "dAvgPos": [1.0] * 10,
            "oNetPts": [0.0] * 10,
            "dNetPts": [0.0] * 10,
            "tNetPts": [0.0] * 10,
            "oUsg": [0.2] * 10,
            "dUsg": [0.2] * 10,
            "plusMinusPoints": [0] * 10,
            "oWPA": [0.0] * 10,
            "dWPA": [0.0] * 10,
            "tWPA": [0.0] * 10,
        }
    )
    game_path = tmp_path / "game_dim.parquet"
    box_path = tmp_path / "box.parquet"
    espn_path = tmp_path / "espn.parquet"
    output = tmp_path / "player_games.parquet"
    games.to_parquet(game_path, index=False)
    box.to_parquet(box_path, index=False)
    espn.to_parquet(espn_path, index=False)

    snapshot = build_player_games(box_path, espn_path, game_path, output, tmp_path / "manifests")
    result = pd.read_parquet(output)
    assert snapshot["passed"]
    assert result["starter"].sum() == 10
    assert set(result["team_side"]) == {"home", "away"}
    assert result["espn_available"].all()
    assert snapshot["issues"]["exact_box_source_rows_dropped"] == 1
