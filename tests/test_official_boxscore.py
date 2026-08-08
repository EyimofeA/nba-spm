from __future__ import annotations

import json

from nba_impact.data.official_boxscore import load_official_boxscore_rows


def test_load_official_boxscore_rows_preserves_starters_and_minutes(tmp_path) -> None:
    payload = {
        "boxScoreTraditional": {
            "gameId": "0022300001",
            "homeTeam": {
                "teamId": 10,
                "players": [
                    {
                        "personId": 1,
                        "firstName": "Home",
                        "familyName": "Starter",
                        "position": "G",
                        "comment": "",
                        "statistics": {"minutes": "31:00"},
                    }
                ],
            },
            "awayTeam": {
                "teamId": 20,
                "players": [
                    {
                        "personId": 2,
                        "firstName": "Away",
                        "familyName": "Bench",
                        "position": "",
                        "comment": "",
                        "statistics": {"minutes": "12:00"},
                    }
                ],
            },
        }
    }
    path = tmp_path / "game_id=0022300001.json"
    path.write_text(json.dumps(payload))

    result, paths = load_official_boxscore_rows(tmp_path)

    assert paths == [path]
    assert result["player_id"].tolist() == [1, 2]
    assert result["starter_position"].tolist() == ["G", ""]
    assert result["minutes"].tolist() == ["31:00", "12:00"]
    assert set(result["player_game_source"]) == {"nba_stats_boxscore_v3_repair"}
