from __future__ import annotations

import json
from types import SimpleNamespace

from nba_impact.data.official_boxscore import ingest_official_boxscores, load_official_boxscore_rows


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


def test_ingest_continues_after_one_invalid_game(tmp_path, monkeypatch) -> None:
    valid = {
        "boxScoreTraditional": {
            "gameId": "0022300002",
            "homeTeam": {"teamId": 10, "players": [{"personId": 1}]},
            "awayTeam": {"teamId": 20, "players": [{"personId": 2}]},
        }
    }

    def endpoint(*, game_id, timeout):
        payload = {} if game_id == "0022300001" else valid
        return SimpleNamespace(get_dict=lambda: payload)

    monkeypatch.setattr(
        "nba_impact.data.official_boxscore.boxscoretraditionalv3.BoxScoreTraditionalV3",
        endpoint,
    )
    snapshot = ingest_official_boxscores(
        ["0022300001", "0022300002"],
        tmp_path / "boxes",
        tmp_path / "manifests",
        max_attempts=1,
        minimum_delay_seconds=0,
    )

    assert snapshot["passed"] is False
    assert snapshot["failed_game_count"] == 1
    assert snapshot["downloaded_game_count"] == 1
    assert (tmp_path / "boxes" / "game_id=0022300002.json").exists()
