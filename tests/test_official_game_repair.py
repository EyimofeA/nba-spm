from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nba_impact.data import official_game_repair as repair


def _payloads() -> dict[str, dict]:
    return {
        "playbyplayv3": {
            "game": {"actions": [{"period": period, "actionId": period} for period in range(1, 5)]}
        },
        "boxscoretraditionalv3": {
            "boxScoreTraditional": {
                "homeTeam": {"players": [{"personId": 1}]},
                "awayTeam": {"players": [{"personId": 2}]},
            }
        },
    }


def test_download_repair_queue_is_resumable(tmp_path: Path, monkeypatch) -> None:
    queue = tmp_path / "repair.parquet"
    pd.DataFrame({"season": [2024], "game_id": ["0022300001"]}).to_parquet(queue, index=False)
    calls: list[str] = []
    payloads = _payloads()

    def fake_request(url: str, **_: object) -> dict:
        endpoint = "playbyplayv3" if "playbyplayv3" in url else "boxscoretraditionalv3"
        calls.append(endpoint)
        return payloads[endpoint]

    monkeypatch.setattr(repair, "_request_json", fake_request)
    first = repair.download_repair_queue(queue, tmp_path / "raw", workers=1)
    second = repair.download_repair_queue(queue, tmp_path / "raw", workers=1)

    assert first["failed_games"] == second["failed_games"] == 0
    assert calls == ["playbyplayv3", "boxscoretraditionalv3"]
    manifest = json.loads((tmp_path / "raw/manifest.json").read_text())
    assert manifest["completed_games"] == 1
    assert all(file["resumed"] for file in manifest["games"][0]["files"].values())
