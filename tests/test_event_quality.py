from __future__ import annotations

import pandas as pd

from nba_impact.data.event_quality import build_event_snapshot


def _write_events(path, games) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for game in games:
        rows.append(
            {
                "gameId": game,
                "actionId": 1,
                "actionNumber": 1,
                "period": 1,
                "clock": "PT12M00.00S",
                "_season": 2025,
                "_season_type": "Regular Season",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_shots(path, games) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for game in games:
        rows.append(
            {
                "GAME_ID": game,
                "GAME_EVENT_ID": 1,
                "PLAYER_ID": 10,
                "TEAM_ID": 20,
                "PERIOD": 1,
                "_season": 2025,
                "_season_type": "Regular Season",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_cdn_events(path, games) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for game in games:
        rows.append(
            {
                "gameId": game,
                "orderNumber": 10,
                "actionNumber": 1,
                "period": 1,
                "clock": "PT12M00.00S",
                "possession": 20,
                "_season": 2025,
                "_season_type": "Regular Season",
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_event_snapshot_detects_cross_source_missing_game(tmp_path) -> None:
    _write_events(tmp_path / "nbastatsv3" / "season=2025" / "regular.parquet", [1, 2])
    _write_shots(tmp_path / "shotdetail" / "season=2025" / "regular.parquet", [1])
    snapshot = build_event_snapshot(tmp_path)
    assert not snapshot["passed"]
    assert snapshot["reconciliation"][0]["missing_games"] == 1


def test_event_snapshot_audits_cdn_order_and_possession_contract(tmp_path) -> None:
    _write_events(tmp_path / "nbastatsv3" / "season=2025" / "regular.parquet", [1, 2])
    _write_cdn_events(tmp_path / "cdnnba" / "season=2025" / "regular.parquet", [1, 2])

    snapshot = build_event_snapshot(tmp_path)

    cdn = next(item for item in snapshot["files"] if item["source"] == "cdnnba")
    assert cdn["passed"]
    assert cdn["row_count"] == 2
    assert cdn["game_count"] == 2
    assert snapshot["passed"]
