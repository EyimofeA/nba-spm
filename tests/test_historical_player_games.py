import json
from pathlib import Path

import pandas as pd

from nba_impact.data.historical_player_games import build_historical_espn_player_games


def _players(game_id: str, team: str, home: int, player_start: int) -> list[dict]:
    return [
        {
            "season": 2019, "game_id": game_id, "player_id": player_start + number,
            "team": team, "home": home, "name": f"{team}-{number}", "starter": int(number < 5),
            "minutes_played": "48:00" if number < 5 else "", "played": int(number < 5),
        }
        for number in range(5)
    ]


def test_historical_espn_player_games_keeps_only_reconciled_games(tmp_path: Path) -> None:
    game_ok, game_bad = "0021800001", "0021800002"
    official = pd.DataFrame([
        {"project_season": 2019, "season_type": "regular", "game_id": game_ok, "game_date": "2018-10-16", "home_team_id": 10, "away_team_id": 20},
        {"project_season": 2019, "season_type": "regular", "game_id": game_bad, "game_date": "2018-10-17", "home_team_id": 30, "away_team_id": 40},
    ])
    official_path = tmp_path / "official.parquet"; official.to_parquet(official_path, index=False)
    espn = pd.DataFrame(_players(game_ok, "HOM", 1, 100) + _players(game_ok, "AWY", 0, 200) + _players(game_bad, "BAD", 1, 300) + _players(game_bad, "AWY", 0, 400))
    espn_path = tmp_path / "espn.parquet"; espn.to_parquet(espn_path, index=False)
    raw = []
    for game_id, teams in ((game_ok, ((10, "HOM"), (20, "AWY"))), (game_bad, ((30, "OTHER"), (40, "AWY")))):
        for team_id, tricode in teams:
            raw.extend([{"gameId": game_id, "period": 1, "teamId": team_id, "teamTricode": tricode}, {"gameId": game_id, "period": 4, "teamId": team_id, "teamTricode": tricode}])
    v3_root = tmp_path / "v3"; partition = v3_root / "project_season=2019"; partition.mkdir(parents=True)
    pd.DataFrame(raw).to_parquet(partition / "regular.parquet", index=False)
    output, quality = tmp_path / "historical.parquet", tmp_path / "quality.parquet"
    snapshot = build_historical_espn_player_games(espn_path, official_path, v3_root, output, quality, tmp_path / "manifests", seasons=(2019,))
    result, ledger = pd.read_parquet(output), pd.read_parquet(quality)
    assert snapshot["game_count"] == 1
    assert len(result) == 10
    assert result["starter"].sum() == 10
    assert set(result["team_id"]) == {10, 20}
    assert ledger.loc[ledger["game_id"].eq(game_bad), "issues"].iloc[0] == "home_team_id_mismatch"


def test_historical_builder_prefers_valid_official_boxscore(tmp_path: Path) -> None:
    game_id = "0021800001"
    official = pd.DataFrame([
        {"project_season": 2019, "season_type": "regular", "game_id": game_id, "game_date": "2018-10-16", "home_team_id": 10, "away_team_id": 20},
    ])
    official_path = tmp_path / "official.parquet"; official.to_parquet(official_path, index=False)
    espn = pd.DataFrame(_players(game_id, "HOM", 1, 100) + _players(game_id, "AWY", 0, 200))
    espn_path = tmp_path / "espn.parquet"; espn.to_parquet(espn_path, index=False)
    raw = []
    for team_id, tricode in ((10, "HOM"), (20, "AWY")):
        raw.extend([{"gameId": game_id, "period": 1, "teamId": team_id, "teamTricode": tricode}, {"gameId": game_id, "period": 4, "teamId": team_id, "teamTricode": tricode}])
    v3_root = tmp_path / "v3"; partition = v3_root / "project_season=2019"; partition.mkdir(parents=True)
    pd.DataFrame(raw).to_parquet(partition / "regular.parquet", index=False)
    box_root = tmp_path / "official_boxes"; box_root.mkdir()
    payload = {
        "boxScoreTraditional": {
            "gameId": game_id,
            "homeTeam": {"teamId": 10, "players": [{"personId": 100 + number, "firstName": "Home", "familyName": str(number), "position": "G" if number < 5 else "", "statistics": {"minutes": "48:00" if number < 5 else ""}} for number in range(5)]},
            "awayTeam": {"teamId": 20, "players": [{"personId": 200 + number, "firstName": "Away", "familyName": str(number), "position": "G" if number < 5 else "", "statistics": {"minutes": "48:00" if number < 5 else ""}} for number in range(5)]},
        }
    }
    (box_root / f"game_id={game_id}.json").write_text(json.dumps(payload))
    output, quality = tmp_path / "historical.parquet", tmp_path / "quality.parquet"
    build_historical_espn_player_games(
        espn_path, official_path, v3_root, output, quality, tmp_path / "manifests",
        seasons=(2019,), official_box_dir=box_root,
    )
    result, ledger = pd.read_parquet(output), pd.read_parquet(quality)
    assert set(result["player_game_source"]) == {"nba_stats_boxscore_v3_repair"}
    assert result["espn_played"].isna().all()
    assert ledger.loc[0, "selected_source"] == "official_boxscore"
    assert bool(ledger.loc[0, "official_box_available"])


def test_historical_builder_rejects_invalid_official_instead_of_falling_back(tmp_path: Path) -> None:
    game_id = "0021800001"
    official = pd.DataFrame([
        {"project_season": 2019, "season_type": "regular", "game_id": game_id, "game_date": "2018-10-16", "home_team_id": 10, "away_team_id": 20},
    ])
    official_path = tmp_path / "official.parquet"; official.to_parquet(official_path, index=False)
    espn = pd.DataFrame(_players(game_id, "HOM", 1, 100) + _players(game_id, "AWY", 0, 200))
    espn_path = tmp_path / "espn.parquet"; espn.to_parquet(espn_path, index=False)
    raw = pd.DataFrame([
        {"gameId": game_id, "period": period, "teamId": team_id, "teamTricode": tricode}
        for period in (1, 4) for team_id, tricode in ((10, "HOM"), (20, "AWY"))
    ])
    v3_root = tmp_path / "v3"; partition = v3_root / "project_season=2019"; partition.mkdir(parents=True)
    raw.to_parquet(partition / "regular.parquet", index=False)
    box_root = tmp_path / "official_boxes"; box_root.mkdir()
    payload = {
        "boxScoreTraditional": {
            "gameId": game_id,
            "homeTeam": {"teamId": 999, "players": [{"personId": 900 + number, "position": "G", "statistics": {"minutes": "48:00"}} for number in range(5)]},
            "awayTeam": {"teamId": 20, "players": [{"personId": 200 + number, "position": "G", "statistics": {"minutes": "48:00"}} for number in range(5)]},
        }
    }
    (box_root / f"game_id={game_id}.json").write_text(json.dumps(payload))
    output, quality = tmp_path / "historical.parquet", tmp_path / "quality.parquet"
    build_historical_espn_player_games(
        espn_path, official_path, v3_root, output, quality, tmp_path / "manifests",
        seasons=(2019,), official_box_dir=box_root,
    )
    ledger = pd.read_parquet(quality)
    assert not bool(ledger.loc[0, "passed"])
    assert ledger.loc[0, "selected_source"] == "official_boxscore"
    assert "home_team_id_mismatch" in ledger.loc[0, "issues"]
