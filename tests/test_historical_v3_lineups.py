from __future__ import annotations

import pandas as pd

from nba_impact.data.historical_v3_lineups import (
    build_historical_v3_lineup_candidate,
    parse_historical_v3_substitutions,
)


def _players(game_id: str = "0022200001") -> pd.DataFrame:
    rows = []
    for team_id, prefix in ((10, "Home"), (20, "Away")):
        for index in range(1, 6):
            player_id = team_id * 100 + index
            rows.append(
                {
                    "game_id": game_id,
                    "season_end": 2023,
                    "season_label": "2022-23",
                    "season_type": "regular",
                    "game_date": "2022-10-18",
                    "team_id": team_id,
                    "player_id": player_id,
                    "player_name": f"{prefix} Player{index}",
                    "starter": True,
                    "minutes_seconds": 2880.0,
                }
            )
    return pd.DataFrame(rows)


def _v3_rows(game_id: int = 22200001) -> pd.DataFrame:
    rows = []
    action_id = 1
    for period in range(1, 5):
        rows.append(
            {
                "gameId": game_id, "actionId": action_id, "actionNumber": action_id,
                "period": period, "clock": "PT12M00.00S", "actionType": "period",
                "description": "period start", "personId": 0, "teamId": 0,
                "scoreHome": 0, "scoreAway": 0,
            }
        )
        action_id += 1
        for team_id in (10, 20):
            for index in range(1, 6):
                rows.append(
                    {
                        "gameId": game_id, "actionId": action_id, "actionNumber": action_id,
                        "period": period, "clock": "PT11M00.00S", "actionType": "turnover",
                        "description": "event", "personId": team_id * 100 + index, "teamId": team_id,
                        "scoreHome": 0, "scoreAway": 0,
                    }
                )
                action_id += 1
    return pd.DataFrame(rows)


def test_historical_builder_emits_only_full_action_coverage_and_exact_minutes(tmp_path) -> None:
    root = tmp_path / "v3"
    v3_path = root / "nbastatsv3" / "project_season=2023" / "regular.parquet"
    v3_path.parent.mkdir(parents=True)
    _v3_rows().to_parquet(v3_path, index=False)
    player_path = tmp_path / "players.parquet"
    _players().to_parquet(player_path, index=False)
    score_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [{
            "project_season": 2023, "season_type": "regular", "game_id": "0022200001",
            "home_team_id": 10, "away_team_id": 20, "home_score": 0, "away_score": 0,
        }]
    ).to_parquet(score_path, index=False)

    report = build_historical_v3_lineup_candidate(
        root, player_path, score_path,
        tmp_path / "stints.parquet", tmp_path / "quality.parquet", tmp_path / "report.json", tmp_path / "manifests",
        project_season=2023,
    )

    quality = pd.read_parquet(tmp_path / "quality.parquet")
    stints = pd.read_parquet(tmp_path / "stints.parquet")
    assert report["passed"]
    assert quality.loc[0, "passed"]
    assert quality.loc[0, "v3_event_coverage_complete"]
    assert quality.loc[0, "max_player_minute_error"] == 0.0
    assert len(stints) == 4
    assert (stints["duration_seconds"] == 720.0).all()


def test_historical_substitution_name_must_be_unique_within_the_team_roster() -> None:
    players = _players().iloc[:0].copy()
    players = pd.concat(
        [
            players,
            pd.DataFrame(
                [
                    {"game_id": "0022200001", "team_id": 10, "player_id": 1, "player_name": "John Williams"},
                    {"game_id": "0022200001", "team_id": 10, "player_id": 2, "player_name": "Jay Williams"},
                ]
            ),
        ],
        ignore_index=True,
    )
    v3 = pd.DataFrame(
        [{
            "game_id": "0022200001", "actionId": 3, "actionNumber": 3, "period": 1,
            "clock": "PT05M00.00S", "actionType": "Substitution", "description": "SUB: Williams FOR Other",
            "personId": 3, "teamId": 10,
        }]
    )

    pairs, failures = parse_historical_v3_substitutions(v3, players)

    assert pairs.empty
    assert failures.loc[0, "reason"] == "incoming_alias_matches_2"
