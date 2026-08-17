from __future__ import annotations

import pandas as pd

from nba_impact.data.official_game_scores import (
    normalize_game_scores,
    scores_from_game_dimension,
    season_label,
)


def test_normalize_game_scores_builds_one_verified_game() -> None:
    frame = pd.DataFrame(
        [
            {"GAME_ID": "0021600001", "GAME_DATE": "2016-10-25", "TEAM_ID": 1,
             "MATCHUP": "CLE vs. NYK", "PTS": 117},
            {"GAME_ID": "0021600001", "GAME_DATE": "2016-10-25", "TEAM_ID": 2,
             "MATCHUP": "NYK @ CLE", "PTS": 88},
        ]
    )

    output, metrics = normalize_game_scores(
        frame, project_season=2017, season_type="regular"
    )

    assert season_label(2017) == "2016-17"
    assert output.iloc[0].to_dict() == {
        "project_season": 2017,
        "season_type": "regular",
        "game_id": "0021600001",
        "game_date": "2016-10-25",
        "home_team_id": 1,
        "away_team_id": 2,
        "home_score": 117,
        "away_score": 88,
    }
    assert metrics["passed"]


def test_scores_from_game_dimension_selects_one_partition() -> None:
    frame = pd.DataFrame(
        [{
            "game_id": "0022400001", "game_date": "2024-10-22",
            "season_end": 2025, "season_type": "regular",
            "home_team_id": 1, "away_team_id": 2,
            "home_score": 105, "away_score": 99,
        }]
    )
    output = scores_from_game_dimension(
        frame, project_season=2025, season_type="regular"
    )
    assert output.iloc[0]["game_id"] == "0022400001"
    assert output.iloc[0]["home_score"] == 105
