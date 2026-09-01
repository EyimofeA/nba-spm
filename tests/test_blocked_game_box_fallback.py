import pandas as pd

from nba_impact.data.blocked_game_box_features import load_nba_player_game_box


def test_nba_player_game_box_normalizes_counts(tmp_path) -> None:
    path = tmp_path / "boxes.parquet"
    pd.DataFrame(
        {
            "gameId": [225000961],
            "personId": [7],
            "points": [12],
            "assists": [3],
            "turnovers": [2],
            "steals": [1],
            "blocks": [1],
            "reboundsOffensive": [2],
            "reboundsDefensive": [4],
            "foulsPersonal": [2],
            "freeThrowsAttempted": [2],
            "freeThrowsMade": [2],
            "fieldGoalsAttempted": [8],
            "fieldGoalsMade": [4],
            "threePointersAttempted": [4],
            "threePointersMade": [2],
        }
    ).to_parquet(path, index=False)
    box = load_nba_player_game_box(path, game_ids=["0225000961"])
    row = box.iloc[0]
    assert row["PLAYER_ID"] == 7
    assert row["FG2A"] == 4
    assert row["FG2M"] == 2
    assert row["FG3A"] == 4
    assert row["FG3M"] == 2
