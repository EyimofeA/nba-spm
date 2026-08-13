from __future__ import annotations

import pandas as pd
import pytest

from nba_impact.models.shot_defense_pilot import chronological_game_split


def test_chronological_game_split_never_splits_games() -> None:
    panel = pd.DataFrame(
        {
            "game_id": ["1", "1", "2", "2", "3", "3", "4", "4"],
            "game_date": pd.to_datetime(
                ["2024-01-01"] * 2 + ["2024-01-02"] * 2
                + ["2024-01-03"] * 2 + ["2024-01-04"] * 2
            ),
        }
    )
    train, split = chronological_game_split(panel, train_fraction=0.5)
    assert set(panel.loc[train, "game_id"]) == {"1", "2"}
    assert set(panel.loc[~train, "game_id"]) == {"3", "4"}
    assert split["train_games"] == 2
    assert split["test_games"] == 2


def test_chronological_game_split_rejects_invalid_fraction() -> None:
    panel = pd.DataFrame({"game_id": ["1", "2"], "game_date": ["2024-01-01", "2024-01-02"]})
    with pytest.raises(ValueError, match="train_fraction"):
        chronological_game_split(panel, train_fraction=1.0)
