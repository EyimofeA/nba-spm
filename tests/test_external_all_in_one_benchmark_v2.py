from __future__ import annotations

import pandas as pd

from research.run_external_all_in_one_benchmark_v2 import (
    component_frame,
    fit_box15_2014_onward,
    name_dimension,
    season_end,
)


def test_season_end_uses_repository_end_year_convention() -> None:
    assert season_end("2017-18") == 2018
    assert season_end("1999-00") == 2000
    assert season_end(2024) == 2024


def test_component_frame_preserves_side_identity() -> None:
    source = pd.DataFrame(
        {"id": [1], "season": ["2017-18"], "off": [2.0], "def": [-0.5]}
    )
    result = component_frame(
        source,
        candidate="test",
        id_column="id",
        season_column="season",
        offense_column="off",
        defense_column="def",
    )
    assert result.iloc[0]["rating_season"] == 2018
    assert result.iloc[0]["net"] == 1.5


def test_name_dimension_removes_ambiguous_season_names() -> None:
    epm = pd.DataFrame(
        {
            "EPM_player_id": [1, 2],
            "EPM_player_name": ["Same Name", "Same Name"],
            "EPM_season": ["2017-18", "2017-18"],
        }
    )
    lebron = pd.DataFrame(
        {"nba_id": [3], "Player": ["Unique Name"], "Season": [2018]}
    )
    result = name_dimension(epm, lebron)
    assert set(result["PLAYER_ID"]) == {3}


def test_restricted_box15_has_complete_scored_seasons_and_side_identity() -> None:
    result = fit_box15_2014_onward()
    assert set(result["rating_season"]) == set(range(2017, 2025))
    assert not result.duplicated(["rating_season", "PLAYER_ID"]).any()
    assert (result["offense"] + result["defense"] - result["net"]).abs().max() < 1e-12
