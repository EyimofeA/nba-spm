from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.rapm_lab.run_raptor_onoff_proxy import _courtmate_chain, _pair_totals


def test_pair_totals_count_each_same_side_pair_once() -> None:
    lineups = np.array([[1, 2, 3, 4, 5], [1, 2, 6, 7, 8]])
    pairs = _pair_totals(
        lineups,
        np.array([10, 10]),
        np.array([2.0, 0.0]),
        np.array([1.0, -1.0]),
    )
    pair = pairs.loc[pairs["player_low"].eq(1) & pairs["player_high"].eq(2)].iloc[0]
    assert pair["shared_possessions"] == 2
    assert pair["shared_points"] == 2
    assert pair["shared_adjustment"] == 0


def test_pair_totals_keep_traded_player_teams_separate() -> None:
    lineups = np.array([[1, 2, 3, 4, 5], [1, 2, 6, 7, 8]])
    pairs = _pair_totals(
        lineups,
        np.array([10, 11]),
        np.array([2.0, 0.0]),
        np.array([0.0, 0.0]),
    )
    pair = pairs.loc[pairs["player_low"].eq(1) & pairs["player_high"].eq(2)]
    assert set(pair["team_id"]) == {10, 11}
    assert pair["shared_possessions"].eq(1).all()


def test_courtmate_without_rating_removes_shared_rows() -> None:
    stints = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3],
            "team_id": [10, 10, 10],
            "possessions": [2.0, 3.0, 2.0],
            "points": [2.0, 4.0, 2.0],
            "opponent_adjustment": [0.0, 0.0, 0.0],
        }
    )
    pairs = pd.DataFrame(
        {
            "player_low": [1, 1, 2],
            "player_high": [2, 3, 3],
            "team_id": [10, 10, 10],
            "shared_points": [2.0, 0.0, 2.0],
            "shared_adjustment": [0.0, 0.0, 0.0],
            "shared_possessions": [1.0, 1.0, 1.0],
        }
    )
    chain = _courtmate_chain(stints, pairs, side="offense", league_rating=100.0)
    assert set(chain["PLAYER_ID"]) == {1, 2, 3}
    assert np.isfinite(chain["courtmates_without"]).all()


def test_second_order_excludes_focal_player_contribution() -> None:
    stints = pd.DataFrame(
        {
            "PLAYER_ID": [1, 2, 3, 4],
            "team_id": [10, 10, 10, 10],
            "possessions": [10.0, 10.0, 10.0, 10.0],
            "points": [10.0, 12.0, 14.0, 16.0],
            "opponent_adjustment": [0.0, 0.0, 0.0, 0.0],
        }
    )
    pairs = pd.DataFrame(
        [
            {
                "player_low": left,
                "player_high": right,
                "team_id": 10,
                "shared_points": 1.0,
                "shared_adjustment": 0.0,
                "shared_possessions": 1.0,
            }
            for left, right in [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
        ]
    )
    original = _courtmate_chain(stints, pairs, side="offense", league_rating=100.0)
    changed = stints.copy()
    changed.loc[changed["PLAYER_ID"].eq(1), "points"] = 30.0
    rerun = _courtmate_chain(changed, pairs, side="offense", league_rating=100.0)
    first = original.loc[original["PLAYER_ID"].eq(1), "second_order"].iloc[0]
    second = rerun.loc[rerun["PLAYER_ID"].eq(1), "second_order"].iloc[0]
    assert np.isclose(first, second)
