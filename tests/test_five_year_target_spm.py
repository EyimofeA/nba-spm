from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_impact.models.five_year_target_spm import (
    _load_contract,
    _paired_bootstrap,
    _target_panels,
)


ROOT = Path(__file__).resolve().parents[1]


def test_contract_keeps_single_season_aio_and_2027_untouched() -> None:
    contract = _load_contract(ROOT / "research/experiments/five_year_target_spm_v1.yml")
    assert contract["aio_contract"]["likelihood_seasons"] == 1
    assert contract["model_contract"]["rating_seasons"][-1] == 2026
    assert contract["evaluation"]["untouched_confirmation_seasons"] == [2027]
    assert contract["model_contract"]["additional_offense_features"] == [
        "zts_pct_points"
    ]


def test_target_panels_use_the_matching_feature_window() -> None:
    features = pd.DataFrame(
        {"PLAYER_ID": [1, 1], "Window_End": [2020, 2021], "x": [2.0, 3.0]}
    )
    five_features = features.assign(x=[20.0, 30.0])
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1],
            "Season": [2020, 2021],
            "target_offense": [1.0, 2.0],
            "target_defense": [0.5, 0.6],
            "target_net": [1.5, 2.6],
            "Poss_Off": [100.0, 121.0],
            "Poss_Def": [100.0, 100.0],
        }
    )
    five = annual.rename(columns={"Season": "window_end"}).rename(
        columns={
            "target_offense": "offense",
            "target_defense": "defense",
            "target_net": "net",
        }
    )
    annual_panel, five_panel = _target_panels(features, annual, five_features, five)
    assert annual_panel[["PLAYER_ID", "Window_End"]].equals(
        five_panel[["PLAYER_ID", "Window_End"]]
    )
    np.testing.assert_allclose(annual_panel["sample_weight"], [10.0, 10.0])
    np.testing.assert_allclose(annual_panel["x"], [2.0, 3.0])
    np.testing.assert_allclose(five_panel["x"], [20.0, 30.0])


def test_target_panels_reject_duplicate_five_year_keys() -> None:
    features = pd.DataFrame({"PLAYER_ID": [1], "Window_End": [2020]})
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1], "Season": [2020], "target_offense": [1.0],
            "target_defense": [0.0], "target_net": [1.0], "Poss_Off": [1.0],
            "Poss_Def": [1.0],
        }
    )
    five = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1], "window_end": [2020, 2020], "offense": [1.0, 1.0],
            "defense": [0.0, 0.0], "net": [1.0, 1.0], "Poss_Off": [1.0, 1.0],
            "Poss_Def": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="Five-year target keys"):
        _target_panels(features, annual, features, five)


def test_paired_bootstrap_uses_identical_games() -> None:
    rows = []
    for season in (2022, 2023):
        for game in ("a", "b"):
            rows.extend(
                [
                    {
                        "test_season": season,
                        "game_id": game,
                        "candidate": "new",
                        "squared_error": 1.0,
                    },
                    {
                        "test_season": season,
                        "game_id": game,
                        "candidate": "old",
                        "squared_error": 4.0,
                    },
                ]
            )
    result = _paired_bootstrap(
        pd.DataFrame(rows),
        challenger="new",
        baseline="old",
        seasons=(2022, 2023),
        draws=50,
        seed=1,
    )
    assert result["observed_equal_season_mse_delta"] == -3.0
    assert result["probability_challenger_better"] == 1.0
