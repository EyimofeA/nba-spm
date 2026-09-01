import pandas as pd

from research.run_target_window_spm_aio import _annual_features, _load_contract


def test_contract_keeps_requested_twelve_folds() -> None:
    contract = _load_contract()
    assert contract["rating_seasons"] == list(range(2014, 2026))
    assert contract["test_seasons"] == list(range(2015, 2027))
    assert contract["spm"]["input_window_seasons"] == 1
    assert contract["spm"]["training_rule"] == "expanding_history_ending_before_rating_season"


def test_annual_features_does_not_pool_across_seasons() -> None:
    annual = pd.DataFrame(
        {
            "PLAYER_ID": [1, 1, 1, 2, 2, 2],
            "Window_End": [2014, 2015, 2016, 2014, 2015, 2016],
            "OffPoss": [10, 20, 30, 5, 10, 15],
            "DefPoss": [10, 20, 30, 5, 10, 15],
            "PTS_p100": [1, 2, 3, 4, 5, 6],
        }
    )
    selected = _annual_features(
        annual,
        {"offense": ("PTS_p100",), "defense": ("PTS_p100",)},
    )
    player = selected.loc[selected["PLAYER_ID"].eq(1)]
    assert player["Window_End"].tolist() == [2014, 2015, 2016]
    assert player["OffPoss"].tolist() == [10, 20, 30]
    assert player["PTS_p100"].tolist() == [1, 2, 3]
