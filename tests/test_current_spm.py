import numpy as np
import pandas as pd

from nba_impact.models.predictive_spm import aggregate_dated_box15_features


def test_dated_box15_excludes_cutoff_games_and_applies_day_decay() -> None:
    frame = pd.DataFrame(
        {
            "game_date": ["2025-01-01", "2025-01-11", "2025-01-21"],
            "PLAYER_ID": [1, 1, 1],
            "OffPoss": [100, 100, 100],
            "DefPoss": [100, 100, 100],
            "PTS": [20, 40, 100],
            "AST": [0, 0, 0],
            "TOV": [0, 0, 0],
            "STL": [0, 0, 0],
            "BLK": [0, 0, 0],
            "OREB": [0, 0, 0],
            "DREB": [0, 0, 0],
            "PF": [0, 0, 0],
            "PFD": [0, 0, 0],
            "FTA": [0, 0, 0],
            "FTM": [0, 0, 0],
            "FG2A": [10, 20, 50],
            "FG2M": [10, 20, 50],
            "FG3A": [0, 0, 0],
            "FG3M": [0, 0, 0],
        }
    )
    result = aggregate_dated_box15_features(
        frame, cutoff_date="2025-01-21", half_life_days=10
    ).iloc[0]
    assert np.isclose(result["OffPoss"], 75.0)
    assert np.isclose(result["PTS_p100"], 100.0 / 3.0)
    assert result["cutoff_date"] == pd.Timestamp("2025-01-21")


def test_dated_box15_rejects_invalid_half_life() -> None:
    with np.testing.assert_raises(ValueError):
        aggregate_dated_box15_features(
            pd.DataFrame(), cutoff_date="2025-01-21", half_life_days=0
        )


def test_dated_box15_rate_prior_shrinks_tiny_sample() -> None:
    frame = pd.DataFrame(
        {
            "game_date": ["2025-01-01", "2025-01-01"],
            "PLAYER_ID": [1, 2],
            "OffPoss": [10, 100],
            "DefPoss": [10, 100],
            "PTS": [10, 10],
            "AST": [0, 0],
            "TOV": [0, 0],
            "STL": [0, 0],
            "BLK": [0, 0],
            "OREB": [0, 0],
            "DREB": [0, 0],
            "PF": [0, 0],
            "PFD": [0, 0],
            "FTA": [0, 0],
            "FTM": [0, 0],
            "FG2A": [5, 50],
            "FG2M": [5, 5],
            "FG3A": [0, 0],
            "FG3M": [0, 0],
        }
    )
    raw = aggregate_dated_box15_features(
        frame, cutoff_date="2025-01-10", half_life_days=10
    )
    stable = aggregate_dated_box15_features(
        frame,
        cutoff_date="2025-01-10",
        half_life_days=10,
        rate_prior_possessions=500,
    )
    raw_gap = abs(raw.loc[0, "PTS_p100"] - raw.loc[1, "PTS_p100"])
    stable_gap = abs(stable.loc[0, "PTS_p100"] - stable.loc[1, "PTS_p100"])
    assert stable_gap < raw_gap
