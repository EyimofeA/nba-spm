import pandas as pd
import pytest
import numpy as np

from nba_impact.models.historical_matched_rapm import (
    MatchedRapmConfig,
    _official_margin_metrics,
    _rating_comparison,
    adapt_v3_terminal_lineups,
    _regular_game_ids,
)
from nba_impact.models.rapm import build_design


def _v3_rows():
    possessions = pd.DataFrame(
        {
            "possession_id": ["g1:p1"],
            "game_id": ["0021700001"],
            "season_end": [2018],
            "game_date": [pd.Timestamp("2017-10-17")],
            "period": [1],
            "possession_number": [1],
            "offense_is_home": [True],
            "points": [2],
            "season_type": ["regular"],
        }
    )
    segments = pd.DataFrame(
        {
            "possession_id": ["g1:p1", "g1:p1"],
            "segment_number": [1, 2],
            **{f"home_player_{i}": [i, i] for i in range(1, 6)},
            **{f"away_player_{i}": [10 + i, 10 + i] for i in range(1, 6)},
        }
    )
    return possessions, segments


def test_v3_adapter_uses_terminal_segment_and_rapm_contract():
    possessions, segments = _v3_rows()
    frame = adapt_v3_terminal_lineups(possessions, segments)
    assert list(frame.columns) == [
        "home_poss", "pts", "a1", "a2", "a3", "a4", "a5",
        "h1", "h2", "h3", "h4", "h5", "season", "date", "period", "num", "gameid",
    ]
    assert frame.loc[0, "gameid"] == "0021700001"
    assert frame.loc[0, "h5"] == 5
    assert frame.loc[0, "a1"] == 11


def test_regular_game_filter_excludes_playoffs():
    frame = pd.DataFrame({"gameid": ["0021700001", "0041700001", "0021700002"]})
    assert _regular_game_ids(frame) == {"0021700001", "0021700002"}


def test_config_keeps_frozen_penalties():
    config = MatchedRapmConfig()
    assert (config.lambda_off, config.lambda_def, config.lambda_home) == (3000.0, 3000.0, 300.0)
    assert config.seasons == (2017, 2018, 2019, 2020, 2021, 2022, 2023)
    assert config.rapm(2018).seasons == (2018,)


def test_v3_adapter_rejects_overlapping_lineup():
    possessions, segments = _v3_rows()
    segments.loc[1, "away_player_1"] = 1
    with pytest.raises(ValueError, match="ten unique players"):
        adapt_v3_terminal_lineups(possessions, segments)


def test_official_margin_metrics_use_common_target():
    frame = pd.DataFrame(
        {
            "home_poss": [1, 1],
            "pts": [1.0, 2.0],
            **{f"a{i}": [10 + i, 10 + i] for i in range(1, 6)},
            **{f"h{i}": [i, i] for i in range(1, 6)},
            "season": [2017, 2017],
            "date": [pd.Timestamp("2016-10-25"), pd.Timestamp("2016-10-26")],
            "period": [1, 1],
            "num": [1, 1],
            "gameid": ["g1", "g2"],
        }
    )
    design = build_design(frame)
    metrics = _official_margin_metrics(
        design,
        np.zeros(design.X.shape[1]),
        0.0,
        np.array([False, True]),
        np.array([True, False]),
        {"g2": 10.0},
    )

    assert metrics["margin_target"] == "official_final_score"
    assert metrics["margin_rmse"] == 10.0
    assert metrics["games_with_reconstructed_margin_mismatch"] == 1
    assert metrics["max_abs_reconstructed_margin_error"] == 8.0


def test_rating_comparison_applies_exposure_to_both_sources_and_sides():
    base = pd.DataFrame(
        {
            "player_id": [1, 2],
            "offense_per_100": [1.0, 2.0],
            "defense_per_100": [0.0, 1.0],
            "net_per_100": [1.0, 3.0],
            "off_possessions": [1500, 1500],
            "def_possessions": [1500, 900],
        }
    )
    legacy = base.copy()
    legacy.loc[0, "off_possessions"] = 999

    result = _rating_comparison(base, legacy, minimum_possessions=1000)

    assert result["minimum_possessions_per_source_side"] == 1000
    assert result["matched_players"] == 0
