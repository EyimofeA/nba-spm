import pandas as pd
import pytest

from nba_impact.models.historical_matched_rapm import (
    MatchedRapmConfig,
    adapt_v3_terminal_lineups,
    _regular_game_ids,
)


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
    assert config.rapm(2018).seasons == (2018,)


def test_v3_adapter_rejects_overlapping_lineup():
    possessions, segments = _v3_rows()
    segments.loc[1, "away_player_1"] = 1
    with pytest.raises(ValueError, match="ten unique players"):
        adapt_v3_terminal_lineups(possessions, segments)

