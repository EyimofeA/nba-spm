from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.data.canonical_impact_contract import (
    _game_reconciliation,
    _normalize_game_id,
    _valid_lineup,
)


def test_game_ids_are_zero_padded() -> None:
    assert _normalize_game_id(pd.Series([29600001, "0029600002"])).tolist() == [
        "0029600001",
        "0029600002",
    ]


def test_lineup_requires_ten_distinct_players() -> None:
    columns = tuple(f"p{i}" for i in range(10))
    valid = {column: index for index, column in enumerate(columns)}
    duplicate = valid | {"p9": 8}
    missing = valid | {"p9": None}
    result = _valid_lineup(pd.DataFrame([valid, duplicate, missing]), columns)
    assert result.tolist() == [True, False, False]


def test_reconciliation_marks_truncated_and_missing_games_for_repair() -> None:
    official = pd.DataFrame(
        {
            "game_id": ["1", "2", "3"],
            "home_score": [100, 100, 100],
            "away_score": [90, 90, 90],
        }
    )
    observed = pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "home_points": [100, 51],
            "away_points": [90, 45],
            "max_period": [4, 2],
            "possession_rows": [190, 95],
        }
    )
    result = _game_reconciliation(
        official, observed, season=2020, source_kind="fixture"
    )
    assert result["score_reconciled"].tolist() == [True, False, False]
    assert result["repair_required"].tolist() == [False, True, True]
    assert result["repair_reason"].tolist() == [
        "none",
        "truncated_before_fourth_period",
        "missing_game",
    ]
    assert np.isnan(result.loc[2, "home_points"])


def test_reconciliation_separates_non_offense_score_points() -> None:
    official = pd.DataFrame({"game_id": ["1"], "home_score": [101], "away_score": [99]})
    observed = pd.DataFrame(
        {
            "game_id": ["1"],
            "home_points": [100],
            "away_points": [100],
            "native_home_points": [101],
            "native_away_points": [99],
            "max_period": [4],
            "possession_rows": [190],
        }
    )
    result = _game_reconciliation(
        official, observed, season=2026, source_kind="fixture"
    )
    assert result.loc[0, "model_score_reconciled"] == np.False_
    assert result.loc[0, "score_reconciled"] == np.True_
    assert result.loc[0, "excluded_home_points"] == 1
    assert result.loc[0, "excluded_away_points"] == -1
    assert result.loc[0, "repair_required"] == np.False_


def test_missing_accounted_score_fails_closed() -> None:
    official = pd.DataFrame(
        {
            "game_id": pd.Series(["1"], dtype="string"),
            "home_score": pd.Series([100], dtype="Int64"),
            "away_score": pd.Series([99], dtype="Int64"),
        }
    )
    observed = pd.DataFrame(
        {
            "game_id": ["1"],
            "home_points": [80.0],
            "away_points": [75.0],
            "max_period": [4],
            "possession_rows": [150],
        }
    )
    result = _game_reconciliation(
        official, observed, season=2000, source_kind="legacy"
    )
    assert result["score_reconciled"].tolist() == [False]
    assert result["repair_required"].tolist() == [True]
    assert result["repair_reason"].tolist() == ["large_score_deficit"]
