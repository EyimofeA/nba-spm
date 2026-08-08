from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_impact.data.statistical_features import build_statistical_feature_windows


def _season_row(player_id: int, **overrides: float) -> dict:
    row = {
        "PLAYER_ID": player_id,
        "OffPoss": 100.0,
        "DefPoss": 100.0,
        "PTS": 20.0,
        "AST": 5.0,
        "TOV": 2.0,
        "STL": 1.0,
        "BLK": 1.0,
        "OREB": 2.0,
        "DREB": 5.0,
        "PF": 2.0,
        "PFD": 3.0,
        "FTA": 5.0,
        "FTM": 4.0,
        "FG2A": 10.0,
        "FG2M": 5.0,
        "FG3A": 5.0,
        "FG3M": 2.0,
        "FGA": 15.0,
        "TOUCHES": 50.0,
        "AVG_SEC_PER_TOUCH": 2.0,
        "AVG_DRIB_PER_TOUCH": 1.0,
        "ShotQualityAvg": 0.5,
        "OnOffRtg": 1.0,
        "OnDefRtg": 110.0,
    }
    row.update(overrides)
    return row


def _write_sources(root: Path, rows: dict[int, list[dict]]) -> None:
    root.mkdir(parents=True)
    for season, season_rows in rows.items():
        pd.DataFrame(season_rows).to_csv(root / f"{season}.csv", index=False)


def test_builder_pools_counts_and_uses_natural_weights(tmp_path) -> None:
    source = tmp_path / "raw"
    _write_sources(
        source,
        {
            2022: [_season_row(1, FG3A=1.0, FG3M=1.0, TOUCHES=10.0, AVG_SEC_PER_TOUCH=5.0)],
            2023: [_season_row(1, FG3A=9.0, FG3M=0.0, TOUCHES=90.0, AVG_SEC_PER_TOUCH=1.0)],
            2024: [_season_row(1, FG3A=10.0, FG3M=5.0, TOUCHES=100.0, AVG_SEC_PER_TOUCH=2.0)],
        },
    )
    run = build_statistical_feature_windows(
        source,
        artifact_root=tmp_path,
        window_ends=(2024,),
    )
    features = pd.read_parquet(run["features_path"])
    row = features.iloc[0]
    assert row["fg3_pct"] == pytest.approx(6.0 / 20.0)
    assert row["avg_seconds_per_touch"] == pytest.approx(1.7)
    assert "AGE" not in features
    assert "MIN" not in features
    assert "GP" not in features


def test_builder_removes_exact_duplicates(tmp_path) -> None:
    source = tmp_path / "raw"
    row = _season_row(1)
    _write_sources(source, {2024: [row, row.copy()]})
    run = build_statistical_feature_windows(
        source,
        artifact_root=tmp_path,
        window_ends=(2024,),
        window_seasons=1,
    )
    assert run["quality"]["duplicate_source_rows_collapsed_on_feature_contract"] == 1
    features = pd.read_parquet(run["features_path"])
    assert features.iloc[0]["PTS_p100"] == pytest.approx(20.0)


def test_builder_rejects_conflicting_player_rows(tmp_path) -> None:
    source = tmp_path / "raw"
    _write_sources(source, {2024: [_season_row(1), _season_row(1, PTS=25.0)]})
    with pytest.raises(ValueError, match="conflicting PLAYER_ID"):
        build_statistical_feature_windows(
            source,
            artifact_root=tmp_path,
            window_ends=(2024,),
            window_seasons=1,
        )
