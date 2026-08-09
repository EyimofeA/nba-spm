from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_impact.data.statistical_features import build_statistical_feature_windows
from nba_impact.data.statistical_features_v2 import build_statistical_features_v2


def _row(player_id: int, *, points: float, fg3m: float, fg3a: float) -> dict:
    return {
        "PLAYER_ID": player_id,
        "OffPoss": 100.0,
        "DefPoss": 100.0,
        "PTS": points,
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
        "FG3A": fg3a,
        "FG3M": fg3m,
        "FGA": 10.0 + fg3a,
        "TOUCHES": 50.0,
        "AVG_SEC_PER_TOUCH": 2.0,
        "AVG_DRIB_PER_TOUCH": 1.0,
        "ShotQualityAvg": 0.5,
    }


def test_v2_stabilizes_small_samples_and_builds_temporal_features(tmp_path: Path) -> None:
    source = tmp_path / "raw"
    source.mkdir()
    for season, points in ((2022, 10.0), (2023, 20.0), (2024, 30.0)):
        pd.DataFrame(
            [
                _row(1, points=points, fg3m=1.0, fg3a=1.0),
                _row(2, points=20.0, fg3m=0.0, fg3a=9.0),
            ]
        ).to_csv(source / f"{season}.csv", index=False)
    v1 = build_statistical_feature_windows(
        source, artifact_root=tmp_path, window_ends=(2024,)
    )
    v2 = build_statistical_features_v2(
        source,
        v1["features_path"],
        artifact_root=tmp_path,
        window_ends=(2024,),
    )
    features = pd.read_parquet(v2["features_path"]).set_index("PLAYER_ID")
    assert features.loc[1, "fg3_pct"] == pytest.approx(1.0)
    assert 0 < features.loc[1, "fg3_pct_eb"] < 1.0
    assert features.loc[1, "PTS_p100_latest"] == pytest.approx(30.0)
    assert features.loc[1, "PTS_p100_trend"] == pytest.approx(10.0)
    assert v2["quality"]["infinite_values"] == 0
    assert Path(v2["audit_path"]).exists()
    assert not {"AGE", "MIN", "GP"}.intersection(features.columns)
