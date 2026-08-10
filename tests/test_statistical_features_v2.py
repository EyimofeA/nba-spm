from __future__ import annotations

import math
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
    expected_proficiency = (2 / (1 + math.exp(-1.0)) - 1) * 1.0
    expected_creation = (
        0.1843 * 5.0
        + 0.0969 * (20.0 + 2.0)
        - 2.3021 * expected_proficiency
        + 0.0582 * 5.0 * (20.0 + 2.0) * expected_proficiency
        - 1.1942
    )
    assert features.loc[1, "shooting_proficiency_2017"] == pytest.approx(
        expected_proficiency
    )
    assert features.loc[1, "box_creation_2017_p100"] == pytest.approx(
        expected_creation
    )
    assert features.loc[1, "effective_fg_pct"] == pytest.approx(6.5 / 11.0)
    assert features.loc[1, "three_point_attempt_rate"] == pytest.approx(1.0 / 11.0)
    assert features.loc[1, "free_throw_rate"] == pytest.approx(5.0 / 11.0)
    assert features["behavioral_passer_score_v1"].notna().all()
    assert features["behavioral_passer_score_v1"].abs().max() <= 26.0
    assert v2["primary_public_inspired_features"] == [
        "shooting_proficiency_2017_eb",
        "box_creation_2017_eb_p100",
        "offensive_load_2017_eb_p100",
        "assist_to_load_2017_eb",
        "turnover_to_load_2017_eb",
        "creation_to_load_2017_eb",
        "behavioral_passer_score_v1",
        "crafted_spacing_stable_v1",
    ]
    assert v2["quality"]["infinite_values"] == 0
    assert Path(v2["audit_path"]).exists()
    assert not {"AGE", "MIN", "GP"}.intersection(features.columns)

    annual_v1 = build_statistical_feature_windows(
        source,
        artifact_root=tmp_path,
        window_ends=(2024,),
        window_seasons=1,
    )
    pd.DataFrame(
        {"PLAYER_ID": [1], "Season": [2024], "zts_pct_points": [2.0]}
    ).to_parquet(tmp_path / "playtype.parquet", index=False)
    annual_v2 = build_statistical_features_v2(
        source,
        annual_v1["features_path"],
        artifact_root=tmp_path,
        window_ends=(2024,),
        pooled_window_seasons=1,
        playtype_features_path=(tmp_path / "playtype.parquet"),
    )
    annual = pd.read_parquet(annual_v2["features_path"]).set_index("PLAYER_ID")
    assert annual.loc[1, "PTS_p100"] == pytest.approx(30.0)
    assert annual.loc[1, "PTS_p100_latest"] == pytest.approx(30.0)
    assert annual_v1["grain"] == "player_single_season"
    assert annual_v2["config"]["pooled_window_seasons"] == 1
    assert annual.loc[1, "zts_pct_points"] == pytest.approx(2.0)
    assert annual_v2["playtype_feature_names"] == ["zts_pct_points"]
