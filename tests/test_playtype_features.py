from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_impact.data.playtype_features import (
    build_playtype_features,
    compute_playtype_features,
)


def test_playtype_features_compute_zts_and_transition_contribution() -> None:
    box = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "Season": 2024, "PTS": 120, "FGA": 100, "FTA": 20, "Minutes": 500},
            {"PLAYER_ID": 2, "Season": 2024, "PTS": 100, "FGA": 100, "FTA": 20, "Minutes": 500},
        ]
    )
    playtypes = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "Season": 2024, "playtype": "iso", "Poss": 50, "Points": 55, "FGA": 45, "fta_estimate": 10},
            {"PLAYER_ID": 1, "Season": 2024, "playtype": "tran", "Poss": 50, "Points": 70, "FGA": 45, "fta_estimate": 10},
            {"PLAYER_ID": 2, "Season": 2024, "playtype": "iso", "Poss": 50, "Points": 45, "FGA": 45, "fta_estimate": 10},
            {"PLAYER_ID": 2, "Season": 2024, "playtype": "tran", "Poss": 50, "Points": 50, "FGA": 45, "fta_estimate": 10},
        ]
    )
    result = compute_playtype_features(box, playtypes).set_index("PLAYER_ID")
    assert result.loc[1, "zts_pct_points"] == pytest.approx(
        result.loc[1, "player_ts_pct"] - result.loc[1, "playtype_expected_ts_pct"]
    )
    # Transition league PPP is 1.2; player 1 is +10 points over expectation.
    assert result.loc[1, "transition_poe_per_75"] == pytest.approx(7.5)
    assert result.loc[1, "transition_share"] == pytest.approx(0.5)


def test_build_playtype_features_writes_unique_versioned_artifact(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "year": 2024, "playtype": "tran", "Poss": 100,
             "Points": 120, "FGA": 80, "FTFreq%": 0.1},
            {"PLAYER_ID": 2, "year": 2024, "playtype": "tran", "Poss": 100,
             "Points": 100, "FGA": 80, "FTFreq%": 0.1},
        ]
    )
    playtype_path = tmp_path / "playtype.csv"
    raw.to_csv(playtype_path, index=False)
    box_dir = tmp_path / "box"
    box_dir.mkdir()
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PTS": 120, "FGA": 100, "FTA": 20, "Minutes": 500},
            {"PLAYER_ID": 2, "PTS": 100, "FGA": 100, "FTA": 20, "Minutes": 500},
        ]
    ).to_csv(box_dir / "2024.csv", index=False)
    run = build_playtype_features(
        playtype_path, box_dir, artifact_root=tmp_path, seasons=(2024,)
    )
    features = pd.read_parquet(run["features_path"])
    assert len(features) == 2
    assert not features.duplicated(["PLAYER_ID", "Season"]).any()
    assert run["quality"]["nonfinite_values"] == 0


def test_build_playtype_features_reads_parquet_player_sheet(tmp_path: Path) -> None:
    playtype_path = tmp_path / "playtype.csv"
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "year": 2026, "playtype": "tran", "Poss": 100,
             "Points": 120, "FGA": 80, "FTFreq%": 0.1},
            {"PLAYER_ID": 2, "year": 2026, "playtype": "tran", "Poss": 100,
             "Points": 100, "FGA": 80, "FTFreq%": 0.1},
        ]
    ).to_csv(playtype_path, index=False)
    box_dir = tmp_path / "box"
    box_dir.mkdir()
    pd.DataFrame(
        [
            {"PLAYER_ID": 1, "PTS": 120, "FGA": 100, "FTA": 20, "Minutes": 500},
            {"PLAYER_ID": 2, "PTS": 100, "FGA": 100, "FTA": 20, "Minutes": 500},
        ]
    ).to_parquet(box_dir / "2026.parquet", index=False)

    run = build_playtype_features(
        playtype_path, box_dir, artifact_root=tmp_path, seasons=(2026,)
    )

    assert pd.read_parquet(run["features_path"])["Season"].tolist() == [2026, 2026]
