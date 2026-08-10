from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_impact.data.assist_quality_features import build_assist_quality_features


def test_assist_quality_repairs_zero_denominator_and_stabilizes(tmp_path: Path) -> None:
    source = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "year": 2024, "OffPoss": 1000, "AssistPoints": 100,
             "FT_AST": 10, "POTENTIAL_AST": 100,
             "expected_teammate_ft_percentage3": 0.75, "assist_ts_pct3": np.inf},
            {"PLAYER_ID": 2, "year": 2024, "OffPoss": 100, "AssistPoints": 0,
             "FT_AST": 0, "POTENTIAL_AST": 0,
             "expected_teammate_ft_percentage3": 0.0, "assist_ts_pct3": np.nan},
        ]
    )
    source["index"] = [1, 2]
    source = pd.concat([source, source.iloc[[0]].assign(index=3)], ignore_index=True)
    path = tmp_path / "assist.csv"
    source.to_csv(path, index=False)
    run = build_assist_quality_features(
        path, artifact_root=tmp_path, seasons=(2024,)
    )
    features = pd.read_parquet(run["features_path"]).set_index("PLAYER_ID")
    assert features.loc[1, "ft_assists_p100_eb"] > 0
    assert np.isfinite(features.to_numpy()).all()
    assert features.loc[2, "assist_points_per_potential_eb"] == pytest.approx(
        features.loc[1, "assist_points_per_potential_eb"], rel=0.5
    )
    assert run["quality"]["source_infinite_values"] == 2
    assert run["quality"]["source_duplicate_keys"] == 1
    assert run["quality"]["output_nonfinite_values"] == 0
