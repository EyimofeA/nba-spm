from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.possession_context import CAUSAL_FEATURE_COLUMNS
from nba_impact.models.expected_possession_points import (
    MODEL_FEATURES,
    build_expected_possession_points,
)


def _context() -> pd.DataFrame:
    rows = []
    for season in (2023, 2024, 2025):
        for game in range(3):
            for possession in range(4):
                points = float((season + game + possession) % 4)
                rows.append(
                    {
                        "possession_id": f"{season}{game:02d}:{possession:03d}",
                        "game_id": f"{season}{game:06d}",
                        "season_start": season,
                        "season_type": "regular",
                        "points": points,
                        "lineup_ready": True,
                        "period": 1 + possession % 4,
                        "is_overtime": False,
                        "seconds_remaining_period_start": 700.0 - possession * 10,
                        "regulation_seconds_remaining_start": 2800.0 - possession * 10,
                        "offense_score_diff_start": float(game - possession),
                        "offense_is_home": bool(possession % 2),
                        "previous_possession_points": float(0 if possession == 0 else (possession - 1) % 4),
                        "is_first_possession": possession == 0,
                        # Forbidden fields may exist in the context but must never be modeled.
                        "offense_team_id": 10 + game,
                        "defense_team_id": 20 + game,
                    }
                )
    return pd.DataFrame(rows)


def test_expected_possession_points_is_chronological_and_player_neutral(tmp_path: Path) -> None:
    source = _context()
    source_path = tmp_path / "context.parquet"
    source.to_parquet(source_path, index=False)
    run = build_expected_possession_points(
        source_path,
        artifact_root=tmp_path,
        test_seasons=(2024, 2025),
        alpha=0.01,
    )
    output = Path(run["artifact_path"])
    predictions = pd.read_parquet(output / "cross_fitted_predictions.parquet")
    folds = pd.read_parquet(output / "fold_metrics.parquet")
    assert json.loads((output / "run.json").read_text())["run_id"] == run["run_id"]
    assert run["estimand_id"] == "player_neutral_expected_possession_points_v1"
    assert run["status"] == "research_null"
    assert set(predictions["season_start"]) == {2024, 2025}
    assert not predictions.duplicated("possession_id").any()
    assert np.isfinite(predictions["expected_points_context"]).all()
    assert (predictions["expected_points_context"] > 0).all()
    assert all(feature in CAUSAL_FEATURE_COLUMNS for feature in MODEL_FEATURES)
    assert not any("team" in feature or "player" in feature for feature in MODEL_FEATURES)
    assert folds["converged"].all()
