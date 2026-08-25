import json

import pandas as pd

from nba_impact.models.expected_shot_quality import run_expected_shot_quality


def _panel() -> pd.DataFrame:
    rows = []
    for season in (2024, 2025, 2026):
        for index in range(80):
            rim = index % 2 == 0
            rows.append(
                {
                    "season_end": season,
                    "shooter_id": 10 + index % 4,
                    "shot_zone": "rim" if rim else "above_break_3",
                    "shot_value": 2 if rim else 3,
                    "shot_made": int((index + season) % 3 != 0),
                    "location_x": 0 if rim else (-220 if index % 4 else 220),
                    "location_y": 20 if rim else 250,
                    "shot_distance_feet": 2 if rim else 25,
                    "period": index % 4 + 1,
                    "regulation_seconds_remaining": 2800 - index * 15,
                    "offense_score_diff_before": index % 11 - 5,
                    "offense_is_home": bool(index % 2),
                }
            )
    return pd.DataFrame(rows)


def test_expected_shot_quality_writes_untouched_player_summary(tmp_path):
    panel = tmp_path / "shots.parquet"
    _panel().to_parquet(panel, index=False)
    run = run_expected_shot_quality(panel, artifact_root=tmp_path / "artifacts", max_iter=100)
    output = tmp_path / "artifacts/models/expected_shot_quality" / run["run_id"]
    metrics = pd.read_parquet(output / "test_metrics.parquet")
    players = pd.read_parquet(output / "player_shot_quality.parquet")
    assert run["test"]["season_end"] == 2026
    assert set(metrics["split"]) == {"all_base", "all", "rim", "non_rim"}
    assert set(players["shot_class"]) == {"all", "rim", "non_rim"}
    assert "shooter_id" not in run["config"]["feature_names"]
    assert json.loads((output / "run.json").read_text())["status"] == "research_baseline"
