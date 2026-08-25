import pandas as pd

from nba_impact.models.shot_quality_lineup import build_lineup_shot_residual


def _panel() -> pd.DataFrame:
    rows = []
    for season in (2024, 2025, 2026):
        for game_number in range(12):
            for action in range(180):
                rim = action % 2 == 0
                rows.append(
                    {
                        "shot_id": f"{season}-{game_number}-{action}",
                        "game_id": f"{season}-{game_number}",
                        "season_end": season,
                        "shooter_id": 1 + action % 5,
                        "shot_zone": "rim" if rim else "above_break_3",
                        "shot_value": 2 if rim else 3,
                        "shot_made": int((action + game_number + season) % 3 != 0),
                        "location_x": 0 if rim else (-220 if action % 4 else 220),
                        "location_y": 20 if rim else 250,
                        "shot_distance_feet": 2 if rim else 25,
                        "period": action % 4 + 1,
                        "regulation_seconds_remaining": 2800 - action * 10,
                        "offense_score_diff_before": action % 11 - 5,
                        "offense_is_home": bool(game_number % 2),
                        **{f"offense_player_{index}": index for index in range(1, 6)},
                        **{f"defense_player_{index}": index + 5 for index in range(1, 6)},
                    }
                )
    return pd.DataFrame(rows)


def test_lineup_shot_residual_writes_rim_and_non_rim_outputs(tmp_path):
    panel = tmp_path / "shots.parquet"
    _panel().to_parquet(panel, index=False)
    run = build_lineup_shot_residual(panel, artifact_root=tmp_path / "artifacts")
    output = tmp_path / "artifacts/models/lineup_shot_residual" / run["run_id"]
    ratings = pd.read_parquet(output / "ratings.parquet")
    metrics = pd.read_parquet(output / "holdout_metrics.parquet")
    assert set(ratings["shot_class"]) == {"all", "rim", "non_rim"}
    assert set(metrics["shot_class"]) == {"all", "rim", "non_rim"}
    assert {
        "lineup_offense_shotmaking_per_100_shots",
        "lineup_defense_contest_per_100_shots",
        "shooter_quality_above_league_per_100_shots",
    }.issubset(ratings.columns)
    assert ratings.loc[ratings["shot_class"].eq("all"), "shooter_attempts"].notna().any()
    assert run["evidence_status"] == "within_season_game_holdout_only"
