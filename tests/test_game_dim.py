from __future__ import annotations

import pandas as pd

from nba_impact.data.game_dim import build_game_dimension


def test_build_game_dimension_preserves_season_semantics(tmp_path) -> None:
    root = tmp_path / "bronze"
    event_path = root / "nbastatsv3" / "season=2025" / "regular.parquet"
    shot_path = root / "shotdetail" / "season=2025" / "regular.parquet"
    event_path.parent.mkdir(parents=True)
    shot_path.parent.mkdir(parents=True)
    events = pd.DataFrame(
        [
            {
                "gameId": 22500001,
                "actionId": action,
                "period": 1,
                "location": location,
                "teamId": team,
                "teamTricode": tricode,
                "scoreHome": home_score,
                "scoreAway": away_score,
            }
            for action, location, team, tricode, home_score, away_score in [
                (1, "h", 10, "HOM", None, None),
                (2, "v", 20, "AWY", 2, 3),
                (3, "h", 10, "HOM", 100, 90),
            ]
        ]
    )
    events.to_parquet(event_path, index=False)
    pd.DataFrame(
        {
            "GAME_ID": [22500001],
            "GAME_DATE": [20251021],
            "HTM": ["HOM"],
            "VTM": ["AWY"],
        }
    ).to_parquet(shot_path, index=False)
    output = tmp_path / "silver" / "game_dim.parquet"
    snapshot = build_game_dimension(root, output, tmp_path / "manifests")
    result = pd.read_parquet(output).iloc[0]
    assert snapshot["passed"]
    assert result["game_id"] == "0022500001"
    assert result["season_start"] == 2025
    assert result["season_end"] == 2026
    assert result["season_label"] == "2025-26"
    assert result["home_margin"] == 10
