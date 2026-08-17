from __future__ import annotations

import pandas as pd

from nba_impact.data.rapm_readiness import audit_rapm_inputs


def test_audit_requires_matching_possessions_and_ordinal_lineups(tmp_path) -> None:
    events = tmp_path / "bronze"
    silver = tmp_path / "silver"
    for source in ("nbastatsv3", "cdnnba"):
        destination = events / source / "season=2023"
        destination.mkdir(parents=True)
        pd.DataFrame(
            {
                "gameId": [1], "actionId": [1], "actionNumber": [1], "period": [1],
                "clock": ["PT12M00.00S"], "teamId": [10], "personId": [1],
                "orderNumber": [1], "possession": [10], "actionType": ["period"],
            }
        ).to_parquet(destination / "regular.parquet", index=False)
    silver.mkdir()
    game = pd.DataFrame({"game_id": ["0000000001"], "season_start": [2023], "season_type": ["regular"]})
    game.to_parquet(silver / "game_dim.parquet", index=False)
    game.to_parquet(silver / "player_games.parquet", index=False)
    pd.DataFrame({"game_id": ["0000000001"], "season_start": [2023], "season_type": ["regular"], "possession_id": ["p"]}).to_parquet(silver / "possessions.parquet", index=False)
    pd.DataFrame({"game_id": ["0000000001"], "possession_segment_id": ["s"], **{f"{side}_player_{i}": [i + (10 if side == "away" else 0)] for side in ("home", "away") for i in range(1, 6)}}).to_parquet(silver / "possession_lineup_segments.parquet", index=False)
    pd.DataFrame().to_parquet(silver / "lineup_game_quality.parquet", index=False)

    report = audit_rapm_inputs(events, silver, (2024,))

    regular = next(row for row in report["rows"] if row["season_type"] == "regular")
    assert regular["passed"]
    assert regular["rapm_ready_games"] == 1
    playoffs = next(row for row in report["rows"] if row["season_type"] == "playoffs")
    assert not playoffs["passed"]
    assert "missing_cdn_ordinal_events" in playoffs["issues"]
