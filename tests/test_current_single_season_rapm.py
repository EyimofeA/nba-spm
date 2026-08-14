from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nba_impact.models.current_single_season_rapm import (
    build_current_single_season_rapm_targets,
)


def _write_current_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    possessions: list[dict] = []
    segments: list[dict] = []
    player_games: list[dict] = []
    for season in (2024, 2025):
        game_id = f"002{season}0001"
        for possession_number in range(1, 5):
            possession_id = f"{game_id}:{possession_number}"
            possessions.append(
                {
                    "possession_id": possession_id,
                    "game_id": game_id,
                    "possession_number": possession_number,
                    "season_end": season,
                    "season_type": "regular",
                    "game_date": pd.Timestamp(f"{season - 1}-11-01"),
                    "period": 1,
                    "offense_is_home": bool(possession_number % 2),
                    "points": float(possession_number % 4),
                }
            )
            for segment_number, offset in ((1, 0), (2, 20)):
                segments.append(
                    {
                        "possession_id": possession_id,
                        "segment_number": segment_number,
                        **{f"home_player_{number}": number + offset for number in range(1, 6)},
                        **{
                            f"away_player_{number}": number + 10 + offset
                            for number in range(1, 6)
                        },
                    }
                )
        for player_id in range(1, 36):
            player_games.append(
                {
                    "player_id": player_id,
                    "player_name": f"Player {player_id}",
                    "game_date": pd.Timestamp(f"{season - 1}-11-01"),
                }
            )
    possessions_path = tmp_path / "possessions.parquet"
    segments_path = tmp_path / "segments.parquet"
    player_games_path = tmp_path / "player_games.parquet"
    names_path = tmp_path / "names.csv"
    pd.DataFrame(possessions).to_parquet(possessions_path, index=False)
    pd.DataFrame(segments).to_parquet(segments_path, index=False)
    pd.DataFrame(player_games).to_parquet(player_games_path, index=False)
    pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Stale"]}).to_csv(
        names_path, index=False
    )
    return possessions_path, segments_path, names_path, player_games_path


def test_current_single_season_targets_use_terminal_lineups_and_write_manifest(
    tmp_path: Path,
) -> None:
    possessions, segments, names, player_games = _write_current_inputs(tmp_path)
    run = build_current_single_season_rapm_targets(
        possessions,
        segments,
        names,
        player_games,
        artifact_root=tmp_path,
        seasons=(2024, 2025),
        lambda_off=10.0,
        lambda_def=10.0,
        lambda_home=1.0,
    )
    output = Path(run["artifact_path"])
    targets = pd.read_parquet(output / "targets.parquet")
    manifest = json.loads((output / "run.json").read_text())
    assert set(targets["Season"]) == {2024, 2025}
    assert not targets.duplicated(["PLAYER_ID", "Season"]).any()
    assert set(targets["PLAYER_ID"]) == {*range(21, 26), *range(31, 36)}
    assert np.allclose(
        targets["target_net"], targets["target_offense"] + targets["target_defense"]
    )
    assert manifest["config"]["lineup_policy"] == "terminal"
    assert manifest["quality"]["maximum_component_identity_error"] < 1e-10


def test_current_single_season_targets_reject_unavailable_season(tmp_path: Path) -> None:
    possessions, segments, names, player_games = _write_current_inputs(tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        build_current_single_season_rapm_targets(
            possessions,
            segments,
            names,
            player_games,
            artifact_root=tmp_path,
            seasons=(2026,),
        )
