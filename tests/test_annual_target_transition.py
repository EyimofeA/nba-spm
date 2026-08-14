from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nba_impact.models.annual_target_transition import (
    build_canonical_annual_target_panel,
)


def _targets(value: float) -> pd.DataFrame:
    rows = []
    for season in (2023, 2024):
        for player_id in range(1, 6):
            offense = value + player_id
            defense = 0.5 * value - 0.25 * player_id
            rows.append(
                {
                    "PLAYER_ID": player_id,
                    "Season": season,
                    "target_offense": offense,
                    "target_defense": defense,
                    "target_net": offense + defense,
                    "Poss_Off": 1000.0,
                    "Poss_Def": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def _write_inputs(tmp_path: Path, *, canonical_offset: float = 0.01) -> tuple[Path, Path, Path, Path]:
    legacy = _targets(0.0)
    canonical = _targets(canonical_offset)
    canonical = pd.concat(
        [
            canonical,
            _targets(canonical_offset)
            .loc[lambda frame: frame["Season"].eq(2024)]
            .assign(Season=2025),
        ],
        ignore_index=True,
    )
    legacy_path = tmp_path / "legacy.parquet"
    canonical_path = tmp_path / "canonical.parquet"
    names_path = tmp_path / "names.csv"
    player_games_path = tmp_path / "player_games.parquet"
    legacy.to_parquet(legacy_path, index=False)
    canonical.to_parquet(canonical_path, index=False)
    pd.DataFrame({"PLAYER_ID": range(1, 6), "PLAYER_NAME": [f"P{x}" for x in range(1, 6)]}).to_csv(names_path, index=False)
    pd.DataFrame(
        {
            "player_id": range(1, 6),
            "player_name": [f"P{x}" for x in range(1, 6)],
            "game_date": pd.to_datetime(["2024-01-01"] * 5),
        }
    ).to_parquet(player_games_path, index=False)
    return legacy_path, canonical_path, names_path, player_games_path


def test_transition_panel_replaces_boundary_with_canonical_source(tmp_path: Path) -> None:
    legacy, canonical, names, player_games = _write_inputs(tmp_path)
    run = build_canonical_annual_target_panel(
        legacy, canonical, names, player_games, artifact_root=tmp_path, transition_season=2024
    )
    panel = pd.read_parquet(Path(run["targets_path"]))
    assert set(panel.loc[panel["Season"].eq(2023), "annual_target_source"]) == {"legacy"}
    assert set(panel.loc[panel["Season"].ge(2024), "annual_target_source"]) == {"canonical_current"}
    assert not panel.duplicated(["PLAYER_ID", "Season"]).any()
    assert run["quality"]["last_season"] == 2025


def test_transition_panel_rejects_incompatible_overlap(tmp_path: Path) -> None:
    legacy, canonical, names, player_games = _write_inputs(tmp_path, canonical_offset=100.0)
    with pytest.raises(ValueError, match="failed"):
        build_canonical_annual_target_panel(
            legacy, canonical, names, player_games, artifact_root=tmp_path, transition_season=2024
        )
