#!/usr/bin/env python3
"""Build canonical PULSE release and walk-forward validation."""

import json
from pathlib import Path

from nba_impact.models.canonical_pulse import build_canonical_pulse


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    run = build_canonical_pulse(
        ROOT / "configs/models/pulse_v1.yml",
        ROOT / "artifacts/research/historical_box15_extension/historical_box15_extension_v1_08ff4c34ff/annual_box15_features.parquet",
        ROOT / "artifacts/models/canonical_stint_rapm/canonical_stint_rapm_v1_45606a4b1b/targets.parquet",
        ROOT / "data/lake/silver/canonical_lineup_stints",
        ROOT / "artifacts",
        historical_player_sheets=ROOT / "data/lake/bronze/historical_player_sheets/year_totals",
        gabriel_player_sheets=ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals",
        player_games_path=ROOT / "data/lake/silver/player_games.parquet",
    )
    print(json.dumps(run, indent=2))
