#!/usr/bin/env python3
"""Build the frozen 1997-2026 PULSE release."""

from pathlib import Path

from nba_impact.models.pulse import build_pulse_release


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    run = build_pulse_release(
        ROOT / "configs/models/pulse_v1.yml",
        features_path=ROOT
        / "artifacts/research/historical_box15_extension/historical_box15_extension_v1_08ff4c34ff/annual_box15_features.parquet",
        targets_path=ROOT
        / "artifacts/research/target_window_spm_aio/target_window_spm_aio_v1_be05a18f9b/targets.parquet",
        validation_run=ROOT
        / "artifacts/research/target_window_spm_aio/target_window_spm_aio_v1_be05a18f9b",
        historical_player_sheets=ROOT
        / "data/lake/bronze/historical_player_sheets/year_totals",
        gabriel_player_sheets=ROOT
        / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals",
        legacy_player_aliases=ROOT / "configs/data/legacy_player_id_aliases_v1.csv",
        player_games_path=ROOT / "data/lake/silver/player_games.parquet",
        possession_cache=ROOT / "rapm/data/possession_cache",
        silver_possessions=ROOT / "data/lake/silver/possessions.parquet",
        silver_lineups=ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        artifact_root=ROOT / "artifacts",
    )
    print(run["run_id"])
