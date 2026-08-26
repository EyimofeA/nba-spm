#!/usr/bin/env python3
"""Run the sparse function-first five-year SPM challenger."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.sparse_function_spm import run_sparse_function_spm


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data/lake/bronze"
FULL_RUN = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79"
)
REFERENCE = (
    ROOT
    / "artifacts/research/spm_target_horizon_full"
    / "spm_target_horizon_full_v1_f0777db1d4"
)


def main() -> None:
    run = run_sparse_function_spm(
        player_sheet_dir=(
            BRONZE / "gabriel_player_sheets/revision=54b57cf/year_totals"
        ),
        five_year_targets_path=FULL_RUN / "five_year_targets.parquet",
        reference_features_path=REFERENCE / "features_5y.parquet",
        full_predictions_path=FULL_RUN / "spm_predictions.parquet",
        annual_ratings_path=FULL_RUN / "aio_ratings.parquet",
        html_root=BRONZE / "basketball_reference/player_totals",
        identity_root=(
            BRONZE / "gabriel_player_sheets/revision=54b57cf/year_totals"
        ),
        schedule_root=BRONZE / "official_game_schedule_1997_2026",
        contract_path=ROOT / "research/experiments/sparse_function_spm_v1.json",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
