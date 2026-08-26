#!/usr/bin/env python3
"""Run the frozen hand-selected twelve-feature SPM challenger."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.hand_selected_sparse_spm import run_hand_selected_sparse_spm


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data/lake/bronze"
PLAYER_SHEETS = BRONZE / "gabriel_player_sheets/revision=54b57cf/year_totals"
SITE_DATA = BRONZE / "gabriel_site_data/revision=782ec8b"
FULL_RUN = ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79"


def main() -> None:
    run = run_hand_selected_sparse_spm(
        player_sheet_dir=PLAYER_SHEETS,
        playtype_path=SITE_DATA / "playtype.csv",
        dfg_path=SITE_DATA / "dfg.csv",
        rim_dfg_path=SITE_DATA / "rimdfg.csv",
        hustle_path=SITE_DATA / "hustle.csv",
        five_year_targets_path=FULL_RUN / "five_year_targets.parquet",
        full_predictions_path=FULL_RUN / "spm_predictions.parquet",
        annual_ratings_path=FULL_RUN / "aio_ratings.parquet",
        html_root=BRONZE / "basketball_reference/player_totals",
        identity_root=PLAYER_SHEETS,
        schedule_root=BRONZE / "official_game_schedule_1997_2026",
        contract_path=ROOT / "research/experiments/hand_selected_sparse_spm_v1.json",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
