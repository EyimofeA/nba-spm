#!/usr/bin/env python3
"""Run the five-year SPM teammate-context residual test."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.five_year_spm_context import run_five_year_spm_context


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data/lake/bronze"
PLAYER_SHEETS = BRONZE / "gabriel_player_sheets/revision=54b57cf/year_totals"
SITE_DATA = BRONZE / "gabriel_site_data/revision=782ec8b"
FIVE_YEAR_RUN = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79"
)
FACTOR_RUN = (
    ROOT
    / "research/rapm_lab/outputs/factor_reconstruction"
    / "factor_rapm_reconstruction_ts_v2_e8c10de3b2"
)


def main() -> None:
    run = run_five_year_spm_context(
        player_sheet_dir=PLAYER_SHEETS,
        playtype_path=SITE_DATA / "playtype.csv",
        dfg_path=SITE_DATA / "dfg.csv",
        rim_dfg_path=SITE_DATA / "rimdfg.csv",
        hustle_path=SITE_DATA / "hustle.csv",
        baseline_predictions_path=FIVE_YEAR_RUN / "spm_predictions.parquet",
        five_year_targets_path=FIVE_YEAR_RUN / "five_year_targets.parquet",
        annual_targets_path=FACTOR_RUN / "model_panel.parquet",
        contract_path=ROOT
        / "research/experiments/five_year_spm_teammate_context_v1.json",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
