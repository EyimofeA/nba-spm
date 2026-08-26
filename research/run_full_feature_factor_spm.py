#!/usr/bin/env python3
"""Run the full-feature factor ceiling and overall-SPM context ablation."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.full_feature_factor_spm import run_full_feature_factor_spm


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data/lake/bronze"
PLAYER_SHEETS = BRONZE / "gabriel_player_sheets/revision=54b57cf/year_totals"
SITE_DATA = BRONZE / "gabriel_site_data/revision=782ec8b"
FACTOR_RUN = (
    ROOT
    / "research/rapm_lab/outputs/factor_reconstruction"
    / "factor_rapm_reconstruction_ts_v2_e8c10de3b2"
)
FIVE_YEAR_SPM_RUN = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79"
)


def main() -> None:
    run = run_full_feature_factor_spm(
        player_sheet_dir=PLAYER_SHEETS,
        playtype_path=SITE_DATA / "playtype.csv",
        dfg_path=SITE_DATA / "dfg.csv",
        rim_dfg_path=SITE_DATA / "rimdfg.csv",
        hustle_path=SITE_DATA / "hustle.csv",
        factor_panel_path=FACTOR_RUN / "model_panel.parquet",
        feature_reference_manifest_path=FIVE_YEAR_SPM_RUN / "run.json",
        contract_path=ROOT
        / "research/experiments/factor_target_full_feature_spm_v1.json",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
