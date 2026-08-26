#!/usr/bin/env python3
"""Run sparse factor-target SPM and teammate-context ablations."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.factor_target_spm import run_factor_target_spm


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data/lake/bronze"
PLAYER_SHEETS = BRONZE / "gabriel_player_sheets/revision=54b57cf/year_totals"
SITE_DATA = BRONZE / "gabriel_site_data/revision=782ec8b"
FACTOR_RUN = (
    ROOT
    / "research/rapm_lab/outputs/factor_reconstruction"
    / "factor_rapm_reconstruction_ts_v2_e8c10de3b2"
)


def main() -> None:
    run = run_factor_target_spm(
        player_sheet_dir=PLAYER_SHEETS,
        playtype_path=SITE_DATA / "playtype.csv",
        dfg_path=SITE_DATA / "dfg.csv",
        rim_dfg_path=SITE_DATA / "rimdfg.csv",
        hustle_path=SITE_DATA / "hustle.csv",
        factor_panel_path=FACTOR_RUN / "model_panel.parquet",
        contract_path=ROOT / "research/experiments/factor_target_sparse_spm_v1.json",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
