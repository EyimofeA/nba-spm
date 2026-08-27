"""Build the canonical 2014-26 frozen full-SPM feature panels."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.data.full_spm_features import build_full_spm_feature_panels


ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "artifacts/research/full_feature_2014_2026/features"


def main() -> None:
    run = build_full_spm_feature_panels(
        annual_features_path=(
            FEATURE_ROOT
            / "statistical_impact/statistical_features_v2_cb03edaf32/features.parquet"
        ),
        feature_contract_path=(
            ROOT
            / "artifacts/models/five_year_target_spm"
            / "five_year_target_spm_v1_65550acb79/run.json"
        ),
        player_sheet_dir=(
            ROOT
            / "data/lake/bronze/gabriel_player_sheets"
            / "revision=54b57cf/year_totals"
        ),
        coverage_paths={
            "playtype": FEATURE_ROOT / "playtype_impact/playtype_features_v1_182fb7e27a/features.parquet",
            "dfg": FEATURE_ROOT / "observed_defense_dashboards/observed_defense_dashboards_v1_606bfaa097d7/dfg.csv",
            "rim_dfg": FEATURE_ROOT / "observed_defense_dashboards/observed_defense_dashboards_v1_606bfaa097d7/rim_dfg.csv",
            "hustle": (
                ROOT
                / "data/lake/bronze/gabriel_site_data"
                / "revision=782ec8b/hustle.csv"
            ),
            "matchup_defense": FEATURE_ROOT / "matchup_defense/matchup_defense_features_v1_72fcc2f921/features.parquet",
        },
        output_root=ROOT / "artifacts/research/full_feature_2014_2026/panels",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
