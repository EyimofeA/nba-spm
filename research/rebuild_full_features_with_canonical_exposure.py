"""Rebuild the 2014-26 full SPM panel with canonical possession exposure.

The Gabriel player sheets omit OffPoss/DefPoss for some low-exposure rows. This
script fills only those missing reliability denominators from the canonical
zero-prior RAPM target panel, then rebuilds annual and five-year features.
"""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.data.full_spm_features import build_full_spm_feature_panels
from nba_impact.data.statistical_features import build_statistical_feature_windows
from nba_impact.data.statistical_features_v2 import build_statistical_features_v2


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = ROOT / "artifacts/research/full_feature_2014_2026"
FEATURE_ROOT = RESEARCH_ROOT / "features"
PLAYER_SHEETS = (
    ROOT
    / "data/lake/bronze/gabriel_player_sheets"
    / "revision=54b57cf/year_totals"
)
TARGETS = (
    ROOT
    / "artifacts/models/canonical_annual_target_panel"
    / "canonical_annual_target_panel_v1_2d9ff74ca3/targets.parquet"
)


def main() -> None:
    source_overrides = {
        season: PLAYER_SHEETS / f"{season}.parquet" for season in range(2012, 2027)
    }
    v1 = build_statistical_feature_windows(
        PLAYER_SHEETS,
        artifact_root=RESEARCH_ROOT,
        window_ends=tuple(range(2014, 2027)),
        window_seasons=1,
        source_overrides=source_overrides,
        exposure_overrides_path=TARGETS,
    )
    v2 = build_statistical_features_v2(
        PLAYER_SHEETS,
        v1["features_path"],
        artifact_root=RESEARCH_ROOT,
        window_ends=tuple(range(2014, 2027)),
        pooled_window_seasons=1,
        source_overrides=source_overrides,
        exposure_overrides_path=TARGETS,
        playtype_features_path=(
            FEATURE_ROOT
            / "playtype_impact/playtype_features_v1_182fb7e27a/features.parquet"
        ),
        defensive_tracking_features_path=(
            FEATURE_ROOT
            / "defensive_tracking/defensive_tracking_features_v1_42c93b2aa6/features.parquet"
        ),
        matchup_defense_features_path=(
            FEATURE_ROOT
            / "matchup_defense/matchup_defense_features_v1_72fcc2f921/features.parquet"
        ),
    )
    panel = build_full_spm_feature_panels(
        annual_features_path=v2["features_path"],
        feature_contract_path=(
            ROOT
            / "artifacts/models/five_year_target_spm"
            / "five_year_target_spm_v1_65550acb79/run.json"
        ),
        player_sheet_dir=PLAYER_SHEETS,
        coverage_paths={
            "playtype": FEATURE_ROOT / "playtype_impact/playtype_features_v1_182fb7e27a/features.parquet",
            "dfg": FEATURE_ROOT / "observed_defense_dashboards/observed_defense_dashboards_v1_606bfaa097d7/dfg.csv",
            "rim_dfg": FEATURE_ROOT / "observed_defense_dashboards/observed_defense_dashboards_v1_606bfaa097d7/rim_dfg.csv",
            "hustle": ROOT / "data/lake/bronze/gabriel_site_data/revision=782ec8b/hustle.csv",
            "matchup_defense": FEATURE_ROOT / "matchup_defense/matchup_defense_features_v1_72fcc2f921/features.parquet",
        },
        output_root=RESEARCH_ROOT / "panels",
    )
    print(
        json.dumps(
            {
                "v1_run_id": v1["run_id"],
                "v2_run_id": v2["run_id"],
                "panel_run_id": panel["run_id"],
                "v1_exposure_cells_filled": v1["quality"][
                    "missing_possession_exposure_cells_filled"
                ],
                "v2_exposure_cells_filled": v2["quality"][
                    "missing_possession_exposure_cells_filled"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
