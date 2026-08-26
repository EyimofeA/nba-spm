"""Run the frozen five-year-target SPM and one-season AIO comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_impact.models.five_year_target_spm import build_five_year_target_spm


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--input-root", type=Path, required=True)
    value.add_argument(
        "--five-year-rolling-targets",
        type=Path,
        default=(
            ROOT
            / "research/rapm_lab/outputs/rolling_5y_2014_2026"
            / "rolling_5y_rapm_2014_2026_a7754bfb77/rolling_ratings.parquet"
        ),
    )
    value.add_argument(
        "--reference-manifest",
        type=Path,
        default=(
            ROOT
            / "artifacts/research/spm_target_horizon_full"
            / "spm_target_horizon_full_v1_f0777db1d4/run.json"
        ),
    )
    value.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/experiments/five_year_target_spm_v1.yml",
    )
    value.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    return value


def main() -> None:
    args = parser().parse_args()
    source = args.input_root
    run = build_five_year_target_spm(
        annual_features_path=(
            source
            / "artifacts/features/statistical_impact"
            / "statistical_features_v2_b808fc1bf1/features.parquet"
        ),
        annual_targets_path=(
            source
            / "artifacts/models/canonical_annual_target_panel"
            / "canonical_annual_target_panel_v1_2d9ff74ca3/targets.parquet"
        ),
        five_year_reference_features_path=args.reference_manifest.parent / "features_5y.parquet",
        five_year_reference_targets_path=args.reference_manifest.parent / "targets.parquet",
        five_year_rolling_targets_path=args.five_year_rolling_targets,
        player_sheet_dir=source / "data/raw/playersheets/year_totals",
        reference_manifest_path=args.reference_manifest,
        legacy_cache_dir=source / "rapm/data/possession_cache",
        current_possessions_path=source / "data/lake/silver/possessions.parquet",
        current_segments_path=source / "data/lake/silver/possession_lineup_segments.parquet",
        player_games_path=source / "data/lake/silver/player_games.parquet",
        contract_path=args.contract,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
