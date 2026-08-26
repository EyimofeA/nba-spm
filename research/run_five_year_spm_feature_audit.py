"""Run the frozen five-year SPM redundancy and next-season importance audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_impact.models.five_year_spm_feature_audit import (
    run_five_year_spm_feature_audit,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "artifacts/research/spm_target_horizon_full"
    / "spm_target_horizon_full_v1_f0777db1d4"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-repeats", type=int, default=5)
    parser.add_argument("--individual-repeats", type=int, default=1)
    args = parser.parse_args()
    run = run_five_year_spm_feature_audit(
        features_path=REFERENCE / "features_5y.parquet",
        targets_path=REFERENCE / "targets.parquet",
        reference_run_path=REFERENCE / "run.json",
        artifact_root=ROOT / "artifacts",
        group_repeats=args.group_repeats,
        individual_repeats=args.individual_repeats,
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
