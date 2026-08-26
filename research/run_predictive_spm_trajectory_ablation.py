"""Run the frozen predictive-SPM age and opportunity ablation."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.predictive_trajectory_ablation import (
    build_predictive_trajectory_ablation,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run = build_predictive_trajectory_ablation(
        ROOT / "research/experiments/predictive_spm_trajectory_ablation_v1.yml",
        ROOT
        / "artifacts/models/predictive_spm/predictive_spm_v1_9392b98d58/predictions.parquet",
        ROOT / "data/raw/playersheets/year_totals",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
