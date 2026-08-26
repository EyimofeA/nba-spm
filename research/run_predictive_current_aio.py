"""Run the frozen five-year predictive current-strength AIO experiment."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.predictive_current_aio import build_predictive_current_aio


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    run = build_predictive_current_aio(
        ROOT / "research/experiments/predictive_current_aio_2026_v1.yml",
        ROOT
        / "artifacts/models/predictive_spm_trajectory_ablation"
        / "predictive_spm_trajectory_ablation_v1_8d310a2ad6"
        / "selected_predictions.parquet",
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        ROOT / "rapm/data/all_names.csv",
        ROOT / "data/lake/silver/player_games.parquet",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
