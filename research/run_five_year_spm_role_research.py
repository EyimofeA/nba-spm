"""Run the five-year SPM role and zone-shotmaking experiment."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.five_year_spm_role_research import (
    run_five_year_spm_role_research,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "artifacts/research/spm_target_horizon_full"
    / "spm_target_horizon_full_v1_f0777db1d4"
)
SKILL_RUN = (
    ROOT
    / "artifacts/models/predictive_player_skills"
    / "predictive_player_skills_2026_v1_a7eb0386fe"
    / "run.json"
)


def main() -> None:
    run = run_five_year_spm_role_research(
        features_path=REFERENCE / "features_5y.parquet",
        targets_path=REFERENCE / "targets.parquet",
        reference_run_path=REFERENCE / "run.json",
        role_dir=ROOT / "web/public/data",
        existing_skill_run_path=SKILL_RUN,
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
