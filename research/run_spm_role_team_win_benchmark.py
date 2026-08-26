#!/usr/bin/env python3
"""Run the downstream team-win gate for five-year SPM role variants."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.spm_role_team_win_benchmark import run_spm_role_team_win_benchmark


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    role_run = (
        ROOT / "artifacts/research/five_year_spm_role_research"
        / "five_year_spm_role_research_v1_3edacae610"
    )
    raw = ROOT / "data/lake/bronze"
    identity_root = raw / "gabriel_player_sheets/revision=54b57cf/year_totals"
    run = run_spm_role_team_win_benchmark(
        role_run / "predictions.parquet",
        html_root=raw / "basketball_reference/player_totals",
        identity_paths={season: identity_root / f"{season}.parquet" for season in range(2020, 2024)},
        schedule_root=raw / "official_game_schedule_1997_2026",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
