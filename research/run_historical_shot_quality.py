#!/usr/bin/env python3
"""Run the 2014-15 row-level expected-shot research prototype."""

from __future__ import annotations

import json
from pathlib import Path

from nba_impact.models.historical_shot_quality import run_historical_shot_quality


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pbp = ROOT / "research/rapm_lab/external/external/merged_playbyplay/pbp_data"
    run = run_historical_shot_quality(
        ROOT / "data/lake/bronze/historical_shot_context/2015/shot_logs.csv",
        tuple(sorted(pbp.glob("2015_*.csv"))),
        player_sheet_path=ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals/2015.parquet",
        artifact_root=ROOT / "artifacts",
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
