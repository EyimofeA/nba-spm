#!/usr/bin/env python3
"""Run the local-only matchup model comparison."""

import json
from pathlib import Path

from nba_impact.models.matchup_research import build_matchup_research


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    sources = {int(key): ROOT / value for key, value in json.loads(
        (ROOT / "configs/models/matchup_elo_v1_sources.json").read_text()
    ).items()}
    print(build_matchup_research(
        source_overrides=sources,
        schedule_path=ROOT / "data/lake/bronze/official_game_schedule_1997_2026/schedule_1997_2026.parquet",
        artifact_root=ROOT / "artifacts",
    ))
