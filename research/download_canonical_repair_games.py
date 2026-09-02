#!/usr/bin/env python3
"""Download official V3 evidence for every canonical repair-queue game."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_impact.data.official_game_repair import download_repair_queue


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "artifacts/data/canonical_impact_contract/canonical_impact_1997_2026_v1_550602a978"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-queue", type=Path, default=DEFAULT_RUN / "repair_queue.parquet")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/lake/bronze/official_game_repair_v3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()
    result = download_repair_queue(
        args.repair_queue,
        args.output_root,
        workers=args.workers,
        retries=args.retries,
    )
    print(json.dumps({key: result[key] for key in ("requested_games", "completed_games", "failed_games")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
