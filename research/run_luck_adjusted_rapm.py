"""Run the frozen four-arm shooting-luck RAPM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_impact.models.luck_adjusted_rapm import run_luck_adjusted_rapm


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/experiments/luck_adjusted_rapm_spm_v1.yml",
    )
    args = parser.parse_args()
    print(json.dumps(run_luck_adjusted_rapm(args.contract, project_root=ROOT), indent=2))


if __name__ == "__main__":
    main()

