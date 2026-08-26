#!/usr/bin/env python3
"""Run the exact annual SPM sample-weight ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_impact.models.spm_weight_ablation import run_spm_weight_ablation


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--reference-oof", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    run = run_spm_weight_ablation(
        args.features,
        args.reference_oof,
        args.reference_run,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({"run_id": run["run_id"], **run["quality"]}, indent=2))


if __name__ == "__main__":
    main()
