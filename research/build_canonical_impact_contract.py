#!/usr/bin/env python3
"""Build the canonical CourtSignal data coverage artifact."""

from pathlib import Path

from nba_impact.data.canonical_impact_contract import build_canonical_impact_contract


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    run = build_canonical_impact_contract(
        ROOT / "configs/data/canonical_impact_1997_2026_v1.yml",
        artifact_root=ROOT / "artifacts/data/canonical_impact_contract",
    )
    print(run["run_id"])
