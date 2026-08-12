"""Paths owned by the new NBA Impact package.

The package deliberately does not import the legacy RAPM path module. This keeps
the clean-room pipeline independent and prevents imports from creating legacy
output/viewer directories as a side effect.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("NBA_IMPACT_DATA_ROOT", PROJECT_ROOT / "data" / "lake"))
BRONZE_ROOT = DATA_ROOT / "bronze"
OFFICIAL_BOXSCORE_ROOT = BRONZE_ROOT / "nba_stats_boxscores"
SILVER_ROOT = DATA_ROOT / "silver"
MANIFEST_ROOT = DATA_ROOT / "manifests"
ARTIFACT_ROOT = Path(os.environ.get("NBA_IMPACT_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts"))
REGISTRY_PATH = Path(
    os.environ.get("NBA_IMPACT_REGISTRY", ARTIFACT_ROOT / "registry" / "nba_impact.duckdb")
)
LEGACY_POSSESSION_CACHE = PROJECT_ROOT / "rapm" / "data" / "possession_cache"
PLAYER_NAMES = PROJECT_ROOT / "rapm" / "data" / "all_names.csv"
LEGACY_PLAYER_SHEETS = PROJECT_ROOT / "data" / "raw" / "playersheets" / "year_totals"


def ensure_owned_dirs() -> None:
    for path in (DATA_ROOT, BRONZE_ROOT, SILVER_ROOT, MANIFEST_ROOT, ARTIFACT_ROOT, REGISTRY_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)
