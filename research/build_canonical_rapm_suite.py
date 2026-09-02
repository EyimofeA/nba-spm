#!/usr/bin/env python3
"""Build selected rolling and full-history RAPM from canonical lineup stints."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

try:
    from build_canonical_rapm_targets import fit_window
except ModuleNotFoundError:
    from research.build_canonical_rapm_targets import fit_window


ROOT = Path(__file__).resolve().parents[1]
STINT_ROOT = ROOT / "data/lake/silver/canonical_lineup_stints"
OUTPUT_ROOT = ROOT / "artifacts/models/canonical_rapm_suite"


def main() -> None:
    manifest = STINT_ROOT / "manifest.json"
    identity = hashlib.sha256(
        json.dumps(
            {
                "source": sha256_file(manifest),
                "windows": [3, 5, 30],
                "penalties": [3000, 4500, 300],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"canonical_rapm_suite_v1_{identity}"
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    files = {}
    for horizon in (3, 5):
        rows = []
        for end in range(1997 + horizon - 1, 2027):
            path = checkpoints / f"rolling_{horizon}y_{end}.parquet"
            if not path.exists():
                ratings = fit_window(tuple(range(end - horizon + 1, end + 1)), 4500)
                ratings["window_start"] = end - horizon + 1
                ratings["window_end"] = end
                ratings.to_parquet(path, index=False)
            rows.append(pd.read_parquet(path))
        destination = output / f"rolling_{horizon}y.parquet"
        pd.concat(rows, ignore_index=True).to_parquet(destination, index=False)
        files[f"rolling_{horizon}y"] = destination.name
    full_path = output / "full_history.parquet"
    if not full_path.exists():
        full = fit_window(tuple(range(1997, 2027)), 4500)
        full["window_start"] = 1997
        full["window_end"] = 2026
        full.to_parquet(full_path, index=False)
    files["full_history"] = full_path.name
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest_sha256": sha256_file(manifest),
        "penalties": {"offense": 3000, "defense": 4500, "home": 300},
        "files": files,
        "status": "release_candidate",
    }
    write_json_atomic(run, output / "run.json")
    print(output)


if __name__ == "__main__":
    main()
