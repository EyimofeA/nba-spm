#!/usr/bin/env python3
"""Publish CourtSignal reconstruction leaderboards as lazy JSON shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "web/local-data/rapm-lab.json"
OUTPUT = ROOT / "web/public/data/reconstructions"


def write_json(path: Path, value: object) -> dict[str, object]:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> int:
    source = json.loads(SOURCE.read_text())
    replications = [
        {
            "metric": row["metric"],
            "matched_rows": row["matched_rows"],
            "pearson": row["pearson"],
            "r_squared": row["r_squared"],
            "run_id": row["run_id"],
        }
        for row in source["replications"]
    ]
    files: dict[str, dict[str, object]] = {}
    boards = []
    for board in source["replication_leaderboards"]:
        filename = f"{board['id']}.json"
        receipt = write_json(OUTPUT / filename, board["rows"])
        files[filename] = receipt
        boards.append(
            {
                "id": board["id"],
                "title": board["title"],
                "season": board["season"],
                "metric": board["metric"],
                "source": "CourtSignal reconstruction",
                "columns": board["columns"],
                "rows": len(board["rows"]),
                "url": f"/data/reconstructions/{filename}?v={receipt['sha256'][:12]}",
            }
        )
    catalog = {
        "schema": "courtsignal_reconstructions_v1",
        "generated_at": source["generated_at"],
        "replications": replications,
        "boards": boards,
        "files": files,
    }
    write_json(OUTPUT / "catalog.json", catalog)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
