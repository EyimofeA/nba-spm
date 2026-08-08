"""Content-addressed manifests for immutable dataset snapshots."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .quality import audit_possession_file


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_possession_snapshot(cache_dir: str | Path, seasons: list[int]) -> dict:
    cache = Path(cache_dir)
    files = []
    for season in seasons:
        path = cache / f"matchups_{season}.parquet"
        report = audit_possession_file(path, expected_season=season)
        files.append(
            {
                "season": season,
                "path": str(path.resolve()),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256_file(path) if path.exists() else None,
                "quality": report.to_dict(),
            }
        )
    identity = hashlib.sha256(
        json.dumps(
            [(item["season"], item["sha256"], item["quality"]["row_count"]) for item in files],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "snapshot_id": f"legacy_possessions_{identity}",
        "dataset": "legacy_possessions",
        "grain": "offensive possession",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "passed": all(item["quality"]["passed"] for item in files),
        "files": files,
    }


def write_json_atomic(payload: dict, destination: str | Path) -> Path:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return output

