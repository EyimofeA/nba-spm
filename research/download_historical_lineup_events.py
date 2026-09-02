#!/usr/bin/env python3
"""Download the historical event and lineup sources needed to repair RAPM.

The downloads are private research inputs. The script resumes partial files,
retries transient failures, verifies the expected byte count when available,
and writes one manifest per source file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = "dfa8fa43f89ae2ca6c18db524edc2050a6bb2286"
HF_API = "https://huggingface.co/api/datasets/cdechoch/nba-data-archive/tree/main/per_season/nbastats"
HF_RAW = (
    "https://huggingface.co/datasets/cdechoch/nba-data-archive/resolve/"
    f"{HF_REVISION}/per_season/nbastats"
)
LINEUP_RAW = "https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_hf_sizes() -> dict[str, int]:
    with urllib.request.urlopen(HF_API, timeout=60) as response:
        rows = json.load(response)
    return {Path(row["path"]).name: int(row["size"]) for row in rows}


def download(url: str, destination: Path, expected_bytes: int | None) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, 9):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url,
            headers={"Range": f"bytes={offset}-", "User-Agent": "CourtSignal research"},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                append = offset > 0 and response.status == 206
                if offset and not append:
                    offset = 0
                with partial.open("ab" if append else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if expected_bytes is not None and partial.stat().st_size != expected_bytes:
                raise OSError(
                    f"expected {expected_bytes} bytes, received {partial.stat().st_size}"
                )
            partial.replace(destination)
            manifest = {
                "url": url,
                "path": str(destination.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "attempts": attempt,
            }
            destination.with_suffix(destination.suffix + ".manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            return manifest
        except Exception:
            if attempt == 8:
                raise
            time.sleep(min(2**attempt, 60))
    raise RuntimeError("unreachable")


def tasks(start: int, end: int) -> list[tuple[str, Path, int | None]]:
    sizes = expected_hf_sizes()
    output: list[tuple[str, Path, int | None]] = []
    for project_season in range(start, end + 1):
        source_year = project_season - 1
        event_name = f"{source_year}.parquet"
        if event_name in sizes:
            output.append(
                (
                    f"{HF_RAW}/{event_name}",
                    ROOT
                    / "data/lake/bronze/canonical_historical_events"
                    / f"season={project_season}"
                    / "regular.parquet",
                    sizes[event_name],
                )
            )
        output.append(
            (
                f"{LINEUP_RAW}/lineup-final{project_season}/data.rds",
                ROOT
                / "data/lake/bronze/canonical_historical_lineups"
                / f"season={project_season}"
                / "regular.rds",
                None,
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1997)
    parser.add_argument("--end", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    work = tasks(args.start, args.end)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(download, url, destination, size): destination
            for url, destination, size in work
            if not destination.exists()
        }
        for future in as_completed(pending):
            manifest = future.result()
            print(manifest["path"], manifest["bytes"], flush=True)


if __name__ == "__main__":
    main()
