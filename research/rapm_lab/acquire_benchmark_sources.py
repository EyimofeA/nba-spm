"""Fetch the requested annual source tables without fitting any models."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.external_impact_benchmark import parse_bpm_html, parse_xrapm_html

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/rapm_lab/data/external/benchmark_20260903"


def main() -> None:
    records, frames = [], []
    targets = [
        ("xrapm", year, "https://xrapm.com/table_pages/" + ("xRAPM.html" if year == 2026 else f"xRAPM_{year}.html"))
        for year in range(1997, 2027)
    ] + [
        ("bpm", year, f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html")
        for year in range(2014, 2027)
    ]
    with requests.Session() as session:
        for source, year, url in targets:
            path = OUTPUT / source / f"{year}.html"
            if not path.exists():
                response = session.get(url, timeout=(10, 30))
                response.raise_for_status()  # No retries or access-control workarounds.
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response.content)
                time.sleep(1)
            frame = (parse_xrapm_html(path.read_text(), year, exclude_ambiguous_names=True)
                     if source == "xrapm" else parse_bpm_html(path.read_text(), year))
            frame["source"] = source
            frames.append(frame)
            records.append({
                "source": source, "season": year, "url": url, "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path), "rows": len(frame),
                "excluded_ambiguous_names": frame.attrs.get("excluded_ambiguous_names", []),
                "checked_at": datetime.now(timezone.utc).isoformat(),
            })
            write_json_atomic({"files": records, "complete": False}, OUTPUT / "manifest.json")
            print(f"{source} {year}: {len(frame)} rows", flush=True)
    pd.concat(frames, ignore_index=True).to_parquet(OUTPUT / "annual_sources.parquet", index=False)
    write_json_atomic({"files": records, "complete": True}, OUTPUT / "manifest.json")
    print(json.dumps({"output": str(OUTPUT), "pages": len(records)}))


if __name__ == "__main__":
    main()
