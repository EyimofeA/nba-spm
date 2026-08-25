#!/usr/bin/env python3
"""Profile staged data before merge into feature catalog."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from paths import FEATURES_DIR, PLAYERSHEETS_YEAR_TOTALS

BLOCKLIST_COLS = {"OnOffRtg", "OnDefRtg", "PLUS_MINUS", "team_ortg", "team_drtg", "team_wins"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_staging(staging_dir: Path) -> dict:
    report = {"path": str(staging_dir), "files": [], "blocklist_hits": [], "ok": True}
    csvs = list(staging_dir.glob("**/*.csv"))
    if not csvs:
        report["ok"] = False
        report["error"] = "no_csv_files"
        return report

    for p in csvs:
        info = {"file": p.name, "sha256": file_sha256(p), "rows": 0, "cols": []}
        try:
            df = pd.read_csv(p, nrows=5000, low_memory=False)
            info["rows"] = len(df)
            info["cols"] = list(df.columns[:50])
            hits = BLOCKLIST_COLS.intersection(df.columns)
            if hits:
                report["blocklist_hits"].extend(sorted(hits))
                report["ok"] = False
            if "PLAYER_ID" in df.columns:
                ref = PLAYERSHEETS_YEAR_TOTALS / "2024.csv"
                if ref.exists():
                    ref_ids = set(pd.read_csv(ref, usecols=["PLAYER_ID"])["PLAYER_ID"].astype(int))
                    join_rate = df["PLAYER_ID"].astype(int).isin(ref_ids).mean()
                    info["join_rate_2024"] = round(float(join_rate), 4)
                    if join_rate < 0.5:
                        report["ok"] = False
        except Exception as e:
            info["error"] = str(e)
            report["ok"] = False
        report["files"].append(info)

    out = staging_dir / "profile.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: data_ingest.py --profile staging/<run_id>/", flush=True)
        sys.exit(1)
    d = Path(sys.argv[-1])
    if not d.is_absolute():
        d = FEATURES_DIR / d if not d.exists() else d
    r = profile_staging(d)
    print(json.dumps(r, indent=2), flush=True)
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
