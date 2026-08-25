#!/usr/bin/env python3
"""Refresh trusted data sources into features/staging/."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from paths import FEATURES_DIR, PLAYERSHEETS_YEAR_TOTALS, PROJECT_ROOT, ensure_dirs

ensure_dirs()

FETCH_LOG = FEATURES_DIR / "staging" / "fetch_log.jsonl"
CACHE_DIR = FEATURES_DIR / "staging" / "cache"
GITHUB_RAW = "https://raw.githubusercontent.com/gabriel1200/{repo}/master/{path}"

REPOS = {
    "playtype": ("site_Data", "playtype.csv"),
}


def log_fetch(url: str, status: int, nbytes: int, duration: float) -> None:
    FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "url": url,
        "status": status,
        "bytes": nbytes,
        "duration_s": round(duration, 2),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(FETCH_LOG, "a") as f:
        f.write(json.dumps(row) + "\n")


def fetch_url(url: str, dest: Path, sleep_s: float = 3.0) -> bool:
    time.sleep(sleep_s)
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SportsAnalytics-Foundry/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        log_fetch(url, resp.status, len(data), time.time() - t0)
        return True
    except Exception as e:
        log_fetch(url, -1, 0, time.time() - t0)
        print(f"FETCH_FAIL {url}: {e}", flush=True)
        return False


def refresh_github_assets(run_id: str) -> Path:
    staging = FEATURES_DIR / "staging" / run_id
    staging.mkdir(parents=True, exist_ok=True)
    for name, (repo, path) in REPOS.items():
        url = GITHUB_RAW.format(repo=repo, path=path)
        fetch_url(url, staging / f"{name}.csv")
    manifest = {
        "run_id": run_id,
        "sources": list(REPOS.keys()),
        "playersheets_dir": str(PLAYERSHEETS_YEAR_TOTALS),
        "n_playersheet_seasons": len(list(PLAYERSHEETS_YEAR_TOTALS.glob("*.csv")))
        if PLAYERSHEETS_YEAR_TOTALS.exists()
        else 0,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"CURATOR_DONE {staging} seasons={manifest['n_playersheet_seasons']}", flush=True)
    return staging


def main() -> None:
    run_id = datetime.now(timezone.utc).strftime("curator_%Y%m%d_%H%M")
    staging = refresh_github_assets(run_id)
    subprocess.run(
        ["python3", str(PROJECT_ROOT / "rapm" / "src" / "data_ingest.py"), "--profile", str(staging)],
        check=False,
    )


if __name__ == "__main__":
    main()
