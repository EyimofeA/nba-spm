"""BBRef schedule/box crawler v2 -- single storage convention, fully resumable.

Storage
-------
data/downloads/bbref_games/NBA_{year}_games.gz : newline-separated bare box
ids (e.g. 197510230CLE), harvested from the season page AND monthly pages.
data/downloads/bbref_boxes/{id}.html.gz : fetched box pages.
data/downloads/MANIFEST.csv : append-only audit log.

Floor guard: crawl aborts if total ids < 10000 (silent-harvest-miss
protection); 10000-20000 warns loudly but continues.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
DL = LAB_ROOT / "data" / "downloads"
GAMES = DL / "bbref_games"
BOXES = DL / "bbref_boxes"
MANIFEST = DL / "MANIFEST.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
MONTHS = ["october", "november", "december", "january", "february", "march", "april", "may", "june"]
ID_RE = re.compile(r"/boxscores/([0-9]{9}[A-Z]{3})\.html")
FLOOR_ABORT = 10_000
FLOOR_WARN = 20_000


def fetch(url: str, timeout: int = 40, quiet_404: bool = False):
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if not (quiet_404 and error.code == 404):
            print(f"  fetch_error {url}: HTTP {error.code}", flush=True)
            log_event("fetch_error", url, f"HTTP {error.code}")
        return None
    except Exception as error:  # noqa: BLE001
        print(f"  fetch_error {url}: {type(error).__name__}", flush=True)
        log_event("fetch_error", url, type(error).__name__)
        return None


def polite(low: float, high: float) -> None:
    time.sleep(random.uniform(low, high))


def log_event(stage: str, item: str, status: str) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["stage", "item", "bytes", "status", "fetched_at"])
        writer.writerow([stage, item, 0, status, datetime.now(timezone.utc).isoformat()])


def read_ids(year: int):
    path = GAMES / f"NBA_{year}_games.gz"
    if not path.exists():
        return set()
    text = gzip.decompress(path.read_bytes()).decode("utf-8", errors="ignore")
    lines = {line.strip().rsplit("/", 1)[-1].replace(".html", "")
             for line in text.splitlines() if line.strip()}
    legacy = set(re.findall(r"([0-9]{9}[A-Z]{3})", text))
    return lines | legacy


def write_ids(year: int, ids) -> None:
    GAMES.mkdir(parents=True, exist_ok=True)
    path = GAMES / f"NBA_{year}_games.gz"
    path.write_bytes(gzip.compress("\n".join(sorted(ids)).encode("utf-8")))


def harvest_season(year: int):
    ids = read_ids(year)
    before = len(ids)
    errored = False

    def absorb(payload):
        nonlocal ids, errored
        if payload is None:
            errored = True
            return
        ids |= set(ID_RE.findall(payload.decode("utf-8", errors="ignore")))

    absorb(fetch(f"https://www.basketball-reference.com/leagues/NBA_{year}_games.html"))
    polite(2.6, 4.2)
    for month in MONTHS:
        absorb(fetch(
            f"https://www.basketball-reference.com/leagues/NBA_{year}_games-{month}.html",
            quiet_404=True,
        ))
        polite(2.4, 4.0)

    write_ids(year, ids)
    print(f"[harvest] {year}: {len(ids)} ids (was {before})", flush=True)
    return len(ids), errored


def run_harvest(start: int, end: int):
    counts = {}
    for year in range(start, end + 1):
        count, errored = harvest_season(year)
        counts[year] = count
        if errored:
            polite(6.0, 10.0)
    return counts


def run_box_crawl() -> None:
    all_ids = set()
    for path in GAMES.glob("NBA_*_games.gz"):
        all_ids |= read_ids(int(re.search(r"NBA_(\d{4})_", path.name).group(1)))
    total = len(all_ids)
    print(f"[crawl] universe: {total} ids | boxes present: {len(list(BOXES.glob('*.html.gz')))}", flush=True)
    if total < FLOOR_ABORT:
        print(f"ABORT: {total} ids below hard floor {FLOOR_ABORT}.", flush=True)
        log_event("crawl", "floor_abort", str(total))
        return
    if total < FLOOR_WARN:
        print(f"WARN: {total} ids below soft floor {FLOOR_WARN}; crawling anyway.", flush=True)
    BOXES.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for i, box_id in enumerate(sorted(all_ids)):
        target = BOXES / f"{box_id}.html.gz"
        if target.exists():
            continue
        payload = fetch(f"https://www.basketball-reference.com/boxscores/{box_id}.html")
        if payload:
            target.write_bytes(gzip.compress(payload))
            log_event("bbref_boxes", box_id, "ok")
            fetched += 1
            if fetched % 250 == 0:
                print(f"[crawl] {i + 1}/{total} processed this run", flush=True)
        polite(2.8, 4.4)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1957)
    parser.add_argument("--end", type=int, default=1996)
    parser.add_argument("--harvest-only", action="store_true")
    args = parser.parse_args()

    counts = run_harvest(args.start, args.end)
    total = sum(counts.values())
    print(f"[harvest] TOTAL ids across {args.start}-{args.end}: {total}", flush=True)
    if args.harvest_only:
        return 0
    run_box_crawl()
    print("TOPUP_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
