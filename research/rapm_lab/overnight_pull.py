"""Overnight download queue — chained behind gabes-pull, fully resumable.

Stages (sequential, politeness delays throughout):
  0  wait for gabes-pull PULL_COMPLETE marker
  1  xrapm.com homepage (Engelmann values)
  2  Engelmann Substack: archive index + plus-minus/RAPM posts
  3  Dunks & Threes EPM pages, 2014-2025
  4  CraftedNBA rating pages (DRIP)
  5  BBRef season games-pages 1957-1996 (results + box-score link harvest)
  6  BBRef per-game box pages 1957-1996 (raw HTML, gzipped, multi-night resumable)
  7  stats.nba.com reachability probe; if reachable, BoxScoreTraditional crawl
     1997-2013 using game ids from gabes old_data

Every fetched artifact lands under research/rapm_lab/data/downloads/ with an entry in
MANIFEST.csv. Re-running skips anything already present.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
DL = LAB_ROOT / "data" / "downloads"
GABES_LOG = LAB_ROOT / "external" / "external" / "pull.log"
MANIFEST = DL / "MANIFEST.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FIELDS = ["stage", "item", "bytes", "status", "fetched_at"]


def log_event(stage: str, item: str, nbytes: int, status: str) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with MANIFEST.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({
            "stage": stage, "item": item, "bytes": nbytes,
            "status": status,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"[{stage}] {status}: {item} ({nbytes} bytes)", flush=True)


def fetch(url: str, timeout: int = 40) -> bytes | None:
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except Exception as error:  # noqa: BLE001 - queue must survive individual failures
        log_event("fetch_error", url, 0, f"{type(error).__name__}: {error}"[:160])
        return None


def save(stage: str, name: str, payload: bytes, gzipped: bool = False) -> bool:
    target = DL / stage / (name + (".gz" if gzipped else ""))
    if target.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if gzipped:
        target.write_bytes(gzip.compress(payload))
    else:
        target.write_bytes(payload)
    log_event(stage, str(target.relative_to(DL)), len(payload), "ok")
    return True


def polite(low: float, high: float) -> None:
    time.sleep(random.uniform(low, high))


def stage_wait_for_gabes(max_hours: float = 8.0) -> None:
    print("[wait] polling for gabes-pull completion...", flush=True)
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        if GABES_LOG.exists() and "PULL_COMPLETE" in GABES_LOG.read_text():
            print("[wait] gabes complete", flush=True)
            return
        time.sleep(60)
    print("[wait] gave up waiting on gabes; continuing queue anyway", flush=True)


def stage_xrapm() -> None:
    payload = fetch("https://www.xrapm.com/")
    if payload:
        save("xrapm", "home.html", payload)


def stage_engelmann(max_posts: int = 25) -> None:
    payload = fetch("https://jeremiasengelmann.substack.com/archive?sort=new")
    if not payload:
        return
    save("engelmann", "archive.html", payload)
    links = sorted(set(re.findall(rb'https://jeremiasengelmann\.substack\.com/p/[a-z0-9\-]+', payload)))
    keywords = ("plus-minus", "rapm", "predict", "build", "rating")
    picked = [l.decode() for l in links if any(k in l.decode() for k in keywords)][:max_posts]
    for i, url in enumerate(picked):
        slug = url.rsplit("/", 1)[-1]
        post = fetch(url)
        if post:
            save("engelmann", f"post_{i:02d}_{slug}.html", post)
        polite(2.0, 4.0)


def stage_dunks(years: range = range(2014, 2026)) -> None:
    for year in years:
        payload = fetch(f"https://dunksandthrees.com/epm?season={year}", timeout=60)
        if payload:
            save("dunks_epm", f"epm_{year}.html", payload)
        polite(3.0, 6.0)


def stage_crafted(paths: tuple[str, ...] = ("/", "/ratings", "/player-ratings", "/drip")) -> None:
    for path in paths:
        payload = fetch("https://craftednba.com" + path, timeout=60)
        if payload:
            name = path.strip("/").replace("/", "_") or "home"
            save("craftednba", f"{name}.html", payload)
        polite(2.0, 4.0)


def stage_bbref_games(start_year: int = 1957, end_year: int = 1996) -> list[str]:
    all_links: list[str] = []
    for year in range(start_year, end_year + 1):
        marker = DL / "bbref_games" / f"NBA_{year}_games.html.gz"
        if marker.exists():
            links = re.findall(rb'/boxscores/[0-9]{9}[A-Z]{3}\.html', gzip.decompress(marker.read_bytes()))
            all_links += [l.decode() for l in links]
            continue
        payload = fetch(f"https://www.basketball-reference.com/leagues/NBA_{year}_games.html")
        if not payload:
            polite(5.0, 8.0)
            continue
        save("bbref_games", f"NBA_{year}_games.html", payload, gzipped=True)
        links = re.findall(rb'/boxscores/[0-9]{9}[A-Z]{3}\.html', payload)
        all_links += [l.decode() for l in links]
        print(f"[bbref_games] {year}: {len(links)} box links", flush=True)
        polite(3.5, 6.0)
    uniq = sorted(set(all_links))
    (DL / "bbref_boxes" / "link_queue.txt").parent.mkdir(parents=True, exist_ok=True)
    (DL / "bbref_boxes" / "link_queue.txt").write_text("\n".join(uniq))
    print(f"[bbref_games] total unique box links: {len(uniq)}", flush=True)
    return uniq


def stage_bbref_boxes(links: list[str], delay: tuple[float, float] = (2.8, 4.5)) -> None:
    out_dir = DL / "bbref_boxes"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, link in enumerate(links):
        name = link.rstrip("/").rsplit("/", 1)[-1]
        target = out_dir / (name + ".html.gz")
        if target.exists():
            continue
        payload = fetch("https://www.basketball-reference.com" + link)
        if payload:
            target.write_bytes(gzip.compress(payload))
            log_event("bbref_boxes", name, len(payload), "ok")
            done += 1
            if done % 200 == 0:
                print(f"[bbref_boxes] {i + 1}/{len(links)} fetched this run", flush=True)
        polite(*delay)


def stage_stats_nba(games: list[str] | None = None, cap: int = 40000) -> None:
    probe = fetch("https://stats.nba.com/stats/playbyplayv2?GameID=0029700001&StartPeriod=1&EndPeriod=14", timeout=20)
    if not probe or b"PlayByPlay" not in probe[:2000]:
        print("[stats_nba] endpoint still unreachable; skipping stage", flush=True)
        log_event("stats_nba", "probe", 0, "unreachable")
        return
    print("[stats_nba] endpoint REACHABLE tonight; crawling box scores", flush=True)
    out_dir = DL / "stats_nba_boxes"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for game_id in (games or [])[:cap]:
        target = out_dir / f"{game_id}.json.gz"
        if target.exists():
            continue
        payload = fetch(
            f"https://stats.nba.com/stats/boxscoretraditionalv2?GameID={game_id}"
            "&StartPeriod=1&EndPeriod=14&StartRange=0&EndRange=28800&RangeType=0",
            timeout=30,
        )
        if payload and b"resultSets" in payload[:2000]:
            target.write_bytes(gzip.compress(payload))
            log_event("stats_nba", game_id, len(payload), "ok")
            done += 1
            if done % 200 == 0:
                print(f"[stats_nba] {done} box scores fetched", flush=True)
        else:
            log_event("stats_nba", game_id, len(payload or b""), "bad_payload")
        polite(1.6, 2.6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="confirm that the full resumable download queue should run",
    )
    parser.add_argument(
        "--skip-gabes-wait",
        action="store_true",
        help="start the remaining stages without waiting for the Gabriel mirror",
    )
    parser.add_argument("--max-wait-hours", type=float, default=8.0)
    args = parser.parse_args()
    if not args.execute:
        parser.error("pass --execute to start network downloads")

    DL.mkdir(parents=True, exist_ok=True)
    print("QUEUE started", flush=True)
    log_event("queue", "started", 0, "ok")

    if not args.skip_gabes_wait:
        stage_wait_for_gabes(args.max_wait_hours)
    stage_xrapm()
    stage_engelmann()
    stage_dunks()
    stage_crafted()

    links = stage_bbref_games(1957, 1996)

    old_data_ids: list[str] = []
    old_dir = LAB_ROOT / "external" / "external" / "merged_playbyplay" / "old_data"
    if old_dir.exists():
        import pandas as pd

        for path in sorted(old_dir.glob("NBA97*.parquet"))[:1]:
            frame = pd.read_parquet(path, columns=["game_id"])
            old_data_ids = sorted({str(g).zfill(10) for g in frame["game_id"].unique()})
            print(f"[stats_nba] harvested {len(old_data_ids)} game ids from {path.name}", flush=True)

    stage_stats_nba(old_data_ids)
    stage_bbref_boxes(links)

    print("OVERNIGHT_COMPLETE", flush=True)
    log_event("queue", "OVERNIGHT_COMPLETE", 0, "ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
