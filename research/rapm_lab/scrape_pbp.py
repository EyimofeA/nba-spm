"""Resumable historical play-by-play scraper (NBA Stats endpoints, 1997-2013).

Safety rails before any long run:
  --probe-games 0029700001 0020300001   fetch specific old games, VALIDATE the
                                        response shape (resultSets/PlayByPlay,
                                        non-empty rows, plausible event count)
                                        and print samples, then exit.
  --max-games N                         dry run: at most N games per season.
Every saved file is validated JSON with a non-empty PlayByPlay rowSet; error
pages are never written silently. A CSV manifest (season, game_id, bytes,
sha256, status, fetched_at) makes runs resumable and auditable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}
LEAGUE_GAME_LOG = (
    "https://stats.nba.com/stats/leaguegamelog?Counter=1000&DateFrom=&DateTo="
    "&Direction=ASC&LeagueID=00&PlayerOrTeam=T&Season={season}"
    "&SeasonType=Regular+Season&Sorter=DATE"
)
PLAY_BY_PLAY = "https://stats.nba.com/stats/playbyplayv2?GameID={game_id}&StartPeriod=1&EndPeriod=14"
MANIFEST_FIELDS = ["season", "game_id", "bytes", "sha256", "events", "status", "fetched_at"]


def get_json(url: str, retries: int = 4) -> dict:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(min(60.0, 4.0 * (attempt + 1)))
    raise RuntimeError(f"GET failed after {retries} attempts: {url}: {last_error}")


def validate_pbp(payload: dict) -> int:
    """Return event count; raise on anything that is not real play-by-play."""
    result_sets = payload.get("resultSets")
    if not result_sets:
        raise ValueError("payload has no resultSets")
    table = result_sets[0]
    rows = table.get("rowSet") or []
    if not rows:
        raise ValueError("PlayByPlay rowSet is empty")
    headers = table.get("headers") or []
    for required in ("EVENTNUM", "PERIOD", "PCTIMESTRING"):
        if required not in headers:
            raise ValueError(f"missing expected PBP column {required}")
    return len(rows)


def season_games(season_label: str) -> list[str]:
    payload = get_json(LEAGUE_GAME_LOG.format(season=season_label))
    table = payload["resultSets"][0]
    headers = table["headers"]
    index = headers.index("GAME_ID")
    return sorted({str(row[index]) for row in table["rowSet"]})


def append_manifest(path: Path, row: dict) -> None:
    is_new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def fetch_game(game_id: str, season_label: str, out_dir: Path, manifest: Path, sleep: tuple[float, float]) -> None:
    target = out_dir / f"season={season_label}" / f"{game_id}.json"
    if target.exists():
        return
    payload = get_json(PLAY_BY_PLAY.format(game_id=game_id))
    events = validate_pbp(payload)  # raises on junk; junk is never written
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    target.write_bytes(raw)
    append_manifest(
        manifest,
        {
            "season": season_label,
            "game_id": game_id,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "events": events,
            "status": "ok",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    time.sleep(random.uniform(*sleep))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", nargs="*", default=[f"{year}-{str(year + 1)[-2:]}" for year in range(1997, 2014)])
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "pbp")
    parser.add_argument("--sleep-min", type=float, default=1.2)
    parser.add_argument("--sleep-max", type=float, default=2.2)
    parser.add_argument("--max-games", type=int, default=0, help="dry-run cap per season; 0 = no cap")
    parser.add_argument("--probe-games", nargs="*", default=[], help="fetch+validate these game IDs and exit")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    if args.probe_games:
        failures = 0
        for game_id in args.probe_games:
            url = PLAY_BY_PLAY.format(game_id=game_id)
            try:
                payload = get_json(url)
                events = validate_pbp(payload)
                table = payload["resultSets"][0]
                headers = table["headers"]
                home_i, away_i, period_i = headers.index("HOMEDESCRIPTION"), headers.index("VISITORDESCRIPTION"), headers.index("PERIOD")
                sample = next((r[home_i] or r[away_i] for r in reversed(table["rowSet"]) if (r[home_i] or r[away_i])), "")
                print(f"PROBE OK  {game_id}: events={events} sample='{sample}' max_period={max(r[period_i] for r in table['rowSet'])}", flush=True)
                time.sleep(1.5)
            except Exception as error:  # noqa: BLE001 - probe must report and continue
                failures += 1
                print(f"PROBE FAIL {game_id}: {error}", flush=True)
        return 1 if failures else 0

    total_done = 0
    for season_label in args.seasons:
        games = season_games(season_label)
        if args.max_games:
            games = games[: args.max_games]
        done = 0
        for number, game_id in enumerate(games, start=1):
            fetch_game(game_id, season_label, out_dir, manifest, (args.sleep_min, args.sleep_max))
            done += 1
            total_done += 1
            if done % 25 == 0 or done == len(games):
                print(f"SEASON {season_label} progress {done}/{len(games)} (total {total_done})", flush=True)
    print(f"COMPLETE total_games={total_done}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
