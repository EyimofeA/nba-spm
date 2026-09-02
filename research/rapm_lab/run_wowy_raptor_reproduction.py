"""Reproduce public DARKO WOWY averages and the published RAPTOR table.

This runner makes two deliberately narrow claims:

1. DARKO publishes player-game Final Cut WOWY rows.  Its season-average table
   is the simple, unweighted arithmetic mean of those rows.
2. FiveThirtyEight publishes the complete modern RAPTOR player-season table.
   The local copy should be semantically identical to the official CSV.

Neither check claims to reproduce the private DARKO state-space model or the
unpublished coefficients in FiveThirtyEight's courtmate-chain on/off model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from nba_api.stats.static import players as nba_players

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.rapm_lab.run_external_reproduction_benchmark import (
    parse_darko_wowy,
    sha256_file,
)


EXTERNAL = ROOT / "research/rapm_lab/data/external"
DARKO_ROOT = EXTERNAL / "darko_wowy"
DARKO_HISTORY = DARKO_ROOT / "game_history"
RAPTOR_ROOT = EXTERNAL / "fivethirtyeight_raptor"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/wowy_raptor_reproduction"

DARKO_HISTORY_URL = "https://www.darko.app/api/player/{player_id}/wowy-history"
RAPTOR_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "nba-raptor/modern_RAPTOR_by_player.csv"
)
RAPTOR_TEAM_URL = (
    "https://raw.githubusercontent.com/fivethirtyeight/data/master/"
    "nba-raptor/modern_RAPTOR_by_team.csv"
)
DARKO_COMPONENTS = {
    "offense": ("wowy_orapm", "reference_offense"),
    "defense": ("wowy_drapm", "reference_defense"),
    "net": ("wowy_rapm", "reference_net"),
}


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise ValueError(f"Invalid DARKO history payload: {path}")
    return payload


def _validate_history(payload: dict, player_id: int) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("DARKO history payload has no rows list")
    wrong = [row.get("nba_id") for row in rows if int(row.get("nba_id", -1)) != player_id]
    if wrong:
        raise ValueError(f"DARKO history returned another player for {player_id}")
    if payload.get("truncated") is True:
        raise ValueError(f"DARKO history is truncated for {player_id}")


def fetch_darko_history(
    player_id: int,
    destination: Path,
    *,
    attempts: int = 6,
    timeout_seconds: int = 45,
) -> tuple[int, str, int]:
    """Fetch one complete player history with retries and atomic replacement."""
    if destination.exists():
        payload = _read_json(destination)
        _validate_history(payload, player_id)
        return player_id, "cached", len(payload["rows"])

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        DARKO_HISTORY_URL.format(player_id=player_id),
        headers={"User-Agent": "CourtSignal research reproduction/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read()
            payload = json.loads(raw)
            _validate_history(payload, player_id)
            temporary = destination.with_suffix(".json.part")
            temporary.write_bytes(raw)
            os.replace(temporary, destination)
            return player_id, "downloaded", len(payload["rows"])
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"DARKO history failed for {player_id}: {last_error}")


def download_darko_histories(
    player_ids: list[int],
    *,
    workers: int = 8,
    attempts: int = 6,
) -> dict:
    counters = {"cached": 0, "downloaded": 0, "rows": 0}
    failures: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                fetch_darko_history,
                player_id,
                DARKO_HISTORY / f"{player_id}.json",
                attempts=attempts,
            ): player_id
            for player_id in player_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            player_id = futures[future]
            try:
                _, status, rows = future.result()
                counters[status] += 1
                counters["rows"] += rows
            except Exception as error:  # report every failed ID after the batch
                failures[player_id] = str(error)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"DARKO histories {completed:,}/{len(futures):,} | "
                    f"downloaded={counters['downloaded']:,} cached={counters['cached']:,} "
                    f"failed={len(failures):,}",
                    flush=True,
                )
    if failures:
        preview = dict(list(sorted(failures.items()))[:10])
        raise RuntimeError(f"{len(failures)} DARKO histories failed: {preview}")
    return counters


def published_darko_panel() -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted(DARKO_ROOT.glob("season_*.html"))
    if not paths:
        raise FileNotFoundError(f"No DARKO season snapshots below {DARKO_ROOT}")
    frame = pd.concat([parse_darko_wowy(path) for path in paths], ignore_index=True)
    if frame[["PLAYER_ID", "season"]].duplicated().any():
        raise ValueError("Published DARKO panel has duplicate player-season keys")
    return frame, paths


def reproduced_darko_panel(player_ids: list[int]) -> pd.DataFrame:
    frames = []
    for player_id in player_ids:
        path = DARKO_HISTORY / f"{player_id}.json"
        if not path.exists():
            continue
        payload = _read_json(path)
        _validate_history(payload, player_id)
        rows = pd.DataFrame(payload["rows"])
        required = ["nba_id", "season", *[source for source, _ in DARKO_COMPONENTS.values()]]
        if rows.empty or not set(required).issubset(rows.columns):
            continue
        frames.append(rows[required].copy())
    if not frames:
        raise ValueError("No DARKO game histories were available")
    games = pd.concat(frames, ignore_index=True)
    games = games.rename(columns={"nba_id": "PLAYER_ID"})
    values = [source for source, _ in DARKO_COMPONENTS.values()]
    games[values] = games[values].apply(pd.to_numeric, errors="raise")
    reproduced = (
        games.groupby(["PLAYER_ID", "season"], as_index=False)
        .agg(
            reproduced_offense=("wowy_orapm", "mean"),
            reproduced_defense=("wowy_drapm", "mean"),
            reproduced_net=("wowy_rapm", "mean"),
            reproduced_games=("wowy_rapm", "count"),
        )
    )
    reproduced["identity_error"] = (
        reproduced["reproduced_offense"]
        + reproduced["reproduced_defense"]
        - reproduced["reproduced_net"]
    ).abs()
    names = {int(row["id"]): row["full_name"] for row in nba_players.get_players()}
    reproduced["player_name"] = reproduced["PLAYER_ID"].map(names)
    return reproduced


def darko_reproduction_metrics(
    published: pd.DataFrame,
    reproduced: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = published.merge(
        reproduced,
        on=["PLAYER_ID", "season"],
        how="left",
        validate="one_to_one",
    )
    metrics = []
    for component, (_, published_column) in DARKO_COMPONENTS.items():
        reproduced_column = f"reproduced_{component}"
        complete = merged.dropna(subset=[published_column, reproduced_column])
        difference = complete[reproduced_column] - complete[published_column]
        metrics.append(
            {
                "source": "DARKO WOWY",
                "check": "season_average_from_player_games",
                "component": component,
                "published_rows": int(published[published_column].notna().sum()),
                "matched_rows": int(len(complete)),
                "coverage": float(len(complete) / max(1, published[published_column].notna().sum())),
                "pearson": float(complete[[published_column, reproduced_column]].corr().iloc[0, 1]),
                "spearman": float(
                    complete[[published_column, reproduced_column]].corr(method="spearman").iloc[0, 1]
                ),
                "maximum_absolute_error": float(difference.abs().max()),
                "mean_absolute_error": float(difference.abs().mean()),
            }
        )
    return pd.DataFrame(metrics), merged


def download_official_raptor(
    path: Path,
    *,
    url: str = RAPTOR_URL,
    attempts: int = 5,
) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "CourtSignal research reproduction/1.0"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=45) as response:
                raw = response.read()
            temporary = path.with_suffix(".csv.part")
            temporary.write_bytes(raw)
            os.replace(temporary, path)
            return
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Official RAPTOR download failed: {last_error}")


def raptor_table_reproduction(
    local_path: Path,
    official_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = pd.read_csv(local_path)
    official = pd.read_csv(official_path)
    keys = ["player_id", "season"]
    if local[keys].duplicated().any() or official[keys].duplicated().any():
        raise ValueError("RAPTOR player-season keys are not unique")
    if set(local.columns) != set(official.columns):
        raise ValueError("Local and official RAPTOR schemas differ")
    merged = local.merge(
        official,
        on=keys,
        how="outer",
        suffixes=("_local", "_official"),
        indicator=True,
        validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        counts = merged["_merge"].value_counts().to_dict()
        raise ValueError(f"Local and official RAPTOR row sets differ: {counts}")
    metrics = []
    for component in ("offense", "defense", "total"):
        column = f"raptor_onoff_{component}"
        left = f"{column}_local"
        right = f"{column}_official"
        complete = merged.loc[merged["_merge"].eq("both")].dropna(subset=[left, right])
        difference = complete[left] - complete[right]
        metrics.append(
            {
                "source": "FiveThirtyEight RAPTOR",
                "check": "official_csv_semantic_identity",
                "component": "net" if component == "total" else component,
                "published_rows": int(official[column].notna().sum()),
                "matched_rows": int(len(complete)),
                "coverage": float(len(complete) / max(1, official[column].notna().sum())),
                "pearson": float(complete[[left, right]].corr().iloc[0, 1]),
                "spearman": float(complete[[left, right]].corr(method="spearman").iloc[0, 1]),
                "maximum_absolute_error": float(difference.abs().max()),
                "mean_absolute_error": float(difference.abs().mean()),
            }
        )
    return pd.DataFrame(metrics), merged


def _run_identity(config: dict, inputs: dict[str, str]) -> str:
    payload = json.dumps({"config": config, "inputs": inputs}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:10]


def _hash_file_set(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-darko", action="store_true")
    parser.add_argument(
        "--darko-universe",
        choices=("all", "published"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument(
        "--limit-darko-players",
        type=int,
        default=None,
        help="Deterministic low-ID pilot. Omit for the complete modern panel.",
    )
    args = parser.parse_args()

    published, snapshot_paths = published_darko_panel()
    if args.darko_universe == "all":
        player_ids = sorted(int(row["id"]) for row in nba_players.get_players())
    else:
        player_ids = sorted(published["PLAYER_ID"].astype(int).unique())
    if args.limit_darko_players is not None:
        player_ids = player_ids[: args.limit_darko_players]
    if args.download_darko:
        download_summary = download_darko_histories(
            player_ids,
            workers=args.workers,
            attempts=args.attempts,
        )
    else:
        download_summary = {"cached": 0, "downloaded": 0, "rows": 0}

    reproduced = reproduced_darko_panel(player_ids)
    history_paths = [DARKO_HISTORY / f"{player_id}.json" for player_id in player_ids]
    history_paths = [path for path in history_paths if path.exists()]
    if not args.download_darko:
        download_summary = {
            "cached": len(history_paths),
            "downloaded": 0,
            "rows": int(reproduced["reproduced_games"].sum()),
        }
    darko_metrics, darko_matches = darko_reproduction_metrics(published, reproduced)

    official_raptor = RAPTOR_ROOT / "modern_RAPTOR_by_player.csv"
    download_official_raptor(official_raptor)
    local_raptor = ROOT / "data/raw/site_Data/full_raptor.csv"
    if not local_raptor.exists():
        local_raptor = Path("/Users/eadebayo/Downloads/Data/modern_RAPTOR_by_player.csv")
    if not local_raptor.exists():
        raise FileNotFoundError("No local public RAPTOR CSV was available for the identity check")
    raptor_metrics, raptor_matches = raptor_table_reproduction(local_raptor, official_raptor)

    inputs = {
        **{str(path.relative_to(ROOT)): sha256_file(path) for path in snapshot_paths},
        str(official_raptor.relative_to(ROOT)): sha256_file(official_raptor),
        str(local_raptor): sha256_file(local_raptor),
        "darko_game_history_file_set": _hash_file_set(history_paths),
    }
    config = {
        "darko_aggregation": "unweighted arithmetic mean of published player-game rows",
        "darko_player_count": len(player_ids),
        "darko_full_expected_player_count": int(published["PLAYER_ID"].nunique()),
        "darko_seasons": sorted(published["season"].astype(int).unique().tolist()),
        "darko_reconstruction_seasons": sorted(
            reproduced["season"].astype(int).unique().tolist()
        ),
        "darko_player_universe": args.darko_universe,
        "raptor_check": "semantic identity of local and official public CSV",
        "claims_excluded": [
            "DARKO causal daily model reproduction",
            "DARKO Final Cut smoothing reproduction",
            "FiveThirtyEight RAPTOR on/off algorithm reproduction",
        ],
    }
    identity = _run_identity(config, inputs)
    output = OUTPUT_ROOT / f"wowy_raptor_reproduction_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([darko_metrics, raptor_metrics], ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    reproduced.to_parquet(output / "darko_reconstructions.parquet", index=False)
    darko_matches.to_parquet(output / "darko_matches.parquet", index=False)
    raptor_matches.to_parquet(output / "raptor_table_matches.parquet", index=False)
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "exact_public_output_reproduction",
        "config": config,
        "inputs": inputs,
        "download_summary": download_summary,
        "metrics": metrics.to_dict(orient="records"),
        "forbidden_interpretation": (
            "Exact public aggregation and table identity do not reproduce DARKO's private model "
            "or FiveThirtyEight's unpublished RAPTOR-on/off coefficients."
        ),
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True))
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.12g}"))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
