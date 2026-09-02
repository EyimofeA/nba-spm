"""Build the single local-only Matchup Lab payload.

The public build never reads ``web/local-data``. This script packages derived
research outputs only; raw scorer-defender rows remain outside the web tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
RUN_ROOT = PROJECT / "artifacts/research/matchup_research"
OUTPUT = ROOT / "local-data/matchup-lab.json"
PUBLIC_DATA = ROOT / "public/data"


def _records(frame: pd.DataFrame, digits: int = 6) -> list[dict]:
    work = frame.copy()
    numeric = work.select_dtypes(include=["number"]).columns
    work[numeric] = work[numeric].round(digits)
    return json.loads(work.replace({np.nan: None}).to_json(orient="records"))


def _identity() -> tuple[dict[tuple[int, int], tuple[str, str | None]], dict[int, str]]:
    by_season: dict[tuple[int, int], tuple[str, str | None]] = {}
    for path in PUBLIC_DATA.glob("leaderboard-*.json"):
        for row in json.loads(path.read_text()):
            by_season[(int(row["PLAYER_ID"]), int(row["Season"]))] = (
                str(row["PLAYER_NAME"]), row.get("TEAM_ABBREVIATION")
            )
    names = {player: value[0] for (player, _), value in by_season.items()}
    return by_season, names


def _add_identity(
    frame: pd.DataFrame,
    by_season: dict[tuple[int, int], tuple[str, str | None]],
    names: dict[int, str],
) -> pd.DataFrame:
    work = frame.copy()
    season = pd.to_numeric(work.get("Season"), errors="coerce")
    resolved = [
        by_season.get(
            (int(player), int(value)) if pd.notna(value) else (-1, -1),
            (names.get(int(player), f"Player {int(player)}"), None),
        )
        for player, value in zip(work["PLAYER_ID"], season)
    ]
    work.insert(1, "PLAYER_NAME", [value[0] for value in resolved])
    work.insert(2, "TEAM_ABBREVIATION", [value[1] for value in resolved])
    return work


def main() -> None:
    runs = sorted(
        (path for path in RUN_ROOT.glob("matchup_research_v1_*") if (path / "run.json").exists()),
        key=lambda path: (path / "run.json").stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError("No matchup research run is available.")
    run = runs[-1]
    manifest = json.loads((run / "run.json").read_text())
    required = [
        "metrics.parquet", "paired_bootstrap.parquet", "player_ratings.parquet",
        "channel_ratings.parquet", "player_history.parquet", "frequent_pairs.parquet",
        "sequential_selection.parquet",
    ]
    missing = [name for name in required if not (run / name).exists()]
    if missing:
        raise FileNotFoundError(f"Latest matchup run is incomplete: {missing}")

    by_season, names = _identity()
    players = _add_identity(pd.read_parquet(run / "player_ratings.parquet"), by_season, names)
    channels = _add_identity(pd.read_parquet(run / "channel_ratings.parquet"), by_season, names)
    history = _add_identity(pd.read_parquet(run / "player_history.parquet"), by_season, names)
    pairs = pd.read_parquet(run / "frequent_pairs.parquet").rename(
        columns={"person_id": "SCORER_ID", "matchups_person_id": "DEFENDER_ID"}
    )
    pairs.insert(1, "SCORER_NAME", pairs["SCORER_ID"].map(names).fillna("Unknown"))
    pairs.insert(3, "DEFENDER_NAME", pairs["DEFENDER_ID"].map(names).fillna("Unknown"))

    payload = {
        "run_id": manifest["run_id"],
        "scope": "localhost_only",
        "status": manifest["status"],
        "seasons": manifest["config"]["seasons"],
        "latest_season": int(players["Season"].max()),
        "quality": manifest["quality"],
        "forbidden_interpretation": manifest["forbidden_interpretation"],
        "validation": _records(pd.read_parquet(run / "metrics.parquet")),
        "bootstrap": _records(pd.read_parquet(run / "paired_bootstrap.parquet")),
        "sequential_selection": _records(pd.read_parquet(run / "sequential_selection.parquet")),
        "players": _records(players),
        "channels": _records(channels),
        "history": _records(history),
        "pairs": _records(pairs),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")))
    print(OUTPUT)


if __name__ == "__main__":
    main()
