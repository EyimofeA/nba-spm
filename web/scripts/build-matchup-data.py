"""Create small browser-ready matchup Elo files from the pinned research artifact.

This intentionally joins only public leaderboard identity fields. It neither
modifies ratings data nor makes matchup Elo part of the production API.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT.parent / "artifacts/models/matchup_elo/matchup_elo_time_decay_v1_f71da3382c/ratings.parquet"
OUTPUT = ROOT / "local-data"
PUBLIC_DATA = ROOT / "public/data"
NAMES = ROOT.parent / "rapm/data/all_names.csv"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    matchup = pd.read_parquet(ARTIFACT)
    identity: dict[tuple[int, int], tuple[str, str | None]] = {}
    for path in PUBLIC_DATA.glob("leaderboard-*.json"):
        for row in json.loads(path.read_text()):
            identity[(int(row["PLAYER_ID"]), int(row["Season"]))] = (
                str(row["PLAYER_NAME"]),
                row.get("TEAM_ABBREVIATION"),
            )
    names = {
        int(row.PLAYER_ID): str(row.PLAYER_NAME)
        for row in pd.read_csv(NAMES).itertuples(index=False)
    }

    for season, frame in matchup.groupby("Season", sort=True):
        rows = []
        for row in frame.itertuples(index=False):
            name, team = identity.get(
                (int(row.PLAYER_ID), int(row.Season)),
                (names.get(int(row.PLAYER_ID), f"Player {int(row.PLAYER_ID)}"), None),
            )
            rows.append(
                {
                    "PLAYER_ID": int(row.PLAYER_ID),
                    "PLAYER_NAME": name,
                    "TEAM_ABBREVIATION": team,
                    "Season": int(row.Season),
                    "offense_elo": round(float(row.offense_elo), 6),
                    "defense_elo": round(float(row.defense_elo), 6),
                    "net_elo": round(float(row.net_elo), 6),
                    "offense_matchup_possessions": round(float(row.offense_matchup_possessions), 3),
                    "defense_matchup_possessions": round(float(row.defense_matchup_possessions), 3),
                    "window_start_season": int(row.window_start_season),
                    "window_end_season": int(row.window_end_season),
                    "time_decay": float(row.time_decay),
                }
            )
        (OUTPUT / f"matchup-elo-{season}.json").write_text(
            json.dumps(rows, separators=(",", ":")),
        )


if __name__ == "__main__":
    main()
