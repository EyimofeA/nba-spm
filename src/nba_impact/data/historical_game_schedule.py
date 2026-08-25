"""Acquire and validate historical NBA regular-season game/team metadata."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pandas as pd
import requests


ENDPOINT = "https://stats.nba.com/stats/leaguegamelog"


def season_label(season_end: int) -> str:
    return f"{season_end - 1}-{str(season_end)[-2:]}"


def parse_league_game_log(payload: dict, *, season_end: int) -> pd.DataFrame:
    result = payload["resultSets"][0]
    rows = pd.DataFrame(result["rowSet"], columns=result["headers"])
    required = {"GAME_ID", "GAME_DATE", "TEAM_ID", "MATCHUP"}
    if missing := sorted(required - set(rows.columns)):
        raise ValueError(f"League game log is missing {missing}.")
    rows["GAME_ID"] = rows["GAME_ID"].astype(str).str.zfill(10)
    if not rows.groupby("GAME_ID").size().eq(2).all():
        raise ValueError("Every regular-season game must have exactly two team rows.")
    home = rows.loc[rows["MATCHUP"].astype(str).str.contains(" vs. ")].copy()
    away = rows.loc[rows["MATCHUP"].astype(str).str.contains(" @ ")].copy()
    if len(home) * 2 != len(rows) or len(away) != len(home):
        raise ValueError("Every game must have one home and one away matchup row.")
    games = home[["GAME_ID", "GAME_DATE", "TEAM_ID"]].rename(
        columns={"GAME_ID": "game_id", "GAME_DATE": "game_date", "TEAM_ID": "home_team_id"}
    ).merge(
        away[["GAME_ID", "TEAM_ID"]].rename(
            columns={"GAME_ID": "game_id", "TEAM_ID": "away_team_id"}
        ),
        on="game_id",
        validate="one_to_one",
    )
    games["project_season"] = int(season_end)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games[["home_team_id", "away_team_id"]] = games[["home_team_id", "away_team_id"]].astype(int)
    return games[
        ["project_season", "game_id", "game_date", "home_team_id", "away_team_id"]
    ].sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)


def acquire_historical_game_schedule(
    output_root: Path,
    *,
    seasons: tuple[int, ...] = tuple(range(1997, 2027)),
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
        }
    )
    frames = []
    for season in seasons:
        raw_path = output_root / f"leaguegamelog_{season}.json.gz"
        if raw_path.exists():
            with gzip.open(raw_path, "rt") as handle:
                payload = json.load(handle)
        else:
            error = None
            for attempt in range(5):
                try:
                    response = session.get(
                        ENDPOINT,
                        params={
                            "Counter": 0,
                            "DateFrom": "",
                            "DateTo": "",
                            "Direction": "DESC",
                            "LeagueID": "00",
                            "PlayerOrTeam": "T",
                            "Season": season_label(season),
                            "SeasonType": "Regular Season",
                            "Sorter": "DATE",
                        },
                        timeout=45,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    with gzip.open(raw_path, "wt") as handle:
                        json.dump(payload, handle, separators=(",", ":"))
                    error = None
                    break
                except (requests.RequestException, ValueError) as caught:
                    error = caught
                    time.sleep(2**attempt)
            if error is not None:
                raise error
            time.sleep(0.4)
        frame = parse_league_game_log(payload, season_end=season)
        frame.to_parquet(output_root / f"schedule_{season}.parquet", index=False)
        frames.append(frame)
    schedule = pd.concat(frames, ignore_index=True)
    if schedule["game_id"].duplicated().any():
        raise ValueError("Game IDs must be unique across seasons.")
    first_season = min(seasons)
    last_season = max(seasons)
    schedule.to_parquet(
        output_root / f"schedule_{first_season}_{last_season}.parquet", index=False
    )
    return schedule
