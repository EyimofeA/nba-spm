#!/usr/bin/env python3
"""Fetch official finals, missing player boxes, and optional canonical lineup RDS.

Unskips 2024 and any other rating/outcome seasons needed for the V3 2017-2026
window. Canonical RDS conversion still needs R or pyreadr; this script only
downloads. 2027 unused.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from nba_impact.data.game_dim import canonical_game_id
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.official_game_scores import normalize_game_scores, season_label
from research.run_pulse_search_h1_official_final import _normalize_box


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data/lake/bronze"
SCORES_ROOT = BRONZE / "official_game_scores"
BOX_ROOT = BRONZE / "nba_player_game_logs"
LINEUP_ROOT = BRONZE / "canonical_historical_lineups"
EVENT_ROOT = BRONZE / "canonical_historical_events"
ESPN_BOX = BRONZE / "llimllib_nba_data/espn/player_box.parquet"
V3_NESTED = BRONZE / "nba_data_archive_scoring/revision=dfa8fa43/nbastatsv3"
HF_REVISION = "dfa8fa43f89ae2ca6c18db524edc2050a6bb2286"
LINEUP_RAW = "https://raw.githubusercontent.com/ramirobentes/nba_pbp_data/main"
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def _get_json(url: str, timeout: int = 90) -> dict:
    last_error = None
    for attempt in range(1, 6):
        try:
            response = requests.get(url, headers=NBA_HEADERS, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"GET failed for {url}: {last_error}")


def _result_frame(payload: dict) -> pd.DataFrame:
    result = payload["resultSets"][0]
    return pd.DataFrame(result["rowSet"], columns=result["headers"])


def fetch_league_game_log(project_season: int, season_type: str = "Regular Season") -> pd.DataFrame:
    label = season_label(project_season)
    kind = season_type.replace(" ", "+")
    url = (
        "https://stats.nba.com/stats/leaguegamelog"
        f"?Counter=0&DateFrom=&DateTo=&Direction=ASC&LeagueID=00"
        f"&PlayerOrTeam=T&Season={label}&SeasonType={kind}&Sorter=DATE"
    )
    return _result_frame(_get_json(url))


def fetch_player_game_logs(project_season: int) -> pd.DataFrame:
    label = season_label(project_season)
    url = (
        "https://stats.nba.com/stats/playergamelogs"
        f"?DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=00"
        f"&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome="
        f"&PORound=0&PaceAdjust=N&PerMode=Totals&Period=0&PlusMinus=N"
        f"&Rank=N&Season={label}&SeasonSegment=&SeasonType=Regular+Season"
        f"&ShotClockRange=&VsConference=&VsDivision="
    )
    return _normalize_box(_result_frame(_get_json(url, timeout=120)))


def write_official_scores(seasons: tuple[int, ...]) -> dict:
    frames = []
    metrics = []
    SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        destination = SCORES_ROOT / f"project_season={season}" / "regular.parquet"
        if destination.exists() and len(pd.read_parquet(destination)) >= 700:
            scores = pd.read_parquet(destination)
            frames.append(scores)
            metrics.append({"season": season, "status": "existing", "games": int(len(scores))})
            print(f"official scores {season}: existing {len(scores)}", flush=True)
            continue
        raw = fetch_league_game_log(season)
        scores, quality = normalize_game_scores(raw, project_season=season, season_type="regular")
        if not quality["passed"]:
            raise ValueError(f"Official scores failed for {season}: {quality}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(destination, index=False)
        frames.append(scores)
        metrics.append({"season": season, "status": "downloaded", **quality})
        print(f"official scores {season}: {quality['games']} games", flush=True)
        time.sleep(0.7)
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("game_id")
    combined_path = SCORES_ROOT / "official_game_scores.parquet"
    combined.to_parquet(combined_path, index=False)
    return {
        "games": int(len(combined)),
        "seasons": metrics,
        "sha256": sha256_file(combined_path),
    }


def write_player_logs(seasons: tuple[int, ...]) -> dict:
    metrics = []
    BOX_ROOT.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        destination = BOX_ROOT / f"season={season}" / "regular.parquet"
        if destination.exists() and len(pd.read_parquet(destination)) >= 15000:
            metrics.append({"season": season, "status": "existing", "rows": int(len(pd.read_parquet(destination)))})
            print(f"player logs {season}: existing", flush=True)
            continue
        frame = fetch_player_game_logs(season)
        if len(frame) < 10000:
            raise ValueError(f"Player logs for {season} too small: {len(frame)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        metrics.append({"season": season, "status": "downloaded", "rows": int(len(frame))})
        print(f"player logs {season}: {len(frame)} rows", flush=True)
        time.sleep(0.7)
    return {"seasons": metrics}


def download_file(url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1000:
        return {"path": str(destination), "status": "existing", "bytes": destination.stat().st_size}
    partial = destination.with_suffix(destination.suffix + ".partial")
    with requests.get(url, stream=True, timeout=300, headers={"User-Agent": "CourtSignal research"}) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    partial.replace(destination)
    return {"path": str(destination), "status": "downloaded", "bytes": destination.stat().st_size}


def download_canonical_inputs(start: int, end: int) -> dict:
    rows = []
    for season in range(start, end + 1):
        lineup = LINEUP_ROOT / f"season={season}" / "regular.rds"
        try:
            rows.append({"kind": "lineup", "season": season, **download_file(
                f"{LINEUP_RAW}/lineup-final{season}/data.rds", lineup
            )})
            print(f"lineup rds {season}: {rows[-1]['bytes']} bytes", flush=True)
        except Exception as exc:
            rows.append({"kind": "lineup", "season": season, "status": "failed", "error": str(exc)})
            print(f"lineup rds {season}: failed {exc}", flush=True)
        event = EVENT_ROOT / f"season={season}" / "regular.parquet"
        source_year = season - 1
        try:
            rows.append({"kind": "event", "season": season, **download_file(
                f"https://huggingface.co/datasets/cdechoch/nba-data-archive/resolve/{HF_REVISION}/per_season/nbastats/{source_year}.parquet",
                event,
            )})
            print(f"events {season}: {rows[-1]['bytes']} bytes", flush=True)
        except Exception as exc:
            rows.append({"kind": "event", "season": season, "status": "failed", "error": str(exc)})
            print(f"events {season}: failed {exc}", flush=True)
    return {"files": rows}


def classic_event_scores(season: int) -> pd.DataFrame:
    path = EVENT_ROOT / f"season={season}" / "regular.parquet"
    if not path.exists():
        return pd.DataFrame()
    events = pd.read_parquet(
        path,
        columns=[
            "GAME_ID", "EVENTNUM", "SCORE", "PLAYER1_TEAM_ID",
            "PLAYER1_TEAM_ABBREVIATION", "_season_type",
        ],
    )
    events = events.loc[events["_season_type"].eq("rg")].copy()
    events["game_id"] = events["GAME_ID"].map(canonical_game_id)
    last = (
        events.dropna(subset=["SCORE"])
        .sort_values(["game_id", "EVENTNUM"], kind="stable")
        .groupby("game_id", as_index=False)
        .tail(1)
    )
    parts = last["SCORE"].astype(str).str.split(r"\s*-\s*", expand=True)
    last["away_score"] = pd.to_numeric(parts[0], errors="raise").astype("Int64")
    last["home_score"] = pd.to_numeric(parts[1], errors="raise").astype("Int64")
    teams = (
        events.dropna(subset=["PLAYER1_TEAM_ID", "PLAYER1_TEAM_ABBREVIATION"])
        .assign(
            team_id=lambda frame: pd.to_numeric(frame["PLAYER1_TEAM_ID"], errors="raise").astype("int64"),
            tricode=lambda frame: frame["PLAYER1_TEAM_ABBREVIATION"].astype(str).str.strip(),
        )
        .drop_duplicates(["game_id", "tricode"])
    )
    lookup = {
        game_id: dict(zip(frame["tricode"], frame["team_id"]))
        for game_id, frame in teams.groupby("game_id")
    }
    lineup_path = LINEUP_ROOT / f"season={season}" / "regular.rds"
    if not lineup_path.exists():
        return pd.DataFrame()
    import pyreadr

    rows = next(iter(pyreadr.read_r(str(lineup_path)).values()))
    home = rows.loc[rows["location_team"].eq("home"), ["game_id", "team", "opp"]].drop_duplicates("game_id")
    home["game_id"] = home["game_id"].map(canonical_game_id)
    home["home_tricode"] = home["team"].astype(str).str.strip()
    home["away_tricode"] = home["opp"].astype(str).str.strip()
    output = last[["game_id", "home_score", "away_score"]].merge(home, on="game_id", how="inner")
    output["home_team_id"] = [
        lookup.get(game_id, {}).get(tricode)
        for game_id, tricode in zip(output["game_id"], output["home_tricode"])
    ]
    output["away_team_id"] = [
        lookup.get(game_id, {}).get(tricode)
        for game_id, tricode in zip(output["game_id"], output["away_tricode"])
    ]
    output["project_season"] = season
    output["season_type"] = "regular"
    output["game_date"] = pd.to_datetime(f"{season - 1}-10-01") + pd.to_timedelta(
        output.groupby("project_season").cumcount(), unit="D"
    )
    output["game_date"] = output["game_date"].dt.date.astype(str)
    output["score_source"] = "classic_pbp_last_SCORE"
    return output.dropna(subset=["home_team_id", "away_team_id", "home_score", "away_score"])


def espn_official_scores(seasons: tuple[int, ...]) -> pd.DataFrame:
    if not ESPN_BOX.exists():
        return pd.DataFrame()
    espn = pd.read_parquet(ESPN_BOX)
    espn["game_id"] = espn["game_id"].map(canonical_game_id)
    espn = espn.loc[espn["season"].isin(seasons) & espn["game_id"].str.startswith("002")].copy()
    espn["pts"] = pd.to_numeric(espn["pts"], errors="coerce")
    espn["home"] = pd.to_numeric(espn["home"], errors="coerce")
    espn["team_id"] = pd.to_numeric(espn["team_id"], errors="coerce")
    espn = espn.dropna(subset=["team_id", "pts", "home"])
    espn["team_id"] = espn["team_id"].astype("int64")
    team = espn.groupby(["season", "game_id", "team_id", "home"], as_index=False)["pts"].sum()
    home = team.loc[team["home"].eq(1)].rename(columns={"team_id": "home_team_id", "pts": "home_score", "season": "project_season"})
    away = team.loc[team["home"].eq(0)].rename(columns={"team_id": "away_team_id", "pts": "away_score", "season": "project_season"})
    output = home.merge(away, on=["project_season", "game_id"], how="inner")
    output["season_type"] = "regular"
    output["game_date"] = pd.to_datetime(output["project_season"].map(lambda year: f"{year - 1}-10-01")) + pd.to_timedelta(
        output.groupby("project_season").cumcount(), unit="D"
    )
    output["game_date"] = output["game_date"].dt.date.astype(str)
    output["score_source"] = "espn_box_team_point_sums"
    return output[
        ["project_season", "season_type", "game_id", "game_date", "home_team_id", "away_team_id", "home_score", "away_score", "score_source"]
    ]


def v3_official_scores(seasons: tuple[int, ...]) -> pd.DataFrame:
    from research.run_pulse_search_h1_official_final import v3_terminal_scores

    parts = []
    for season in seasons:
        path = V3_NESTED / f"project_season={season}" / "regular.parquet"
        if not path.exists():
            continue
        frame = v3_terminal_scores(pd.read_parquet(path), season)
        frame["score_source"] = "v3_terminal_last_scoreHome_scoreAway"
        frame["game_date"] = pd.to_datetime(f"{season - 1}-10-01") + pd.to_timedelta(
            range(len(frame)), unit="D"
        )
        frame["game_date"] = frame["game_date"].dt.date.astype(str)
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_offline_official_scores(seasons: tuple[int, ...] = tuple(range(2014, 2027))) -> dict:
    frames = []
    sources = []
    espn = espn_official_scores(seasons)
    if not espn.empty:
        frames.append(espn)
        sources.append({"source": "espn", "games": int(len(espn))})
    classic_parts = [classic_event_scores(season) for season in seasons]
    classic = pd.concat([part for part in classic_parts if not part.empty], ignore_index=True) if any(
        not part.empty for part in classic_parts
    ) else pd.DataFrame()
    if not classic.empty:
        frames.append(classic)
        sources.append({"source": "classic_pbp", "games": int(len(classic))})
    v3 = v3_official_scores(tuple(season for season in seasons if season >= 2017))
    if not v3.empty:
        frames.append(v3)
        sources.append({"source": "v3", "games": int(len(v3))})
    if not frames:
        raise FileNotFoundError("No offline official-score sources available.")
    combined = pd.concat(frames, ignore_index=True)
    # Prefer ESPN box, then classic PBP, then V3 terminal.
    rank = {"espn_box_team_point_sums": 0, "classic_pbp_last_SCORE": 1, "v3_terminal_last_scoreHome_scoreAway": 2}
    combined["_rank"] = combined["score_source"].map(rank).fillna(9)
    combined = combined.sort_values(["game_id", "_rank"], kind="stable").drop_duplicates("game_id")
    combined = combined.drop(columns=["_rank", "score_source"], errors="ignore")
    SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    for season, group in combined.groupby("project_season"):
        destination = SCORES_ROOT / f"project_season={int(season)}" / "regular.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(destination, index=False)
    path = SCORES_ROOT / "official_game_scores.parquet"
    combined.to_parquet(path, index=False)
    return {
        "games": int(len(combined)),
        "seasons": sorted(int(year) for year in combined["project_season"].unique()),
        "sources": sources,
        "sha256": sha256_file(path),
        "path": str(path),
    }


def main() -> None:
    seasons = tuple(range(2014, 2027))
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_scores": None,
        "player_logs": None,
        "canonical_inputs": None,
    }
    try:
        report["official_scores"] = build_offline_official_scores(seasons)
        print(f"offline official scores: {report['official_scores']['games']} games", flush=True)
    except Exception as exc:
        report["official_scores"] = {"status": "failed", "error": str(exc)}
        print(f"offline official scores failed: {exc}", flush=True)
        try:
            report["official_scores"] = write_official_scores(tuple(range(2017, 2027)))
        except Exception as api_exc:
            report["official_scores"] = {"status": "failed", "error": str(api_exc)}
            print(f"LeagueGameLog also failed: {api_exc}", flush=True)
    try:
        report["player_logs"] = write_player_logs(tuple(range(2017, 2025)))
    except Exception as exc:
        report["player_logs"] = {"status": "failed", "error": str(exc)}
        print(f"player logs failed: {exc}", flush=True)
    try:
        report["canonical_inputs"] = download_canonical_inputs(2014, 2026)
    except Exception as exc:
        report["canonical_inputs"] = {"status": "failed", "error": str(exc)}
        print(f"canonical inputs failed: {exc}", flush=True)
    destination = ROOT / "artifacts/research/pulse_search/fetch_support.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report, destination)
    print(json.dumps(report, indent=2, default=str)[:4000], flush=True)


if __name__ == "__main__":
    main()
