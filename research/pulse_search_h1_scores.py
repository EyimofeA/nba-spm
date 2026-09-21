#!/usr/bin/env python3
"""H1 score and player-game helpers for the PULSE search pilots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.game_dim import canonical_game_id
from nba_impact.data.player_game import minutes_to_seconds


ROOT = Path(__file__).resolve().parents[1]
V3_PARENT = ROOT / "data/lake/bronze/nba_data_archive_scoring/revision=dfa8fa43"
V3_NESTED = V3_PARENT / "nbastatsv3"
SHOTDETAIL = V3_PARENT / "shotdetail"
BOX_LOGS = ROOT / "data/lake/bronze/llimllib_nba_data/player_game_logs.parquet"
SILVER = ROOT / "data/lake/silver/pulse_search"
WEB = ROOT / "web/public/data"
OUTPUT = ROOT / "artifacts/research/pulse_search"
SCORES = SILVER / "v3_terminal_scores.parquet"

def minutes_value(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return 0.0
    try:
        return minutes_to_seconds(value)
    except ValueError:
        numeric = pd.to_numeric(text, errors="coerce")
        if pd.isna(numeric) or numeric < 0:
            return 0.0
        return float(numeric) * 60.0 if numeric <= 80 else float(numeric)


def _normalize_box(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "gameId": "game_id",
        "GAME_ID": "game_id",
        "teamId": "team_id",
        "TEAM_ID": "team_id",
        "personId": "player_id",
        "PLAYER_ID": "player_id",
        "firstName": "first_name",
        "FIRST_NAME": "first_name",
        "familyName": "family_name",
        "FAMILY_NAME": "family_name",
        "position": "starter_position",
        "START_POSITION": "starter_position",
        "minutes": "minutes",
        "MIN": "minutes",
        "playerName": "player_name",
        "PLAYER_NAME": "player_name",
        "points": "points",
        "PTS": "points",
    }
    out = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns}).copy()
    missing = sorted({"game_id", "team_id", "player_id"} - set(out.columns))
    if missing:
        raise ValueError(f"Player box rows missing {missing}; columns={list(frame.columns)}")
    out["game_id"] = out["game_id"].map(canonical_game_id)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="raise").astype("int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="raise").astype("int64")
    if "starter_position" not in out:
        out["starter_position"] = ""
    out["starter_position"] = out["starter_position"].fillna("").astype(str).str.strip()
    out["starter"] = out["starter_position"].ne("")
    out["minutes_seconds"] = out["minutes"].map(minutes_value) if "minutes" in out else 0.0
    if "points" in out:
        out["points"] = pd.to_numeric(out["points"], errors="coerce").fillna(0)
    if "player_name" not in out:
        first = out["first_name"].fillna("").astype(str).str.strip() if "first_name" in out else ""
        last = out["family_name"].fillna("").astype(str).str.strip() if "family_name" in out else ""
        out["player_name"] = (first + " " + last).str.strip()
    return out


def fetch_season_box_logs(season: int) -> pd.DataFrame:
    from nba_api.stats.endpoints import playergamelogs

    label = f"{season - 1}-{str(season)[-2:]}"
    raw = playergamelogs.PlayerGameLogs(
        season_nullable=label,
        season_type_nullable="Regular Season",
        timeout=60,
    ).get_data_frames()[0]
    return _normalize_box(raw)


def shotdetail_dates(seasons: tuple[int, ...]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = SHOTDETAIL / f"project_season={season}" / "regular.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["GAME_ID", "GAME_DATE"])
        frame["game_id"] = frame["GAME_ID"].map(canonical_game_id)
        frame["game_date"] = pd.to_datetime(
            pd.to_numeric(frame["GAME_DATE"], errors="raise").astype("Int64").astype(str),
            format="%Y%m%d",
            errors="raise",
        )
        frames.append(frame.groupby("game_id", as_index=False)["game_date"].first())
    if not frames:
        return pd.DataFrame(columns=["game_id", "game_date"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("game_id")


def v3_terminal_scores(v3: pd.DataFrame, project_season: int) -> pd.DataFrame:
    work = v3.copy()
    work["game_id"] = work["gameId"].map(canonical_game_id)
    ordered = work.sort_values(["game_id", "actionId"], kind="stable")
    last = ordered.groupby("game_id", as_index=False).tail(1)
    filled = ordered.dropna(subset=["scoreHome", "scoreAway"]).groupby("game_id", as_index=False).tail(1)
    last = last.drop(columns=["scoreHome", "scoreAway"]).merge(
        filled[["game_id", "scoreHome", "scoreAway"]],
        on="game_id",
        how="left",
    )
    teams = (
        work.loc[(work["teamId"] > 0) & work["location"].isin(["h", "v"])]
        .groupby(["game_id", "location"], as_index=False)
        .agg(team_id=("teamId", "first"))
    )
    home = teams.loc[teams["location"].eq("h"), ["game_id", "team_id"]].rename(
        columns={"team_id": "home_team_id"}
    )
    away = teams.loc[teams["location"].eq("v"), ["game_id", "team_id"]].rename(
        columns={"team_id": "away_team_id"}
    )
    output = last[["game_id", "scoreHome", "scoreAway"]].merge(home, on="game_id", how="inner").merge(
        away, on="game_id", how="inner"
    )
    output["home_score"] = pd.to_numeric(output["scoreHome"], errors="coerce").astype("Int64")
    output["away_score"] = pd.to_numeric(output["scoreAway"], errors="coerce").astype("Int64")
    output["project_season"] = int(project_season)
    output["season_type"] = "regular"
    if output["game_id"].duplicated().any() or output[["home_team_id", "away_team_id"]].isna().any().any():
        raise ValueError(f"V3 terminal scores failed for {project_season}")
    return output.drop(columns=["scoreHome", "scoreAway"])


def box_team_scores(box: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    if "points" not in box.columns or box.empty:
        return scores.assign(box_home_score=pd.NA, box_away_score=pd.NA, box_v3_match=pd.NA)
    team = box.groupby(["game_id", "team_id"], as_index=False)["points"].sum()
    home = scores.merge(
        team.rename(columns={"team_id": "home_team_id", "points": "box_home_score"}),
        on=["game_id", "home_team_id"],
        how="left",
    )
    both = home.merge(
        team.rename(columns={"team_id": "away_team_id", "points": "box_away_score"}),
        on=["game_id", "away_team_id"],
        how="left",
    )
    both["box_v3_match"] = both["box_home_score"].eq(both["home_score"]) & both["box_away_score"].eq(
        both["away_score"]
    )
    return both


def build_score_reference(seasons: tuple[int, ...], box: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dates = shotdetail_dates(seasons)
    parts = []
    for season in seasons:
        v3 = pd.read_parquet(V3_NESTED / f"project_season={season}" / "regular.parquet")
        parts.append(v3_terminal_scores(v3, season))
    scores = pd.concat(parts, ignore_index=True)
    scores = scores.merge(dates, on="game_id", how="left")
    missing_dates = int(scores["game_date"].isna().sum())
    if missing_dates:
        order = scores.sort_values(["project_season", "game_id"], kind="stable").copy()
        fallback = pd.to_datetime(
            order["project_season"].map(lambda year: f"{year - 1}-10-01")
        ) + pd.to_timedelta(order.groupby("project_season").cumcount(), unit="D")
        order["game_date"] = order["game_date"].fillna(fallback)
        scores = order
    scores["game_date"] = pd.to_datetime(scores["game_date"], errors="raise").dt.date.astype(str)
    scores = box_team_scores(box, scores)
    scores["v3_home_score"] = scores["home_score"]
    scores["v3_away_score"] = scores["away_score"]
    scores["home_score"] = scores["home_score"].fillna(scores["box_home_score"])
    scores["away_score"] = scores["away_score"].fillna(scores["box_away_score"])
    box_present = scores["box_home_score"].notna() & scores["box_away_score"].notna()
    scores.loc[box_present, "home_score"] = scores.loc[box_present, "box_home_score"].astype("Int64")
    scores.loc[box_present, "away_score"] = scores.loc[box_present, "box_away_score"].astype("Int64")
    if scores[["home_score", "away_score", "home_team_id", "away_team_id"]].isna().any().any():
        raise ValueError("Score reference still has null finals after V3 and box fill.")
    compared = scores.loc[scores["box_home_score"].notna()]
    metrics = {
        "games": int(len(scores)),
        "missing_dates_before_fallback": missing_dates,
        "box_compared_games": int(len(compared)),
        "box_v3_matches": int(compared["box_v3_match"].sum()) if len(compared) else 0,
        "box_v3_mismatches": int((~compared["box_v3_match"]).sum()) if len(compared) else 0,
        "score_source": "v3_terminal_last_scoreHome_scoreAway",
        "official_proxy": "box_player_point_sums_when_present_else_v3_terminal",
    }
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORES, index=False)
    return scores, metrics


def build_player_games(scores: pd.DataFrame, box_path: Path, destination: Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    parts = []
    if box_path.exists():
        parts.append(_normalize_box(pd.read_parquet(box_path)))
    games = scores.loc[scores["season_type"].eq("regular"), [
        "game_id", "project_season", "game_date", "home_team_id", "away_team_id"
    ]].copy()
    games["game_id"] = games["game_id"].map(canonical_game_id)
    box = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not box.empty:
        merged = box.merge(games, on="game_id", how="inner")
        covered = set(int(year) for year in merged["project_season"].unique())
        parts = [merged]
    else:
        covered = set()
        parts = []
    missing = [season for season in seasons if season not in covered]
    for season in missing:
        fetched = fetch_season_box_logs(season)
        fetched = fetched.merge(games.loc[games["project_season"].eq(season)], on="game_id", how="inner")
        if fetched.empty:
            raise ValueError(f"No player-game rows for season {season}.")
        parts.append(fetched)
        print(f"fetched NBA player-game logs for {season}: {len(fetched)} rows", flush=True)
    merged = pd.concat(parts, ignore_index=True)
    merged["season_end"] = merged["project_season"].astype(int)
    merged["season_label"] = merged["season_end"].map(lambda year: f"{year - 1}-{str(year)[-2:]}")
    merged["season_type"] = "regular"
    output = merged[[
        "game_id", "season_end", "season_label", "season_type", "game_date",
        "team_id", "player_id", "player_name", "starter", "minutes_seconds",
    ]].drop_duplicates(["game_id", "player_id"])
    starter_counts = output.groupby(["game_id", "team_id"])["starter"].sum()
    bad = int(starter_counts.ne(5).sum())
    if bad > 0.05 * max(len(starter_counts), 1):
        raise ValueError(f"{bad} team-games lack five starters.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(destination, index=False)
    return output
