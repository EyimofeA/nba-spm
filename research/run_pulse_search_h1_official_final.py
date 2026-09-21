#!/usr/bin/env python3
"""Smallest decisive PULSE H1 pilot: official-final vs technical-FT-excluded targets.

Rebuilds 2025-2026 regular-season V3 possession/lineup candidates, fits the
PULSE update (3000/4500/300) with and without identified technical free-throw
points, and scores identical next-season games on V3-terminal finals and on
box-sum finals.

LeagueGameLog is blocked in this environment. V3 last scoreHome/scoreAway is
the score-conserving possession target. Box player-point sums are an
independent official-final proxy. Public PULSE priors are descriptive refits
through 2026; pulse_* arms are leaky diagnostics that can only reject.

This is a research candidate. It does not replace canonical stints or spend 2027.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.game_dim import canonical_game_id
from nba_impact.data.historical_v3_lineups import build_historical_v3_lineup_candidate
from nba_impact.data.historical_v3_possession_lineups import (
    build_historical_v3_possession_lineup_candidate,
)
from nba_impact.data.historical_v3_possessions import build_historical_v3_possession_candidates
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.player_game import minutes_to_seconds
from nba_impact.data.possessions import LINEUP_COLUMNS
from nba_impact.models.canonical_pulse import game_metrics, predict_next_season_games, stint_prior_center
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import build_stint_design, fit_stint_center_path


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


def technical_free_throws(v3: pd.DataFrame) -> pd.DataFrame:
    work = v3.copy()
    work["game_id"] = work["gameId"].map(canonical_game_id)
    action = work["actionType"].astype(str).str.strip().str.casefold()
    subtype = work["subType"].astype(str) if "subType" in work else pd.Series("", index=work.index)
    description = work["description"].astype(str)
    made = action.eq("free throw")
    technical = subtype.str.contains("technical", case=False, na=False)
    missed = description.str.contains(r"^\s*MISS\b", case=False, na=False)
    selected = work.loc[made & technical & ~missed, ["game_id", "actionId", "teamId"]].copy()
    selected["event_order"] = pd.to_numeric(selected["actionId"], errors="raise").astype("int64")
    selected["team_id"] = pd.to_numeric(selected["teamId"], errors="coerce")
    selected["points"] = 1
    return selected[["game_id", "event_order", "team_id", "points"]]


def apply_technical_points(possessions: pd.DataFrame, assigned: pd.DataFrame, tech: pd.DataFrame) -> pd.DataFrame:
    mapped = assigned.merge(tech, on=["game_id", "event_order"], how="inner")
    output = possessions.copy()
    if mapped.empty:
        output["technical_points"] = 0
        output["points_excl_tech"] = output["points"]
        return output
    by_possession = mapped.groupby("possession_id", as_index=False)["points"].sum().rename(
        columns={"points": "technical_points"}
    )
    output = output.merge(by_possession, on="possession_id", how="left")
    output["technical_points"] = output["technical_points"].fillna(0).astype(int)
    output["points_excl_tech"] = output["points"] - output["technical_points"]
    if (output["points_excl_tech"] < 0).any():
        raise ValueError("Technical-FT exclusion produced negative possession points.")
    return output


def _attach_possession_meta(possessions: pd.DataFrame, segments: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    meta_columns = ["possession_id", "offense_is_home", "season_end", "game_id", "technical_points", *extra]
    meta = possessions.loc[:, [column for column in meta_columns if column in possessions.columns]]
    overlap = [column for column in meta.columns if column != "possession_id" and column in segments.columns]
    return segments.drop(columns=overlap).merge(meta, on="possession_id", how="inner")


def segments_to_stints(possessions: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    frame = _attach_possession_meta(possessions, segments)
    first = frame.groupby("possession_id", as_index=False)["segment_number"].min().rename(
        columns={"segment_number": "first_segment"}
    )
    frame = frame.merge(first, on="possession_id", how="left")
    frame["is_first"] = frame["segment_number"].eq(frame["first_segment"])
    frame["home_possessions"] = np.where(frame["offense_is_home"] & frame["is_first"], 1, 0)
    frame["away_possessions"] = np.where((~frame["offense_is_home"]) & frame["is_first"], 1, 0)
    frame["home_points"] = np.where(frame["offense_is_home"], frame["points"], 0)
    frame["away_points"] = np.where(~frame["offense_is_home"], frame["points"], 0)
    frame["home_tech"] = np.where(frame["offense_is_home"] & frame["is_first"], frame["technical_points"], 0)
    frame["away_tech"] = np.where((~frame["offense_is_home"]) & frame["is_first"], frame["technical_points"], 0)
    grouped = frame.groupby(["game_id", "season_end", *LINEUP_COLUMNS], as_index=False, sort=False).agg(
        home_possessions=("home_possessions", "sum"),
        away_possessions=("away_possessions", "sum"),
        home_points=("home_points", "sum"),
        away_points=("away_points", "sum"),
        home_technical_points_excluded=("home_tech", "sum"),
        away_technical_points_excluded=("away_tech", "sum"),
    )
    grouped["season"] = grouped["season_end"].astype(int)
    grouped["home_points_excl"] = grouped["home_points"] - grouped["home_technical_points_excluded"]
    grouped["away_points_excl"] = grouped["away_points"] - grouped["away_technical_points_excluded"]
    if (grouped[["home_points_excl", "away_points_excl"]] < 0).any().any():
        raise ValueError("Stint technical-FT exclusion produced negative points.")
    return grouped


def segments_to_terminal(possessions: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Assign each possession's points to its last lineup. Blocked H6 comparison."""
    extra = ("points",) if "points" in possessions.columns else ()
    frame = _attach_possession_meta(possessions, segments, extra=extra)
    last = frame.groupby("possession_id", as_index=False)["segment_number"].max().rename(
        columns={"segment_number": "last_segment"}
    )
    frame = frame.merge(last, on="possession_id", how="inner")
    frame = frame.loc[frame["segment_number"].eq(frame["last_segment"])].copy()
    points = frame["points"] if "points" in frame else frame["points_seg"]
    frame["home_possessions"] = np.where(frame["offense_is_home"], 1, 0)
    frame["away_possessions"] = np.where(~frame["offense_is_home"], 1, 0)
    frame["home_points"] = np.where(frame["offense_is_home"], points, 0)
    frame["away_points"] = np.where(~frame["offense_is_home"], points, 0)
    frame["home_technical_points_excluded"] = np.where(frame["offense_is_home"], frame["technical_points"], 0)
    frame["away_technical_points_excluded"] = np.where(~frame["offense_is_home"], frame["technical_points"], 0)
    grouped = frame.groupby(["game_id", "season_end", *LINEUP_COLUMNS], as_index=False, sort=False).agg(
        home_possessions=("home_possessions", "sum"),
        away_possessions=("away_possessions", "sum"),
        home_points=("home_points", "sum"),
        away_points=("away_points", "sum"),
        home_technical_points_excluded=("home_technical_points_excluded", "sum"),
        away_technical_points_excluded=("away_technical_points_excluded", "sum"),
    )
    grouped["season"] = grouped["season_end"].astype(int)
    grouped["home_points_excl"] = grouped["home_points"] - grouped["home_technical_points_excluded"]
    grouped["away_points_excl"] = grouped["away_points"] - grouped["away_technical_points_excluded"]
    return grouped


def public_priors(season: int, family: str = "pulse") -> pd.DataFrame:
    rows = json.loads((WEB / f"leaderboard-{season}.json").read_text())
    frame = pd.DataFrame(rows)
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
    frame["Window_End"] = season
    if family == "pulse":
        return frame.rename(columns={
            "pulse_prior_offense": "prior_offense_per_100",
            "pulse_prior_defense": "prior_defense_per_100",
        })
    if family == "rich":
        return frame.rename(columns={
            "rich_spm_offense": "prior_offense_per_100",
            "rich_spm_defense": "prior_defense_per_100",
        })
    raise ValueError(family)


def official_margins(scores: pd.DataFrame, season: int) -> pd.DataFrame:
    games = scores.loc[scores["project_season"].eq(season) & scores["season_type"].eq("regular")].copy()
    games["game_id"] = games["game_id"].map(canonical_game_id)
    games["v3_margin"] = games["home_score"].astype(float) - games["away_score"].astype(float)
    if "box_home_score" in games and games["box_home_score"].notna().any():
        games["box_margin"] = games["box_home_score"].astype(float) - games["box_away_score"].astype(float)
        games["official_margin"] = games["box_margin"].where(games["box_margin"].notna(), games["v3_margin"])
    else:
        games["box_margin"] = games["v3_margin"]
        games["official_margin"] = games["v3_margin"]
    return games[[
        "game_id", "official_margin", "v3_margin", "box_margin", "home_score", "away_score"
    ]]


def score_games(source, target, beta, intercept, official: pd.DataFrame, name: str, rating_season: int) -> pd.DataFrame:
    games = predict_next_season_games(source, target, beta, intercept)
    games = games.merge(official, on="game_id", how="inner", validate="one_to_one")
    output = games.rename(columns={"actual_margin": "excl_margin"})
    output["actual_margin"] = output["official_margin"]
    output["candidate"] = name
    output["rating_season"] = rating_season
    output["outcome_season"] = rating_season + 1
    output["squared_error_official"] = (output["official_margin"] - output["predicted_margin"]) ** 2
    output["squared_error_v3"] = (output["v3_margin"] - output["predicted_margin"]) ** 2
    output["squared_error_excl"] = (output["excl_margin"] - output["predicted_margin"]) ** 2
    return output


def fit_stint_arm(
    stints: pd.DataFrame,
    point_columns: tuple[str, str],
    priors,
    season: int,
    scale: float,
    *,
    lambda_off: float = 3000.0,
    lambda_def: float = 4500.0,
):
    frame = stints.copy()
    frame["home_points"] = frame[point_columns[0]]
    frame["away_points"] = frame[point_columns[1]]
    design = build_stint_design(frame)
    zero = np.zeros(design.X.shape[1])
    config = RapmConfig(
        (season,),
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_home=300.0,
        data_scope="pulse_search_h1_v3_candidate",
    )
    rapm_beta, rapm_intercept = fit_stint_center_path(design, config, zero, center_scales=(0.0,))[0.0]
    center, coverage = stint_prior_center(design, priors, season)
    pulse_beta, pulse_intercept = fit_stint_center_path(
        design, config, center, center_scales=(scale,)
    )[scale]
    return design, {
        "rapm": (rapm_beta, rapm_intercept),
        "pulse": (pulse_beta, pulse_intercept),
    }, coverage, center


def paired_delta(games: pd.DataFrame, left: str, right: str, value: str) -> dict:
    wide = games.loc[games["candidate"].isin([left, right])].pivot(
        index=["outcome_season", "game_id"], columns="candidate", values=value
    )
    if wide.isna().any().any():
        raise ValueError(f"{left} and {right} did not score identical games on {value}.")
    delta = wide[left] - wide[right]
    rng = np.random.default_rng(20260921)
    values = [group.to_numpy() for _, group in delta.groupby(level="outcome_season")]
    draws = np.array([
        np.mean([part[rng.integers(0, len(part), len(part))].mean() for part in values])
        for _ in range(2000)
    ])
    observed = float(np.mean([part.mean() for part in values]))
    return {
        "mean": observed,
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "games": int(len(wide)),
    }


def build_sources(seasons: tuple[int, ...], scores: pd.DataFrame, players: Path) -> list[dict]:
    possession_root = SILVER / "possession_candidates"
    quality_poss = SILVER / "possession_quality.parquet"
    if not all((possession_root / f"project_season={season}" / "regular.parquet").exists() for season in seasons):
        build_historical_v3_possession_candidates(
            V3_NESTED,
            SCORES,
            possession_root,
            quality_poss,
            SILVER / "manifests",
            seasons=seasons,
            season_types=("regular",),
        )
    summaries = []
    for season in seasons:
        season_root = SILVER / f"project_season={season}"
        season_root.mkdir(parents=True, exist_ok=True)
        stints_path = season_root / "lineup_stints.parquet"
        lineup_quality = season_root / "lineup_quality.parquet"
        if not stints_path.exists():
            report = build_historical_v3_lineup_candidate(
                V3_PARENT,
                players,
                SCORES,
                stints_path,
                lineup_quality,
                season_root / "lineup_report.json",
                SILVER / "manifests",
                project_season=season,
                season_type="regular",
            )
            print(
                f"lineups {season}: passed={report['passed_game_count']} "
                f"quarantined={report['quarantined_game_count']}",
                flush=True,
            )
        attached_poss = season_root / "possessions.parquet"
        segments = season_root / "segments.parquet"
        assigned = season_root / "assigned_actions.parquet"
        attach_quality = season_root / "attach_quality.parquet"
        if not attached_poss.exists():
            attach = build_historical_v3_possession_lineup_candidate(
                V3_PARENT,
                possession_root / f"project_season={season}" / "regular.parquet",
                quality_poss,
                stints_path,
                lineup_quality,
                attached_poss,
                segments,
                assigned,
                attach_quality,
                season_root / "attach_report.json",
                SILVER / "manifests",
                project_season=season,
                season_type="regular",
            )
            print(
                f"attach {season}: emitted={attach['emitted_game_count']} "
                f"rejected={attach['rejected_after_attachment_count']}",
                flush=True,
            )
        v3 = pd.read_parquet(V3_NESTED / f"project_season={season}" / "regular.parquet")
        possessions = apply_technical_points(
            pd.read_parquet(attached_poss),
            pd.read_parquet(assigned),
            technical_free_throws(v3),
        )
        stints = segments_to_stints(possessions, pd.read_parquet(segments))
        terminal = segments_to_terminal(possessions, pd.read_parquet(segments))
        stints.to_parquet(season_root / "canonical_like_stints.parquet", index=False)
        terminal.to_parquet(season_root / "terminal_like_stints.parquet", index=False)
        possessions.to_parquet(season_root / "possessions_with_tech.parquet", index=False)
        summaries.append({
            "season": season,
            "possession_games": int(possessions["game_id"].nunique()),
            "stints": int(len(stints)),
            "terminal_rows": int(len(terminal)),
            "technical_points": int(possessions["technical_points"].sum()),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2025,2026")
    args = parser.parse_args()
    seasons = tuple(int(item) for item in args.seasons.split(",") if item.strip())
    SILVER.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    box = _normalize_box(pd.read_parquet(BOX_LOGS)) if BOX_LOGS.exists() else pd.DataFrame()
    scores, score_metrics = build_score_reference(seasons, box)
    print(json.dumps({"score_metrics": score_metrics}, indent=2), flush=True)
    players = SILVER / "player_games.parquet"
    if not players.exists():
        build_player_games(scores, BOX_LOGS, players, seasons)
    summaries = build_sources(seasons, scores, players)
    print(json.dumps(summaries, indent=2), flush=True)

    rating_seasons = [season for season in seasons if season + 1 in seasons]
    rows = []
    fold_metrics = []
    for rating_season in rating_seasons:
        source_stints = pd.read_parquet(SILVER / f"project_season={rating_season}" / "canonical_like_stints.parquet")
        target_stints = pd.read_parquet(SILVER / f"project_season={rating_season + 1}" / "canonical_like_stints.parquet")
        priors = public_priors(rating_season)
        official = official_margins(scores, rating_season + 1)
        for target_name, columns in {
            "excl": ("home_points_excl", "away_points_excl"),
            "incl": ("home_points", "away_points"),
        }.items():
            source_design, fits, coverage, _center = fit_stint_arm(
                source_stints, columns, priors, rating_season, scale=1.0
            )
            target_frame = target_stints.copy()
            target_frame["home_points"] = target_frame[columns[0]]
            target_frame["away_points"] = target_frame[columns[1]]
            target_design = build_stint_design(target_frame)
            for model, (beta, intercept) in fits.items():
                scored = score_games(
                    source_design, target_design, beta, intercept, official,
                    f"{model}_{target_name}", rating_season,
                )
                rows.append(scored)
                for actual, label in (("official_margin", "official"), ("excl_margin", "excl"), ("v3_margin", "v3")):
                    metrics = game_metrics(scored.assign(actual_margin=scored[actual]))
                    fold_metrics.append({
                        "candidate": f"{model}_{target_name}",
                        "actual": label,
                        "rating_season": rating_season,
                        "outcome_season": rating_season + 1,
                        "prior_players_with_prior": coverage["players_with_prior"],
                        **metrics,
                    })
            for scale in (0.5, 0.75):
                _, scaled, _, _ = fit_stint_arm(source_stints, columns, priors, rating_season, scale=scale)
                scored = score_games(
                    source_design, target_design, *scaled["pulse"], official,
                    f"pulse_{target_name}_scale{scale}", rating_season,
                )
                rows.append(scored)
                metrics = game_metrics(scored.assign(actual_margin=scored["official_margin"]))
                fold_metrics.append({
                    "candidate": f"pulse_{target_name}_scale{scale}",
                    "actual": "official",
                    "rating_season": rating_season,
                    "outcome_season": rating_season + 1,
                    **metrics,
                })
        print(f"H1 fold {rating_season}->{rating_season + 1} complete", flush=True)

    games = pd.concat(rows, ignore_index=True)
    folds = pd.DataFrame(fold_metrics)
    summary = folds.groupby(["candidate", "actual"], as_index=False).agg(
        folds=("outcome_season", "nunique"),
        games=("games", "sum"),
        equal_season_mse=("mse", "mean"),
        mean_correlation=("correlation", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    core = games.loc[games["candidate"].isin(["pulse_incl", "pulse_excl", "rapm_incl", "rapm_excl"])]
    intervals = {
        "pulse_incl_minus_pulse_excl_official_mse": paired_delta(
            core, "pulse_incl", "pulse_excl", "squared_error_official"
        ),
        "rapm_incl_minus_rapm_excl_official_mse": paired_delta(
            core, "rapm_incl", "rapm_excl", "squared_error_official"
        ),
        "pulse_incl_minus_rapm_incl_official_mse": paired_delta(
            core, "pulse_incl", "rapm_incl", "squared_error_official"
        ),
        "pulse_excl_minus_rapm_excl_official_mse": paired_delta(
            core, "pulse_excl", "rapm_excl", "squared_error_official"
        ),
    }
    run_id = "pulse_search_h1_official_final_v1"
    destination = OUTPUT / run_id
    destination.mkdir(parents=True, exist_ok=True)
    games.to_parquet(destination / "game_predictions.parquet", index=False)
    folds.to_parquet(destination / "fold_metrics.parquet", index=False)
    summary.to_parquet(destination / "summary.parquet", index=False)
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_pilot_not_promotion",
        "hypothesis": "Score and fit the official-final target, including technical free throws.",
        "seasons": list(seasons),
        "score_metrics": score_metrics,
        "season_build": summaries,
        "source_hashes": {
            "scores": sha256_file(SCORES),
            "runner": sha256_file(Path(__file__)),
        },
        "limitations": [
            "V3 historical possession/lineup candidates, not canonical CDN/Gabriel stints.",
            "LeagueGameLog blocked; official proxy is box player-point sums, else V3 terminal scores.",
            "Public PULSE priors are descriptive refits through 2026; pulse_* arms are leaky diagnostics.",
            "Pilot folds only; not the frozen 12-fold gate.",
            "2027 unused.",
        ],
        "summary": summary.to_dict("records"),
        "paired_intervals": intervals,
    }
    write_json_atomic(run, destination / "run.json")
    print(summary.to_string(index=False))
    print(json.dumps(intervals, indent=2))
    print(destination, flush=True)


if __name__ == "__main__":
    main()
