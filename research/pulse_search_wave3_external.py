#!/usr/bin/env python3
"""External ratings and official-final common-panel scoring for PULSE wave 3."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

from nba_impact.models.canonical_pulse import game_metrics
from nba_impact.models.external_impact_benchmark import normalize_player_name, parse_xrapm_html
from nba_impact.models.stint_rapm import build_stint_design, stint_ratings
from research.run_pulse_search_fetch_support import download_file
from research.run_pulse_search_h1_official_final import OUTPUT, score_games
from research.run_pulse_search_real_protocol import OFFICIAL, load_official_margins, load_stints


EXTERNAL_ROOT = OUTPUT / "external_sources_v1"
XRAPM_URL = "https://xrapm.com/table_pages/{name}"
EPM_RECORD = re.compile(
    r"\{season:(\d+),game_dt:\"[^\"]+\",player_id:(\d+),player_name:\"([^\"]+)\","
    r"[^}]*?off:(-?\d+(?:\.\d+)?),def:(-?\d+(?:\.\d+)?),tot:(-?\d+(?:\.\d+)?)"
)
HEADERS = {"User-Agent": "CourtSignal research"}


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    summary = folds.groupby("candidate", as_index=False).agg(
        folds=("outcome_season", "nunique"),
        games=("games", "sum"),
        equal_season_mse=("mse", "mean"),
        mean_correlation=("correlation", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    return summary.sort_values("equal_season_rmse")


def common_coefficients(ratings: pd.DataFrame, source, common: set[int], home: float) -> np.ndarray:
    keep = np.isin(source.players, list(common))
    selected = ratings.set_index("PLAYER_ID").reindex(source.players[keep])
    beta = np.zeros(source.X.shape[1])
    n = len(source.players)
    for offset, side, weights, sign in (
        (0, "offense", source.off_possessions, 1),
        (n, "defense", source.def_possessions, -1),
    ):
        values = selected[side].to_numpy(float)
        values = sign * (values - np.average(values, weights=weights[keep])) / 100
        beta[offset + np.flatnonzero(keep)] = values
    beta[-1] = home
    return beta


def fetch_xrapm(seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        name = "xRAPM.html" if season == 2026 else f"xRAPM_{season}.html"
        path = EXTERNAL_ROOT / "xrapm" / f"{season}.html"
        try:
            download_file(XRAPM_URL.format(name=name), path)
            frame = parse_xrapm_html(path.read_text(errors="ignore"), season, exclude_ambiguous_names=True)
            rows.append(frame)
            print(f"xRAPM {season}: {len(frame)}", flush=True)
        except Exception as exc:
            print(f"xRAPM {season} failed: {exc}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fetch_expected_epm(seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        path = EXTERNAL_ROOT / "epm" / f"{season}.html"
        url = f"https://dunksandthrees.com/epm?season={season}"
        try:
            download_file(url, path)
            found = [
                {
                    "rating_season": int(match.group(1)),
                    "PLAYER_ID": int(match.group(2)),
                    "player_name": match.group(3),
                    "offense": float(match.group(4)),
                    "defense": float(match.group(5)),
                    "net": float(match.group(6)),
                    "candidate": "EPM expected",
                }
                for match in EPM_RECORD.finditer(path.read_text(errors="ignore"))
            ]
            if not found:
                raise ValueError("no EPM records parsed")
            rows.append(pd.DataFrame(found).drop_duplicates(["PLAYER_ID", "rating_season"]))
            print(f"EPM expected {season}: {len(rows[-1])}", flush=True)
        except Exception as exc:
            print(f"EPM {season} failed: {exc}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _darko_column_map(frame: pd.DataFrame) -> dict[str, str] | None:
    columns = {column.lower(): column for column in frame.columns}
    needed = {"nba_id": None, "season": None, "o_dpm": None, "d_dpm": None}
    for key in list(needed):
        for name, original in columns.items():
            compact_key = key.replace("_", "")
            if compact_key in name.replace("_", "") or key in name:
                needed[key] = original
    if not all(needed.values()):
        return None
    return needed


def fetch_darko() -> pd.DataFrame:
    urls = [
        "https://docs.google.com/spreadsheets/d/1mYwuQfyDsO2dKjW0WD63nIs_n4tT1QSK-Yb-DpaI3ls/export?format=csv",
        "https://www.darko.app/data/dpm.csv",
        "https://raw.githubusercontent.com/kmedved/darko/master/dpm.csv",
    ]
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            response = requests.get(url, timeout=30, headers=HEADERS)
            if response.status_code != 200 or len(response.content) < 1000:
                print(f"DARKO miss {url}: {response.status_code} {len(response.content)}", flush=True)
                continue
            path = EXTERNAL_ROOT / "darko" / "attempt.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            frame = pd.read_csv(path)
            needed = _darko_column_map(frame)
            if needed is None:
                print(f"DARKO columns unmatched from {url}: {list(frame.columns)[:12]}", flush=True)
                continue
            output = pd.DataFrame(
                {
                    "PLAYER_ID": pd.to_numeric(frame[needed["nba_id"]], errors="coerce"),
                    "rating_season": pd.to_numeric(frame[needed["season"]], errors="coerce"),
                    "offense": pd.to_numeric(frame[needed["o_dpm"]], errors="coerce"),
                    "defense": pd.to_numeric(frame[needed["d_dpm"]], errors="coerce"),
                    "candidate": "DARKO DPM",
                }
            ).dropna()
            output["PLAYER_ID"] = output["PLAYER_ID"].astype(int)
            output["rating_season"] = output["rating_season"].astype(int)
            output["net"] = output["offense"] + output["defense"]
            print(f"DARKO from {url}: {len(output)}", flush=True)
            return output.drop_duplicates(["PLAYER_ID", "rating_season"])
        except Exception as exc:
            print(f"DARKO {url} failed: {exc}", flush=True)
    return pd.DataFrame()


def name_match_xrapm(xrapm: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    source = xrapm.copy()
    source["rating_season"] = source["season"].astype(int)
    source["normalized_name"] = source["player_name_xrapm"].map(normalize_player_name)
    matched = source.merge(names, on=["rating_season", "normalized_name"], how="left")
    output = matched.dropna(subset=["PLAYER_ID"]).copy()
    output["PLAYER_ID"] = output["PLAYER_ID"].astype(int)
    output["offense"] = output["xrapm_offense"]
    output["defense"] = output["xrapm_defense"]
    output["net"] = output["xrapm_net"]
    output["candidate"] = "xRAPM"
    return output[["PLAYER_ID", "rating_season", "candidate", "offense", "defense", "net"]]


def load_external_ratings() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("wave3 external sources", flush=True)
    epm = fetch_expected_epm(tuple(range(2015, 2026)))
    xrapm_raw = fetch_xrapm(tuple(range(2015, 2026)))
    darko = fetch_darko()
    names = pd.DataFrame()
    if not epm.empty:
        names = epm[["rating_season", "PLAYER_ID", "player_name"]].copy()
        names["normalized_name"] = names["player_name"].map(normalize_player_name)
        names = names.dropna().drop_duplicates(["rating_season", "normalized_name"])
    xrapm = (
        name_match_xrapm(xrapm_raw, names)
        if not xrapm_raw.empty and not names.empty
        else pd.DataFrame()
    )
    frames = [epm.drop(columns=["player_name"], errors="ignore"), xrapm, darko]
    present = [frame for frame in frames if not frame.empty]
    external = pd.concat(present, ignore_index=True) if present else pd.DataFrame()
    return epm, xrapm, darko, external


def _season_tables(internal_ratings: dict, source, rating_season: int, year_ext: pd.DataFrame) -> dict:
    tables = {}
    for label, key in (
        ("nine-year Box15 / PULSE-equivalent", "box15"),
        ("hgb_hustle_k1.0", "hgb_hustle_k1.0"),
    ):
        beta = internal_ratings[rating_season][key][0]
        tables[label] = stint_ratings(source, beta)[["PLAYER_ID", "offense", "defense"]]
    if year_ext.empty:
        return tables
    for name, group in year_ext.groupby("candidate"):
        tables[name] = group[["PLAYER_ID", "offense", "defense"]]
    return tables


def score_common_panel(internal_ratings: dict[int, dict], external: pd.DataFrame) -> dict:
    official_scores = pd.read_parquet(OFFICIAL)
    games = []
    folds = []
    if external.empty:
        return {"status": "unscored", "reason": "no_external_player_seasons"}
    names = sorted(set(external["candidate"]))
    internal_names = ["nine-year Box15 / PULSE-equivalent", "hgb_hustle_k1.0"]
    for rating_season in range(2015, 2026):
        if rating_season not in internal_ratings:
            continue
        source = build_stint_design(load_stints(rating_season, ("home_points_excl", "away_points_excl")))
        target = build_stint_design(load_stints(rating_season + 1, ("home_points_excl", "away_points_excl")))
        official = load_official_margins(rating_season + 1, official_scores)
        rapm = internal_ratings[rating_season]["rapm"]
        home = float(rapm[0][-1])
        intercept = float(rapm[1])
        year_ext = external.loc[external["rating_season"].eq(rating_season)]
        tables = _season_tables(internal_ratings, source, rating_season, year_ext)
        available = [name for name in [*internal_names, *names] if name in tables]
        finite_ids = [
            set(tables[name].loc[np.isfinite(tables[name][["offense", "defense"]]).all(axis=1), "PLAYER_ID"])
            for name in available
        ]
        positive = set(source.players[np.minimum(source.off_possessions, source.def_possessions) > 0])
        common = set.intersection(positive, *finite_ids) if finite_ids else set()
        if len(common) < 50:
            print(f"common panel {rating_season}: only {len(common)} players; skip", flush=True)
            continue
        for name in available:
            beta = common_coefficients(tables[name], source, common, home)
            scored = score_games(source, target, beta, intercept, official, name, rating_season)
            scored["scope"] = "official_final_overlap"
            games.append(scored)
            folds.append(
                {
                    "candidate": name,
                    "rating_season": rating_season,
                    "outcome_season": rating_season + 1,
                    "matched_players": len(common),
                    **game_metrics(scored.assign(actual_margin=scored["official_margin"])),
                }
            )
        print(f"common panel {rating_season}->{rating_season + 1}: {len(common)} players", flush=True)
    if not games:
        return {"status": "unscored", "reason": "no_common_player_support"}
    games_frame = pd.concat(games, ignore_index=True)
    folds_frame = pd.DataFrame(folds)
    summary = summarize_folds(folds_frame)
    return {
        "status": "scored",
        "summary": summary.to_dict("records"),
        "games": int(games_frame.drop_duplicates(["outcome_season", "game_id"]).shape[0]),
        "folds": int(folds_frame["outcome_season"].nunique()),
        "candidates": summary["candidate"].tolist(),
        "games_frame": games_frame,
        "folds_frame": folds_frame,
        "summary_frame": summary,
    }
