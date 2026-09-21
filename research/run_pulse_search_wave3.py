#!/usr/bin/env python3
"""Wave 3: nine-year PULSE labels, hustle+HGB recalibration, official-final panel.

Research-only. Does not promote. Does not spend 2027.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_impact.data.game_dim import canonical_game_id
from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.canonical_pulse import game_metrics, predict_next_season_games, stint_prior_center
from nba_impact.models.external_impact_benchmark import normalize_player_name, parse_xrapm_html
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import build_stint_design, stint_ratings
from research.build_canonical_lineup_stints import build_season, convert_lineups
from research.build_canonical_rapm_targets import fit_window
from research.pulse_search_protocol import (
    HUSTLE_BLOCK,
    PULSE_RMSE_GATE,
    PULSE_SLOPE_GATE,
    add_z_increment,
    fit_box15_prior,
    fit_precision_jobs,
    fold_internal_affine,
    hustle_rates,
    keep_calibrated_variant,
    scale_side_center,
)
from research.run_pulse_search_fetch_support import (
    EVENT_ROOT,
    classic_event_scores,
    download_canonical_inputs,
    download_file,
)
from research.run_pulse_search_h1_official_final import OUTPUT, paired_delta, score_games
from research.run_pulse_search_real_protocol import (
    CANONICAL,
    FEATURES,
    HUSTLE,
    OFFICIAL,
    load_official_margins,
    load_stints,
)


NINE_YEAR_ROOT = OUTPUT / "canonical_nine_year_rapm_v1"
WAVE3_ROOT = OUTPUT / "pulse_search_wave3_v1"
EXTERNAL_ROOT = OUTPUT / "external_sources_v1"
XRAPM_URL = "https://xrapm.com/table_pages/{name}"
EPM_RECORD = re.compile(
    r"\{season:(\d+),game_dt:\"[^\"]+\",player_id:(\d+),player_name:\"([^\"]+)\","
    r"[^}]*?off:(-?\d+(?:\.\d+)?),def:(-?\d+(?:\.\d+)?),tot:(-?\d+(?:\.\d+)?)"
)
HEADERS = {"User-Agent": "CourtSignal research"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lineup_native_scores(season: int) -> pd.DataFrame:
    rows = pd.read_csv(convert_lineups(season))
    home = rows.loc[rows["location_team"].eq("home")].copy()
    home["game_id"] = home["game_id"].map(canonical_game_id)
    game = home.groupby("game_id", as_index=False).agg(
        home_score=("pts_team", "sum"),
        away_score=("pts_opp", "sum"),
        home_tricode=("team", "first"),
        away_tricode=("opp", "first"),
        game_date=("game_date", "first"),
    )
    events = pd.read_parquet(
        EVENT_ROOT / f"season={season}" / "regular.parquet",
        columns=["GAME_ID", "PLAYER1_TEAM_ID", "PLAYER1_TEAM_ABBREVIATION", "_season_type"],
    )
    events = events.loc[events["_season_type"].eq("rg")].copy()
    events["game_id"] = events["GAME_ID"].map(canonical_game_id)
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
    game["home_team_id"] = [
        lookup.get(game_id, {}).get(str(tricode).strip())
        for game_id, tricode in zip(game["game_id"], game["home_tricode"])
    ]
    game["away_team_id"] = [
        lookup.get(game_id, {}).get(str(tricode).strip())
        for game_id, tricode in zip(game["game_id"], game["away_tricode"])
    ]
    game["project_season"] = season
    game["season_type"] = "regular"
    game["game_date"] = pd.to_datetime(game["game_date"], errors="coerce").dt.date.astype(str)
    game["score_source"] = "lineup_native_point_sums"
    return game.dropna(subset=["home_team_id", "away_team_id", "home_score", "away_score"])


def historical_official_scores(season: int) -> pd.DataFrame:
    native = lineup_native_scores(season)
    try:
        classic = classic_event_scores(season)
    except Exception as exc:
        print(f"classic scores {season} failed: {exc}", flush=True)
        classic = pd.DataFrame()
    if classic.empty:
        return native
    classic = classic.copy()
    classic["game_id"] = classic["game_id"].map(canonical_game_id)
    native["game_id"] = native["game_id"].map(canonical_game_id)
    columns = [
        "project_season", "season_type", "game_id", "game_date",
        "home_team_id", "away_team_id", "home_score", "away_score", "score_source",
    ]
    classic = classic.loc[:, [column for column in columns if column in classic.columns]]
    filled = native.merge(classic, on="game_id", how="left", suffixes=("", "_classic"))
    for column in ("home_score", "away_score", "home_team_id", "away_team_id", "game_date"):
        classic_column = f"{column}_classic"
        if classic_column in filled.columns:
            filled[column] = filled[classic_column].combine_first(filled[column])
    filled["score_source"] = np.where(
        filled["home_score_classic"].notna() if "home_score_classic" in filled.columns else False,
        "classic_pbp_last_SCORE",
        filled["score_source"],
    )
    filled["project_season"] = season
    filled["season_type"] = "regular"
    return filled[columns].dropna(subset=["home_team_id", "away_team_id", "home_score", "away_score"])


def merge_official_scores(new_rows: pd.DataFrame) -> dict:
    existing = pd.read_parquet(OFFICIAL) if OFFICIAL.exists() else pd.DataFrame()
    combined = pd.concat([existing, new_rows], ignore_index=True) if not existing.empty else new_rows
    combined["game_id"] = combined["game_id"].map(canonical_game_id)
    combined = combined.drop_duplicates("game_id", keep="first")
    OFFICIAL.parent.mkdir(parents=True, exist_ok=True)
    for season, group in combined.groupby("project_season"):
        destination = OFFICIAL.parent / f"project_season={int(season)}" / "regular.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        group.to_parquet(destination, index=False)
    combined.to_parquet(OFFICIAL, index=False)
    return {
        "games": int(len(combined)),
        "seasons": sorted(int(year) for year in combined["project_season"].unique()),
        "sha256": sha256_file(OFFICIAL),
    }


def prepare_historical_stints(start: int = 1997, end: int = 2013) -> dict:
    download = download_canonical_inputs(start, end)
    scores_rows = []
    quality = []
    official_table = pd.read_parquet(OFFICIAL) if OFFICIAL.exists() else pd.DataFrame()
    for season in range(start, end + 1):
        destination = CANONICAL / f"season={season}" / "regular.parquet"
        if destination.exists():
            quality.append({"season": season, "status": "existing"})
            continue
        scores_rows.append(historical_official_scores(season))
        official_table = pd.concat([official_table, scores_rows[-1]], ignore_index=True)
        official_table = official_table.drop_duplicates("game_id", keep="first")
        try:
            quality.append(build_season(season, official_table, CANONICAL))
            print(
                f"canonical {season}: stints={quality[-1]['stints']} "
                f"native={quality[-1]['native_score_reconciliation']:.3f}",
                flush=True,
            )
        except Exception as exc:
            quality.append({"season": season, "status": "failed", "error": str(exc)})
            print(f"canonical {season} failed: {exc}", flush=True)
    if scores_rows:
        merge_official_scores(pd.concat(scores_rows, ignore_index=True))
    return {"download": {"failed": sum(1 for row in download["files"] if row.get("status") == "failed")}, "stints": quality}


def fit_nine_year_targets(start_end: int = 2005, last_end: int = 2025) -> dict:
    checkpoints = NINE_YEAR_ROOT / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    missing_seasons = []
    rows = []
    for end in range(start_end, last_end + 1):
        path = checkpoints / f"target_9y_{end}.parquet"
        seasons = tuple(range(end - 8, end + 1))
        absent = [season for season in seasons if not (CANONICAL / f"season={season}" / "regular.parquet").exists()]
        if path.exists():
            rows.append(pd.read_parquet(path))
            continue
        if absent:
            missing_seasons.append({"window_end": end, "missing": absent})
            print(f"nine-year {end}: missing stints {absent}", flush=True)
            continue
        ratings = fit_window(seasons, 3000).rename(
            columns={"offense": "target_offense", "defense": "target_defense", "net": "target_net"}
        )
        ratings["Window_End"] = end
        ratings["horizon"] = 9
        ratings["target_variant"] = "normal"
        ratings.to_parquet(path, index=False)
        rows.append(ratings)
        print(f"Nine-year RAPM ending {end}: {len(ratings)} players", flush=True)
    if not rows:
        raise FileNotFoundError("No nine-year RAPM windows could be fit.")
    targets = pd.concat(rows, ignore_index=True)
    destination = NINE_YEAR_ROOT / "targets.parquet"
    targets.to_parquet(destination, index=False)
    run = {
        "run_id": NINE_YEAR_ROOT.name,
        "created_at": _now(),
        "target_horizon": 9,
        "target_variant": "normal",
        "windows": sorted(int(value) for value in targets["Window_End"].unique()),
        "target_rows": int(len(targets)),
        "missing_windows": missing_seasons,
        "skipped_manifest_passed": True,
        "skipped_annual_rapm": True,
        "sha256": sha256_file(destination),
    }
    write_json_atomic(run, NINE_YEAR_ROOT / "run.json")
    return run


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    summary = folds.groupby("candidate", as_index=False).agg(
        folds=("outcome_season", "nunique"),
        games=("games", "sum"),
        equal_season_mse=("mse", "mean"),
        mean_correlation=("correlation", "mean"),
        mean_calibration_slope=("calibration_slope", "mean"),
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    return summary.sort_values("equal_season_rmse")


def calibration_jobs(hustle_center: np.ndarray, n_players: int) -> list[dict]:
    jobs = []
    for scale in (0.5, 0.75, 1.0, 1.25):
        jobs.append({"name": f"hgb_hustle_scale{scale}", "center": hustle_center, "prior_scale": scale})
    for off_scale, def_scale in ((1.0, 0.7), (1.0, 1.3), (0.7, 1.0), (1.3, 1.0)):
        jobs.append(
            {
                "name": f"hgb_hustle_side_o{off_scale}_d{def_scale}",
                "center": scale_side_center(
                    hustle_center, n_players, offense_scale=off_scale, defense_scale=def_scale
                ),
                "prior_scale": 1.0,
            }
        )
    for off_mult, def_mult in ((1.0, 1.5), (1.5, 1.0), (0.75, 1.25)):
        jobs.append(
            {
                "name": f"hgb_hustle_prec_o{off_mult}_d{def_mult}",
                "center": hustle_center,
                "prior_scale": 1.0,
                "offense_penalty_mult": off_mult,
                "defense_penalty_mult": def_mult,
            }
        )
    return jobs


def run_twelve_folds(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    features = pd.read_parquet(FEATURES)
    hustle = pd.read_csv(HUSTLE)
    official_scores = pd.read_parquet(OFFICIAL)
    rows = []
    fold_metrics = []
    prior_quality = []
    season_fits: dict[int, dict] = {}
    for rating_season in range(2014, 2026):
        source = build_stint_design(load_stints(rating_season, ("home_points_excl", "away_points_excl")))
        target = build_stint_design(load_stints(rating_season + 1, ("home_points_excl", "away_points_excl")))
        official = load_official_margins(rating_season + 1, official_scores)
        ridge, quality = fit_box15_prior(features, targets, training_before=rating_season, learner="ridge")
        hgb, _ = fit_box15_prior(features, targets, training_before=rating_season, learner="hgb")
        quality["rating_season"] = rating_season
        prior_quality.append(quality)
        ridge_center, coverage = stint_prior_center(source, ridge, rating_season)
        hgb_center, _ = stint_prior_center(source, hgb, rating_season)
        rates = hustle_rates(hustle, rating_season, source.players, source.def_possessions)
        hustle_center = add_z_increment(hgb_center, source, rates, HUSTLE_BLOCK, 1.0)
        n = len(source.players)
        jobs = [
            {"name": "rapm", "center": np.zeros(source.X.shape[1]), "prior_scale": 1.0},
            {"name": "box15", "center": ridge_center, "prior_scale": 1.0},
            {"name": "hgb", "center": hgb_center, "prior_scale": 1.0},
            {"name": "hgb_hustle_k1.0", "center": hustle_center, "prior_scale": 1.0},
            *calibration_jobs(hustle_center, n),
        ]
        config = RapmConfig(
            (rating_season,),
            lambda_off=3000.0,
            lambda_def=4500.0,
            lambda_home=300.0,
            data_scope="pulse_search_wave3_nine_year",
        )
        fits = fit_precision_jobs(source, config, jobs)
        for name, (beta, intercept) in fits.items():
            scored = score_games(source, target, beta, intercept, official, name, rating_season)
            rows.append(scored)
            fold_metrics.append(
                {
                    "candidate": name,
                    "actual": "official",
                    "rating_season": rating_season,
                    "outcome_season": rating_season + 1,
                    "prior_players_with_prior": coverage["players_with_prior"],
                    **game_metrics(scored.assign(actual_margin=scored["official_margin"])),
                }
            )
        season_fits[rating_season] = fits
        print(f"wave3 fold {rating_season}->{rating_season + 1}", flush=True)
    games = pd.concat(rows, ignore_index=True)
    affine = fold_internal_affine(games, "hgb_hustle_k1.0")
    games = pd.concat([games, affine], ignore_index=True)
    for season, group in affine.groupby("outcome_season"):
        fold_metrics.append(
            {
                "candidate": "hgb_hustle_k1.0_affine",
                "actual": "official",
                "rating_season": int(season) - 1,
                "outcome_season": int(season),
                **game_metrics(group.assign(actual_margin=group["official_margin"])),
            }
        )
    folds = pd.DataFrame(fold_metrics)
    return games, folds, pd.DataFrame(prior_quality), season_fits


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
            text = path.read_text(errors="ignore")
            found = []
            for match in EPM_RECORD.finditer(text):
                found.append(
                    {
                        "rating_season": int(match.group(1)),
                        "PLAYER_ID": int(match.group(2)),
                        "player_name": match.group(3),
                        "offense": float(match.group(4)),
                        "defense": float(match.group(5)),
                        "net": float(match.group(6)),
                        "candidate": "EPM expected",
                    }
                )
            if not found:
                raise ValueError("no EPM records parsed")
            rows.append(pd.DataFrame(found).drop_duplicates(["PLAYER_ID", "rating_season"]))
            print(f"EPM expected {season}: {len(rows[-1])}", flush=True)
        except Exception as exc:
            print(f"EPM {season} failed: {exc}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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
            columns = {column.lower(): column for column in frame.columns}
            needed = {"nba_id": None, "season": None, "o_dpm": None, "d_dpm": None}
            for key in list(needed):
                for name, original in columns.items():
                    if key.replace("_", "") in name.replace("_", "") or key in name:
                        needed[key] = original
            if not all(needed.values()):
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


def score_common_panel(internal_ratings: dict[int, dict[str, pd.DataFrame]], external: pd.DataFrame) -> dict:
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
        tables = {}
        for label, key in (
            ("nine-year Box15 / PULSE-equivalent", "box15"),
            ("hgb_hustle_k1.0", "hgb_hustle_k1.0"),
        ):
            beta = internal_ratings[rating_season][key][0]
            tables[label] = stint_ratings(source, beta)[["PLAYER_ID", "offense", "defense"]]
        year_ext = external.loc[external["rating_season"].eq(rating_season)] if not external.empty else pd.DataFrame()
        for name, group in year_ext.groupby("candidate") if not year_ext.empty else []:
            tables[name] = group[["PLAYER_ID", "offense", "defense"]]
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
    summary = summarize(folds_frame)
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=("stints", "targets", "all"))
    args = parser.parse_args()
    WAVE3_ROOT.mkdir(parents=True, exist_ok=True)
    print("wave3 historical stints", flush=True)
    stint_report = prepare_historical_stints(1997, 2013)
    if args.stage == "stints":
        write_json_atomic({"created_at": _now(), "stints": stint_report}, WAVE3_ROOT / "stints.json")
        print(json.dumps(stint_report, default=str)[:4000], flush=True)
        return
    print("wave3 nine-year RAPM", flush=True)
    target_report = fit_nine_year_targets(2005, 2025)
    if args.stage == "targets":
        print(json.dumps(target_report, default=str)[:4000], flush=True)
        return
    targets = pd.read_parquet(NINE_YEAR_ROOT / "targets.parquet")
    games, folds, prior_quality, season_fits = run_twelve_folds(targets)
    summary = summarize(folds)
    box15 = summary.loc[summary["candidate"].eq("box15")].iloc[0]
    kept = []
    for row in summary.itertuples(index=False):
        if row.candidate in {"box15", "rapm"}:
            continue
        if keep_calibrated_variant(row.equal_season_rmse, row.mean_calibration_slope, float(box15.equal_season_rmse)):
            kept.append(row.candidate)
    intervals = {}
    for candidate in summary["candidate"]:
        if candidate == "box15":
            continue
        try:
            intervals[f"{candidate}_minus_box15"] = paired_delta(games, candidate, "box15", "squared_error_official")
        except Exception as exc:
            intervals[f"{candidate}_minus_box15"] = {"status": "unscored", "error": str(exc)}
    print(summary.to_string(index=False), flush=True)

    print("wave3 external sources", flush=True)
    epm = fetch_expected_epm(tuple(range(2015, 2026)))
    xrapm_raw = fetch_xrapm(tuple(range(2015, 2026)))
    darko = fetch_darko()
    names = pd.DataFrame()
    if not epm.empty:
        names = epm[["rating_season", "PLAYER_ID", "player_name"]].copy()
        names["normalized_name"] = names["player_name"].map(normalize_player_name)
        names = names.dropna().drop_duplicates(["rating_season", "normalized_name"])
    xrapm = name_match_xrapm(xrapm_raw, names) if not xrapm_raw.empty and not names.empty else pd.DataFrame()
    external = pd.concat([frame for frame in (epm.drop(columns=["player_name"], errors="ignore"), xrapm, darko) if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in (epm, xrapm, darko)
    ) else pd.DataFrame()

    common = score_common_panel(season_fits, external)

    games.to_parquet(WAVE3_ROOT / "game_predictions.parquet", index=False)
    folds.to_parquet(WAVE3_ROOT / "fold_metrics.parquet", index=False)
    summary.to_parquet(WAVE3_ROOT / "summary.parquet", index=False)
    prior_quality.to_parquet(WAVE3_ROOT / "prior_quality.parquet", index=False)
    if common.get("status") == "scored":
        common["games_frame"].to_parquet(WAVE3_ROOT / "common_game_predictions.parquet", index=False)
        common["folds_frame"].to_parquet(WAVE3_ROOT / "common_fold_metrics.parquet", index=False)
        common["summary_frame"].to_parquet(WAVE3_ROOT / "common_summary.parquet", index=False)
        common_public = {key: value for key, value in common.items() if not key.endswith("_frame")}
    else:
        common_public = common

    best = summary.iloc[0]
    dual = bool(
        keep_calibrated_variant(float(best.equal_season_rmse), float(best.mean_calibration_slope), PULSE_RMSE_GATE)
        and common_public.get("status") == "scored"
    )
    run = {
        "run_id": WAVE3_ROOT.name,
        "created_at": _now(),
        "status": "research_pilot_not_promotion",
        "win_condition_beaten": False,
        "dual_gate_note": "Promotion remains false even if a local RMSE/slope pair looks better.",
        "stints": stint_report,
        "nine_year_targets": target_report,
        "box15_nine_year_rmse": float(box15.equal_season_rmse),
        "box15_nine_year_slope": float(box15.mean_calibration_slope),
        "kept_calibrated_variants": kept,
        "best_candidate": best.candidate,
        "best_rmse": float(best.equal_season_rmse),
        "best_slope": float(best.mean_calibration_slope),
        "beat_13614_without_worse_slope": keep_calibrated_variant(
            float(best.equal_season_rmse), float(best.mean_calibration_slope), PULSE_RMSE_GATE
        ),
        "paired_intervals": intervals,
        "summary": summary.to_dict("records"),
        "common_panel": common_public,
        "external_coverage": {
            "epm_rows": int(len(epm)),
            "xrapm_rows": int(len(xrapm)),
            "darko_rows": int(len(darko)),
            "lebron_rows": 0,
        },
        "limitations": [
            "Nine-year RAPM is rebuilt on reconstructed canonical stints; 2014 stint hash does not match frozen pulse_canonical_v1_cd3c14750a.",
            "EPM is end-of-season expected EPM from dunksandthrees.com/epm, not the locked actual EPM_All_Seasons.csv.",
            "DARKO/LEBRON source CSVs are used only if a public file parsed.",
            "2027 unused. No promotion.",
        ],
        "pulse_rmse_gate": PULSE_RMSE_GATE,
        "pulse_slope_gate": PULSE_SLOPE_GATE,
        "would_have_beaten_dual_if_common_matched": dual,
    }
    (WAVE3_ROOT / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    print(json.dumps({
        "best": run["best_candidate"],
        "rmse": run["best_rmse"],
        "slope": run["best_slope"],
        "kept": kept,
        "common": common_public.get("status"),
        "dual": run["win_condition_beaten"],
    }), flush=True)


if __name__ == "__main__":
    main()
