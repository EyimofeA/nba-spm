"""Fit and validate full-span joint player, age, and coach RAPM."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.age_adjusted_rapm import build_age_design, fit_age_adjusted_rapm, predict_age_adjusted_rapm
from nba_impact.models.aging_resolution import load_age_panel
from nba_impact.models.coach_rapm import (
    build_coach_game_ledger,
    coach_age_ratings,
    fit_joint_coach_age_rapm,
    predict_joint_coach_age_rapm,
)
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import RapmConfig, build_design, load_unified_terminal_possessions
from research.rapm_lab.run_age_adjusted_rapm import _paired_rmse_bootstrap
from research.rapm_lab.run_coach_rapm import _download_coaches


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/coach_age_full_1997_2026_v1.json"
CACHE = ROOT / "rapm/data/possession_cache"
POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"
TEAM_DIM = ROOT / "data/lake/silver/team_dim.parquet"
AGE_DIR = ROOT / "data/raw/playersheets/year_totals"
HISTORICAL_SCHEDULE = ROOT / "data/lake/bronze/official_game_schedule_1997_2026/schedule_1997_2023.parquet"
CURRENT_SCORES = ROOT / "data/lake/bronze/official_game_scores"
EXTERNAL = ROOT / "data/lake/bronze/external_coaches/xrapm"
OUTPUT = ROOT / "research/rapm_lab/outputs/full_coach_age_rapm"
XRAPM_URL = "https://xrapm.com/table_pages/RAPM_coach.html"


def _schedule() -> pd.DataFrame:
    frames = [pd.read_parquet(HISTORICAL_SCHEDULE)]
    for season in (2024, 2025, 2026):
        frame = pd.read_parquet(CURRENT_SCORES / f"project_season={season}/regular.parquet")
        frame["game_date"] = pd.to_datetime(frame["game_date"])
        frames.append(frame[["project_season", "game_id", "game_date", "home_team_id", "away_team_id"]])
    games = pd.concat(frames, ignore_index=True)
    games["game_id"] = games["game_id"].astype(str).str.zfill(10)
    if games.duplicated("game_id").any():
        raise ValueError("Historical schedule contains duplicate games.")
    return games


def _external_coach_ratings() -> tuple[pd.DataFrame, str]:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL / "RAPM_coach.html"
    response = requests.get(XRAPM_URL, timeout=45)
    response.raise_for_status()
    path.write_text(response.text)
    body = response.text.split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    cells = []
    for line in body.splitlines():
        value = line.strip()
        if not value.startswith("<td>"):
            continue
        value = value[4:].split("<td>", 1)[0].split("</td>", 1)[0]
        cells.append(value.strip())
    # The source omits the empty team cell from every malformed row.
    if len(cells) % 7:
        raise ValueError("xRAPM coach table does not contain complete seven-cell rows.")
    rows = [cells[index : index + 7] for index in range(0, len(cells), 7)]
    frame = pd.DataFrame(rows, columns=["coach", "offense_raw", "offense_se", "defense_raw", "defense_se", "net_raw", "net_se"])
    frame["team"] = ""
    for source, target in (("offense_raw", "xrapm_offense"), ("defense_raw", "xrapm_points_allowed_defense"), ("net_raw", "xrapm_net")):
        frame[target] = pd.to_numeric(frame[source].str.extract(r"^([-+]?\d+(?:\.\d+)?)")[0], errors="raise")
    frame["xrapm_defense"] = -frame["xrapm_points_allowed_defense"]
    return frame[["coach", "team", "xrapm_offense", "xrapm_defense", "xrapm_net"]], sha256_file(path)


def _name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _margin(frame: pd.DataFrame, prediction: np.ndarray) -> tuple[dict, pd.DataFrame]:
    return game_margin_metrics(frame.reset_index(drop=True), prediction)


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    seasons = tuple(contract["seasons"])
    started = time.perf_counter()
    coaches, coach_hashes = _download_coaches(seasons)
    games = _schedule()
    ledger, audit = build_coach_game_ledger(coaches, games, pd.read_parquet(TEAM_DIM))
    frame = load_unified_terminal_possessions(CACHE, POSSESSIONS, SEGMENTS, seasons, transition_season=2024, game_types=("regular",))
    game_teams = games[["game_id", "home_team_id", "away_team_id"]].rename(columns={"game_id": "gameid"})
    frame = frame.merge(game_teams, on="gameid", how="left", validate="many_to_one")
    lookup = ledger[["game_id", "team_id", "coach_id", "coach_name"]]
    for side in ("home", "away"):
        side_lookup = lookup.rename(columns={"game_id": "gameid", "team_id": f"{side}_team_id", "coach_id": f"{side}_coach_id", "coach_name": f"{side}_coach_name"})
        frame = frame.merge(side_lookup, on=["gameid", f"{side}_team_id"], how="left", validate="many_to_one")
    if frame[["home_coach_id", "away_coach_id"]].isna().any(axis=None):
        raise ValueError("Coach assignment is incomplete for RAPM games.")
    offense_coach = np.where(frame["home_poss"].astype(bool), frame["home_coach_id"], frame["away_coach_id"]).astype(str)
    defense_coach = np.where(frame["home_poss"].astype(bool), frame["away_coach_id"], frame["home_coach_id"]).astype(str)
    design = build_design(frame, include_home=True)
    age_design = build_age_design(frame, load_age_panel(AGE_DIR, seasons), minimum_age=contract["minimum_age"], maximum_age=contract["maximum_age"], reference_age=contract["reference_age"])
    penalties = contract["player_penalties"]
    config = RapmConfig(seasons=seasons, lambda_off=penalties["lambda_off"], lambda_def=penalties["lambda_def"], lambda_home=penalties["lambda_home"], data_scope="full_joint_coach_age_rapm")

    selection_season = contract["selection_test_season"]
    selection_train = design.seasons < selection_season
    selection_test = design.seasons == selection_season
    baseline = fit_age_adjusted_rapm(design, age_design, config, age_penalty=contract["age_penalty"], row_mask=selection_train)
    baseline_pred = predict_age_adjusted_rapm(baseline, design, age_design, row_mask=selection_test)
    selection_rows = [{"model": "player_plus_age", "coach_penalty": np.nan, **_margin(frame.loc[selection_test], baseline_pred)[0]}]
    for coach_penalty in contract["coach_penalties"]:
        print(f"selection coach penalty={coach_penalty:g}", flush=True)
        fit = fit_joint_coach_age_rapm(design, age_design, offense_coach, defense_coach, config, age_penalty=contract["age_penalty"], coach_penalty=coach_penalty, row_mask=selection_train)
        prediction = predict_joint_coach_age_rapm(fit, design, age_design, offense_coach, defense_coach, row_mask=selection_test)
        selection_rows.append({"model": "player_plus_age_plus_coach", "coach_penalty": float(coach_penalty), **_margin(frame.loc[selection_test], prediction)[0]})
    selection = pd.DataFrame(selection_rows)
    selected_penalty = float(selection.loc[selection["model"].eq("player_plus_age_plus_coach")].sort_values(["margin_rmse", "coach_penalty"]).iloc[0]["coach_penalty"])

    diagnostic_season = contract["reused_diagnostic_season"]
    train = design.seasons < diagnostic_season
    test = design.seasons == diagnostic_season
    print("diagnostic baseline and selected coach model", flush=True)
    baseline = fit_age_adjusted_rapm(design, age_design, config, age_penalty=contract["age_penalty"], row_mask=train)
    baseline_prediction = predict_age_adjusted_rapm(baseline, design, age_design, row_mask=test)
    fit = fit_joint_coach_age_rapm(design, age_design, offense_coach, defense_coach, config, age_penalty=contract["age_penalty"], coach_penalty=selected_penalty, row_mask=train)
    coach_prediction = predict_joint_coach_age_rapm(fit, design, age_design, offense_coach, defense_coach, row_mask=test)
    baseline_metrics, baseline_games = _margin(frame.loc[test], baseline_prediction)
    coach_metrics, coach_games = _margin(frame.loc[test], coach_prediction)
    bootstrap = _paired_rmse_bootstrap(baseline_games, coach_games, draws=contract["paired_game_bootstrap_draws"], seed=contract["paired_game_bootstrap_seed"])

    print("final full-span coach-age fit", flush=True)
    full_fit = fit_joint_coach_age_rapm(design, age_design, offense_coach, defense_coach, config, age_penalty=contract["age_penalty"], coach_penalty=selected_penalty)
    ratings = coach_age_ratings(full_fit).merge(coaches[["coach_id", "coach"]].drop_duplicates("coach_id"), on="coach_id", how="left", validate="one_to_one")
    context = coaches.groupby("coach_id", as_index=False).agg(seasons=("season", "nunique"), teams=("team_tricode", "nunique"), listed_games=("games", "sum"))
    ratings = ratings.merge(context, on="coach_id", how="left", validate="one_to_one")
    external, external_hash = _external_coach_ratings()
    ratings["name_key"] = ratings["coach"].map(_name_key)
    external["name_key"] = external["coach"].map(_name_key)
    comparison = ratings.merge(external.drop(columns="coach"), on="name_key", how="inner", validate="one_to_one", suffixes=("", "_external"))
    comparison_metrics = {f"{side}_correlation": float(comparison[[side, f"xrapm_{side}"]].corr().iloc[0, 1]) for side in ("offense", "defense", "net")}

    identity = hashlib.sha256(json.dumps({"contract": sha256_file(CONTRACT), "runner": sha256_file(Path(__file__)), "xrapm": external_hash}, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT / f"coach_age_full_1997_2026_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    selection.to_parquet(output / "selection.parquet", index=False)
    pd.DataFrame([{"model": "player_plus_age", **baseline_metrics}, {"model": "player_plus_age_plus_coach", **coach_metrics}]).to_parquet(output / "diagnostic.parquet", index=False)
    ratings.to_parquet(output / "coach_ratings.parquet", index=False)
    comparison.to_parquet(output / "xrapm_comparison.parquet", index=False)
    audit.to_parquet(output / "coach_schedule_audit.parquet", index=False)
    manifest = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "seasons": list(seasons),
        "possessions": int(len(frame)),
        "games": int(frame["gameid"].nunique()),
        "coaches": int(len(ratings)),
        "age_slot_coverage": float(age_design.known_slots / age_design.total_slots),
        "selected_coach_penalty": selected_penalty,
        "selection": selection.to_dict("records"),
        "diagnostic": {"baseline": baseline_metrics, "coach": coach_metrics, "paired_bootstrap": bootstrap},
        "external_comparison": {"source": XRAPM_URL, "matched_coaches": int(len(comparison)), **comparison_metrics},
        "coach_source_hashes": coach_hashes,
        "paths": {"ratings": "coach_ratings.parquet", "comparison": "xrapm_comparison.parquet", "selection": "selection.parquet", "diagnostic": "diagnostic.parquet", "audit": "coach_schedule_audit.parquet"},
        "forbidden_interpretation": "Coach coefficients are association after player and lineup-age adjustment, not causal coaching value.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
