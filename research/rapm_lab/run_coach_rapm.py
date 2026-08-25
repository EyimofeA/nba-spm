"""Acquire a coach ledger and run a bounded joint player-coach RAPM test."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.coach_rapm import (
    build_coach_game_ledger,
    coach_ratings,
    fit_joint_coach_rapm,
    parse_bbref_coaches,
    predict_joint_coach_rapm,
)
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_unified_terminal_possessions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "rapm/data/possession_cache"
POSSESSIONS = REPO_ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = REPO_ROOT / "data/lake/silver/possession_lineup_segments.parquet"
TEAM_DIM = REPO_ROOT / "data/lake/silver/team_dim.parquet"
SCORES = REPO_ROOT / "data/lake/bronze/official_game_scores"
COACH_DATA = REPO_ROOT / "data/lake/bronze/external_coaches/basketball_reference"
OUTPUT_ROOT = REPO_ROOT / "research/rapm_lab/outputs/coach_rapm"
COACH_PENALTIES = (3000.0, 10000.0, 30000.0, 100000.0)


def _download_coaches(seasons: tuple[int, ...]) -> tuple[pd.DataFrame, dict[str, str]]:
    COACH_DATA.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 CourtSignal research contact"})
    frames = []
    hashes = {}
    for season in seasons:
        path = COACH_DATA / f"NBA_{season}_coaches.html"
        if not path.exists():
            url = f"https://www.basketball-reference.com/leagues/NBA_{season}_coaches.html"
            error = None
            for attempt in range(4):
                try:
                    response = session.get(url, timeout=30)
                    response.raise_for_status()
                    path.write_text(response.text)
                    error = None
                    break
                except requests.RequestException as caught:
                    error = caught
                    time.sleep(2**attempt)
            if error is not None:
                raise error
            time.sleep(0.25)
        html = path.read_text()
        frames.append(parse_bbref_coaches(html, season=season))
        hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    coaches = pd.concat(frames, ignore_index=True)
    coaches.to_parquet(COACH_DATA / "coach_seasons.parquet", index=False)
    return coaches, hashes


def _official_games(seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        path = SCORES / f"project_season={season}/regular.parquet"
        frame = pd.read_parquet(path)
        rows.append(frame)
    games = pd.concat(rows, ignore_index=True)
    games["game_id"] = games["game_id"].astype(str).str.zfill(10)
    games["game_date"] = pd.to_datetime(games["game_date"])
    if games.duplicated("game_id").any():
        raise ValueError("Official regular-season game IDs must be unique.")
    return games


def _evaluate_stage(
    frame: pd.DataFrame,
    design,
    offense_coach: np.ndarray,
    defense_coach: np.ndarray,
    *,
    train_end: int,
    test_season: int,
    penalties: tuple[float, ...],
) -> tuple[pd.DataFrame, dict[float, object]]:
    train = design.seasons <= train_end
    test = design.seasons == test_season
    config = RapmConfig(
        seasons=tuple(range(2017, train_end + 1)),
        lambda_off=3000,
        lambda_def=3000,
        lambda_home=300,
        data_scope="joint_player_coach_rapm",
    )
    baseline_beta, baseline_intercept = fit_coefficients(design, config, row_mask=train)
    baseline_prediction = baseline_intercept + np.asarray(design.X @ baseline_beta).ravel()
    baseline_metrics, _ = game_margin_metrics(
        frame.loc[test].reset_index(drop=True), baseline_prediction[test]
    )
    rows = [
        {
            "model": "player_rapm",
            "coach_penalty": np.nan,
            "train_end": train_end,
            "test_season": test_season,
            **baseline_metrics,
        }
    ]
    fits = {}
    for penalty in penalties:
        print(f"coach stage={test_season} lambda={penalty:g}", flush=True)
        fit = fit_joint_coach_rapm(
            design,
            offense_coach,
            defense_coach,
            config,
            coach_penalty=penalty,
            row_mask=train,
        )
        prediction = predict_joint_coach_rapm(
            fit, design, offense_coach, defense_coach
        )
        metrics, _ = game_margin_metrics(
            frame.loc[test].reset_index(drop=True), prediction[test]
        )
        rows.append(
            {
                "model": "player_plus_coach_rapm",
                "coach_penalty": penalty,
                "train_end": train_end,
                "test_season": test_season,
                **metrics,
            }
        )
        fits[penalty] = fit
    return pd.DataFrame(rows), fits


def run(output_root: Path = OUTPUT_ROOT) -> dict:
    seasons = tuple(range(2017, 2027))
    coach_seasons, coach_hashes = _download_coaches(seasons)
    games = _official_games(seasons)
    team_dim = pd.read_parquet(TEAM_DIM)
    ledger, ledger_audit = build_coach_game_ledger(coach_seasons, games, team_dim)
    ledger.to_parquet(COACH_DATA / "coach_games.parquet", index=False)
    ledger_audit.to_parquet(COACH_DATA / "coach_game_audit.parquet", index=False)
    frame = load_unified_terminal_possessions(
        CACHE,
        POSSESSIONS,
        SEGMENTS,
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    game_teams = games[
        ["game_id", "home_team_id", "away_team_id"]
    ].rename(columns={"game_id": "gameid"})
    frame = frame.merge(game_teams, on="gameid", how="left", validate="many_to_one")
    coach_lookup = ledger[["game_id", "team_id", "coach_id", "coach_name"]]
    home = coach_lookup.rename(
        columns={
            "game_id": "gameid",
            "team_id": "home_team_id",
            "coach_id": "home_coach_id",
            "coach_name": "home_coach_name",
        }
    )
    away = coach_lookup.rename(
        columns={
            "game_id": "gameid",
            "team_id": "away_team_id",
            "coach_id": "away_coach_id",
            "coach_name": "away_coach_name",
        }
    )
    frame = frame.merge(home, on=["gameid", "home_team_id"], how="left", validate="many_to_one")
    frame = frame.merge(away, on=["gameid", "away_team_id"], how="left", validate="many_to_one")
    missing_games = frame.loc[
        frame[["home_coach_id", "away_coach_id"]].isna().any(axis=1), "gameid"
    ].nunique()
    if missing_games:
        raise ValueError(f"Coach ledger is missing {missing_games} RAPM games.")
    offense_coach = np.where(
        frame["home_poss"].astype(bool), frame["home_coach_id"], frame["away_coach_id"]
    ).astype(str)
    defense_coach = np.where(
        frame["home_poss"].astype(bool), frame["away_coach_id"], frame["home_coach_id"]
    ).astype(str)
    design = build_design(frame, include_home=True)
    selection, _ = _evaluate_stage(
        frame,
        design,
        offense_coach,
        defense_coach,
        train_end=2024,
        test_season=2025,
        penalties=COACH_PENALTIES,
    )
    selected_penalty = float(
        selection.loc[selection["model"].eq("player_plus_coach_rapm")]
        .sort_values(["margin_rmse", "coach_penalty"], kind="stable")
        .iloc[0]["coach_penalty"]
    )
    diagnostic, fits = _evaluate_stage(
        frame,
        design,
        offense_coach,
        defense_coach,
        train_end=2025,
        test_season=2026,
        penalties=(selected_penalty,),
    )
    selected_fit = fits[selected_penalty]
    ratings = coach_ratings(selected_fit).merge(
        coach_seasons[["coach_id", "coach"]].drop_duplicates("coach_id"),
        on="coach_id",
        how="left",
        validate="one_to_one",
    )
    context = (
        coach_seasons.groupby("coach_id", as_index=False)
        .agg(seasons=("season", "nunique"), teams=("team_tricode", "nunique"), listed_games=("games", "sum"))
    )
    ratings = ratings.merge(context, on="coach_id", how="left", validate="one_to_one")
    metrics = pd.concat(
        [selection.assign(stage="selection"), diagnostic.assign(stage="diagnostic")],
        ignore_index=True,
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "coach_sources": coach_hashes,
                "possessions": sha256_file(POSSESSIONS),
                "segments": sha256_file(SEGMENTS),
                "penalties": COACH_PENALTIES,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"coach_rapm_v1_{identity}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output / "metrics.parquet", index=False)
    ratings.to_parquet(output / "coach_ratings.parquet", index=False)
    ledger_audit.to_parquet(output / "coach_game_audit.parquet", index=False)
    baseline_selection = selection.loc[selection["model"].eq("player_rapm")].iloc[0]
    model_selection = selection.loc[
        selection["coach_penalty"].eq(selected_penalty)
    ].iloc[0]
    baseline_diagnostic = diagnostic.loc[diagnostic["model"].eq("player_rapm")].iloc[0]
    model_diagnostic = diagnostic.loc[
        diagnostic["model"].eq("player_plus_coach_rapm")
    ].iloc[0]
    manifest = {
        "run_id": run_id,
        "status": "diagnostic_only",
        "model_family": "joint_player_coach_offense_defense_ridge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "coach-associated residual points per 100 after player lineup adjustment",
        "config": {
            "seasons": list(seasons),
            "selection": {"train": [2017, 2024], "test": 2025},
            "diagnostic": {"train": [2017, 2025], "test": 2026},
            "player_penalties": {"offense": 3000, "defense": 3000, "home": 300},
            "coach_penalty_candidates": list(COACH_PENALTIES),
            "selected_coach_penalty": selected_penalty,
        },
        "coverage": {
            "coach_season_rows": int(len(coach_seasons)),
            "coach_game_rows": int(len(ledger)),
            "coaches": int(coach_seasons["coach_id"].nunique()),
            "multi_season_coaches": int(context["seasons"].gt(1).sum()),
            "multi_team_coaches": int(context["teams"].gt(1).sum()),
            "rapm_games": int(frame["gameid"].nunique()),
            "rapm_possessions": int(len(frame)),
        },
        "comparison": {
            "selection_rmse_delta_vs_player_rapm": float(
                model_selection["margin_rmse"] - baseline_selection["margin_rmse"]
            ),
            "selection_correlation_delta_vs_player_rapm": float(
                model_selection["margin_correlation"] - baseline_selection["margin_correlation"]
            ),
            "diagnostic_rmse_delta_vs_player_rapm": float(
                model_diagnostic["margin_rmse"] - baseline_diagnostic["margin_rmse"]
            ),
            "diagnostic_correlation_delta_vs_player_rapm": float(
                model_diagnostic["margin_correlation"] - baseline_diagnostic["margin_correlation"]
            ),
        },
        "paths": {
            "metrics": str((output / "metrics.parquet").relative_to(REPO_ROOT)),
            "coach_ratings": str((output / "coach_ratings.parquet").relative_to(REPO_ROOT)),
            "coach_game_audit": str((output / "coach_game_audit.parquet").relative_to(REPO_ROOT)),
            "coach_game_ledger": str((COACH_DATA / "coach_games.parquet").relative_to(REPO_ROOT)),
        },
        "caveats": [
            "Coach identity is assigned from Basketball Reference season game counts and the official chronological schedule.",
            "Coach effects remain highly confounded with franchise, roster, assistants, and organizational context.",
            "Most coaches do not switch teams, so portable coach identification is weak even with ridge regularization.",
            "2025 selects the coach penalty; 2026 is reused diagnostic evidence. Season 2027 is not loaded.",
        ],
        "forbidden_interpretation": "Causal coach talent, portable scheme value, or a production ranking.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
