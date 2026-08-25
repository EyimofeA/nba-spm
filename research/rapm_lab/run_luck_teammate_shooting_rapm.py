"""Fit FT/3P luck-adjusted RAPM and teammate-only eFG RAPM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.possession_outcome_rapm import assign_events_to_possessions, canonical_terminal_frame
from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients, load_current_player_names, ratings_table
from nba_impact.models.shooting_luck_rapm import leave_game_out_empirical_bayes_rate, replace_shooter_with_dummy
from research.rapm_lab.run_rubberband_progress_rapm import _paired_rmse_bootstrap


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/rapm_lab/outputs/luck_teammate_shooting_rapm"


def run() -> dict:
    possessions = pd.read_parquet(ROOT / "data/lake/silver/possessions.parquet")
    segments = pd.read_parquet(ROOT / "data/lake/silver/possession_lineup_segments.parquet")
    events = pd.read_parquet(ROOT / "data/lake/silver/event_states.parquet")
    seasons = (2024, 2025, 2026)
    base = canonical_terminal_frame(possessions, segments, seasons=seasons)
    relevant = events.loc[
        events["season_type"].eq("regular")
        & events["season_end"].isin(seasons)
        & events["actionType"].isin(["Made Shot", "Missed Shot", "Free Throw"])
    ].copy()
    mapped = assign_events_to_possessions(base, relevant).merge(
        base[["possession_id", "offense_team_id"]], on="possession_id", validate="many_to_one"
    )
    mapped = mapped.loc[
        pd.to_numeric(mapped["teamId"], errors="coerce").eq(mapped["offense_team_id"])
    ].copy()
    description = mapped["description"].astype("string")
    field_goal = mapped["actionType"].isin(["Made Shot", "Missed Shot"])
    shot_value = pd.to_numeric(mapped["shotValue"], errors="coerce").where(
        lambda value: value.isin([2, 3]),
        np.where(description.str.contains("3PT", case=False, na=False), 3.0, 2.0),
    )
    mapped["shot_value"] = shot_value
    mapped["made"] = np.where(
        mapped["actionType"].eq("Made Shot"),
        1.0,
        np.where(
            mapped["actionType"].eq("Free Throw"),
            ~description.str.contains("MISS", case=False, na=False),
            0.0,
        ),
    ).astype(float)
    mapped["player_id"] = pd.to_numeric(mapped["personId"], errors="coerce")
    attempts = mapped.loc[
        mapped["player_id"].notna()
        & (mapped["actionType"].eq("Free Throw") | (field_goal & mapped["shot_value"].eq(3)))
    ].copy()
    attempts["player_id"] = attempts["player_id"].astype(int)
    attempts["season"] = attempts["season_end"].astype(int)
    attempts["attempt_type"] = np.where(attempts["actionType"].eq("Free Throw"), "ft", "3p")
    expected_parts = []
    for attempt_type, group in attempts.groupby("attempt_type", sort=False):
        group = group.copy()
        group["expected_make"] = leave_game_out_empirical_bayes_rate(
            group, prior_attempts=50.0 if attempt_type == "ft" else 100.0
        )
        group["point_value"] = 1.0 if attempt_type == "ft" else 3.0
        expected_parts.append(group)
    attempts = pd.concat(expected_parts, ignore_index=True)
    attempts["actual_points"] = attempts["made"] * attempts["point_value"]
    attempts["expected_points"] = attempts["expected_make"] * attempts["point_value"]
    luck = attempts.groupby("possession_id", as_index=False).agg(
        actual_luck_points=("actual_points", "sum"),
        expected_luck_points=("expected_points", "sum"),
    )
    adjusted = base.merge(luck, on="possession_id", how="left", validate="one_to_one")
    adjusted[["actual_luck_points", "expected_luck_points"]] = adjusted[["actual_luck_points", "expected_luck_points"]].fillna(0.0)
    adjusted["luck_adjusted_pts"] = adjusted["pts"] - adjusted["actual_luck_points"] + adjusted["expected_luck_points"]
    config = RapmConfig(seasons=seasons, lambda_off=3000, lambda_def=3000, lambda_home=300, data_scope="ft_3p_luck_adjusted_rapm")
    normal_design = build_design(base, include_home=True)
    luck_frame = adjusted.assign(pts=adjusted["luck_adjusted_pts"])
    luck_design = build_design(luck_frame, include_home=True)
    train = normal_design.seasons < 2026
    test = normal_design.seasons == 2026
    normal_beta, normal_intercept = fit_coefficients(normal_design, config, row_mask=train)
    luck_beta, luck_intercept = fit_coefficients(luck_design, config, row_mask=train)
    normal_metrics, normal_games = game_margin_metrics(base.loc[test].reset_index(drop=True), normal_intercept + np.asarray(normal_design.X[test] @ normal_beta).ravel())
    luck_metrics, luck_games = game_margin_metrics(base.loc[test].reset_index(drop=True), luck_intercept + np.asarray(luck_design.X[test] @ luck_beta).ravel())
    full_normal, _ = fit_coefficients(normal_design, config)
    full_luck, _ = fit_coefficients(luck_design, config)
    names = load_current_player_names(ROOT / "rapm/data/all_names.csv", ROOT / "data/lake/silver/player_games.parquet")
    normal_ratings = ratings_table(normal_design, full_normal, names=names).rename(columns={"offense_per_100": "normal_offense", "defense_per_100": "normal_defense", "net_per_100": "normal_net"})
    luck_ratings = ratings_table(luck_design, full_luck, names=names).rename(columns={"offense_per_100": "luck_offense", "defense_per_100": "luck_defense", "net_per_100": "luck_net"})
    ratings = normal_ratings.merge(luck_ratings[["player_id", "luck_offense", "luck_defense", "luck_net"]], on="player_id", validate="one_to_one")
    ratings["luck_net_change"] = ratings["luck_net"] - ratings["normal_net"]

    shots = mapped.loc[mapped["actionType"].isin(["Made Shot", "Missed Shot"])].copy()
    shots["shooter_id"] = pd.to_numeric(shots["personId"], errors="coerce")
    shots = shots.dropna(subset=["shooter_id"])
    shots["shooter_id"] = shots["shooter_id"].astype(int)
    shots["efg_value"] = np.where(shots["actionType"].eq("Made Shot"), shots["shot_value"] / 2.0, 0.0)
    shots = shots.merge(base, on="possession_id", suffixes=("_event", ""), validate="many_to_one")
    home_lineup = shots[[f"h{i}" for i in range(1, 6)]].to_numpy(dtype=int)
    away_lineup = shots[[f"a{i}" for i in range(1, 6)]].to_numpy(dtype=int)
    offense_lineup = np.where(
        shots["home_poss"].to_numpy(dtype=bool)[:, None],
        home_lineup,
        away_lineup,
    )
    shooter_present = (
        offense_lineup == shots["shooter_id"].to_numpy(dtype=int)[:, None]
    ).any(axis=1)
    source_missing_shooter = int((~shooter_present).sum())
    shots = shots.loc[shooter_present].copy()
    teammate_frame, missing_shooter = replace_shooter_with_dummy(shots)
    if missing_shooter:
        raise ValueError("A source-confirmed shooter was not removed from the offensive five.")
    home_lineup = teammate_frame[[f"h{i}" for i in range(1, 6)]].to_numpy(dtype=int)
    away_lineup = teammate_frame[[f"a{i}" for i in range(1, 6)]].to_numpy(dtype=int)
    offense_lineup = np.where(
        teammate_frame["home_poss"].to_numpy(dtype=bool)[:, None],
        home_lineup,
        away_lineup,
    )
    shooter_still_present = (
        offense_lineup
        == teammate_frame["shooter_id"].to_numpy(dtype=int)[:, None]
    ).any(axis=1)
    teammate_frame = teammate_frame.loc[~shooter_still_present].copy()
    teammate_frame["pts"] = teammate_frame["efg_value"]
    teammate_design = build_design(teammate_frame, include_home=True)
    teammate_beta, _ = fit_coefficients(teammate_design, config)
    teammate = ratings_table(teammate_design, teammate_beta, names=names)
    teammate = teammate.loc[teammate["player_id"].ne(0)].rename(columns={"offense_per_100": "teammate_efg_offense", "defense_per_100": "shot_defense", "net_per_100": "teammate_efg_net"})

    identity = hashlib.sha256(json.dumps({"runner": sha256_file(Path(__file__)), "events": sha256_file(ROOT / "data/lake/silver/event_states.parquet")}, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT / f"luck_teammate_shooting_rapm_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(output / "luck_adjusted_ratings.parquet", index=False)
    teammate.to_parquet(output / "teammate_efg_ratings.parquet", index=False)
    manifest = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seasons": list(seasons),
        "luck_adjustment": "Replace realized offensive 3P and FT makes with leave-current-game-out player-season empirical-Bayes expectations.",
        "diagnostic": {"normal": normal_metrics, "luck_adjusted": luck_metrics, "paired_bootstrap": _paired_rmse_bootstrap(normal_games, luck_games, draws=2000, seed=20260827)},
        "quality": {"possessions": len(base), "mapped_events": len(mapped), "luck_attempts": len(attempts), "teammate_shots": len(teammate_frame), "source_shots_excluded_missing_shooter": source_missing_shooter, "post_filter_missing_shooter": missing_shooter},
        "paths": {"luck": "luck_adjusted_ratings.parquet", "teammate_efg": "teammate_efg_ratings.parquet"},
        "forbidden_interpretation": "Shot quality, causal teammate shooting effect, or an untouched confirmation result.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
