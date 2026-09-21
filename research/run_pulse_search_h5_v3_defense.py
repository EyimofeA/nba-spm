#!/usr/bin/env python3
"""PULSE search H5: V3 rim-protection and box rebound conversion as prior increments.

Canonical mechanism features are absent in this checkout. This pilot builds
same-season 2025 defender-attributed rim suppression and box DREB conversion
from the H1 V3 attach tables, then adds them to the leaky public PULSE prior.

It cannot claim the 12-fold win condition. A loss here exhausts H5 on available
data. 2027 unused.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.possessions import AWAY_LINEUP_COLUMNS, HOME_LINEUP_COLUMNS
from nba_impact.models.canonical_pulse import game_metrics, stint_prior_center
from nba_impact.models.rapm import RapmConfig
from nba_impact.models.stint_rapm import build_stint_design, fit_stint_center_path, stint_ratings
from research.run_pulse_search_h1_official_final import (
    BOX_LOGS,
    OUTPUT,
    SILVER,
    _normalize_box,
    fit_stint_arm,
    official_margins,
    paired_delta,
    public_priors,
    score_games,
)


GABRIEL = ROOT / "data/lake/bronze/gabriel_site_data/revision=782ec8b"
RIM_FEET = 4.0


def field_goal_shots(assigned: pd.DataFrame) -> pd.DataFrame:
    shots = assigned.loc[pd.to_numeric(assigned["isFieldGoal"], errors="coerce").fillna(0).eq(1)].copy()
    shots["shot_distance"] = pd.to_numeric(shots["shotDistance"], errors="coerce").fillna(0.0)
    shots["made"] = shots["shotResult"].astype(str).str.casefold().eq("made")
    return shots


def attribute_rim_defense(shots: pd.DataFrame, home_team_id: pd.Series) -> pd.DataFrame:
    work = shots.copy()
    work["home_team_id"] = home_team_id.reindex(work.index)
    work["offense_is_home"] = pd.to_numeric(work["possession"], errors="coerce").eq(
        pd.to_numeric(work["home_team_id"], errors="coerce")
    )
    rim = work.loc[work["shot_distance"].le(RIM_FEET)].copy()
    if rim.empty:
        return pd.DataFrame(columns=["PLAYER_ID", "rim_dfga", "rim_fgm"])
    rows = []
    for offense_home, columns in ((True, AWAY_LINEUP_COLUMNS), (False, HOME_LINEUP_COLUMNS)):
        block = rim.loc[rim["offense_is_home"].eq(offense_home)]
        if block.empty:
            continue
        stacked = block.melt(
            id_vars=["made"],
            value_vars=list(columns),
            value_name="PLAYER_ID",
        )
        rows.append(stacked[["PLAYER_ID", "made"]])
    players = pd.concat(rows, ignore_index=True)
    players["PLAYER_ID"] = pd.to_numeric(players["PLAYER_ID"], errors="raise").astype(int)
    players["rim_dfga"] = 0.2
    players["rim_fgm"] = np.where(players["made"], 0.2, 0.0)
    return players.groupby("PLAYER_ID", as_index=False).agg(
        rim_dfga=("rim_dfga", "sum"),
        rim_fgm=("rim_fgm", "sum"),
    )


def box_rebound_conversion(box: pd.DataFrame, season: int) -> pd.DataFrame:
    frame = box.loc[box["game_id"].str.startswith(f"002{str(season - 1)[-2:]}")].copy()
    if "reboundsDefensive" not in frame.columns:
        return pd.DataFrame(columns=["PLAYER_ID", "dreb", "dreb_chances"])
    team = frame.groupby(["game_id", "team_id"], as_index=False).agg(
        team_dreb=("reboundsDefensive", "sum"),
        team_oreb=("reboundsOffensive", "sum"),
    )
    opp = team.rename(columns={
        "team_id": "opp_team_id",
        "team_oreb": "opp_oreb",
        "team_dreb": "opp_dreb",
    })
    games = team.merge(opp, on="game_id")
    games = games.loc[games["team_id"].ne(games["opp_team_id"])]
    chances = games.groupby("team_id", as_index=False).agg(
        team_dreb=("team_dreb", "sum"),
        opp_oreb=("opp_oreb", "sum"),
    )
    chances["team_chances"] = chances["team_dreb"] + chances["opp_oreb"]
    players = frame.groupby("player_id", as_index=False).agg(
        dreb=("reboundsDefensive", "sum"),
        minutes=("minutes_seconds", "sum"),
        team_id=("team_id", "first"),
    )
    team_minutes = frame.groupby("team_id", as_index=False)["minutes_seconds"].sum().rename(
        columns={"minutes_seconds": "team_minutes"}
    )
    players = players.merge(team_minutes, on="team_id", how="left").merge(
        chances[["team_id", "team_chances"]], on="team_id", how="left"
    )
    share = players["minutes"] / players["team_minutes"].where(players["team_minutes"].gt(0))
    players["dreb_chances"] = share * players["team_chances"]
    players["PLAYER_ID"] = players["player_id"].astype(int)
    return players[["PLAYER_ID", "dreb", "dreb_chances"]]


def defense_features(assigned: pd.DataFrame, scores: pd.DataFrame, box: pd.DataFrame, season: int, possessions: np.ndarray, players: np.ndarray) -> pd.DataFrame:
    shots = field_goal_shots(assigned)
    if "home_team_id" not in shots.columns:
        shots = shots.merge(scores[["game_id", "home_team_id"]], on="game_id", how="left")
    rim = attribute_rim_defense(shots, shots["home_team_id"])
    expected = float(rim["rim_fgm"].sum() / rim["rim_dfga"].sum()) if rim["rim_dfga"].sum() else 0.0
    reb = box_rebound_conversion(box, season)
    panel = pd.DataFrame({"PLAYER_ID": players.astype(int), "Poss_Def": possessions})
    panel = panel.merge(rim, on="PLAYER_ID", how="left").merge(reb, on="PLAYER_ID", how="left")
    panel[["rim_dfga", "rim_fgm", "dreb", "dreb_chances", "Poss_Def"]] = panel[
        ["rim_dfga", "rim_fgm", "dreb", "dreb_chances", "Poss_Def"]
    ].fillna(0.0)
    poss = panel["Poss_Def"].clip(lower=1.0)
    panel["rim_dfga_p100"] = 100.0 * panel["rim_dfga"] / poss
    observed = panel["rim_fgm"] / panel["rim_dfga"].where(panel["rim_dfga"].gt(0))
    panel["rim_points_saved_p100"] = 200.0 * (expected - observed.fillna(expected)) * panel["rim_dfga"] / poss
    reliability = panel["rim_dfga"] / (panel["rim_dfga"] + 40.0)
    panel["rim_protection_workload_value"] = (
        reliability * panel["rim_points_saved_p100"] * np.sqrt(panel["rim_dfga_p100"].clip(lower=0.0))
    )
    conversion = panel["dreb"] / panel["dreb_chances"].where(panel["dreb_chances"].gt(0))
    center = float(np.average(conversion.dropna(), weights=panel.loc[conversion.notna(), "dreb_chances"])) if conversion.notna().any() else 0.0
    reb_rel = panel["dreb_chances"] / (panel["dreb_chances"] + 100.0)
    panel["dreb_conversion_above_expected_eb"] = reb_rel * (conversion.fillna(center) - center)
    return panel


def add_defense_features(center: np.ndarray, design, features: pd.DataFrame, columns: tuple[str, ...], scale: float) -> np.ndarray:
    n = len(design.players)
    lookup = features.set_index("PLAYER_ID")
    extra = np.zeros(n)
    for column in columns:
        values = lookup[column].reindex(design.players).fillna(0.0).to_numpy(float)
        extra += values
    extra = extra / max(len(columns), 1)
    std = float(np.sqrt(np.average((extra - np.average(extra, weights=design.def_possessions)) ** 2, weights=design.def_possessions)))
    if std > 1e-12:
        extra = extra / std
    extra -= np.average(extra, weights=design.def_possessions)
    out = center.copy()
    out[n : 2 * n] += -scale * extra / 100.0
    return out


def main() -> None:
    rating_season = 2025
    scores = pd.read_parquet(SILVER / "v3_terminal_scores.parquet")
    source_stints = pd.read_parquet(SILVER / f"project_season={rating_season}" / "canonical_like_stints.parquet")
    target_stints = pd.read_parquet(SILVER / f"project_season={rating_season + 1}" / "canonical_like_stints.parquet")
    assigned = pd.read_parquet(SILVER / f"project_season={rating_season}" / "assigned_actions.parquet")
    box = _normalize_box(pd.read_parquet(BOX_LOGS))
    priors = public_priors(rating_season, "pulse")
    official = official_margins(scores, rating_season + 1)
    columns = ("home_points_excl", "away_points_excl")
    source_design, baseline, coverage, center = fit_stint_arm(
        source_stints, columns, priors, rating_season, scale=1.0
    )
    target_frame = target_stints.copy()
    target_frame["home_points"] = target_frame[columns[0]]
    target_frame["away_points"] = target_frame[columns[1]]
    target_design = build_stint_design(target_frame)
    rapm_ratings = stint_ratings(source_design, baseline["rapm"][0])
    features = defense_features(
        assigned, scores, box, rating_season, source_design.def_possessions, source_design.players
    )
    features = features.merge(
        rapm_ratings[["PLAYER_ID", "defense"]].rename(columns={"defense": "rapm_defense"}),
        on="PLAYER_ID",
        how="left",
    )
    rim = pd.read_csv(GABRIEL / "rimdfg.csv")
    rim = rim.loc[pd.to_numeric(rim["year"], errors="coerce").eq(rating_season)].copy()
    rim["PLAYER_ID"] = pd.to_numeric(rim["PLAYER_ID"], errors="coerce")
    rim = rim.dropna(subset=["PLAYER_ID"])
    rim["PLAYER_ID"] = rim["PLAYER_ID"].astype(int)
    diff = rim["DIFF%"].fillna(rim["Diff%"]) if "DIFF%" in rim else rim["Diff%"]
    rim["diff_pp"] = pd.to_numeric(diff, errors="coerce")
    rim["diff_pp"] = rim["diff_pp"].where(rim["diff_pp"].abs().gt(1.0), 100.0 * rim["diff_pp"])
    poss_lookup = pd.Series(source_design.def_possessions, index=source_design.players)
    rim["Poss_Def"] = rim["PLAYER_ID"].map(poss_lookup)
    rim = rim.loc[rim["Poss_Def"].ge(200)].copy()
    rim["rim_dfga_p100"] = 100.0 * pd.to_numeric(rim["DFGA"], errors="coerce") / rim["Poss_Def"]
    rim["gabriel_rim_points_saved_p100"] = -2.0 * rim["rim_dfga_p100"] * rim["diff_pp"] / 100.0
    rim["gabriel_rim_protection_workload_value"] = rim["gabriel_rim_points_saved_p100"] * np.sqrt(
        rim["rim_dfga_p100"].clip(lower=0.0)
    )
    hustle = pd.read_csv(GABRIEL / "hustle.csv")
    hustle = hustle.loc[pd.to_numeric(hustle["year"], errors="coerce").eq(rating_season)].copy()
    hustle["PLAYER_ID"] = pd.to_numeric(hustle["PLAYER_ID"], errors="coerce")
    hustle = hustle.dropna(subset=["PLAYER_ID"])
    hustle["PLAYER_ID"] = hustle["PLAYER_ID"].astype(int)
    hustle["Poss_Def"] = hustle["PLAYER_ID"].map(poss_lookup)
    hustle = hustle.loc[hustle["Poss_Def"].ge(200)].copy()
    hustle["gabriel_contested_2pt_p100"] = (
        100.0 * pd.to_numeric(hustle["CONTESTED_SHOTS_2PT"], errors="coerce") / hustle["Poss_Def"]
    )
    features = features.merge(
        rim[["PLAYER_ID", "gabriel_rim_points_saved_p100", "gabriel_rim_protection_workload_value"]],
        on="PLAYER_ID",
        how="left",
    ).merge(
        hustle[["PLAYER_ID", "gabriel_contested_2pt_p100"]],
        on="PLAYER_ID",
        how="left",
    )
    for column in (
        "gabriel_rim_points_saved_p100",
        "gabriel_rim_protection_workload_value",
        "gabriel_contested_2pt_p100",
    ):
        features[column] = features[column].fillna(0.0)

    rows = []
    fold_metrics = []

    def record(name: str, beta, intercept):
        scored = score_games(source_design, target_design, beta, intercept, official, name, rating_season)
        rows.append(scored)
        metrics = game_metrics(scored.assign(actual_margin=scored["official_margin"]))
        fold_metrics.append({"candidate": name, "actual": "official", **metrics})

    record("pulse_excl", *baseline["pulse"])
    config = RapmConfig(
        (rating_season,),
        lambda_off=3000.0,
        lambda_def=4500.0,
        lambda_home=300.0,
        data_scope="pulse_search_h5_v3_candidate",
    )
    for name, cols, scale in (
        ("h5_rim_k1", ("rim_protection_workload_value",), 1.0),
        ("h5_rim_k0.5", ("rim_protection_workload_value",), 0.5),
        ("h5_reb_k1", ("dreb_conversion_above_expected_eb",), 1.0),
        ("h5_rim_reb_k1", ("rim_protection_workload_value", "dreb_conversion_above_expected_eb"), 1.0),
        ("h5_rim_saved_k1", ("rim_points_saved_p100",), 1.0),
        ("h5_gabriel_rim_k1", ("gabriel_rim_protection_workload_value",), 1.0),
        ("h5_gabriel_rim_k0.5", ("gabriel_rim_protection_workload_value",), 0.5),
        ("h5_gabriel_saved_k1", ("gabriel_rim_points_saved_p100",), 1.0),
        ("h5_gabriel_contest_k1", ("gabriel_contested_2pt_p100",), 1.0),
        ("h5_gabriel_rim_contest_k1", ("gabriel_rim_protection_workload_value", "gabriel_contested_2pt_p100"), 1.0),
    ):
        shifted = add_defense_features(center, source_design, features, cols, scale)
        beta, intercept = fit_stint_center_path(source_design, config, shifted, center_scales=(1.0,))[1.0]
        record(name, beta, intercept)

    valid = features["Poss_Def"].gt(200) & features["rapm_defense"].notna()
    if int(valid.sum()) > 20:
        x = np.column_stack([
            np.ones(int(valid.sum())),
            features.loc[valid, "rim_protection_workload_value"].to_numpy(float),
            features.loc[valid, "dreb_conversion_above_expected_eb"].to_numpy(float),
        ])
        y = features.loc[valid, "rapm_defense"].to_numpy(float)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        predicted = (
            coef[0]
            + coef[1] * features["rim_protection_workload_value"]
            + coef[2] * features["dreb_conversion_above_expected_eb"]
        )
        lookup = pd.Series(predicted.to_numpy(float), index=features["PLAYER_ID"].to_numpy())
        hybrid = priors.copy()
        hybrid["prior_defense_per_100"] = hybrid["PLAYER_ID"].map(lookup).fillna(
            hybrid["prior_defense_per_100"]
        )
        feat_center, _ = stint_prior_center(source_design, hybrid, rating_season)
        beta, intercept = fit_stint_center_path(source_design, config, feat_center, center_scales=(1.0,))[1.0]
        record("h5_lstsq_def_replace", beta, intercept)

    games = pd.concat(rows, ignore_index=True)
    folds = pd.DataFrame(fold_metrics)
    folds["rmse"] = np.sqrt(folds["mse"])
    baseline_rmse = float(folds.loc[folds["candidate"].eq("pulse_excl"), "rmse"].iloc[0])
    folds["rmse_minus_pulse_excl"] = folds["rmse"] - baseline_rmse
    intervals = {}
    for candidate in folds["candidate"]:
        if candidate == "pulse_excl":
            continue
        intervals[f"{candidate}_minus_pulse_excl"] = paired_delta(
            games, candidate, "pulse_excl", "squared_error_official"
        )
    run_id = "pulse_search_h5_v3_defense_v1"
    destination = OUTPUT / run_id
    destination.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination / "defense_features.parquet", index=False)
    games.to_parquet(destination / "game_predictions.parquet", index=False)
    folds.to_parquet(destination / "fold_metrics.parquet", index=False)
    run = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_pilot_not_promotion",
        "hypothesis": "Add V3 equal-share rim protection and box DREB conversion to the PULSE prior.",
        "rim_feet": RIM_FEET,
        "baseline_pulse_excl_rmse": baseline_rmse,
        "feature_summary": {
            "players": int(len(features)),
            "mean_rim_dfga": float(features["rim_dfga"].mean()),
            "mean_rim_points_saved_p100": float(features["rim_points_saved_p100"].mean()),
            "mean_dreb_conversion_residual": float(features["dreb_conversion_above_expected_eb"].mean()),
        },
        "folds": folds.to_dict("records"),
        "paired_intervals": intervals,
        "source_hashes": {"runner": sha256_file(Path(__file__))},
        "limitations": [
            "Equal-share lineup attribution is not Synergy or tracking matchup data.",
            "One leaky 2025-2026 fold on V3 candidates.",
            "Feature-to-RAPM least squares is same-season and diagnostic only.",
            "2027 unused.",
        ],
    }
    write_json_atomic(run, destination / "run.json")
    print(folds.sort_values("rmse").to_string(index=False))
    print(json.dumps(run["feature_summary"], indent=2))
    print(destination, flush=True)


if __name__ == "__main__":
    main()
