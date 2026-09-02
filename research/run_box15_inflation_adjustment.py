#!/usr/bin/env python3
"""Test current-era, defense-faced adjustment for Box15 scoring."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

try:
    from run_target_window_spm_aio import (
        HISTORICAL_BOX,
        _evaluate_priors,
        _fit_box_priors,
        _game_metrics,
        _panel,
    )
except ModuleNotFoundError:  # Imported as research.run_* by tests.
    from research.run_target_window_spm_aio import (
        HISTORICAL_BOX,
        _evaluate_priors,
        _fit_box_priors,
        _game_metrics,
        _panel,
    )


ROOT = Path(__file__).resolve().parents[1]
STINT_ROOT = ROOT / "data/lake/silver/canonical_lineup_stints"
BASELINE_RUN = ROOT / (
    "artifacts/research/target_window_spm_aio/"
    "target_window_spm_aio_v1_be05a18f9b"
)
OUTPUT_ROOT = ROOT / "artifacts/research/box15_inflation_adjustment"
RATING_SEASONS = tuple(range(2014, 2026))
REFERENCE_SEASON = 2026
BOOTSTRAP_DRAWS = 5_000
BOOTSTRAP_SEED = 20260902
BASELINE = "box15_9y_normal_aio"
CHALLENGER = "box15_era_adjusted_9y_normal_aio"


def season_opponent_context(
    stints: pd.DataFrame, *, season: int
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Return each player's possession-weighted defense faced in one season."""
    rows = stints.reset_index(drop=True).reset_index(names="stint_row")
    team_rows = []
    exposure_rows = []
    for offense, defense in (("home", "away"), ("away", "home")):
        offense_team = f"{offense}_team_id"
        defense_team = f"{defense}_team_id"
        possessions = f"{offense}_possessions"
        points = rows[f"{offense}_points"] + rows[f"{offense}_technical_points_excluded"]
        team_rows.append(
            pd.DataFrame(
                {
                    "offense_team": rows[offense_team],
                    "defense_team": rows[defense_team],
                    "possessions": rows[possessions],
                    "points": points,
                }
            )
        )
        slots = [f"{offense}_player_{slot}" for slot in range(1, 6)]
        exposure = rows[["stint_row", defense_team, possessions, *slots]].melt(
            id_vars=["stint_row", defense_team, possessions],
            value_vars=slots,
            value_name="PLAYER_ID",
        )
        exposure = exposure.rename(
            columns={defense_team: "defense_team", possessions: "offensive_possessions"}
        ).drop(columns="variable")
        exposure = exposure.loc[exposure["offensive_possessions"].gt(0)].dropna(
            subset=["PLAYER_ID"]
        )
        exposure_rows.append(exposure.drop_duplicates(["stint_row", "PLAYER_ID"]))

    plays = pd.concat(team_rows, ignore_index=True)
    defense = plays.groupby("defense_team", as_index=False).agg(
        points_allowed=("points", "sum"),
        defensive_possessions=("possessions", "sum"),
    )
    if defense["defensive_possessions"].le(0).any():
        raise ValueError(f"Season {season} has a team with zero defensive possessions.")
    defense["team_drtg"] = (
        100.0 * defense["points_allowed"] / defense["defensive_possessions"]
    )
    league_ortg = 100.0 * plays["points"].sum() / plays["possessions"].sum()

    exposure = pd.concat(exposure_rows, ignore_index=True).merge(
        defense[["defense_team", "team_drtg"]],
        on="defense_team",
        how="left",
        validate="many_to_one",
    )
    if exposure["team_drtg"].isna().any():
        raise ValueError(f"Season {season} has exposure to an unrated defense.")
    exposure["weighted_drtg"] = (
        exposure["offensive_possessions"] * exposure["team_drtg"]
    )
    player = exposure.groupby("PLAYER_ID", as_index=False).agg(
        context_offensive_possessions=("offensive_possessions", "sum"),
        weighted_drtg=("weighted_drtg", "sum"),
    )
    player["defense_faced_drtg"] = (
        player["weighted_drtg"] / player["context_offensive_possessions"]
    )
    player["PLAYER_ID"] = player["PLAYER_ID"].astype(int)
    player["Window_End"] = int(season)
    return player.drop(columns="weighted_drtg"), {
        "Window_End": int(season),
        "league_ortg": float(league_ortg),
        "league_drtg": float(
            100.0 * defense["points_allowed"].sum() / defense["defensive_possessions"].sum()
        ),
        "teams": int(len(defense)),
        "players": int(len(player)),
    }


def apply_inflation_adjustment(
    features: pd.DataFrame,
    player_context: pd.DataFrame,
    season_context: pd.DataFrame,
    *,
    reference_ortg: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Scale PTS/100 by current ORTG divided by defense-faced DRTG."""
    adjusted = features.merge(
        player_context,
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    ).merge(
        season_context[["Window_End", "league_drtg"]],
        on="Window_End",
        how="left",
        validate="many_to_one",
    )
    if adjusted["league_drtg"].isna().any():
        missing = sorted(adjusted.loc[adjusted["league_drtg"].isna(), "Window_End"].unique())
        raise ValueError(f"Missing season context for {missing}.")
    matched = adjusted["defense_faced_drtg"].notna()
    defense_faced = adjusted["defense_faced_drtg"].fillna(adjusted["league_drtg"])
    if defense_faced.le(0).any():
        raise ValueError("Defense-faced DRTG must be positive.")
    adjusted["inflation_multiplier"] = float(reference_ortg) / defense_faced
    adjusted["PTS_p100"] = adjusted["PTS_p100"] * adjusted["inflation_multiplier"]
    weight = adjusted["OffPoss"].clip(lower=0)
    quality = {
        "rows": int(len(adjusted)),
        "matched_rows": int(matched.sum()),
        "row_match_rate": float(matched.mean()),
        "offensive_possession_match_rate": float(weight.loc[matched].sum() / weight.sum()),
        "fallback_rows": int((~matched).sum()),
        "minimum_multiplier": float(adjusted["inflation_multiplier"].min()),
        "maximum_multiplier": float(adjusted["inflation_multiplier"].max()),
    }
    return adjusted.drop(
        columns=[
            "context_offensive_possessions",
            "defense_faced_drtg",
            "league_drtg",
            "inflation_multiplier",
        ]
    ), quality


def paired_bootstrap(games: pd.DataFrame) -> dict[str, float | int]:
    baseline = games.loc[games["candidate"].eq(BASELINE)]
    challenger = games.loc[games["candidate"].eq(CHALLENGER)]
    seasons = sorted(set(baseline["test_season"]) & set(challenger["test_season"]))
    differences = []
    for season in seasons:
        left = baseline.loc[baseline["test_season"].eq(season)].set_index("game_id").sort_index()
        right = challenger.loc[challenger["test_season"].eq(season)].set_index("game_id").sort_index()
        if not left.index.equals(right.index):
            raise ValueError(f"Game coverage differs in {season}.")
        differences.append(left["squared_error"].to_numpy() - right["squared_error"].to_numpy())
    observed = float(np.mean([values.mean() for values in differences]))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        draws[draw] = np.mean(
            [values[rng.integers(0, len(values), len(values))].mean() for values in differences]
        )
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "draws": BOOTSTRAP_DRAWS,
        "folds": len(seasons),
        "baseline_minus_challenger_mse": observed,
        "lower_95": float(low),
        "upper_95": float(high),
        "probability_challenger_better": float(np.mean(draws > 0)),
    }


def main() -> int:
    features = pd.read_parquet(HISTORICAL_BOX)
    targets = pd.read_parquet(BASELINE_RUN / "targets.parquet")
    targets = targets.loc[targets["horizon"].eq(9) & targets["target_variant"].eq("normal")]
    player_parts = []
    season_rows = []
    for season in range(int(features["Window_End"].min()), REFERENCE_SEASON + 1):
        player, season_row = season_opponent_context(
            pd.read_parquet(STINT_ROOT / f"season={season}/regular.parquet"),
            season=season,
        )
        player_parts.append(player)
        season_rows.append(season_row)
    player_context = pd.concat(player_parts, ignore_index=True)
    season_context = pd.DataFrame(season_rows)
    reference_ortg = float(
        season_context.loc[season_context["Window_End"].eq(REFERENCE_SEASON), "league_ortg"].iloc[0]
    )
    adjusted, adjustment_quality = apply_inflation_adjustment(
        features,
        player_context,
        season_context,
        reference_ortg=reference_ortg,
    )
    priors, prior_metrics = _fit_box_priors(
        _panel(adjusted, targets),
        RATING_SEASONS,
        {"offense": 300.0, "defense": 1000.0},
    )
    priors["candidate"] = "box15_era_adjusted_9y_normal"
    challenger_games, ratings, coverage = _evaluate_priors(
        priors,
        {
            "rating_seasons": list(RATING_SEASONS),
            "aio": {"lambda_off": 3000, "lambda_def": 4500, "lambda_home": 300},
        },
    )
    baseline_games = pd.read_parquet(BASELINE_RUN / "game_predictions.parquet")
    baseline_games = baseline_games.loc[
        baseline_games["candidate"].isin(["box15_9y_normal", BASELINE, "zero_prior_rapm"])
    ]
    games = pd.concat(
        [
            baseline_games,
            challenger_games.loc[~challenger_games["candidate"].eq("zero_prior_rapm")],
        ],
        ignore_index=True,
    )
    folds, summary = _game_metrics(games)
    bootstrap = paired_bootstrap(games)

    sources = {
        "runner": Path(__file__),
        "features": HISTORICAL_BOX,
        "targets": BASELINE_RUN / "targets.parquet",
        "baseline_games": BASELINE_RUN / "game_predictions.parquet",
        "stint_manifest": STINT_ROOT / "manifest.json",
    }
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"box15_inflation_adjustment_v1_{digest}"
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "season_context.parquet": season_context,
        "priors.parquet": priors,
        "prior_metrics.parquet": prior_metrics,
        "ratings.parquet": ratings,
        "coverage.parquet": coverage,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
    }.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "reused_historical_diagnostic",
        "question": "Does current-era, defense-faced scoring improve the frozen Box15 prior?",
        "transformation": "PTS_p100 * (2026 league ORTG / possession-weighted DRTG of defenses faced)",
        "reference_season": REFERENCE_SEASON,
        "reference_league_ortg": reference_ortg,
        "unmatched_player_season_fallback": "same-season league DRTG",
        "adjustment_quality": adjustment_quality,
        "target": "nine-year normal RAPM ending in the rating season, followed by the frozen one-season lineup update",
        "test_seasons": list(range(2015, 2027)),
        "summary": summary.to_dict("records"),
        "paired_bootstrap": bootstrap,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": hashes[name]}
            for name, path in sources.items()
        },
        "warning": "All 2015-2026 folds are reused historical evidence. Season 2027 remains untouched.",
    }
    write_json_atomic(run, output / "run.json")
    print(summary.to_string(index=False))
    print(json.dumps({"adjustment_quality": adjustment_quality, "paired_bootstrap": bootstrap}, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
