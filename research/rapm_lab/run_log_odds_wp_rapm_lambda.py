"""Tune clipped log-odds WP-RAPM on chronological next-season games."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_player_names,
    ratings_table,
)
from nba_impact.models.win_probability_rapm import build_log_odds_wp_target


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/log_odds_wp_rapm_lambda_v1.json"
WP_TARGET = ROOT / "research/rapm_lab/outputs/wp_spm_aio/checkpoints/wp_target_2014_2026.parquet"
PULSE_GAMES = ROOT / "artifacts/models/pulse/pulse_canonical_v1_cd3c14750a/validation_games.parquet"
REFERENCE_GAMES = ROOT / "research/rapm_lab/outputs/wp_rapm_vs_pulse/wp_rapm_vs_pulse_v1_3d2995995c/game_predictions.parquet"
OUTPUT_ROOT = ROOT / "research/rapm_lab/outputs/log_odds_wp_rapm_lambda"
NAMES = ROOT / "rapm/data/all_names.csv"
PLAYER_GAMES = ROOT / "data/lake/silver/player_games.parquet"


def _candidate(epsilon: float, lambda_off: int, lambda_def: int) -> str:
    return f"logit_e{epsilon:g}_o{lambda_off}_d{lambda_def}"


def _game_predictions(design, beta: np.ndarray, intercept: float, mask: np.ndarray) -> pd.DataFrame:
    predicted = intercept + np.asarray(design.X[mask] @ beta).ravel()
    home_sign = np.where(design.home_offense[mask], 1.0, -1.0)
    return pd.DataFrame(
        {
            "game_id": design.game_ids[mask],
            "raw_prediction": predicted * home_sign,
        }
    ).groupby("game_id", as_index=False).sum()


def _affine(frame: pd.DataFrame) -> tuple[float, float]:
    x = frame["raw_prediction"].to_numpy(float)
    design = np.column_stack([np.ones(len(frame)), x])
    intercept, slope = np.linalg.lstsq(
        design, frame["actual_margin"].to_numpy(float), rcond=None
    )[0]
    return float(intercept), float(slope)


def _past_only_predictions(games: pd.DataFrame, scored_seasons: list[int]) -> pd.DataFrame:
    rows = []
    for candidate, candidate_games in games.groupby("candidate", sort=False):
        for season in scored_seasons:
            train = candidate_games.loc[candidate_games["outcome_season"].lt(season)]
            test = candidate_games.loc[candidate_games["outcome_season"].eq(season)].copy()
            intercept, slope = _affine(train)
            test["predicted_margin"] = intercept + slope * test["raw_prediction"]
            test["calibration_intercept"] = intercept
            test["calibration_slope"] = slope
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def _metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, season), frame in predictions.groupby(["candidate", "outcome_season"]):
        error = frame["actual_margin"] - frame["predicted_margin"]
        rows.append(
            {
                "candidate": candidate,
                "outcome_season": int(season),
                "games": len(frame),
                "mse": float(np.mean(error**2)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "correlation": float(frame["actual_margin"].corr(frame["predicted_margin"])),
            }
        )
    return pd.DataFrame(rows)


def _publication_ratings(
    source: pd.DataFrame,
    contract: dict,
    selected: pd.Series,
) -> pd.DataFrame:
    target, _ = build_log_odds_wp_target(source, epsilon=float(selected["epsilon"]))
    target["pts"] = target["offense_log_odds_change"]
    design = build_design(target, include_home=True)
    names = load_current_player_names(NAMES, PLAYER_GAMES)
    rows = []
    for season in (2024, 2025, 2026):
        mask = design.seasons == season
        config = RapmConfig(
            seasons=(season,),
            lambda_off=float(selected["lambda_offense"]),
            lambda_def=float(selected["lambda_defense"]),
            lambda_home=float(contract["lambda_home"]),
            data_scope="annual_clipped_log_odds_wp_rapm",
        )
        beta, _ = fit_coefficients(design, config, row_mask=mask)
        ratings = ratings_table(design, beta, names=names)
        n_players = len(design.players)
        exposure = pd.DataFrame(
            {
                "player_id": design.players,
                "off_possessions": np.asarray(
                    np.abs(design.X[mask, :n_players]).sum(axis=0)
                ).ravel(),
                "def_possessions": np.asarray(
                    np.abs(design.X[mask, n_players : 2 * n_players]).sum(axis=0)
                ).ravel(),
            }
        )
        ratings = ratings.drop(
            columns=["off_possessions", "def_possessions"]
        ).merge(exposure, on="player_id", validate="one_to_one")
        ratings["Season"] = season
        ratings["epsilon"] = float(selected["epsilon"])
        ratings["lambda_offense"] = int(selected["lambda_offense"])
        ratings["lambda_defense"] = int(selected["lambda_defense"])
        rows.append(ratings)
    return pd.concat(rows, ignore_index=True)


def _paired_bootstrap(
    predictions: pd.DataFrame,
    reference_column: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for draw in range(draws):
        fold_deltas = []
        for _, frame in predictions.groupby("outcome_season"):
            take = rng.integers(0, len(frame), len(frame))
            actual = frame["actual_margin"].to_numpy()[take]
            logit_error = actual - frame["logit_wp_rapm"].to_numpy()[take]
            reference_error = actual - frame[reference_column].to_numpy()[take]
            fold_deltas.append(np.mean(logit_error**2 - reference_error**2))
        deltas[draw] = np.mean(fold_deltas)
    return {
        "left": "logit_wp_rapm",
        "right": reference_column,
        "draws": draws,
        "mean_mse_delta": float(deltas.mean()),
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "probability_left_better": float(np.mean(deltas < 0)),
    }


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    if max(contract["outcome_seasons"]) >= contract["untouched_confirmation_season"]:
        raise ValueError("Season 2027 must remain untouched.")

    source = pd.read_parquet(WP_TARGET)
    pulse = pd.read_parquet(PULSE_GAMES)
    actual = pulse.loc[
        pulse["candidate"].eq("pulse"),
        ["outcome_season", "game_id", "actual_margin"],
    ].drop_duplicates(["outcome_season", "game_id"])

    game_frames = []
    diagnostics = []
    specifications = {
        (float(epsilon), int(lambda_off), int(lambda_def))
        for epsilon in contract["epsilons"]
        for lambda_off in contract["lambda_offense"]
        for lambda_def in contract["lambda_defense"]
    }
    refinement = contract["refinement"]
    specifications.update(
        {
            (float(refinement["epsilon"]), int(lambda_off), int(lambda_def))
            for lambda_off in refinement["lambda_offense"]
            for lambda_def in refinement["lambda_defense"]
        }
    )
    for epsilon in contract["epsilons"]:
        target, conservation = build_log_odds_wp_target(source, epsilon=float(epsilon))
        if conservation["conservation_error"].abs().max() > 1e-10:
            raise AssertionError("Log-odds changes must telescope within games.")
        target["pts"] = target["offense_log_odds_change"]
        design = build_design(target, include_home=True)
        terminal = target.groupby("gameid", sort=False).tail(1)["home_log_odds_change"].abs().sum()
        diagnostics.append(
            {
                "epsilon": float(epsilon),
                "target_abs_p50": float(target["home_log_odds_change"].abs().quantile(0.5)),
                "target_abs_p99": float(target["home_log_odds_change"].abs().quantile(0.99)),
                "terminal_share_of_absolute_target": float(terminal / target["home_log_odds_change"].abs().sum()),
            }
        )
        pairs = sorted(
            (lambda_off, lambda_def)
            for candidate_epsilon, lambda_off, lambda_def in specifications
            if candidate_epsilon == float(epsilon)
        )
        for lambda_off, lambda_def in pairs:
            candidate = _candidate(float(epsilon), int(lambda_off), int(lambda_def))
            for rating_season in contract["rating_seasons"]:
                train = design.seasons == rating_season
                config = RapmConfig(
                    seasons=(int(rating_season),),
                    lambda_off=float(lambda_off),
                    lambda_def=float(lambda_def),
                    lambda_home=float(contract["lambda_home"]),
                    data_scope="annual_clipped_log_odds_wp_rapm",
                )
                beta, intercept = fit_coefficients(design, config, row_mask=train)
                outcome_season = int(rating_season) + 1
                game = _game_predictions(
                    design, beta, intercept, design.seasons == outcome_season
                )
                game["rating_season"] = int(rating_season)
                game["outcome_season"] = outcome_season
                game["candidate"] = candidate
                game["epsilon"] = float(epsilon)
                game["lambda_offense"] = int(lambda_off)
                game["lambda_defense"] = int(lambda_def)
                game_frames.append(game)

    games = pd.concat(game_frames, ignore_index=True).merge(
        actual, on=["outcome_season", "game_id"], validate="many_to_one"
    )
    observed_games = games.groupby(["candidate", "outcome_season"]).size().unstack()
    if observed_games.nunique(axis="index").ne(1).any():
        raise AssertionError("Candidate game coverage differs within a season.")
    if int(observed_games.min().min()) < 1_200:
        raise AssertionError("Common PULSE and WP game coverage fell below 1,200.")

    predictions = _past_only_predictions(games, contract["scored_outcome_seasons"])
    folds = _metrics(predictions)
    summary = folds.groupby("candidate", as_index=False).agg(
        folds=("outcome_season", "nunique"),
        mean_mse=("mse", "mean"),
        equal_season_rmse=("mse", lambda value: float(np.sqrt(value.mean()))),
        mean_correlation=("correlation", "mean"),
    )
    candidate_parameters = games[
        ["candidate", "epsilon", "lambda_offense", "lambda_defense"]
    ].drop_duplicates("candidate")
    summary = summary.merge(candidate_parameters, on="candidate", validate="one_to_one")
    summary = summary.sort_values("mean_mse", kind="stable").reset_index(drop=True)

    best_candidate = str(summary.iloc[0]["candidate"])
    best_predictions = predictions.loc[
        predictions["candidate"].eq(best_candidate),
        ["outcome_season", "game_id", "actual_margin", "predicted_margin"],
    ].rename(columns={"predicted_margin": "logit_wp_rapm"})
    references = pd.read_parquet(REFERENCE_GAMES)[
        [
            "outcome_season",
            "game_id",
            "PULSE_predicted_margin",
            "RAPM_predicted_margin",
            "WP-RAPM_predicted_margin",
        ]
    ].rename(
        columns={
            "PULSE_predicted_margin": "pulse",
            "RAPM_predicted_margin": "rapm",
            "WP-RAPM_predicted_margin": "raw_wp_rapm",
        }
    )
    comparison_games = best_predictions.merge(
        references, on=["outcome_season", "game_id"], validate="one_to_one"
    )
    if len(comparison_games) != len(best_predictions):
        raise AssertionError("Benchmark comparison lost common scored games.")
    benchmark_rows = []
    for model in ("pulse", "rapm", "raw_wp_rapm", "logit_wp_rapm"):
        for season, frame in comparison_games.groupby("outcome_season"):
            error = frame["actual_margin"] - frame[model]
            benchmark_rows.append(
                {
                    "model": model,
                    "outcome_season": int(season),
                    "games": len(frame),
                    "mse": float(np.mean(error**2)),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "correlation": float(frame["actual_margin"].corr(frame[model])),
                }
            )
    benchmark_folds = pd.DataFrame(benchmark_rows)
    benchmark_summary = benchmark_folds.groupby("model", as_index=False).agg(
        folds=("outcome_season", "nunique"),
        mean_mse=("mse", "mean"),
        equal_season_rmse=("mse", lambda value: float(np.sqrt(value.mean()))),
        mean_correlation=("correlation", "mean"),
    )
    paired_comparisons = [
        _paired_bootstrap(
            comparison_games,
            reference,
            draws=int(contract["bootstrap_draws"]),
            seed=int(contract["bootstrap_seed"]),
        )
        for reference in ("raw_wp_rapm", "rapm", "pulse")
    ]

    chronological = []
    for season in contract["scored_outcome_seasons"]:
        past = games.loc[games["outcome_season"].lt(season)]
        training_scores = []
        for candidate, frame in past.groupby("candidate", sort=False):
            intercept, slope = _affine(frame)
            error = frame["actual_margin"] - (intercept + slope * frame["raw_prediction"])
            training_scores.append((float(np.mean(error**2)), candidate))
        _, selected = min(training_scores)
        row = folds.loc[
            folds["candidate"].eq(selected) & folds["outcome_season"].eq(season)
        ].iloc[0]
        parameters = candidate_parameters.set_index("candidate").loc[selected]
        chronological.append(
            {
                "outcome_season": int(season),
                "selected_candidate": selected,
                "epsilon": float(parameters["epsilon"]),
                "lambda_offense": int(parameters["lambda_offense"]),
                "lambda_defense": int(parameters["lambda_defense"]),
                "games": int(row["games"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "correlation": float(row["correlation"]),
            }
        )
    chronological_frame = pd.DataFrame(chronological)

    public_ratings = _publication_ratings(source, contract, summary.iloc[0])

    identity = hashlib.sha256(
        json.dumps(
            {
                "contract": sha256_file(CONTRACT),
                "runner": sha256_file(Path(__file__)),
                "wp_model": sha256_file(ROOT / "src/nba_impact/models/win_probability_rapm.py"),
                "rapm_model": sha256_file(ROOT / "src/nba_impact/models/rapm.py"),
                "target": sha256_file(WP_TARGET),
                "pulse_games": sha256_file(PULSE_GAMES),
                "reference_games": sha256_file(REFERENCE_GAMES),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"log_odds_wp_rapm_lambda_v1_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "game_predictions.parquet", index=False)
    folds.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "candidate_summary.parquet", index=False)
    chronological_frame.to_parquet(output / "chronological_selection.parquet", index=False)
    pd.DataFrame(diagnostics).to_parquet(output / "target_diagnostics.parquet", index=False)
    benchmark_folds.to_parquet(output / "benchmark_fold_metrics.parquet", index=False)
    benchmark_summary.to_parquet(output / "benchmark_summary.parquet", index=False)
    public_ratings.to_parquet(output / "public_ratings_2024_2026.parquet", index=False)
    run_record = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": contract["status"],
        "target": contract["target"],
        "games_per_scored_fold": {
            str(k): int(v) for k, v in predictions.groupby("outcome_season")["game_id"].nunique().items()
        },
        "best_fixed_candidate_reused_diagnostic": summary.iloc[0].to_dict(),
        "chronological_selection": chronological,
        "chronological_equal_season_rmse": float(np.sqrt(chronological_frame["mse"].mean())),
        "benchmark_summary": benchmark_summary.to_dict("records"),
        "paired_comparisons": paired_comparisons,
        "target_diagnostics": diagnostics,
        "public_ratings": "public_ratings_2024_2026.parquet",
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run_record, output / "run.json")
    print(summary.head(12).to_string(index=False))
    print("\nChronological selection")
    print(chronological_frame.to_string(index=False))
    print(f"\nwrote {output}")
    return run_record


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
