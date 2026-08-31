"""Tune one-season offense/defense ridge and test score-state controls."""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients, load_unified_terminal_possessions
from nba_impact.models.rubberband_score_state import (
    annotate_offense_margin_before,
    fit_score_state_rapm,
    predict_score_state_rapm,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "research/experiments/single_season_rapm_sweep_v1.json"
OUTPUT = ROOT / "research/rapm_lab/outputs/single_season_rapm_sweep"


def _config(seasons: tuple[int, ...], off: float, defense: float, home: float) -> RapmConfig:
    return RapmConfig(
        seasons=seasons,
        lambda_off=off,
        lambda_def=defense,
        lambda_home=home,
        data_scope="single_season_rapm_sweep",
    )


def _summary(metrics: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        metrics.groupby(group, as_index=False)
        .agg(
            seasons=("test_season", "nunique"),
            equal_season_mse=("margin_rmse", lambda values: float(np.mean(np.square(values)))),
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            mean_predicted_margin_sd=("predicted_margin_sd", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values(["equal_season_mse", group], kind="stable")
    )


def _game_metrics(games: pd.DataFrame) -> dict[str, float | int]:
    error = games["actual_margin"] - games["predicted_margin"]
    variance = float(np.var(games["predicted_margin"], ddof=0))
    slope = (
        float(np.cov(games["actual_margin"], games["predicted_margin"], ddof=0)[0, 1] / variance)
        if variance > 0
        else float("nan")
    )
    return {
        "games": int(len(games)),
        "margin_rmse": float(np.sqrt(np.mean(error**2))),
        "margin_mae": float(np.mean(np.abs(error))),
        "margin_correlation": float(games[["actual_margin", "predicted_margin"]].corr().iloc[0, 1]),
        "actual_margin_sd": float(games["actual_margin"].std(ddof=0)),
        "predicted_margin_sd": float(games["predicted_margin"].std(ddof=0)),
        "calibration_intercept": float(games["actual_margin"].mean() - slope * games["predicted_margin"].mean()),
        "calibration_slope": slope,
    }


def _paired_bootstrap(games: pd.DataFrame, *, reference: str, candidate: str, draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    groups = [part.reset_index(drop=True) for _, part in games.groupby("test_season")]
    differences = np.empty(draws, dtype=float)
    for draw in range(draws):
        reference_errors: list[np.ndarray] = []
        candidate_errors: list[np.ndarray] = []
        for part in groups:
            take = rng.integers(0, len(part), len(part))
            actual = part["actual_margin"].to_numpy()[take]
            reference_errors.append((actual - part[reference].to_numpy()[take]) ** 2)
            candidate_errors.append((actual - part[candidate].to_numpy()[take]) ** 2)
        differences[draw] = np.mean(np.concatenate(reference_errors)) - np.mean(np.concatenate(candidate_errors))
    return {
        "reference": reference,
        "candidate": candidate,
        "reference_minus_candidate_mse": float(differences.mean()),
        "lower_95": float(np.quantile(differences, 0.025)),
        "upper_95": float(np.quantile(differences, 0.975)),
        "probability_candidate_better": float(np.mean(differences > 0)),
    }


def run() -> dict:
    contract = json.loads(CONTRACT.read_text())
    seasons = tuple(int(value) for value in contract["seasons"])
    selection_seasons = set(int(value) for value in contract["selection_seasons"])
    diagnostic_seasons = set(int(value) for value in contract["diagnostic_seasons"])
    evaluation_seasons = selection_seasons | diagnostic_seasons
    started = time.perf_counter()
    frame = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    frame = annotate_offense_margin_before(frame)
    design = build_design(frame, include_home=True)
    margins = frame["offense_margin_before"].to_numpy(dtype=float)
    home = float(contract["lambda_home"])
    candidates = [
        {
            "candidate": f"o{int(off)}_d{int(defense)}",
            "lambda_off": float(off),
            "lambda_def": float(defense),
        }
        for off, defense in itertools.product(contract["lambda_off_grid"], contract["lambda_def_grid"])
    ]

    lambda_rows: list[dict] = []
    for candidate in candidates:
        config = _config(seasons, candidate["lambda_off"], candidate["lambda_def"], home)
        for test_season in sorted(evaluation_seasons):
            train = design.seasons == test_season - 1
            test = design.seasons == test_season
            beta, intercept = fit_coefficients(design, config, row_mask=train)
            prediction = intercept + np.asarray(design.X[test] @ beta).ravel()
            metrics, _ = game_margin_metrics(frame.loc[test].reset_index(drop=True), prediction)
            lambda_rows.append({**candidate, "test_season": test_season, **metrics})
        print(f"completed {candidate['candidate']}", flush=True)
    lambda_metrics = pd.DataFrame(lambda_rows)
    lambda_selection = _summary(lambda_metrics.loc[lambda_metrics["test_season"].isin(selection_seasons)], "candidate")
    selected_name = str(lambda_selection.iloc[0]["candidate"])
    selected = next(value for value in candidates if value["candidate"] == selected_name)

    state_rows: list[dict] = []
    score = contract["score_bucket"]
    state_candidates: list[float | None] = [None, *(float(value) for value in contract["score_state_penalties"])]
    selected_config = _config(seasons, selected["lambda_off"], selected["lambda_def"], home)
    for penalty in state_candidates:
        label = "none" if penalty is None else f"bucket_p{int(penalty)}"
        for test_season in sorted(evaluation_seasons):
            train = design.seasons == test_season - 1
            test = design.seasons == test_season
            if penalty is None:
                beta, intercept = fit_coefficients(design, selected_config, row_mask=train)
                prediction = intercept + np.asarray(design.X[test] @ beta).ravel()
            else:
                fit = fit_score_state_rapm(
                    design,
                    margins,
                    selected_config,
                    minimum=int(score["minimum"]),
                    maximum=int(score["maximum"]),
                    bucket_width=int(score["width"]),
                    state_penalty=penalty,
                    row_mask=train,
                )
                prediction = predict_score_state_rapm(
                    fit,
                    design,
                    margins,
                    row_mask=test,
                    include_score_state=False,
                )
            metrics, _ = game_margin_metrics(frame.loc[test].reset_index(drop=True), prediction)
            state_rows.append({"score_control": label, "state_penalty": penalty, "test_season": test_season, **metrics})
        print(f"completed score control {label}", flush=True)
    state_metrics = pd.DataFrame(state_rows)
    state_selection = _summary(state_metrics.loc[state_metrics["test_season"].isin(selection_seasons)], "score_control")
    selected_state = str(state_selection.iloc[0]["score_control"])

    baseline_config = _config(seasons, 3000.0, 3000.0, home)
    diagnostic_game_rows: list[pd.DataFrame] = []
    diagnostic_metric_rows: list[dict] = []
    selection_plain_games: list[pd.DataFrame] = []
    for test_season in sorted(evaluation_seasons):
        train = design.seasons == test_season - 1
        test = design.seasons == test_season
        current: dict[str, pd.DataFrame] = {}
        for label, config in (("baseline", baseline_config), ("selected_plain", selected_config)):
            beta, intercept = fit_coefficients(design, config, row_mask=train)
            prediction = intercept + np.asarray(design.X[test] @ beta).ravel()
            metrics, games = game_margin_metrics(frame.loc[test].reset_index(drop=True), prediction)
            current[label] = games.rename(columns={"predicted_margin": label})
            if test_season in diagnostic_seasons:
                diagnostic_metric_rows.append({"model": label, "test_season": test_season, **metrics})
        if selected_state == "none":
            current["selected_state"] = current["selected_plain"].rename(columns={"selected_plain": "selected_state"})
        else:
            penalty = float(selected_state.removeprefix("bucket_p"))
            fit = fit_score_state_rapm(
                design,
                margins,
                selected_config,
                minimum=int(score["minimum"]),
                maximum=int(score["maximum"]),
                bucket_width=int(score["width"]),
                state_penalty=penalty,
                row_mask=train,
            )
            prediction = predict_score_state_rapm(fit, design, margins, row_mask=test, include_score_state=False)
            metrics, games = game_margin_metrics(frame.loc[test].reset_index(drop=True), prediction)
            current["selected_state"] = games.rename(columns={"predicted_margin": "selected_state"})
            if test_season in diagnostic_seasons:
                diagnostic_metric_rows.append({"model": "selected_state", "test_season": test_season, **metrics})
        merged = current["baseline"].merge(
            current["selected_plain"][["gameid", "actual_margin", "selected_plain"]],
            on=["gameid", "actual_margin"], validate="one_to_one"
        ).merge(
            current["selected_state"][["gameid", "actual_margin", "selected_state"]],
            on=["gameid", "actual_margin"], validate="one_to_one"
        )
        merged["test_season"] = test_season
        if test_season in selection_seasons:
            selection_plain_games.append(merged)
        else:
            diagnostic_game_rows.append(merged)

    selection_games = pd.concat(selection_plain_games, ignore_index=True)
    calibration_variance = float(np.var(selection_games["selected_state"], ddof=0))
    calibration_slope = float(
        np.cov(selection_games["actual_margin"], selection_games["selected_state"], ddof=0)[0, 1]
        / calibration_variance
    )
    calibration_intercept = float(
        selection_games["actual_margin"].mean() - calibration_slope * selection_games["selected_state"].mean()
    )
    diagnostic_games = pd.concat(diagnostic_game_rows, ignore_index=True)
    diagnostic_games["selected_calibrated"] = calibration_intercept + calibration_slope * diagnostic_games["selected_state"]
    for season, part in diagnostic_games.groupby("test_season"):
        metrics = _game_metrics(
            part[["actual_margin", "selected_calibrated"]].rename(columns={"selected_calibrated": "predicted_margin"})
        )
        diagnostic_metric_rows.append({"model": "selected_calibrated", "test_season": int(season), **metrics})

    diagnostic_metrics = pd.DataFrame(diagnostic_metric_rows)
    diagnostic_summary = _summary(diagnostic_metrics, "model")
    bootstrap = pd.DataFrame(
        [
            _paired_bootstrap(
                diagnostic_games,
                reference="baseline",
                candidate="selected_plain",
                draws=int(contract["bootstrap_draws"]),
                seed=int(contract["bootstrap_seed"]),
            ),
            _paired_bootstrap(
                diagnostic_games,
                reference="selected_plain",
                candidate="selected_state",
                draws=int(contract["bootstrap_draws"]),
                seed=int(contract["bootstrap_seed"]) + 1,
            ),
            _paired_bootstrap(
                diagnostic_games,
                reference="selected_state",
                candidate="selected_calibrated",
                draws=int(contract["bootstrap_draws"]),
                seed=int(contract["bootstrap_seed"]) + 2,
            ),
        ]
    )

    identity = hashlib.sha256(
        json.dumps({"contract": sha256_file(CONTRACT), "runner": sha256_file(Path(__file__))}, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT / f"single_season_rapm_sweep_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "lambda_fold_metrics.parquet": lambda_metrics,
        "lambda_selection.parquet": lambda_selection,
        "score_state_fold_metrics.parquet": state_metrics,
        "score_state_selection.parquet": state_selection,
        "diagnostic_fold_metrics.parquet": diagnostic_metrics,
        "diagnostic_summary.parquet": diagnostic_summary,
        "diagnostic_game_predictions.parquet": diagnostic_games,
        "paired_bootstrap.parquet": bootstrap,
    }
    for name, value in outputs.items():
        value.to_parquet(output / name, index=False)
    manifest = {
        "run_id": output.name,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": contract,
        "selected_lambda": selected,
        "selected_score_control": selected_state,
        "selection_calibration": {"intercept": calibration_intercept, "slope": calibration_slope},
        "diagnostic_summary": diagnostic_summary.to_dict("records"),
        "paired_bootstrap": bootstrap.to_dict("records"),
        "quality": {"possessions": int(len(frame)), "games": int(frame["gameid"].nunique()), "identical_diagnostic_games": True},
        "paths": {name.removesuffix(".parquet"): name for name in outputs},
        "forbidden_interpretation": "Independent confirmation or a production RAPM penalty change.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
