#!/usr/bin/env python3
"""Benchmark dynamic CourtSignal shooting gravity against public LASER."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.shot_model_suite import dynamic_shooting_gravity


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "laser_breaker_v1"
CONTRACT = ROOT / "research/experiments/laser_breaker_v1.yml"
LASER = ROOT / "data/external/laser_2014_2025.tsv"
SHOT_RUN = ROOT / "artifacts/research/shot_model_suite/shot_model_suite_v1_2494cca535"
ANNUAL = SHOT_RUN / "annual_shooting_threat.parquet"
TARGETS = (
    "next_three_point_pct",
    "next_three_pa_p100",
    "next_contested_share",
    "next_gravity_composite",
)


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(character for character in text if not unicodedata.combining(character)).casefold().replace(".", "").replace("'", "").replace("-", " ")


def _zscore(series: pd.Series) -> pd.Series:
    scale = series.std(ddof=0)
    return (series - series.mean()) / scale if scale else series * 0.0


def _contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("The LASER benchmark contract changed.")
    if contract["information_cutoff"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return contract


def _benchmark_panel(annual: pd.DataFrame) -> pd.DataFrame:
    player = annual.copy()
    player["distance_attempts"] = player[
        ["very_tight_FG3A", "tight_FG3A", "open_FG3A", "wide_open_FG3A"]
    ].sum(axis=1)
    player["contested_attempts"] = player["very_tight_FG3A"] + player["tight_FG3A"]
    player["next_three_point_pct"] = np.divide(
        player["FG3M"],
        player["FG3A"],
        out=np.full(len(player), np.nan),
        where=player["FG3A"].to_numpy() > 0,
    )
    player["next_three_pa_p100"] = np.divide(
        100.0 * player["FG3A"],
        player["OffPoss"],
        out=np.full(len(player), np.nan),
        where=player["OffPoss"].to_numpy() > 0,
    )
    player["next_contested_share"] = np.divide(
        player["contested_attempts"],
        player["distance_attempts"],
        out=np.full(len(player), np.nan),
        where=player["distance_attempts"].to_numpy() > 0,
    )
    future = player[
        [
            "PLAYER_ID",
            "Season",
            "next_three_point_pct",
            "next_three_pa_p100",
            "next_contested_share",
            "FG3A",
            "OffPoss",
            "distance_attempts",
        ]
    ].copy()
    future["Season"] -= 1
    laser = pd.read_csv(LASER, sep="\t")
    laser["Season"] = laser["SEASON"].str[-2:].astype(int) + 2000
    laser["normalized_name"] = laser["NAME"].map(_normalize_name)
    names = annual[["PLAYER_ID", "PLAYER_NAME", "Season"]].copy()
    names["normalized_name"] = names["PLAYER_NAME"].map(_normalize_name)
    names = names.drop_duplicates(["Season", "normalized_name"])
    laser = laser.merge(
        names[["PLAYER_ID", "Season", "normalized_name"]],
        on=["Season", "normalized_name"],
        how="left",
        validate="one_to_one",
    )
    if laser["PLAYER_ID"].isna().any():
        missing = laser.loc[laser["PLAYER_ID"].isna(), "NAME"].unique().tolist()
        raise ValueError(f"LASER names did not resolve: {missing}")
    output = laser.merge(
        future,
        on=["PLAYER_ID", "Season"],
        how="inner",
        validate="one_to_one",
    )
    output["laser_z"] = output.groupby("Season")["LASER"].transform(_zscore)
    for target in TARGETS[:3]:
        output[f"{target}_z"] = output.groupby("Season")[target].transform(_zscore)
    output["next_gravity_composite"] = output[
        [f"{target}_z" for target in TARGETS[:3]]
    ].mean(axis=1)
    return output


def _weights(frame: pd.DataFrame, target: str) -> np.ndarray:
    field = {
        "next_three_point_pct": "FG3A",
        "next_three_pa_p100": "OffPoss",
        "next_contested_share": "distance_attempts",
        "next_gravity_composite": "OffPoss",
    }[target]
    return frame[field].to_numpy(dtype=float)


def _score(
    frame: pd.DataFrame,
    *,
    train_origins: tuple[int, ...],
    score_origins: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_rows = []
    for target in TARGETS:
        target_field = f"{target}_z" if target != "next_gravity_composite" else target
        required = [target_field, "laser_z", "court_signal_gravity"]
        train = frame.loc[frame["Season"].isin(train_origins)].dropna(subset=required)
        score = frame.loc[frame["Season"].isin(score_origins)].dropna(subset=required).copy()
        for candidate, feature in (
            ("laser", "laser_z"),
            ("court_signal_gravity", "court_signal_gravity"),
        ):
            model = Ridge(alpha=10.0).fit(
                train[[feature]],
                train[target_field],
                sample_weight=_weights(train, target),
            )
            score[f"prediction_{candidate}"] = model.predict(score[[feature]])
            for season, fold in score.groupby("Season"):
                weight = _weights(fold, target)
                actual = fold[target_field].to_numpy(dtype=float)
                prediction = fold[f"prediction_{candidate}"].to_numpy(dtype=float)
                metric_rows.append(
                    {
                        "target": target,
                        "origin_season": int(season),
                        "outcome_season": int(season) + 1,
                        "candidate": candidate,
                        "players": len(fold),
                        "weighted_mse": float(np.average((actual - prediction) ** 2, weights=weight)),
                        "correlation": float(np.corrcoef(actual, prediction)[0, 1]),
                    }
                )
        prediction_rows.append(
            score[
                [
                    "PLAYER_ID",
                    "NAME",
                    "Season",
                    target_field,
                    "FG3A",
                    "OffPoss",
                    "distance_attempts",
                    "prediction_laser",
                    "prediction_court_signal_gravity",
                ]
            ].assign(target=target, target_field=target_field)
        )
    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def _paired_bootstrap(predictions: pd.DataFrame, draws: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(20260829)
    rows = []
    for target, target_frame in predictions.groupby("target"):
        season_values = []
        for _, frame in target_frame.groupby("Season"):
            actual = frame[frame["target_field"].iloc[0]].to_numpy(dtype=float)
            weight = _weights(frame, target)
            laser_error = (actual - frame["prediction_laser"].to_numpy()) ** 2
            court_error = (actual - frame["prediction_court_signal_gravity"].to_numpy()) ** 2
            season_values.append((laser_error, court_error, weight))
        point = np.mean(
            [np.average(court - laser, weights=weight) for laser, court, weight in season_values]
        )
        samples = np.empty(draws)
        for draw in range(draws):
            fold_delta = []
            for laser, court, weight in season_values:
                indices = rng.integers(0, len(weight), len(weight))
                fold_delta.append(
                    np.average(court[indices] - laser[indices], weights=weight[indices])
                )
            samples[draw] = np.mean(fold_delta)
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "target": target,
                "court_signal_minus_laser_mse": float(point),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_court_signal_better": float(np.mean(samples < 0)),
                "bootstrap_draws": draws,
                "resampling_unit": "player_within_outcome_season",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    contract = _contract()
    annual = pd.read_parquet(ANNUAL)
    if annual["Season"].max() >= 2027:
        raise ValueError("Season 2027 entered the shooting benchmark.")
    benchmark = _benchmark_panel(annual)
    development = tuple(contract["information_cutoff"]["development_origins"])
    selection = tuple(contract["information_cutoff"]["selection_origins"])
    diagnostics = tuple(contract["information_cutoff"]["diagnostic_origins"])
    grid_rows = []
    gravity_by_spec = {}
    for half_life in contract["selection_grid"]["half_life_seasons"]:
        for prior in contract["selection_grid"]["context_prior_attempts"]:
            gravity = dynamic_shooting_gravity(
                annual,
                half_life_seasons=float(half_life),
                prior_attempts=float(prior),
            )
            gravity_by_spec[(float(half_life), float(prior))] = gravity
            candidate = benchmark.merge(
                gravity[["PLAYER_ID", "Window_End", "court_signal_gravity"]],
                left_on=["PLAYER_ID", "Season"],
                right_on=["PLAYER_ID", "Window_End"],
                validate="one_to_one",
            )
            metrics, _ = _score(
                candidate,
                train_origins=development,
                score_origins=selection,
            )
            summary = metrics.groupby("candidate")["weighted_mse"].mean()
            grid_rows.append(
                {
                    "half_life_seasons": float(half_life),
                    "prior_attempts": float(prior),
                    "court_signal_composite_mse": float(
                        metrics.loc[
                            metrics["target"].eq("next_gravity_composite")
                            & metrics["candidate"].eq("court_signal_gravity"),
                            "weighted_mse",
                        ].mean()
                    ),
                    "laser_composite_mse": float(
                        metrics.loc[
                            metrics["target"].eq("next_gravity_composite")
                            & metrics["candidate"].eq("laser"),
                            "weighted_mse",
                        ].mean()
                    ),
                    "mean_all_target_mse": float(summary["court_signal_gravity"]),
                }
            )
    grid = pd.DataFrame(grid_rows).sort_values(
        ["court_signal_composite_mse", "half_life_seasons", "prior_attempts"],
        kind="stable",
    )
    selected = grid.iloc[0]
    selected_key = (
        float(selected["half_life_seasons"]),
        float(selected["prior_attempts"]),
    )
    gravity = gravity_by_spec[selected_key]
    scored = benchmark.merge(
        gravity,
        left_on=["PLAYER_ID", "Season"],
        right_on=["PLAYER_ID", "Window_End"],
        validate="one_to_one",
    )
    metrics, predictions = _score(
        scored,
        train_origins=(*development, *selection),
        score_origins=diagnostics,
    )
    bootstrap = _paired_bootstrap(predictions)
    diagnostic_wins = bootstrap.set_index("target")["court_signal_minus_laser_mse"].lt(0)
    decisive_composite = float(
        bootstrap.set_index("target").loc[
            "next_gravity_composite", "bootstrap_95_high"
        ]
    ) < 0
    gate_passed = bool(diagnostic_wins.sum() >= 3 and decisive_composite)
    reporting = contract["reporting"]
    leaderboard_season = int(reporting["leaderboard_season"])
    leaderboard = gravity.loc[
        gravity["Window_End"].eq(leaderboard_season)
        & gravity["active_in_window_end"]
        & gravity["gravity_effective_attempts"].ge(
            float(reporting["minimum_effective_attempts"])
        )
    ].sort_values(
        "court_signal_gravity", ascending=False, kind="stable"
    )
    sources = {
        "contract": CONTRACT,
        "laser": LASER,
        "annual_shooting": ANNUAL,
        "shot_run": SHOT_RUN / "run.json",
        "runner": Path(__file__),
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "selected_half_life_seasons": selected_key[0],
        "selected_prior_attempts": selected_key[1],
        "leaderboard_reporting": reporting,
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/laser_breaker" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "selection_grid.parquet": grid,
        "gravity_panel.parquet": gravity,
        "benchmark_panel.parquet": scored,
        "diagnostic_metrics.parquet": metrics,
        "diagnostic_predictions.parquet": predictions,
        "paired_bootstrap.parquet": bootstrap,
        "leaderboard_2026.parquet": leaderboard,
    }
    for name, frame in files.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "court_signal_better_than_laser_gate": gate_passed,
        "contract": contract,
        "config": config,
        "quality": {
            "public_laser_rows": int(len(pd.read_csv(LASER, sep="\t"))),
            "resolved_laser_name_coverage": 1.0,
            "common_rows_with_next_season": len(benchmark),
            "season_2027_loaded": False,
        },
        "files": {
            name: {"rows": len(frame), "sha256": sha256_file(output / name)}
            for name, frame in files.items()
        },
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(grid.head(12).to_string(index=False))
    print(metrics.groupby(["target", "candidate"])[["weighted_mse", "correlation"]].mean().to_string())
    print(bootstrap.to_string(index=False))
    print(f"gate_passed={gate_passed}")


if __name__ == "__main__":
    main()
