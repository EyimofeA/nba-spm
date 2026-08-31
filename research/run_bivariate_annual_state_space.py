#!/usr/bin/env python3
"""Test joint offense-defense covariance in the annual RAPM state-space model."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.linalg import cho_factor, cho_solve

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.bivariate_state_space import filter_bivariate_annual_rapm
from nba_impact.models.rapm import load_legacy_possessions
from run_aio_prior_canonical_followup import _annual_from_frame, _solve

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "bivariate_annual_state_space_v1"
CONTRACT = ROOT / "research/experiments/bivariate_annual_state_space_v1.yml"
TARGETS = ROOT / "artifacts/models/canonical_annual_target_panel/canonical_annual_target_panel_v1_2d9ff74ca3/targets.parquet"
MATRIX_ROOT = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
SEASONS = tuple(range(2019, 2027))
SELECTION_ORIGINS = (2022, 2023)
DIAGNOSTIC_ORIGINS = (2024, 2025)
PROCESS_CORRELATIONS = (-0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 0.90)
PHI = 0.90
PROCESS_SD = 0.25


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Bivariate state-space contract ID changed.")
    if tuple(contract["seasons"]) != SEASONS or contract["season_2027"] != "forbidden":
        raise ValueError("Bivariate state-space season boundary changed.")
    return contract


def _annual_bundles() -> tuple[dict, dict[int, float], pd.DataFrame]:
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    centered_y_ss = {}
    for season in range(2019, 2024):
        frame = load_legacy_possessions(
            POSSESSION_CACHE, (season,), game_types=("regular",)
        )
        annual[season] = _annual_from_frame(frame, season)
        centered = frame["pts"].to_numpy(dtype=float) - float(frame["pts"].mean())
        centered_y_ss[season] = float(centered @ centered)
    for season in range(2024, 2027):
        rolling_y_ss = float(
            json.loads(
                (MATRIX_ROOT / f"5y_end_{season}/manifest.json").read_text()
            )["train"]["centered_y_sum_squares"]
        )
        centered_y_ss[season] = rolling_y_ss - sum(
            centered_y_ss[value] for value in range(season - 4, season)
        )
        if centered_y_ss[season] <= 0:
            raise ValueError(f"Recovered annual y'y is invalid for {season}.")
    return annual, centered_y_ss, reconstruction


def _measurement_panel(
    annual: dict, centered_y_ss: dict[int, float], targets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    quality_rows = []
    for season in SEASONS:
        bundle = annual[season]
        n = len(bundle.players)
        penalty = np.concatenate([np.full(n, 3000.0), np.full(n, 3000.0), np.asarray([300.0])])
        xtx = bundle.xtx.toarray().astype(float)
        system = xtx.copy()
        system.flat[:: system.shape[0] + 1] += penalty
        factor = cho_factor(system, lower=True, check_finite=False)
        inverse = cho_solve(factor, np.eye(len(system)), check_finite=False)
        beta = cho_solve(factor, bundle.xty_centered, check_finite=False)
        residual_ss = float(
            centered_y_ss[season]
            - 2.0 * beta @ bundle.xty_centered
            + beta @ xtx @ beta
        )
        effective_df = float(np.sum(xtx * inverse.T))
        residual_variance = residual_ss / max(bundle.rows - effective_df, 1.0)
        if residual_variance <= 0:
            raise ValueError(f"Annual residual variance is invalid for {season}.")
        covariance = residual_variance * (inverse @ xtx @ inverse)
        off_weight = np.zeros(2 * n + 1)
        def_weight = np.zeros(2 * n + 1)
        off_weight[:n] = bundle.off_possessions / bundle.off_possessions.sum()
        def_weight[n : 2 * n] = bundle.def_possessions / bundle.def_possessions.sum()
        cov_off_mean = covariance @ off_weight
        cov_def_mean = covariance @ def_weight
        var_off_mean = float(off_weight @ cov_off_mean)
        var_def_mean = float(def_weight @ cov_def_mean)
        cov_means = float(off_weight @ cov_def_mean)
        off_variance = np.diag(covariance)[:n] + var_off_mean - 2.0 * cov_off_mean[:n]
        def_variance = np.diag(covariance)[n : 2 * n] + var_def_mean - 2.0 * cov_def_mean[n : 2 * n]
        raw_cross = covariance[np.arange(n), n + np.arange(n)]
        centered_raw_cross = raw_cross - cov_def_mean[:n] - cov_off_mean[n : 2 * n] + cov_means
        published_cross = -centered_raw_cross
        off_variance = np.clip(off_variance * 10_000.0, 1e-9, None)
        def_variance = np.clip(def_variance * 10_000.0, 1e-9, None)
        published_cross *= 10_000.0
        limit = 0.99 * np.sqrt(off_variance * def_variance)
        published_cross = np.clip(published_cross, -limit, limit)
        zero = np.zeros(2 * n + 1)
        fitted, _ = _solve(bundle, zero, scale=0.0)
        point = pd.DataFrame(
            {
                "PLAYER_ID": bundle.players,
                "Season": season,
                "recomputed_offense": 100.0 * fitted[:n],
                "recomputed_defense": -100.0 * fitted[n : 2 * n],
                "observation_variance_offense": off_variance,
                "observation_variance_defense": def_variance,
                "observation_covariance_offense_defense": published_cross,
            }
        )
        target = targets.loc[targets["Season"].eq(season)]
        point = point.merge(
            target[["PLAYER_ID", "Season", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]],
            on=["PLAYER_ID", "Season"],
            how="inner",
            validate="one_to_one",
        )
        error = max(
            float((point["recomputed_offense"] - point["target_offense"]).abs().max()),
            float((point["recomputed_defense"] - point["target_defense"]).abs().max()),
        )
        if error > 1e-6:
            raise ValueError(f"Annual point estimate mismatch in {season}: {error}")
        rows.append(point.drop(columns=["recomputed_offense", "recomputed_defense"]))
        quality_rows.append(
            {
                "Season": season,
                "players": len(point),
                "maximum_point_estimate_error": error,
                "median_observation_correlation": float(
                    np.median(published_cross / np.sqrt(off_variance * def_variance))
                ),
                "residual_variance": residual_variance,
                "effective_degrees_of_freedom": effective_df,
                "observation_scale_status": "recovered_annual_homoskedastic_ridge",
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(quality_rows)


def _metrics(
    trajectory: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    origins: tuple[int, ...],
    candidate: str,
    scope: str,
) -> pd.DataFrame:
    rows = []
    eligible = targets.loc[np.minimum(targets["Poss_Off"], targets["Poss_Def"]).ge(1000)]
    for origin in origins:
        forecast = trajectory.loc[
            trajectory["Season"].eq(origin),
            ["PLAYER_ID", "forecast_next_offense", "forecast_next_defense", "forecast_next_net"],
        ]
        actual = eligible.loc[
            eligible["Season"].eq(origin + 1),
            ["PLAYER_ID", "target_offense", "target_defense", "target_net"],
        ]
        merged = forecast.merge(actual, on="PLAYER_ID", how="inner", validate="one_to_one")
        for component in ("offense", "defense", "net"):
            truth = merged[f"target_{component}"].to_numpy(dtype=float)
            prediction = merged[f"forecast_next_{component}"].to_numpy(dtype=float)
            rows.append(
                {
                    "candidate": candidate,
                    "scope": scope,
                    "origin_season": origin,
                    "target_season": origin + 1,
                    "component": component,
                    "players": len(merged),
                    "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
                    "correlation": float(np.corrcoef(truth, prediction)[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def _paired_predictions(
    challenger: pd.DataFrame,
    baseline: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    origins: tuple[int, ...],
    scope: str,
) -> pd.DataFrame:
    eligible = targets.loc[
        np.minimum(targets["Poss_Off"], targets["Poss_Def"]).ge(1000)
    ]
    rows = []
    for origin in origins:
        actual = eligible.loc[
            eligible["Season"].eq(origin + 1), ["PLAYER_ID", "target_net"]
        ]
        left = challenger.loc[
            challenger["Season"].eq(origin), ["PLAYER_ID", "forecast_next_net"]
        ].rename(columns={"forecast_next_net": "challenger"})
        right = baseline.loc[
            baseline["Season"].eq(origin), ["PLAYER_ID", "forecast_next_net"]
        ].rename(columns={"forecast_next_net": "baseline"})
        frame = actual.merge(left, on="PLAYER_ID", validate="one_to_one").merge(
            right, on="PLAYER_ID", validate="one_to_one"
        )
        frame["origin_season"] = origin
        frame["target_season"] = origin + 1
        frame["scope"] = scope
        frame["challenger_squared_error"] = (
            frame["target_net"] - frame["challenger"]
        ) ** 2
        frame["baseline_squared_error"] = (
            frame["target_net"] - frame["baseline"]
        ) ** 2
        frame["mse_delta"] = (
            frame["challenger_squared_error"] - frame["baseline_squared_error"]
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _paired_bootstrap(frame: pd.DataFrame, *, draws: int, seed: int) -> dict:
    seasons = [group["mse_delta"].to_numpy(dtype=float) for _, group in frame.groupby("target_season")]
    point = float(np.mean([values.mean() for values in seasons]))
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        samples[draw] = np.mean(
            [values[rng.integers(0, len(values), len(values))].mean() for values in seasons]
        )
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "mean_mse_delta": point,
        "bootstrap_95_low": float(low),
        "bootstrap_95_high": float(high),
        "probability_challenger_lower_mse": float(np.mean(samples < 0)),
        "draws": draws,
    }


def main() -> None:
    contract = _load_contract()
    targets = pd.read_parquet(TARGETS)
    if targets["Season"].max() >= 2027:
        raise ValueError("Season 2027 entered the bivariate experiment.")
    targets = targets.loc[targets["Season"].isin(SEASONS)].copy()
    annual, centered_y_ss, reconstruction = _annual_bundles()
    measurements, measurement_quality = _measurement_panel(
        annual, centered_y_ss, targets
    )
    trajectories = {}
    metrics = []
    independent = filter_bivariate_annual_rapm(
        measurements,
        phi=PHI,
        process_sd=PROCESS_SD,
        process_correlation=0.0,
        use_observation_covariance=False,
    )
    trajectories["independent"] = independent
    covariance_only = filter_bivariate_annual_rapm(
        measurements,
        phi=PHI,
        process_sd=PROCESS_SD,
        process_correlation=0.0,
        use_observation_covariance=True,
    )
    trajectories["observation_covariance_only"] = covariance_only
    for candidate, trajectory in trajectories.items():
        metrics.append(_metrics(trajectory, targets, origins=SELECTION_ORIGINS, candidate=candidate, scope="selection"))
        metrics.append(_metrics(trajectory, targets, origins=DIAGNOSTIC_ORIGINS, candidate=candidate, scope="diagnostic"))
    selection_rows = []
    rho_trajectories = {}
    for rho in PROCESS_CORRELATIONS:
        candidate = f"bivariate_rho_{rho:+.2f}"
        trajectory = filter_bivariate_annual_rapm(
            measurements,
            phi=PHI,
            process_sd=PROCESS_SD,
            process_correlation=rho,
            use_observation_covariance=True,
        )
        rho_trajectories[rho] = trajectory
        selected_metrics = _metrics(trajectory, targets, origins=SELECTION_ORIGINS, candidate=candidate, scope="selection")
        diagnostic_metrics = _metrics(trajectory, targets, origins=DIAGNOSTIC_ORIGINS, candidate=candidate, scope="diagnostic")
        metrics.extend([selected_metrics, diagnostic_metrics])
        net = selected_metrics.loc[selected_metrics["component"].eq("net")]
        selection_rows.append(
            {
                "process_correlation": rho,
                "mean_selection_net_rmse": float(net["rmse"].mean()),
                "mean_selection_net_correlation": float(net["correlation"].mean()),
            }
        )
    selection = pd.DataFrame(selection_rows).sort_values(
        ["mean_selection_net_rmse", "process_correlation"], kind="stable"
    )
    selected_rho = float(selection.iloc[0]["process_correlation"])
    selected = rho_trajectories[selected_rho].copy()
    selected["candidate"] = "selected_bivariate"
    paired = pd.concat(
        [
            _paired_predictions(
                selected,
                independent,
                targets,
                origins=SELECTION_ORIGINS,
                scope="selection",
            ),
            _paired_predictions(
                selected,
                independent,
                targets,
                origins=DIAGNOSTIC_ORIGINS,
                scope="diagnostic",
            ),
        ],
        ignore_index=True,
    )
    paired_intervals = {
        scope: _paired_bootstrap(
            paired.loc[paired["scope"].eq(scope)], draws=5000, seed=20260829
        )
        for scope in ("selection", "diagnostic")
    }
    metric_frame = pd.concat(metrics, ignore_index=True)
    summary = (
        metric_frame.groupby(["candidate", "scope", "component"], as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_correlation=("correlation", "mean"), folds=("origin_season", "nunique"))
    )
    sources = {
        "contract": CONTRACT,
        "targets": TARGETS,
        "runner": Path(__file__),
        **{f"matrix_{season}": MATRIX_ROOT / f"5y_end_{season}/manifest.json" for season in range(2024, 2027)},
        **{f"possessions_{season}": POSSESSION_CACHE / f"matchups_{season}.parquet" for season in range(2019, 2024)},
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "phi": PHI,
        "process_sd": PROCESS_SD,
        "process_correlation_grid": list(PROCESS_CORRELATIONS),
        "selected_process_correlation": selected_rho,
        "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in sources.items()},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/bivariate_annual_state_space" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "observation_covariance.parquet": measurements,
        "measurement_quality.parquet": measurement_quality,
        "selected_trajectory.parquet": selected,
        "independent_trajectory.parquet": independent,
        "paired_predictions.parquet": paired,
        "candidate_metrics.parquet": metric_frame,
        "summary.parquet": summary,
        "selection.parquet": selection,
        "annual_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    selected_name = f"bivariate_rho_{selected_rho:+.2f}"
    selection_net = summary.loc[(summary["candidate"].eq(selected_name)) & (summary["scope"].eq("selection")) & (summary["component"].eq("net"))].iloc[0]
    diagnostic_net = summary.loc[(summary["candidate"].eq(selected_name)) & (summary["scope"].eq("diagnostic")) & (summary["component"].eq("net"))].iloc[0]
    baseline_diagnostic = summary.loc[(summary["candidate"].eq("independent")) & (summary["scope"].eq("diagnostic")) & (summary["component"].eq("net"))].iloc[0]
    diagnostic_interval = paired_intervals["diagnostic"]
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_bivariate_challenger" if diagnostic_interval["bootstrap_95_high"] < 0 else "research_null",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "observation_rows": len(measurements),
            "maximum_point_estimate_error": float(measurement_quality["maximum_point_estimate_error"].max()),
            "maximum_component_identity_error": float((selected["filtered_offense"] + selected["filtered_defense"] - selected["filtered_net"]).abs().max()),
            "season_2027_loaded": False,
        },
        "results": {
            "selected_process_correlation": selected_rho,
            "selection_net_rmse": float(selection_net["mean_rmse"]),
            "diagnostic_net_rmse": float(diagnostic_net["mean_rmse"]),
            "independent_diagnostic_net_rmse": float(baseline_diagnostic["mean_rmse"]),
            "paired_mse_intervals": paired_intervals,
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {"path": name, "sha256": sha256_file(output / name), "rows": len(frame)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(selection.to_string(index=False))
    print(summary.loc[summary["component"].eq("net")].sort_values(["scope", "mean_rmse"]).to_string(index=False))
    print(json.dumps(run["results"], indent=2))


if __name__ == "__main__":
    main()
