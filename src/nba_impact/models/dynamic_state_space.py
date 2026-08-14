"""Causal annual state-space challenger for normal-RAPM trajectories.

This is a deliberately small Kalman filter, not a player forecast model. Annual
normal RAPM is the noisy observation; a side-specific AR(1) latent state is the
filtered current-strength proxy. The model uses only the named season and past.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import RapmConfig, build_design, load_current_possessions, load_legacy_possessions
from nba_impact.models.rapm_uncertainty import game_cluster_sandwich


COMPONENTS = ("offense", "defense")


def build_annual_observation_variance(
    targets_path: str | Path,
    legacy_cache_dir: str | Path,
    current_possessions_path: str | Path,
    current_segments_path: str | Path,
    *,
    artifact_root: str | Path,
    transition_season: int = 2024,
    seasons: tuple[int, ...] | None = None,
) -> dict:
    """Derive annual side variance from the fast CR0 game-cluster diagnostic."""
    targets = pd.read_parquet(targets_path)
    required = {"PLAYER_ID", "Season", "target_offense", "target_defense", "target_net"}
    if missing := sorted(required - set(targets.columns)):
        raise ValueError(f"Annual target panel is missing columns: {missing}.")
    targets = targets.copy()
    targets["PLAYER_ID"] = pd.to_numeric(targets["PLAYER_ID"], errors="raise").astype(int)
    targets["Season"] = pd.to_numeric(targets["Season"], errors="raise").astype(int)
    available_seasons = tuple(sorted(int(value) for value in targets["Season"].unique()))
    requested_seasons = available_seasons if seasons is None else tuple(sorted(set(seasons)))
    if missing := sorted(set(requested_seasons) - set(available_seasons)):
        raise ValueError(f"Requested annual observation-variance seasons are unavailable: {missing}.")
    identity = hashlib.sha256((sha256_file(targets_path) + str(transition_season)).encode()).hexdigest()[:10]
    run_id = f"annual_rapm_observation_variance_v1_{identity}"
    output = Path(artifact_root) / "models" / "annual_rapm_observation_variance" / run_id
    checkpoint_dir = output / "season_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Legacy-only batches must not load the much larger current canonical panel.
    # Besides saving substantial local work, this keeps `--seasons 2016` a real
    # one-season operation rather than a hidden all-source scan.
    needs_current = any(season >= transition_season for season in requested_seasons)
    current = (
        load_current_possessions(
            current_possessions_path,
            current_segments_path,
            lineup_policy="terminal",
            game_types=("regular",),
        )
        if needs_current
        else None
    )
    rows: list[pd.DataFrame] = []
    quality: list[dict] = []
    for season in requested_seasons:
        checkpoint = checkpoint_dir / f"season={season}.parquet"
        if checkpoint.exists():
            rows.append(pd.read_parquet(checkpoint))
            quality.append({"season": season, "source": "checkpoint", "games": None, "players": int(len(rows[-1])), "maximum_point_estimate_difference": None})
            continue
        if season < transition_season:
            frame = load_legacy_possessions(legacy_cache_dir, (season,), game_types=("regular",))
            source = "legacy"
        else:
            if current is None:  # pragma: no cover - guarded by needs_current above.
                raise RuntimeError("Current possessions were not loaded for a canonical season.")
            frame = current.loc[current["season"].eq(season)].copy()
            source = "canonical_current"
        if frame.empty:
            raise ValueError(f"Annual observation variance has no possessions for {season}.")
        design = build_design(frame)
        covariance, beta, _ = game_cluster_sandwich(
            design,
            RapmConfig(
                seasons=(season,),
                lambda_off=3000.0,
                lambda_def=3000.0,
                lambda_home=300.0,
                game_types=("regular",),
                data_scope=f"{source}_annual_observation_variance",
            ),
        )
        n_players = len(design.players)
        table = pd.DataFrame({"PLAYER_ID": design.players, "Season": season})
        table["observation_variance_offense"] = np.clip(np.diag(covariance)[:n_players] * 10_000.0, 1e-12, None)
        table["observation_variance_defense"] = np.clip(np.diag(covariance)[n_players:2 * n_players] * 10_000.0, 1e-12, None)
        table["observation_covariance_offense_defense"] = -np.diag(covariance[:n_players, n_players:2 * n_players]) * 10_000.0
        table["observation_variance_net"] = np.clip(
            table["observation_variance_offense"]
            + table["observation_variance_defense"]
            + 2.0 * table["observation_covariance_offense_defense"],
            1e-12,
            None,
        )
        table["variance_method"] = "analytic_game_cluster_ridge_cr0_diagnostic"
        table["annual_target_source"] = source
        target = targets.loc[targets["Season"].eq(season), ["PLAYER_ID", "target_offense", "target_defense", "target_net"]]
        merged = table.merge(target, on="PLAYER_ID", how="inner", validate="one_to_one")
        if len(merged) != len(target):
            raise ValueError(f"Annual observation variance player coverage differs for {season}.")
        recomputed = np.concatenate([beta[:n_players, None] * 100.0, -beta[n_players:2 * n_players, None] * 100.0], axis=1)
        expected = merged.set_index("PLAYER_ID").loc[design.players, ["target_offense", "target_defense"]].to_numpy()
        max_difference = float(np.abs(recomputed - expected).max())
        if max_difference > 1e-8:
            raise ValueError(f"Annual variance RAPM point estimate mismatch for {season}: {max_difference}.")
        rows.append(table)
        table.to_parquet(checkpoint.with_suffix(".parquet.partial"), index=False)
        checkpoint.with_suffix(".parquet.partial").replace(checkpoint)
        quality.append({"season": season, "source": source, "games": int(frame["gameid"].nunique()), "players": int(n_players), "maximum_point_estimate_difference": max_difference})
    output_frame = pd.concat(rows, ignore_index=True)
    output_frame.to_parquet(output / "observation_variance.parquet", index=False)
    pd.DataFrame(quality).to_parquet(output / "season_quality.parquet", index=False)
    missing_seasons = sorted(set(available_seasons) - set(output_frame["Season"]))
    max_difference = [item["maximum_point_estimate_difference"] for item in quality if item["maximum_point_estimate_difference"] is not None]
    run = {"run_id": run_id, "model_family": "annual_normal_rapm_observation_variance", "status": "research_measurement_input_complete" if not missing_seasons else "research_measurement_input_partial", "created_at": datetime.now(timezone.utc).isoformat(), "config": {"transition_season": transition_season, "target_sha256": sha256_file(targets_path), "legacy_cache_dir": str(Path(legacy_cache_dir).resolve()), "current_possessions_sha256": sha256_file(current_possessions_path), "current_segments_sha256": sha256_file(current_segments_path), "builder_sha256": sha256_file(Path(__file__))}, "quality": {"rows": int(len(output_frame)), "duplicate_keys": int(output_frame.duplicated(["PLAYER_ID", "Season"]).sum()), "maximum_point_estimate_difference": float(max(max_difference)) if max_difference else None, "missing_seasons": missing_seasons}, "metrics": {"season_quality": quality}, "artifact_path": str(output.resolve()), "observation_variance_path": str((output / "observation_variance.parquet").resolve()), "caveats": ["CR0 covariance is a fast observation-noise diagnostic, not publication uncertainty.", "Ridge bias and lineup connectivity remain outside this variance input.", "Season checkpoints are resumable; do not run the state-space model until all annual seasons are present." ]}
    write_json_atomic(run, output / "run.json")
    return run


def _validate_inputs(targets: pd.DataFrame, observation_variance: pd.DataFrame) -> pd.DataFrame:
    target_required = {"PLAYER_ID", "Season", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"}
    variance_required = {"PLAYER_ID", "Season", "observation_variance_offense", "observation_variance_defense"}
    if missing := sorted(target_required - set(targets.columns)):
        raise ValueError(f"State-space targets are missing columns: {missing}.")
    if missing := sorted(variance_required - set(observation_variance.columns)):
        raise ValueError(f"State-space observation variance is missing columns: {missing}.")
    values = targets.merge(
        observation_variance.loc[:, list(variance_required)],
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    ).copy()
    if values[["observation_variance_offense", "observation_variance_defense"]].isna().any().any():
        raise ValueError("Every annual RAPM target needs side-specific observation variance.")
    numeric = [
        "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def",
        "observation_variance_offense", "observation_variance_defense",
    ]
    values[numeric] = values[numeric].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(values[numeric].to_numpy()).all():
        raise ValueError("State-space inputs contain non-finite values.")
    if (values[["Poss_Off", "Poss_Def", "observation_variance_offense", "observation_variance_defense"]] <= 0).any().any():
        raise ValueError("State-space inputs require positive exposure and observation variance.")
    if values.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("State-space inputs have duplicate player-season keys.")
    error = np.abs(values["target_net"] - values["target_offense"] - values["target_defense"])
    if float(error.max()) > 1e-9:
        raise ValueError("State-space targets must satisfy net = offense + defense.")
    return values.sort_values(["PLAYER_ID", "Season"], kind="stable").reset_index(drop=True)


def _transition_variance(phi: float, process_variance: float, gap: int) -> float:
    if gap < 1:
        raise ValueError("Player seasons must be strictly increasing.")
    if phi == 1.0:
        return process_variance * gap
    return process_variance * (1.0 - phi ** (2 * gap)) / (1.0 - phi**2)


def build_causal_state_space_filter(
    targets: pd.DataFrame,
    observation_variance: pd.DataFrame,
    *,
    phi: float,
    process_sd: float,
) -> pd.DataFrame:
    """Filter each annual component using only observations through that year."""
    if not 0 < phi <= 1:
        raise ValueError("phi must be in (0, 1].")
    if process_sd <= 0:
        raise ValueError("process_sd must be positive.")
    values = _validate_inputs(targets, observation_variance)
    process_variance = float(process_sd**2)
    rows: list[dict] = []
    for player_id, player in values.groupby("PLAYER_ID", sort=False):
        state_mean = {side: 0.0 for side in COMPONENTS}
        # Stationary initial variance makes the first annual observation a
        # regularized update rather than an assumed exact player state.
        state_variance = {
            side: process_variance / (1.0 - phi**2) if phi < 1 else process_variance * 20.0
            for side in COMPONENTS
        }
        previous_season: int | None = None
        for row in player.itertuples(index=False):
            season = int(row.Season)
            gap = 1 if previous_season is None else season - previous_season
            if previous_season is not None and gap <= 0:
                raise ValueError("Player seasons must be strictly increasing.")
            record = {"PLAYER_ID": int(player_id), "Season": season, "previous_observed_season": previous_season, "season_gap": gap}
            for side in COMPONENTS:
                predicted_mean = phi**gap * state_mean[side]
                predicted_variance = phi ** (2 * gap) * state_variance[side] + _transition_variance(phi, process_variance, gap)
                observation = float(getattr(row, f"target_{side}"))
                observation_variance_side = float(getattr(row, f"observation_variance_{side}"))
                gain = predicted_variance / (predicted_variance + observation_variance_side)
                state_mean[side] = predicted_mean + gain * (observation - predicted_mean)
                state_variance[side] = (1.0 - gain) * predicted_variance
                record[f"annual_{side}"] = observation
                record[f"observation_variance_{side}"] = observation_variance_side
                record[f"filtered_{side}"] = state_mean[side]
                record[f"filtered_variance_{side}"] = state_variance[side]
                record[f"kalman_gain_{side}"] = gain
            record["annual_net"] = float(row.target_net)
            record["Poss_Off"] = float(row.Poss_Off)
            record["Poss_Def"] = float(row.Poss_Def)
            record["filtered_net"] = record["filtered_offense"] + record["filtered_defense"]
            rows.append(record)
            previous_season = season
    output = pd.DataFrame(rows)
    output["trajectory_method"] = "causal_annual_ar1_state_space_v1"
    output["phi"] = float(phi)
    output["process_sd"] = float(process_sd)
    identity_error = np.abs(output["filtered_net"] - output["filtered_offense"] - output["filtered_defense"])
    if float(identity_error.max()) > 1e-12:
        raise AssertionError("State-space net must equal offense plus defense.")
    return output


def _forward_metrics(trajectory: pd.DataFrame, targets: pd.DataFrame, *, origins: tuple[int, ...], model: str, minimum_side_possessions: float) -> pd.DataFrame:
    rows: list[dict] = []
    eligible = targets.loc[targets[["Poss_Off", "Poss_Def"]].min(axis=1).ge(minimum_side_possessions)]
    for origin in origins:
        predictions = trajectory.loc[trajectory["Season"].eq(origin)]
        actual = eligible.loc[eligible["Season"].eq(origin + 1)]
        merged = predictions.merge(actual, on="PLAYER_ID", suffixes=("_origin", "_next"), validate="one_to_one")
        prediction = merged["annual_net"] if model == "latest_annual" else merged["filtered_net"]
        target = merged["target_net"].to_numpy(dtype=float)
        values = prediction.to_numpy(dtype=float)
        rows.append({
            "model": model,
            "origin_season": origin,
            "target_season": origin + 1,
            "players": int(len(merged)),
            "net_rmse": float(np.sqrt(np.mean((target - values) ** 2))) if len(merged) else np.nan,
            "net_correlation": float(np.corrcoef(target, values)[0, 1]) if len(merged) >= 2 and np.std(target) > 0 and np.std(values) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def build_state_space_trajectory(
    targets_path: str | Path,
    observation_variance_path: str | Path,
    names_path: str | Path,
    *,
    artifact_root: str | Path,
    candidate_phis: tuple[float, ...] = (0.50, 0.65, 0.80, 0.90),
    candidate_process_sds: tuple[float, ...] = (0.25, 0.50, 1.00, 2.00),
    selection_origins: tuple[int, ...] = (2018, 2019, 2020, 2021),
    diagnostic_origins: tuple[int, ...] = (2022, 2023),
    minimum_side_possessions: float = 1000.0,
) -> dict:
    """Select the state-space challenger only on historical selection origins."""
    targets = pd.read_parquet(targets_path)
    variance = pd.read_parquet(observation_variance_path)
    values = _validate_inputs(targets, variance)
    available = set(values["Season"])
    required = set(selection_origins) | {year + 1 for year in selection_origins}
    required |= set(diagnostic_origins) | {year + 1 for year in diagnostic_origins}
    if missing := sorted(required - available):
        raise ValueError(f"State-space evaluation needs missing annual seasons: {missing}.")
    candidates = []
    for phi in candidate_phis:
        for process_sd in candidate_process_sds:
            trajectory = build_causal_state_space_filter(values, variance, phi=phi, process_sd=process_sd)
            metrics = _forward_metrics(trajectory, values, origins=selection_origins, model="state_space", minimum_side_possessions=minimum_side_possessions)
            candidates.append({"phi": phi, "process_sd": process_sd, "mean_net_rmse": float(metrics["net_rmse"].mean()), "mean_net_correlation": float(metrics["net_correlation"].mean())})
    candidate_table = pd.DataFrame(candidates).sort_values(["mean_net_rmse", "phi", "process_sd"], kind="stable").reset_index(drop=True)
    chosen = candidate_table.iloc[0]
    trajectory = build_causal_state_space_filter(values, variance, phi=float(chosen.phi), process_sd=float(chosen.process_sd))
    baseline = _forward_metrics(trajectory, values, origins=selection_origins, model="latest_annual", minimum_side_possessions=minimum_side_possessions)
    selection = _forward_metrics(trajectory, values, origins=selection_origins, model="state_space", minimum_side_possessions=minimum_side_possessions)
    diagnostic_baseline = _forward_metrics(trajectory, values, origins=diagnostic_origins, model="latest_annual", minimum_side_possessions=minimum_side_possessions)
    diagnostic = _forward_metrics(trajectory, values, origins=diagnostic_origins, model="state_space", minimum_side_possessions=minimum_side_possessions)
    names = pd.read_csv(names_path, usecols=["PLAYER_ID", "PLAYER_NAME"])
    names["PLAYER_ID"] = pd.to_numeric(names["PLAYER_ID"], errors="raise").astype(int)
    trajectory = trajectory.merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    config = {"candidate_phis": list(candidate_phis), "candidate_process_sds": list(candidate_process_sds), "selection_origins": list(selection_origins), "diagnostic_origins": list(diagnostic_origins), "minimum_side_possessions": minimum_side_possessions, "selected_phi": float(chosen.phi), "selected_process_sd": float(chosen.process_sd), "targets_sha256": sha256_file(targets_path), "observation_variance_sha256": sha256_file(observation_variance_path), "names_sha256": sha256_file(names_path), "builder_sha256": sha256_file(Path(__file__))}
    run_id = f"annual_state_space_trajectory_v1_{hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]}"
    output = Path(artifact_root) / "models" / "annual_state_space_trajectory" / run_id
    output.mkdir(parents=True, exist_ok=False)
    trajectory.to_parquet(output / "trajectories.parquet", index=False)
    candidate_table.to_parquet(output / "selection_candidates.parquet", index=False)
    forward = pd.concat([baseline.assign(scope="selection"), selection.assign(scope="selection"), diagnostic_baseline.assign(scope="diagnostic"), diagnostic.assign(scope="diagnostic")], ignore_index=True)
    forward.to_parquet(output / "forward_metrics.parquet", index=False)
    run = {"run_id": run_id, "model_family": "causal_annual_ar1_state_space_trajectory", "estimand_id": "current_latent_strength_v1", "estimand": "filtered_end_of_season_latent_strength_proxy_from_annual_normal_rapm", "status": "research_state_space_challenger", "evidence_status": "reused_annual_target_evaluation", "created_at": datetime.now(timezone.utc).isoformat(), "config": config, "metrics": {"selection_latest_annual_net_rmse": float(baseline.net_rmse.mean()), "selection_state_space_net_rmse": float(selection.net_rmse.mean()), "diagnostic_latest_annual_net_rmse": float(diagnostic_baseline.net_rmse.mean()), "diagnostic_state_space_net_rmse": float(diagnostic.net_rmse.mean()), "selection_state_space_net_correlation": float(selection.net_correlation.mean()), "diagnostic_state_space_net_correlation": float(diagnostic.net_correlation.mean()), "maximum_component_identity_error": float(np.abs(trajectory.filtered_net - trajectory.filtered_offense - trajectory.filtered_defense).max())}, "artifact_path": str(output.resolve()), "trajectories_path": str((output / "trajectories.parquet").resolve()), "caveats": ["This is a causal filtered strength proxy, not retrospective annual impact or a forecast.", "Annual observation variance is an analytic game-cluster RAPM covariance diagnostic, not publication uncertainty.", "Selection and diagnostics reuse historical annual targets; Season 2027 is not included."], "forbidden_interpretation": "Confirmed latent ability, causal value, forecast certainty, or production rating."}
    write_json_atomic(run, output / "run.json")
    return run
