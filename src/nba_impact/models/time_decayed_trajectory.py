"""Filtered annual normal-RAPM trajectories with no future-season leakage.

This is deliberately a small time-decay baseline, not a latent-state model.
It consumes annual zero-prior normal-RAPM estimates and smooths each observed
player-season only with that season and earlier observations.  A later
state-space model must beat this baseline on a frozen forward evaluation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic


COMPONENTS = ("offense", "defense", "net")
COMPONENT_TARGETS = {
    "offense": "target_offense",
    "defense": "target_defense",
    "net": "target_net",
}
COMPONENT_EXPOSURE = {
    "offense": "Poss_Off",
    "defense": "Poss_Def",
}


def _validate_targets(targets: pd.DataFrame) -> pd.DataFrame:
    required = {
        "PLAYER_ID",
        "Season",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
    }
    if missing := sorted(required - set(targets.columns)):
        raise ValueError(f"Annual normal-RAPM targets are missing columns: {missing}.")
    frame = targets.copy()
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="raise").astype(int)
    frame["Season"] = pd.to_numeric(frame["Season"], errors="raise").astype(int)
    numeric = [*COMPONENT_TARGETS.values(), "Poss_Off", "Poss_Def"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if frame.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual normal-RAPM targets have duplicate player-season keys.")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("Annual normal-RAPM targets contain non-finite values.")
    if (frame[["Poss_Off", "Poss_Def"]] <= 0).any().any():
        raise ValueError("Annual normal-RAPM targets require positive side possessions.")
    identity_error = np.abs(
        frame["target_net"] - frame["target_offense"] - frame["target_defense"]
    )
    if float(identity_error.max()) > 1e-9:
        raise ValueError("Annual normal-RAPM target components do not sum to net.")
    return frame.sort_values(["PLAYER_ID", "Season"], kind="stable").reset_index(drop=True)


def build_filtered_time_decay(
    targets: pd.DataFrame,
    *,
    decay: float,
    exposure_power: float,
) -> pd.DataFrame:
    """Smooth annual ratings using only the current and earlier player seasons.

    For player ``i`` in season ``T`` and a component-specific possession count
    ``n`` the filtered value is:

    ``sum_t<=T decay**(T-t) * n_it**exposure_power * annual_it / sum weights``.

    The net output is constructed from filtered offense plus filtered defense,
    not independently smoothed, so the component identity is exact.
    """
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1].")
    if not 0 <= exposure_power <= 1:
        raise ValueError("exposure_power must be in [0, 1].")
    frame = _validate_targets(targets)
    rows: list[dict] = []
    for player_id, player in frame.groupby("PLAYER_ID", sort=False):
        last_season: int | None = None
        weighted_sum = {"offense": 0.0, "defense": 0.0}
        weight_total = {"offense": 0.0, "defense": 0.0}
        observations = 0
        for row in player.itertuples(index=False):
            season = int(row.Season)
            elapsed = 0 if last_season is None else season - last_season
            if last_season is not None and elapsed <= 0:
                raise ValueError("Player seasons must be strictly increasing.")
            historical_multiplier = decay**elapsed
            for component in ("offense", "defense"):
                weighted_sum[component] *= historical_multiplier
                weight_total[component] *= historical_multiplier
                annual_value = float(getattr(row, COMPONENT_TARGETS[component]))
                exposure = float(getattr(row, COMPONENT_EXPOSURE[component]))
                weight = exposure**exposure_power
                weighted_sum[component] += weight * annual_value
                weight_total[component] += weight
            observations += 1
            output = {
                "PLAYER_ID": int(player_id),
                "Season": season,
                "annual_offense": float(row.target_offense),
                "annual_defense": float(row.target_defense),
                "annual_net": float(row.target_net),
                "Poss_Off": float(row.Poss_Off),
                "Poss_Def": float(row.Poss_Def),
                "previous_observed_season": last_season,
                "season_gap": elapsed,
                "observed_seasons": observations,
                "effective_weight_offense": weight_total["offense"],
                "effective_weight_defense": weight_total["defense"],
                "filtered_offense": weighted_sum["offense"] / weight_total["offense"],
                "filtered_defense": weighted_sum["defense"] / weight_total["defense"],
            }
            output["filtered_net"] = (
                output["filtered_offense"] + output["filtered_defense"]
            )
            rows.append(output)
            last_season = season
    output = pd.DataFrame(rows).sort_values(["PLAYER_ID", "Season"], kind="stable")
    output["trajectory_method"] = "filtered_exposure_weighted_time_decay_v1"
    output["decay"] = float(decay)
    output["exposure_power"] = float(exposure_power)
    identity_error = np.abs(
        output["filtered_net"] - output["filtered_offense"] - output["filtered_defense"]
    )
    if float(identity_error.max()) > 1e-12:
        raise AssertionError("Filtered trajectory components do not sum to net.")
    return output.reset_index(drop=True)


def _component_metrics(
    trajectory: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    origins: tuple[int, ...],
    model: str,
    minimum_side_possessions: float,
) -> pd.DataFrame:
    records: list[dict] = []
    observed = _validate_targets(targets)
    eligible = observed.loc[
        observed[["Poss_Off", "Poss_Def"]].min(axis=1).ge(minimum_side_possessions)
    ].copy()
    for origin in origins:
        current = trajectory.loc[trajectory["Season"].eq(origin)].copy()
        next_year = eligible.loc[eligible["Season"].eq(origin + 1)].copy()
        merged = current.merge(
            next_year,
            on="PLAYER_ID",
            suffixes=("_origin", "_next"),
            validate="one_to_one",
        )
        for component in COMPONENTS:
            prediction = (
                merged[f"annual_{component}"]
                if model == "latest_annual"
                else merged[f"filtered_{component}"]
            ).to_numpy(dtype=float)
            actual = merged[f"target_{component}"].to_numpy(dtype=float)
            count = len(merged)
            records.append(
                {
                    "model": model,
                    "origin_season": int(origin),
                    "target_season": int(origin + 1),
                    "component": component,
                    "players": int(count),
                    "rmse": float(np.sqrt(np.mean((actual - prediction) ** 2))) if count else np.nan,
                    "correlation": (
                        float(np.corrcoef(actual, prediction)[0, 1])
                        if count >= 2 and np.std(actual) > 0 and np.std(prediction) > 0
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _mean_season_metric(metrics: pd.DataFrame, *, model: str, component: str, metric: str) -> float:
    rows = metrics.loc[
        metrics["model"].eq(model) & metrics["component"].eq(component), metric
    ]
    return float(rows.mean()) if len(rows) else np.nan


def build_time_decayed_trajectory(
    targets_path: str | Path,
    names_path: str | Path,
    *,
    artifact_root: str | Path,
    candidate_decays: tuple[float, ...] = (0.50, 0.65, 0.80, 0.90),
    candidate_exposure_powers: tuple[float, ...] = (0.0, 0.5, 1.0),
    selection_origins: tuple[int, ...] = (2018, 2019, 2020, 2021),
    diagnostic_origins: tuple[int, ...] = (2022, 2023),
    minimum_side_possessions: float = 1000.0,
) -> dict:
    """Select a simple filtered trajectory only on predeclared past origins."""
    if not candidate_decays or not candidate_exposure_powers:
        raise ValueError("At least one decay and exposure-power candidate is required.")
    source_path = Path(targets_path)
    names_source = Path(names_path)
    raw_targets = pd.read_parquet(source_path)
    raw_exposure = raw_targets[["Poss_Off", "Poss_Def"]].apply(
        pd.to_numeric, errors="raise"
    )
    invalid_side_exposure_rows = int((raw_exposure <= 0).any(axis=1).sum())
    targets = _validate_targets(
        raw_targets.loc[(raw_exposure > 0).all(axis=1)].copy()
    )
    available = set(targets["Season"])
    required = set(selection_origins) | {season + 1 for season in selection_origins}
    required |= set(diagnostic_origins) | {season + 1 for season in diagnostic_origins}
    if missing := sorted(required - available):
        raise ValueError(f"Trajectory evaluation needs missing annual seasons: {missing}.")
    candidate_rows: list[dict] = []
    selected_metrics: list[pd.DataFrame] = []
    baseline = build_filtered_time_decay(targets, decay=1.0, exposure_power=0.0)
    baseline_selection = _component_metrics(
        baseline,
        targets,
        origins=selection_origins,
        model="latest_annual",
        minimum_side_possessions=minimum_side_possessions,
    )
    for decay in candidate_decays:
        for exposure_power in candidate_exposure_powers:
            trajectory = build_filtered_time_decay(
                targets, decay=decay, exposure_power=exposure_power
            )
            metrics = _component_metrics(
                trajectory,
                targets,
                origins=selection_origins,
                model="filtered_time_decay",
                minimum_side_possessions=minimum_side_possessions,
            )
            candidate_rows.append(
                {
                    "decay": float(decay),
                    "exposure_power": float(exposure_power),
                    "mean_net_rmse": _mean_season_metric(
                        metrics,
                        model="filtered_time_decay",
                        component="net",
                        metric="rmse",
                    ),
                    "mean_net_correlation": _mean_season_metric(
                        metrics,
                        model="filtered_time_decay",
                        component="net",
                        metric="correlation",
                    ),
                }
            )
            selected_metrics.append(metrics.assign(decay=decay, exposure_power=exposure_power))
    candidate_table = pd.DataFrame(candidate_rows).sort_values(
        ["mean_net_rmse", "decay", "exposure_power"], kind="stable"
    ).reset_index(drop=True)
    chosen = candidate_table.iloc[0]
    trajectory = build_filtered_time_decay(
        targets,
        decay=float(chosen["decay"]),
        exposure_power=float(chosen["exposure_power"]),
    )
    selection_metrics = _component_metrics(
        trajectory,
        targets,
        origins=selection_origins,
        model="filtered_time_decay",
        minimum_side_possessions=minimum_side_possessions,
    )
    diagnostic_metrics = _component_metrics(
        trajectory,
        targets,
        origins=diagnostic_origins,
        model="filtered_time_decay",
        minimum_side_possessions=minimum_side_possessions,
    )
    diagnostic_baseline = _component_metrics(
        trajectory,
        targets,
        origins=diagnostic_origins,
        model="latest_annual",
        minimum_side_possessions=minimum_side_possessions,
    )
    if names_source.exists():
        names = pd.read_csv(names_source, usecols=["PLAYER_ID", "PLAYER_NAME"])
        names["PLAYER_ID"] = pd.to_numeric(names["PLAYER_ID"], errors="raise").astype(int)
        if names["PLAYER_ID"].duplicated().any():
            raise ValueError("Player-name crosswalk has duplicate IDs.")
        trajectory = trajectory.merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    trajectory = trajectory.sort_values(["PLAYER_ID", "Season"], kind="stable")
    config = {
        "candidate_decays": list(candidate_decays),
        "candidate_exposure_powers": list(candidate_exposure_powers),
        "selection_origins": list(selection_origins),
        "diagnostic_origins": list(diagnostic_origins),
        "minimum_side_possessions": minimum_side_possessions,
        "invalid_side_exposure_rows_excluded": invalid_side_exposure_rows,
        "selected_decay": float(chosen["decay"]),
        "selected_exposure_power": float(chosen["exposure_power"]),
        "targets_sha256": sha256_file(source_path),
        "names_sha256": sha256_file(names_source) if names_source.exists() else None,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    run_id = f"time_decayed_trajectory_v1_{hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]}"
    output = Path(artifact_root) / "models" / "time_decayed_trajectory" / run_id
    output.mkdir(parents=True, exist_ok=False)
    trajectory.to_parquet(output / "trajectories.parquet", index=False)
    candidate_table.to_parquet(output / "selection_candidates.parquet", index=False)
    pd.concat(
        [
            baseline_selection.assign(scope="selection"),
            selection_metrics.assign(scope="selection"),
            diagnostic_baseline.assign(scope="diagnostic"),
            diagnostic_metrics.assign(scope="diagnostic"),
        ],
        ignore_index=True,
    ).to_parquet(output / "forward_metrics.parquet", index=False)
    metrics = {
        "selection_latest_annual_net_rmse": _mean_season_metric(
            baseline_selection, model="latest_annual", component="net", metric="rmse"
        ),
        "selection_filtered_net_rmse": _mean_season_metric(
            selection_metrics, model="filtered_time_decay", component="net", metric="rmse"
        ),
        "diagnostic_latest_annual_net_rmse": _mean_season_metric(
            diagnostic_baseline, model="latest_annual", component="net", metric="rmse"
        ),
        "diagnostic_filtered_net_rmse": _mean_season_metric(
            diagnostic_metrics, model="filtered_time_decay", component="net", metric="rmse"
        ),
        "selection_filtered_net_correlation": _mean_season_metric(
            selection_metrics, model="filtered_time_decay", component="net", metric="correlation"
        ),
        "diagnostic_filtered_net_correlation": _mean_season_metric(
            diagnostic_metrics, model="filtered_time_decay", component="net", metric="correlation"
        ),
        "rows": int(len(trajectory)),
        "players": int(trajectory["PLAYER_ID"].nunique()),
        "invalid_side_exposure_rows_excluded": invalid_side_exposure_rows,
        "maximum_component_identity_error": float(
            np.abs(
                trajectory["filtered_net"]
                - trajectory["filtered_offense"]
                - trajectory["filtered_defense"]
            ).max()
        ),
    }
    run = {
        "run_id": run_id,
        "model_family": "filtered_exposure_weighted_time_decay_trajectory",
        "estimand": "filtered_end_of_season_player_strength_proxy_from_annual_normal_rapm",
        "estimand_id": "filtered_time_decayed_normal_rapm_proxy_v1",
        "status": "research_trajectory_baseline",
        "evidence_status": "reused_annual_target_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "metrics": metrics,
        "artifact_path": str(output.resolve()),
        "trajectories_path": str((output / "trajectories.parquet").resolve()),
        "caveats": [
            "This is a filtered descriptive/current-strength proxy, not retrospective annual impact.",
            "Each trajectory point uses only the named season and earlier annual normal-RAPM estimates.",
            "Annual normal-RAPM targets are noisy; this run has no trajectory uncertainty intervals.",
            "The legacy annual target archive ends in 2024 and is stale for current NBA use.",
            "Season 2027 is not included.",
        ],
        "forbidden_interpretation": "Causal player value, forecast certainty, interpolated evidence, or public production rating.",
    }
    write_json_atomic(run, output / "run.json")
    return run
