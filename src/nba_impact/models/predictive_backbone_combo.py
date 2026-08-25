"""Predeclared backbone combination (experiment predictive_backbone_combo_v1).

Combines the frozen predictive SPM next-season forecast with the AR(1)
state-space filtered rating into a parameter-free 50/50 mean and scores the
combination against every backbone-race comparator on identical rows. No
model is refit here; every input is a pinned artifact from a prior run.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic

SELECTION_SEASONS = (2019, 2020, 2021, 2022, 2023, 2024)
CONFIRMATION_SEASONS = (2025, 2026)
FORECAST_SEASONS = SELECTION_SEASONS + CONFIRMATION_SEASONS
MINIMUM_SIDE_POSSESSIONS = 1000.0
CANDIDATE_WEIGHT = 0.5
SENSITIVITY_WEIGHTS = (0.25, 0.75)

_SPM_COLUMNS = {
    "PLAYER_ID",
    "Target_Season",
    "Window_End",
    "raw_offense",
    "raw_defense",
    "raw_net",
    "target_net",
    "persistence_net",
}
_STATE_SPACE_COLUMNS = {"PLAYER_ID", "Season", "filtered_offense", "filtered_defense", "filtered_net"}
_TIME_DECAY_COLUMNS = {"PLAYER_ID", "Season", "filtered_net"}
_TARGET_PANEL_COLUMNS = {
    "PLAYER_ID",
    "Season",
    "target_offense",
    "target_defense",
    "target_net",
    "Poss_Off",
    "Poss_Def",
}


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{label} is missing required columns: {missing}")


def _weighted_moments(
    prediction: np.ndarray, actual: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    """Return weighted RMSE and weighted correlation for one arm."""
    total = float(weight.sum())
    if total <= 0.0 or len(prediction) == 0:
        return float("nan"), float("nan")
    residual = prediction - actual
    rmse = float(np.sqrt(np.sum(weight * residual**2) / total))
    mean_p = float(np.sum(weight * prediction) / total)
    mean_a = float(np.sum(weight * actual) / total)
    covariance = float(np.sum(weight * (prediction - mean_p) * (actual - mean_a)) / total)
    variance_p = float(np.sum(weight * (prediction - mean_p) ** 2) / total)
    variance_a = float(np.sum(weight * (actual - mean_a) ** 2) / total)
    if variance_p <= 0.0 or variance_a <= 0.0:
        correlation = float("nan")
    else:
        correlation = float(covariance / np.sqrt(variance_p * variance_a))
    return rmse, correlation


def _build_identical_rows(
    spm_predictions_path: str | Path,
    state_space_trajectories_path: str | Path,
    time_decay_trajectories_path: str | Path,
    targets_path: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Inner-join every comparator on (PLAYER_ID, forecast season)."""
    spm = pd.read_parquet(spm_predictions_path)
    _require_columns(spm, _SPM_COLUMNS, "predictive SPM predictions")
    state_space = pd.read_parquet(state_space_trajectories_path)
    _require_columns(state_space, _STATE_SPACE_COLUMNS, "state-space trajectories")
    time_decay = pd.read_parquet(time_decay_trajectories_path)
    _require_columns(time_decay, _TIME_DECAY_COLUMNS, "time-decay trajectories")
    panel = pd.read_parquet(targets_path)
    _require_columns(panel, _TARGET_PANEL_COLUMNS, "canonical annual target panel")

    spm = spm.loc[spm["Target_Season"].isin(FORECAST_SEASONS)].copy()
    spm["PLAYER_ID"] = spm["PLAYER_ID"].astype(int)
    spm["Target_Season"] = spm["Target_Season"].astype(int)
    spm["Window_End"] = spm["Window_End"].astype(int)
    mismatched = spm.loc[spm["Window_End"] != spm["Target_Season"]]
    if len(mismatched):
        raise ValueError(
            "Predictive SPM rows where Window_End differs from Target_Season: "
            f"{len(mismatched)}"
        )
    # Filters enter at the last observed season before the forecast (the
    # race's origin-season convention), never at the forecast season itself.
    spm["Origin_Season"] = spm["Target_Season"] - 1

    state_space = state_space.copy()
    state_space["PLAYER_ID"] = state_space["PLAYER_ID"].astype(int)
    state_space["Season"] = state_space["Season"].astype(int)
    origin = state_space.rename(
        columns={
            "Season": "Origin_Season",
            "filtered_offense": "state_space_offense",
            "filtered_defense": "state_space_defense",
            "filtered_net": "state_space_net",
        }
    )[["PLAYER_ID", "Origin_Season", "state_space_offense", "state_space_defense", "state_space_net"]]
    rows = spm.merge(origin, on=["PLAYER_ID", "Origin_Season"], how="inner", validate="one_to_one")

    time_decay = time_decay.copy()
    time_decay["PLAYER_ID"] = time_decay["PLAYER_ID"].astype(int)
    time_decay["Season"] = time_decay["Season"].astype(int)
    decay_origin = time_decay.rename(
        columns={"Season": "Origin_Season", "filtered_net": "time_decay_net"}
    )[["PLAYER_ID", "Origin_Season", "time_decay_net"]]
    rows = rows.merge(decay_origin, on=["PLAYER_ID", "Origin_Season"], how="inner", validate="one_to_one")

    panel = panel.copy()
    panel["PLAYER_ID"] = panel["PLAYER_ID"].astype(int)
    panel["Season"] = panel["Season"].astype(int)
    panel = panel.rename(
        columns={
            "Season": "Target_Season",
            "target_offense": "panel_target_offense",
            "target_defense": "panel_target_defense",
            "target_net": "panel_target_net",
        }
    )
    rows = rows.merge(
        panel[
            [
                "PLAYER_ID",
                "Target_Season",
                "panel_target_offense",
                "panel_target_defense",
                "panel_target_net",
                "Poss_Off",
                "Poss_Def",
            ]
        ],
        on=["PLAYER_ID", "Target_Season"],
        how="inner",
        validate="one_to_one",
    )

    differences = (rows["target_net"] - rows["panel_target_net"]).abs()
    maximum_target_difference = float(differences.max()) if len(differences) else 0.0
    if maximum_target_difference > 1e-9:
        raise ValueError(
            "Comparator targets disagree with the canonical panel: "
            f"{maximum_target_difference}"
        )

    quality = {
        "spm_prediction_rows": int(len(spm)),
        "rows_with_state_space_origin": int(len(rows)),
        "dropped_without_state_space_origin": int(len(spm) - len(rows)),
        "maximum_target_panel_difference": maximum_target_difference,
    }
    return rows, quality


def _fold_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    """Weighted RMSE/correlation per season, arm, and population."""
    frames = rows.assign(
        weight=np.sqrt(np.minimum(rows["Poss_Off"], rows["Poss_Def"])),
        primary=np.minimum(rows["Poss_Off"], rows["Poss_Def"]).ge(MINIMUM_SIDE_POSSESSIONS),
    )
    arms: dict[str, pd.Series] = {
        "combo_mean_50_50": CANDIDATE_WEIGHT * frames["raw_net"]
        + (1.0 - CANDIDATE_WEIGHT) * frames["state_space_net"],
        "predictive_spm_raw": frames["raw_net"],
        "state_space_filtered": frames["state_space_net"],
        "time_decay_filtered": frames["time_decay_net"],
        "persistence": frames["persistence_net"],
    }
    for weight in SENSITIVITY_WEIGHTS:
        arms[f"combo_w{int(round(weight * 100)):02d}"] = (
            weight * frames["raw_net"] + (1.0 - weight) * frames["state_space_net"]
        )
    records: list[dict] = []
    populations = (
        ("primary", frames.loc[frames["primary"]]),
        ("all_matched", frames),
    )
    for population, scoped in populations:
        actual = scoped["panel_target_net"].to_numpy()
        weights = scoped["weight"].to_numpy()
        for season in sorted(scoped["Target_Season"].unique()):
            seasonal = scoped["Target_Season"].eq(season).to_numpy()
            for arm, series in arms.items():
                values = series.loc[scoped.index].to_numpy()
                rmse, correlation = _weighted_moments(values[seasonal], actual[seasonal], weights[seasonal])
                records.append(
                    {
                        "population": population,
                        "season": int(season),
                        "arm": arm,
                        "players": int(seasonal.sum()),
                        "weighted_rmse": rmse,
                        "weighted_correlation": correlation,
                    }
                )
    metrics = pd.DataFrame.from_records(records)
    metrics["role"] = np.where(metrics["arm"].isin(arms), "comparator", "sensitivity_diagnostic")
    return metrics


def _summarize(metrics: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    primary = metrics.loc[(metrics["population"] == "primary") & (metrics["role"] == "comparator")]
    selection = primary.loc[primary["season"].isin(SELECTION_SEASONS)]
    confirmation = primary.loc[primary["season"].isin(CONFIRMATION_SEASONS)]
    selection_mean = selection.groupby("arm")["weighted_rmse"].mean().to_dict()
    confirmation_rows = confirmation.pivot(index="season", columns="arm", values="weighted_rmse")
    confirmation_mean = confirmation_rows.mean(axis=0).to_dict() if len(confirmation_rows) else {}
    return (
        {arm: float(value) for arm, value in selection_mean.items()},
        {arm: float(value) for arm, value in confirmation_mean.items()},
    )


def _decide(selection_mean: dict[str, float], confirmation_mean: dict[str, float]) -> str:
    candidate = "combo_mean_50_50"
    parents = ("predictive_spm_raw", "state_space_filtered")
    if any(name not in selection_mean for name in (*parents, candidate)):
        return "incomplete_comparators"
    wins_selection = all(selection_mean[candidate] < selection_mean[parent] for parent in parents)
    if not wins_selection:
        return "backbone_state_space_filtered_combo_lost_selection"
    if not confirmation_mean or any(name not in confirmation_mean for name in parents):
        return "backbone_state_space_filtered_confirmation_missing"
    wins_confirmation = all(confirmation_mean[candidate] < confirmation_mean[parent] for parent in parents)
    if wins_confirmation:
        return "backbone_combo_promoted"
    better_parent = min(parents, key=lambda name: confirmation_mean[name])
    if confirmation_mean[parents[0]] == confirmation_mean[parents[1]]:
        return "backbone_state_space_filtered_confirmation_tie"
    return f"backbone_{better_parent}_won_confirmation"


def build_predictive_backbone_combo(
    spm_predictions_path: str | Path,
    state_space_trajectories_path: str | Path,
    time_decay_trajectories_path: str | Path,
    targets_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    """Score the frozen 50/50 combination against the race comparators."""
    rows, quality = _build_identical_rows(
        spm_predictions_path,
        state_space_trajectories_path,
        time_decay_trajectories_path,
        targets_path,
    )
    if rows.empty:
        raise ValueError("Backbone combination produced zero identical rows.")
    metrics = _fold_metrics(rows)
    selection_mean, confirmation_mean = _summarize(metrics)
    decision = _decide(selection_mean, confirmation_mean)

    input_hashes = {
        "spm_predictions": sha256_file(spm_predictions_path),
        "state_space_trajectories": sha256_file(state_space_trajectories_path),
        "time_decay_trajectories": sha256_file(time_decay_trajectories_path),
        "targets": sha256_file(targets_path),
        "source_code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    identity = hashlib.sha256(
        "|".join([input_hashes[name] for name in sorted(input_hashes)] + [str(CANDIDATE_WEIGHT)]).encode()
    ).hexdigest()[:10]
    run_id = f"predictive_backbone_combo_v1_{identity}"
    output = Path(artifact_root) / "models" / "predictive_backbone_combo" / run_id
    output.mkdir(parents=True, exist_ok=False)

    scored = rows.assign(
        weight=np.sqrt(np.minimum(rows["Poss_Off"], rows["Poss_Def"])),
        combo_mean_50_50=CANDIDATE_WEIGHT * rows["raw_net"] + (1.0 - CANDIDATE_WEIGHT) * rows["state_space_net"],
    )
    scored.to_parquet(output / "scored_rows.parquet", index=False)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)

    primary = metrics.loc[(metrics["population"] == "primary") & (metrics["role"] == "comparator")]
    paired_deltas = {
        parent: {
            str(int(season)): float(
                primary.loc[(primary["season"] == season) & (primary["arm"] == "combo_mean_50_50"), "weighted_rmse"].iloc[0]
                - primary.loc[(primary["season"] == season) & (primary["arm"] == parent), "weighted_rmse"].iloc[0]
            )
            for season in sorted(primary["season"].unique())
        }
        for parent in ("predictive_spm_raw", "state_space_filtered")
    }
    run = {
        "run_id": run_id,
        "model_family": "next_season_backbone_combination",
        "estimand": "single_regular_season_normal_rapm_offense_defense_and_net_points_per_100",
        "status": "research_predeclared_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_snapshot_id": None,
        "artifact_path": str(output),
        "inputs": input_hashes,
        "quality": {
            **quality,
            "identical_primary_rows": int(
                (np.minimum(scored["Poss_Off"], scored["Poss_Def"]) >= MINIMUM_SIDE_POSSESSIONS).sum()
            ),
            "identical_all_rows": int(len(scored)),
        },
        "config": {
            "experiment_id": "predictive_backbone_combo_v1",
            "candidate_weight": CANDIDATE_WEIGHT,
            "sensitivity_weights": list(SENSITIVITY_WEIGHTS),
            "selection_seasons": list(SELECTION_SEASONS),
            "confirmation_seasons": list(CONFIRMATION_SEASONS),
            "minimum_side_possessions": MINIMUM_SIDE_POSSESSIONS,
            "weighting": "sqrt(min(Poss_Off, Poss_Def)) at the forecast season",
        },
        "metrics": {
            "selection_mean_weighted_rmse": selection_mean,
            "confirmation_mean_weighted_rmse": confirmation_mean,
            "paired_fold_deltas_combo_minus_parent": paired_deltas,
            "decision": decision,
        },
    }
    write_json_atomic(run, output / "run.json")
    return run
