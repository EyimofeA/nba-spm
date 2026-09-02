"""Reconcile native factor ratings with PULSE and RAPM point estimates.

Native factor RAPM values use different outcome scales.  This module learns a
linear bridge for each side and target, exposes the bridged contributions, and
stores the unmodelled remainder as an exact balancing residual.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import write_json_atomic


FACTOR_COLUMNS = {
    "offense": {
        "shooting_ts": "shooting_ts_offense",
        "turnover_value": "turnover_avoidance_offense",
        "offensive_rebound_value": "opponent_oreb_prevention_offense",
    },
    "defense": {
        "shooting_ts": "shooting_ts_defense",
        "turnover_value": "turnover_avoidance_defense",
        "opponent_oreb_prevention": "opponent_oreb_prevention_defense",
    },
}


def _fit_contributions(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature contributions and fitted values.

    The intercept is intentionally excluded from the named mechanisms.  It is
    absorbed by the balancing residual with all other unmodelled value.
    """
    clean = frame.dropna(subset=[*feature_columns, target_column])
    if len(clean) < max(25, len(feature_columns) * 5):
        raise ValueError(f"Not enough complete rows to calibrate {target_column}.")
    model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
    model.fit(clean[feature_columns], clean[target_column])
    scaler: StandardScaler = model.named_steps["standardscaler"]
    ridge: Ridge = model.named_steps["ridge"]
    standardized = scaler.transform(frame[feature_columns])
    contributions = standardized * ridge.coef_[None, :]
    fitted = contributions.sum(axis=1) + float(ridge.intercept_)
    return contributions, fitted


def _calibrate_target(
    frame: pd.DataFrame,
    *,
    side: str,
    target_prefix: str,
    source_suffix: str,
    alpha: float,
) -> pd.DataFrame:
    mapping = FACTOR_COLUMNS[side]
    source_columns = [f"{column}{source_suffix}" for column in mapping.values()]
    target_column = f"{target_prefix}_{side}"
    contributions, fitted = _fit_contributions(
        frame,
        feature_columns=source_columns,
        target_column=target_column,
        alpha=alpha,
    )
    result = frame[["PLAYER_ID", "Season"]].copy()
    for index, name in enumerate(mapping):
        result[f"{target_prefix}_{side}_{name}_contribution"] = contributions[:, index]
    result[f"{target_prefix}_{side}_mapped"] = fitted
    named = [column for column in result if column.endswith("_contribution")]
    result[f"{target_prefix}_{side}_residual"] = (
        frame[target_column].to_numpy(dtype=float)
        - result[named].sum(axis=1).to_numpy(dtype=float)
    )
    return result


def build_pulse_decomposition(
    *,
    pulse_run: Path,
    factor_target_run: Path,
    factor_prediction_run: Path,
    artifact_root: Path,
    alpha: float = 100.0,
) -> Path:
    """Build a descriptive, exactly balanced PULSE factor ledger.

    The public ledger begins in 2021 because that is the first season with
    chronology-safe Box15 factor predictions.  It is descriptive and does not
    promote the factor specialists as a better prior.
    """
    ratings = pd.read_parquet(pulse_run / "ratings.parquet")
    native = pd.read_parquet(factor_target_run / "annual_factor_targets.parquet")
    predictions = pd.read_parquet(factor_prediction_run / "factor_predictions.parquet")
    box = predictions.loc[predictions["candidate"].eq("box15_factor")].copy()
    box["key"] = box["factor"].astype(str) + "_" + box["component"].astype(str)
    prior_native = box.pivot_table(
        index=["PLAYER_ID", "Window_End"], columns="key", values="prediction", aggfunc="first"
    ).reset_index().rename(columns={"Window_End": "Season"})
    prior_native.columns.name = None

    native = native.rename(columns={"player_id": "PLAYER_ID"})
    needed_native = sorted({column for side in FACTOR_COLUMNS.values() for column in side.values()})
    actual = native[["PLAYER_ID", "Season", *needed_native]].copy()
    prior_native = prior_native.rename(columns={column: f"{column}_prior" for column in needed_native})
    frame = (
        ratings.merge(actual, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one")
        .merge(prior_native, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one")
        .sort_values(["Season", "PLAYER_ID"])
        .reset_index(drop=True)
    )
    if frame.empty:
        raise ValueError("PULSE decomposition has no matched factor rows.")

    outputs: list[pd.DataFrame] = []
    for side in ("offense", "defense"):
        outputs.append(_calibrate_target(
            frame, side=side, target_prefix="rapm", source_suffix="", alpha=alpha
        ))
        outputs.append(_calibrate_target(
            frame, side=side, target_prefix="pulse_prior", source_suffix="_prior", alpha=alpha
        ))
        outputs.append(_calibrate_target(
            frame, side=side, target_prefix="lineup_update", source_suffix="", alpha=alpha
        ))

    ledger = frame[[
        "PLAYER_ID", "PLAYER_NAME", "Season", "Poss_Off", "Poss_Def",
        "rapm_offense", "rapm_defense", "rapm_net",
        "pulse_prior_offense", "pulse_prior_defense", "pulse_prior_net",
        "lineup_update_offense", "lineup_update_defense", "lineup_update_net",
        "pulse_offense", "pulse_defense", "pulse_net",
        *needed_native, *[f"{column}_prior" for column in needed_native],
    ]].copy()
    for output in outputs:
        ledger = ledger.merge(output, on=["PLAYER_ID", "Season"], validate="one_to_one")

    for side in ("offense", "defense"):
        names = list(FACTOR_COLUMNS[side])
        for name in names:
            ledger[f"pulse_{side}_{name}_contribution"] = (
                ledger[f"pulse_prior_{side}_{name}_contribution"]
                + ledger[f"lineup_update_{side}_{name}_contribution"]
            )
        pulse_components = [f"pulse_{side}_{name}_contribution" for name in names]
        ledger[f"pulse_{side}_residual"] = (
            ledger[f"pulse_{side}"] - ledger[pulse_components].sum(axis=1)
        )

    for prefix in ("rapm", "pulse_prior", "lineup_update", "pulse"):
        ledger[f"{prefix}_net_residual"] = (
            ledger[f"{prefix}_net"]
            - ledger[f"{prefix}_offense"]
            - ledger[f"{prefix}_defense"]
        )
        if not np.allclose(ledger[f"{prefix}_net_residual"], 0.0, atol=1e-9):
            raise ValueError(f"{prefix} violates offense plus defense equals net.")
        for side in ("offense", "defense"):
            components = [
                f"{prefix}_{side}_{name}_contribution" for name in FACTOR_COLUMNS[side]
            ]
            reconstructed = ledger[components].sum(axis=1) + ledger[f"{prefix}_{side}_residual"]
            if not np.allclose(reconstructed, ledger[f"{prefix}_{side}"], atol=1e-9):
                raise ValueError(f"{prefix} {side} factor ledger does not reconcile.")

    inputs = [
        pulse_run / "ratings.parquet",
        factor_target_run / "annual_factor_targets.parquet",
        factor_prediction_run / "factor_predictions.parquet",
    ]
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs
    }
    identity = hashlib.sha256(json.dumps({
        "alpha": alpha, "source_hashes": source_hashes, "rows": len(ledger)
    }, sort_keys=True).encode()).hexdigest()[:10]
    run_path = artifact_root / "models" / "pulse_decomposition" / f"pulse_decomposition_v1_{identity}"
    run_path.mkdir(parents=True, exist_ok=True)
    ledger.to_parquet(run_path / "factor_ledger.parquet", index=False)
    manifest = {
        "run_id": run_path.name,
        "model_family": "descriptive_pulse_factor_bridge_with_exact_residual",
        "status": "research_descriptive",
        "season_scope": [int(ledger["Season"].min()), int(ledger["Season"].max())],
        "rows": int(len(ledger)),
        "alpha": float(alpha),
        "source_hashes": source_hashes,
        "factor_sides": FACTOR_COLUMNS,
        "identity_checks": {
            "offense_plus_defense_equals_net": True,
            "factor_contributions_plus_residual_equal_each_target": True,
            "pulse_prior_plus_lineup_update_equals_pulse": bool(np.allclose(
                ledger["pulse_prior_net"] + ledger["lineup_update_net"], ledger["pulse_net"], atol=1e-9
            )),
        },
        "forbidden_interpretation": (
            "The calibrated components are descriptive statistical allocations, not causal credits. "
            "Raw factor values use different units and must not be added without the bridge."
        ),
        "files": {"factor_ledger": "factor_ledger.parquet"},
    }
    write_json_atomic(manifest, run_path / "run.json")
    return run_path
