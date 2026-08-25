"""Run six-sided factor RAPM and a chronological multinomial RAPM pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.possession_outcome_rapm import (
    MULTINOMIAL_CLASSES,
    build_factor_ledger,
    fit_factor_ratings,
    fit_multinomial_lineup,
    multinomial_metrics,
    points_class,
)
from nba_impact.models.rapm import RapmConfig, build_design, fit_coefficients


REPO_ROOT = Path(__file__).resolve().parents[2]
POSSESSIONS = REPO_ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = REPO_ROOT / "data/lake/silver/possession_lineup_segments.parquet"
EVENTS = REPO_ROOT / "data/lake/silver/event_states.parquet"
OUTPUT_ROOT = REPO_ROOT / "research/rapm_lab/outputs/possession_outcomes"
ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3)


def _constant_probabilities(y_train: np.ndarray, rows: int) -> np.ndarray:
    shares = np.bincount(y_train, minlength=4).astype(float)
    shares /= shares.sum()
    return np.tile(shares, (rows, 1))


def _linear_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float | int]:
    actual = frame["pts"].to_numpy(dtype=float)
    error = actual - np.asarray(prediction)
    margin, _ = game_margin_metrics(frame, prediction)
    return {
        "possessions": int(len(frame)),
        "expected_points_rmse": float(math.sqrt(np.mean(error**2))),
        "expected_points_mae": float(np.mean(np.abs(error))),
        **margin,
    }


def _fit_factor_panel(ledger, config: RapmConfig) -> pd.DataFrame:
    factors = {
        "shooting_efg": fit_factor_ratings(
            ledger.shots,
            "efg_value",
            factor="shooting_efg",
            higher_is_good_for_offense=True,
            config=config,
        ),
        "turnover": fit_factor_ratings(
            ledger.possessions,
            "turnover",
            factor="turnover",
            higher_is_good_for_offense=False,
            config=config,
        ),
        "offensive_rebound": fit_factor_ratings(
            ledger.rebound_opportunities,
            "offensive_rebound",
            factor="offensive_rebound",
            higher_is_good_for_offense=True,
            config=config,
        ),
    }
    output: pd.DataFrame | None = None
    for name, frame in factors.items():
        columns = [
            "player_id",
            f"{name}_offense",
            f"{name}_defense",
            f"{name}_net",
            f"{name}_off_exposure",
            f"{name}_def_exposure",
        ]
        output = frame[columns] if output is None else output.merge(
            frame[columns], on="player_id", how="outer", validate="one_to_one"
        )
    assert output is not None
    return output.sort_values("player_id", kind="stable")


def _multinomial_experiment(frame: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict]:
    design = build_design(frame, include_home=True)
    y = points_class(frame["pts"])
    metric_rows = []
    selection_models = {}
    selection_train = design.seasons == 2024
    selection_test = design.seasons == 2025
    selection_three_plus = float(frame.loc[selection_train & (y == 3), "pts"].mean())
    for alpha in ALPHAS:
        print(f"multinomial selection alpha={alpha:g}", flush=True)
        model = fit_multinomial_lineup(
            design.X[selection_train], y[selection_train], alpha=alpha
        )
        probability = model.predict_proba(design.X[selection_test])
        metrics, _ = multinomial_metrics(
            frame.loc[selection_test].reset_index(drop=True),
            probability,
            three_plus_value=selection_three_plus,
        )
        metric_rows.append({"stage": "selection", "model": "multinomial", "alpha": alpha, **metrics})
        selection_models[alpha] = model
    constant = _constant_probabilities(y[selection_train], int(selection_test.sum()))
    constant_metrics, _ = multinomial_metrics(
        frame.loc[selection_test].reset_index(drop=True),
        constant,
        three_plus_value=selection_three_plus,
    )
    metric_rows.append({"stage": "selection", "model": "constant_class_rate", "alpha": np.nan, **constant_metrics})
    linear_config = RapmConfig(
        seasons=(2024,), lambda_off=3000, lambda_def=3000, lambda_home=300,
        data_scope="multinomial_comparator",
    )
    beta, intercept = fit_coefficients(design, linear_config, row_mask=selection_train)
    linear_prediction = intercept + np.asarray(design.X[selection_test] @ beta).ravel()
    metric_rows.append(
        {
            "stage": "selection",
            "model": "linear_points_rapm",
            "alpha": np.nan,
            **_linear_metrics(frame.loc[selection_test].reset_index(drop=True), linear_prediction),
        }
    )
    selection = pd.DataFrame(metric_rows)
    selected_alpha = float(
        selection.loc[selection["model"].eq("multinomial")]
        .sort_values(["log_loss", "alpha"], kind="stable")
        .iloc[0]["alpha"]
    )

    diagnostic_train = design.seasons <= 2025
    diagnostic_test = design.seasons == 2026
    diagnostic_three_plus = float(frame.loc[diagnostic_train & (y == 3), "pts"].mean())
    print(f"multinomial diagnostic alpha={selected_alpha:g}", flush=True)
    final_model = fit_multinomial_lineup(
        design.X[diagnostic_train], y[diagnostic_train], alpha=selected_alpha
    )
    probability = final_model.predict_proba(design.X[diagnostic_test])
    metrics, games = multinomial_metrics(
        frame.loc[diagnostic_test].reset_index(drop=True),
        probability,
        three_plus_value=diagnostic_three_plus,
    )
    diagnostic_rows = [
        {"stage": "diagnostic", "model": "multinomial", "alpha": selected_alpha, **metrics}
    ]
    constant = _constant_probabilities(y[diagnostic_train], int(diagnostic_test.sum()))
    constant_metrics, _ = multinomial_metrics(
        frame.loc[diagnostic_test].reset_index(drop=True),
        constant,
        three_plus_value=diagnostic_three_plus,
    )
    diagnostic_rows.append(
        {"stage": "diagnostic", "model": "constant_class_rate", "alpha": np.nan, **constant_metrics}
    )
    linear_config = RapmConfig(
        seasons=(2024, 2025), lambda_off=3000, lambda_def=3000, lambda_home=300,
        data_scope="multinomial_comparator",
    )
    beta, intercept = fit_coefficients(design, linear_config, row_mask=diagnostic_train)
    linear_prediction = intercept + np.asarray(design.X[diagnostic_test] @ beta).ravel()
    diagnostic_rows.append(
        {
            "stage": "diagnostic",
            "model": "linear_points_rapm",
            "alpha": np.nan,
            **_linear_metrics(frame.loc[diagnostic_test].reset_index(drop=True), linear_prediction),
        }
    )
    all_metrics = pd.concat([selection, pd.DataFrame(diagnostic_rows)], ignore_index=True)
    joblib.dump(final_model, output / "multinomial_model.joblib")
    prediction_rows = frame.loc[
        diagnostic_test,
        ["possession_id", "gameid", "season", "pts", "home_poss"],
    ].reset_index(drop=True)
    for index, outcome in enumerate(("zero", "one", "two", "three_plus")):
        prediction_rows[f"probability_{outcome}"] = probability[:, index]
    prediction_rows["expected_points"] = probability @ np.array(
        [0.0, 1.0, 2.0, diagnostic_three_plus]
    )
    prediction_rows.to_parquet(output / "multinomial_predictions_2026.parquet", index=False)
    games.to_parquet(output / "multinomial_game_predictions_2026.parquet", index=False)
    summary = {
        "selected_alpha": selected_alpha,
        "selection_train_seasons": [2024],
        "selection_test_season": 2025,
        "diagnostic_train_seasons": [2024, 2025],
        "diagnostic_test_season": 2026,
        "three_plus_value_selection": selection_three_plus,
        "three_plus_value_diagnostic": diagnostic_three_plus,
        "classes": MULTINOMIAL_CLASSES.tolist(),
    }
    return all_metrics, summary


def run(output_root: Path = OUTPUT_ROOT) -> dict:
    possessions = pd.read_parquet(POSSESSIONS)
    segments = pd.read_parquet(SEGMENTS)
    events = pd.read_parquet(EVENTS)
    ledger = build_factor_ledger(possessions, segments, events)
    if ledger.possessions["season"].max() >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    identity = hashlib.sha256(
        json.dumps(
            {
                "possessions": sha256_file(POSSESSIONS),
                "segments": sha256_file(SEGMENTS),
                "events": sha256_file(EVENTS),
                "alphas": ALPHAS,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"possession_outcome_rapm_v1_{identity}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=True)
    factor_config = RapmConfig(
        seasons=(2024, 2025, 2026),
        lambda_off=3000,
        lambda_def=3000,
        lambda_home=300,
        data_scope="canonical_three_season_factor_rapm",
    )
    print("fitting six-sided factor RAPM", flush=True)
    factors = _fit_factor_panel(ledger, factor_config)
    factors.to_parquet(output / "factor_ratings.parquet", index=False)
    metrics, multinomial_config = _multinomial_experiment(ledger.possessions, output)
    metrics.to_parquet(output / "multinomial_metrics.parquet", index=False)
    manifest = {
        "run_id": run_id,
        "status": "diagnostic_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_families": [
            "six_sided_factor_rapm",
            "sparse_multinomial_softmax_rapm",
        ],
        "factor_estimands": {
            "shooting_efg": "eFG makes per field-goal attempt, points per 100 shot attempts",
            "turnover": "turnover avoidance and forcing per 100 possessions",
            "offensive_rebound": "offensive rebound conversion and prevention per 100 resolved missed-FG rebound opportunities",
        },
        "factor_sign": "positive is good on offense and defense",
        "factor_quality": ledger.quality,
        "multinomial": multinomial_config,
        "multinomial_metrics": metrics.to_dict(orient="records"),
        "paths": {
            "factor_ratings": str((output / "factor_ratings.parquet").relative_to(REPO_ROOT)),
            "multinomial_metrics": str((output / "multinomial_metrics.parquet").relative_to(REPO_ROOT)),
            "multinomial_predictions_2026": str(
                (output / "multinomial_predictions_2026.parquet").relative_to(REPO_ROOT)
            ),
            "multinomial_model": str((output / "multinomial_model.joblib").relative_to(REPO_ROOT)),
        },
        "caveats": [
            "The six sides are three mechanisms with separate offense and defense ratings; they do not add to total points RAPM.",
            "The shooting target is shot-level eFG, turnover is possession-level, and rebounding is missed-field-goal-opportunity-level.",
            "The softmax candidate uses SGD regularization, not the closed-form RAPM lambda scale.",
            "2025 selects alpha and 2026 is a reused diagnostic. Season 2027 is not loaded.",
        ],
        "forbidden_interpretation": "Production player value, causal mechanisms, or an additive decomposition of total RAPM.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
