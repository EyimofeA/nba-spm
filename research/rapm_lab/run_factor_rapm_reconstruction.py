"""Test whether annual factor RAPMs reconstruct held-out annual points RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.factor_reconstruction import (
    fit_weighted_ridge,
    predict_weighted_ridge,
    weighted_metrics,
)
from nba_impact.models.possession_outcome_rapm import (
    build_factor_ledger,
    fit_factor_ratings,
    fit_weighted_factor_ratings,
)
from nba_impact.models.rapm import RapmConfig, load_current_player_names


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "research/experiments/factor_rapm_reconstruction_ts_v2.json"
POSSESSIONS = REPO_ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = REPO_ROOT / "data/lake/silver/possession_lineup_segments.parquet"
EVENTS = REPO_ROOT / "data/lake/silver/event_states.parquet"
TARGETS = REPO_ROOT / "artifacts/models/canonical_annual_target_panel/canonical_annual_target_panel_v1_4586bd2f72/targets.parquet"
NAMES = REPO_ROOT / "rapm/data/all_names.csv"
PLAYER_GAMES = REPO_ROOT / "data/lake/silver/player_games.parquet"
OUTPUT_ROOT = REPO_ROOT / "research/rapm_lab/outputs/factor_reconstruction"
FACTORS = ("shooting_ts", "turnover", "offensive_rebound")


def _factor_panel(ledger, seasons: tuple[int, ...], contract: dict) -> pd.DataFrame:
    panels = []
    specs = (
        ("shooting_ts", ledger.ts_opportunities, "ts_value", True, "ts_attempt_weight"),
        ("turnover", ledger.possessions, "turnover", False, None),
        ("offensive_rebound", ledger.rebound_opportunities, "offensive_rebound", True, None),
    )
    penalty = contract["factor_penalties"]
    for season in seasons:
        merged: pd.DataFrame | None = None
        for factor, frame, target, higher_is_good, weight in specs:
            rows = frame.loc[frame["season"].eq(season)].copy()
            config = RapmConfig(
                seasons=(season,),
                lambda_off=float(penalty["lambda_off"]),
                lambda_def=float(penalty["lambda_def"]),
                lambda_home=float(penalty["lambda_home"]),
                data_scope="annual_factor_rapm_reconstruction",
            )
            if weight is None:
                ratings = fit_factor_ratings(
                    rows,
                    target,
                    factor=factor,
                    higher_is_good_for_offense=higher_is_good,
                    config=config,
                )
            else:
                ratings = fit_weighted_factor_ratings(
                    rows,
                    target,
                    weight,
                    factor=factor,
                    config=config,
                )
            columns = [
                "player_id",
                f"{factor}_offense",
                f"{factor}_defense",
                f"{factor}_off_exposure",
                f"{factor}_def_exposure",
            ]
            merged = ratings[columns] if merged is None else merged.merge(
                ratings[columns], on="player_id", how="outer", validate="one_to_one"
            )
        assert merged is not None
        merged["Season"] = season
        panels.append(merged)
    return pd.concat(panels, ignore_index=True).rename(columns={"player_id": "PLAYER_ID"})


def _eligible_panel(factors: pd.DataFrame, contract: dict) -> pd.DataFrame:
    targets = pd.read_parquet(TARGETS)
    panel = targets.loc[targets["Season"].isin(contract["seasons"])].merge(
        factors, on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    )
    panel["weight"] = np.sqrt(panel[["Poss_Off", "Poss_Def"]].min(axis=1))
    minimum = float(contract["minimum_target_possessions_per_side"])
    feature_columns = [f"{factor}_{side}" for factor in FACTORS for side in ("offense", "defense")]
    return panel.loc[
        panel[["Poss_Off", "Poss_Def"]].min(axis=1).ge(minimum)
        & panel[feature_columns].notna().all(axis=1)
    ].copy()


def _fit_sides(
    train: pd.DataFrame,
    test: pd.DataFrame,
    factors: tuple[str, ...],
    alpha: float,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    predictions = {}
    coefficients = []
    for side in ("offense", "defense"):
        features = [f"{factor}_{side}" for factor in factors]
        model = fit_weighted_ridge(
            train[features].to_numpy(),
            train[f"target_{side}"].to_numpy(),
            train["weight"].to_numpy(),
            alpha=alpha,
        )
        predictions[side] = predict_weighted_ridge(model, test[features].to_numpy())
        for feature, coefficient in zip(features, model.coefficients):
            coefficients.append(
                {
                    "side": side,
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                    "alpha": alpha,
                }
            )
    predictions["net"] = predictions["offense"] + predictions["defense"]
    return predictions, coefficients


def _score(
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    *,
    stage: str,
    model: str,
    alpha: float | None,
) -> list[dict]:
    rows = []
    for side in ("offense", "defense", "net"):
        rows.append(
            {
                "stage": stage,
                "model": model,
                "side": side,
                "alpha": alpha,
                "players": int(len(frame)),
                **weighted_metrics(
                    frame[f"target_{side}"].to_numpy(),
                    predictions[side],
                    frame["weight"].to_numpy(),
                ),
            }
        )
    return rows


def _mean_predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    output = {}
    for side in ("offense", "defense"):
        mean = float(np.average(train[f"target_{side}"], weights=train["weight"]))
        output[side] = np.full(len(test), mean)
    output["net"] = output["offense"] + output["defense"]
    return output


def _bootstrap_rmse_delta(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    target = frame["target_net"].to_numpy()
    weight = frame["weight"].to_numpy()
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for draw in range(draws):
        index = rng.integers(0, len(frame), len(frame))
        base = weighted_metrics(target[index], baseline[index], weight[index])["weighted_rmse"]
        test = weighted_metrics(target[index], candidate[index], weight[index])["weighted_rmse"]
        deltas[draw] = test - base
    return {
        "draws": draws,
        "observed_rmse_delta": weighted_metrics(target, candidate, weight)["weighted_rmse"]
        - weighted_metrics(target, baseline, weight)["weighted_rmse"],
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "probability_improvement": float(np.mean(deltas < 0)),
    }


def run(output_root: Path = OUTPUT_ROOT) -> dict:
    contract = json.loads(CONTRACT.read_text())
    if contract["status"] != "frozen_research_contract":
        raise ValueError("Factor reconstruction requires a frozen contract.")
    if contract["untouched_confirmation_season"] in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched.")
    identity_payload = {
        "contract": sha256_file(CONTRACT),
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(REPO_ROOT / "src/nba_impact/models/factor_reconstruction.py"),
        "factor_model": sha256_file(
            REPO_ROOT / "src/nba_impact/models/possession_outcome_rapm.py"
        ),
        "possessions": sha256_file(POSSESSIONS),
        "segments": sha256_file(SEGMENTS),
        "events": sha256_file(EVENTS),
        "targets": sha256_file(TARGETS),
    }
    identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"{contract['experiment_id']}_{identity}"
    output = output_root / run_id
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    seasons = tuple(int(value) for value in contract["seasons"])
    ledger = build_factor_ledger(
        pd.read_parquet(POSSESSIONS),
        pd.read_parquet(SEGMENTS),
        pd.read_parquet(EVENTS),
        seasons=seasons,
    )
    factor_panel = _factor_panel(ledger, seasons, contract)
    factor_panel.to_parquet(output / "annual_factor_ratings.parquet", index=False)
    panel = _eligible_panel(factor_panel, contract)
    panel.to_parquet(output / "model_panel.parquet", index=False)

    development = int(contract["development_season"])
    selection_season = int(contract["selection_season"])
    diagnostic_season = int(contract["reused_diagnostic_season"])
    development_rows = panel.loc[panel["Season"].eq(development)]
    selection_rows = panel.loc[panel["Season"].eq(selection_season)]
    metric_rows = []
    for alpha in contract["alphas"]:
        prediction, _ = _fit_sides(development_rows, selection_rows, FACTORS, float(alpha))
        metric_rows.extend(
            _score(selection_rows, prediction, stage="selection", model="all_three", alpha=float(alpha))
        )
    metrics = pd.DataFrame(metric_rows)
    selected_alpha = float(
        metrics.loc[(metrics["stage"] == "selection") & (metrics["side"] == "net")]
        .sort_values(["weighted_rmse", "alpha"], kind="stable")
        .iloc[0]["alpha"]
    )

    train = panel.loc[panel["Season"].isin((development, selection_season))]
    diagnostic = panel.loc[panel["Season"].eq(diagnostic_season)].copy()
    baseline = _mean_predictions(train, diagnostic)
    metric_rows.extend(_score(diagnostic, baseline, stage="diagnostic", model="mean", alpha=None))
    all_predictions, coefficients = _fit_sides(train, diagnostic, FACTORS, selected_alpha)
    metric_rows.extend(
        _score(diagnostic, all_predictions, stage="diagnostic", model="all_three", alpha=selected_alpha)
    )
    for factor in FACTORS:
        prediction, _ = _fit_sides(train, diagnostic, (factor,), selected_alpha)
        metric_rows.extend(
            _score(diagnostic, prediction, stage="diagnostic", model=factor, alpha=selected_alpha)
        )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_parquet(output / "metrics.parquet", index=False)
    pd.DataFrame(coefficients).to_parquet(output / "coefficients.parquet", index=False)

    names = load_current_player_names(NAMES, PLAYER_GAMES)
    diagnostic = diagnostic.merge(names, on="PLAYER_ID", how="left", validate="many_to_one")
    diagnostic["PLAYER_NAME"] = diagnostic["PLAYER_NAME"].fillna(diagnostic["PLAYER_ID"].astype(str))
    for side in ("offense", "defense", "net"):
        diagnostic[f"predicted_{side}"] = all_predictions[side]
        diagnostic[f"residual_{side}"] = diagnostic[f"target_{side}"] - all_predictions[side]
    diagnostic.to_parquet(output / "predictions_2026.parquet", index=False)
    bootstrap = _bootstrap_rmse_delta(
        diagnostic,
        baseline["net"],
        all_predictions["net"],
        draws=int(contract["player_bootstrap_draws"]),
        seed=int(contract["player_bootstrap_seed"]),
    )
    diagnostic_net = metrics.loc[
        (metrics["stage"] == "diagnostic") & (metrics["side"] == "net")
    ].sort_values("weighted_rmse", kind="stable")
    manifest = {
        "run_id": run_id,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "estimand_id": contract["estimand_id"],
        "config": {**contract, **identity_payload, "selected_alpha": selected_alpha},
        "factor_quality": ledger.quality,
        "panel_rows_by_season": {
            str(key): int(value) for key, value in panel["Season"].value_counts().sort_index().items()
        },
        "metrics": metrics.to_dict("records"),
        "diagnostic_net_comparison": diagnostic_net.to_dict("records"),
        "bootstrap_all_three_vs_mean": bootstrap,
        "paths": {
            "annual_factor_ratings": "annual_factor_ratings.parquet",
            "model_panel": "model_panel.parquet",
            "metrics": "metrics.parquet",
            "coefficients": "coefficients.parquet",
            "predictions_2026": "predictions_2026.parquet",
        },
        "forbidden_interpretation": (
            "This tests held-out annual mapping into same-season RAPM. It is not a causal "
            "factor decomposition, a future forecast, or an independent player-value model."
        ),
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
