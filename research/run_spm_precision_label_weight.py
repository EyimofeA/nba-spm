#!/usr/bin/env python3
"""Compare exposure and precision weights for the frozen Box15 SPM prior."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit, _select_alpha
from nba_impact.models.spm_precision_weights import (
    analytic_ridge_label_variance,
    bounded_inverse_variance_weights,
)
from nba_impact.models.statistical_impact import _metrics
from run_aio_prior_bakeoff import _prior_frame

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "spm_precision_label_weight_v1"
CONTRACT = ROOT / "research/experiments/spm_precision_label_weight_v1.yml"
FEATURES = ROOT / "artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_8be676bd0f/five_year_features.parquet"
TARGETS = ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
MATRIX_ROOT = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
WINDOWS = tuple(range(2018, 2027))
RATING_SEASONS = tuple(range(2022, 2027))
EVALUATED_RATING_SEASONS = RATING_SEASONS[:-1]
VARIANTS = ("sqrt_possessions", "bounded_inverse_variance")
MODEL_ORDER = (
    "sqrt_possessions",
    "bounded_inverse_variance",
    "zero_prior_rapm",
    "sqrt_possessions_aio",
    "bounded_inverse_variance_aio",
)
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Precision-weight contract ID changed.")
    cutoff = contract["information_cutoff"]
    if tuple(cutoff["rating_seasons"]) != RATING_SEASONS:
        raise ValueError("Rating seasons differ from the frozen contract.")
    if tuple(cutoff["evaluated_rating_seasons"]) != EVALUATED_RATING_SEASONS:
        raise ValueError("Evaluated seasons differ from the frozen contract.")
    if cutoff["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return contract


def _variance_panel() -> pd.DataFrame:
    rows = [
        analytic_ridge_label_variance(MATRIX_ROOT / f"5y_end_{season}")
        for season in WINDOWS
    ]
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Label-variance panel has duplicate player-window keys.")
    return result


def _load_panel(variance: pd.DataFrame) -> pd.DataFrame:
    features = pd.read_parquet(FEATURES)
    targets = pd.read_parquet(TARGETS)
    if max(features["Window_End"].max(), targets["Window_End"].max()) >= 2027:
        raise ValueError("Season 2027 entered the precision-weight experiment.")
    fields = [
        "PLAYER_ID", "Window_End", "target_offense", "target_defense",
        "target_net", "Poss_Off", "Poss_Def",
    ]
    panel = features.merge(
        targets[fields], on=["PLAYER_ID", "Window_End"], how="inner", validate="one_to_one"
    ).merge(
        variance, on=["PLAYER_ID", "Window_End"], how="inner", validate="one_to_one"
    )
    panel = panel.loc[panel["Window_End"].isin(WINDOWS)].copy()
    required = [*BOX_PIPM_STYLE_FEATURES, "label_variance_offense", "label_variance_defense"]
    if panel[required].isna().any().any() or not np.isfinite(panel[required].to_numpy(dtype=float)).all():
        raise ValueError("Precision-weight panel has missing or nonfinite values.")
    if (panel[["label_variance_offense", "label_variance_defense"]] <= 0).any().any():
        raise ValueError("Precision-weight panel requires positive label variances.")
    panel["sqrt_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    return panel


def _fit_priors(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    weight_rows: list[pd.DataFrame] = []
    for season in RATING_SEASONS:
        train_base = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train_base["Window_End"].nunique() < 3 or test.empty:
            raise ValueError(f"Rating season {season} lacks chronological history.")
        for variant in VARIANTS:
            prior = test[["PLAYER_ID", "Window_End"]].copy()
            for side in ("offense", "defense"):
                train = train_base.copy()
                if variant == "sqrt_possessions":
                    train["sample_weight"] = train["sqrt_weight"]
                else:
                    train["sample_weight"] = bounded_inverse_variance_weights(
                        train[f"label_variance_{side}"]
                    )
                target = f"target_{side}"
                alpha = _select_alpha(
                    train.rename(columns={"Window_End": "Season"}),
                    BOX_PIPM_STYLE_FEATURES,
                    target,
                    ALPHA_GRID,
                )
                model = _fit(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
                prior[side] = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
                metric_rows.append(
                    {
                        "rating_season": season,
                        "candidate": variant,
                        "component": side,
                        "selected_alpha": alpha,
                        **_metrics(
                            test[target].to_numpy(dtype=float),
                            prior[side].to_numpy(dtype=float),
                            test["sqrt_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
                weights = train[["PLAYER_ID", "Window_End"]].copy()
                weights["rating_season"] = season
                weights["candidate"] = variant
                weights["component"] = side
                weights["sample_weight"] = train["sample_weight"].to_numpy()
                weight_rows.append(weights)
            prior["net"] = prior["offense"] + prior["defense"]
            metric_rows.append(
                {
                    "rating_season": season,
                    "candidate": variant,
                    "component": "net",
                    "selected_alpha": None,
                    **_metrics(
                        test["target_net"].to_numpy(dtype=float),
                        prior["net"].to_numpy(dtype=float),
                        test["sqrt_weight"].to_numpy(dtype=float),
                    ),
                }
            )
            prior_rows.append(_prior_frame(prior, variant))
    return pd.concat(prior_rows, ignore_index=True), pd.DataFrame(metric_rows), pd.concat(weight_rows, ignore_index=True)


def main() -> None:
    contract = _load_contract()
    variance = _variance_panel()
    panel = _load_panel(variance)
    priors, target_metrics, training_weights = _fit_priors(panel)
    base.RATING_SEASONS = RATING_SEASONS
    base.EVALUATED_RATING_SEASONS = EVALUATED_RATING_SEASONS
    base.PRIOR_CANDIDATES = VARIANTS
    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = {
        frozenset(("sqrt_possessions", "bounded_inverse_variance")),
        frozenset(("sqrt_possessions_aio", "bounded_inverse_variance_aio")),
    }
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    ratings, games, coverage = base._score_models(priors, annual, MATRIX_ROOT)
    fold_metrics, summary = base._game_metrics_frames(games)
    bootstrap_models, bootstrap_pairs = base.paired_game_bootstrap(
        games, draws=5000, seed=20260829
    )
    sources = {
        "contract": CONTRACT,
        "features": FEATURES,
        "targets": TARGETS,
        "runner": Path(__file__),
        **{f"matrix_{season}": MATRIX_ROOT / f"5y_end_{season}/manifest.json" for season in WINDOWS},
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in sources.items()
        },
        "rating_seasons": list(RATING_SEASONS),
        "evaluated_rating_seasons": list(EVALUATED_RATING_SEASONS),
        "alpha_grid": list(ALPHA_GRID),
        "bootstrap": {"draws": 5000, "seed": 20260829, "unit": "whole game within test season"},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/spm_precision_label_weight" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "label_variance.parquet": variance,
        "training_weights.parquet": training_weights,
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": fold_metrics,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": bootstrap_models,
        "paired_bootstrap.parquet": bootstrap_pairs,
        "prior_coverage.parquet": coverage,
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    pair = bootstrap_pairs.loc[
        bootstrap_pairs["candidate"].eq("bounded_inverse_variance_aio")
        & bootstrap_pairs["reference"].eq("sqrt_possessions_aio")
    ]
    if pair.empty:
        pair = bootstrap_pairs.loc[
            bootstrap_pairs["candidate"].eq("sqrt_possessions_aio")
            & bootstrap_pairs["reference"].eq("bounded_inverse_variance_aio")
        ].copy()
        for field in ("mean_mse_delta", "bootstrap_95_low", "bootstrap_95_high"):
            pair[field] *= -1
        pair[["bootstrap_95_low", "bootstrap_95_high"]] = pair[["bootstrap_95_high", "bootstrap_95_low"]]
    result = pair.iloc[0]
    status = "research_challenger" if result["bootstrap_95_high"] < 0 else "research_null"
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "variance_rows": len(variance),
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float((ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()),
            "season_2027_loaded": False,
        },
        "decision": {
            "challenger": "bounded_inverse_variance_aio",
            "reference": "sqrt_possessions_aio",
            "mean_mse_delta": float(result["mean_mse_delta"]),
            "bootstrap_95_low": float(result["bootstrap_95_low"]),
            "bootstrap_95_high": float(result["bootstrap_95_high"]),
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {"path": name, "sha256": sha256_file(output / name), "rows": len(frame)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print(json.dumps(run["decision"], indent=2))


if __name__ == "__main__":
    main()
