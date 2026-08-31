#!/usr/bin/env python3
"""Test a cross-fitted residual correction over the frozen Box15 SPM."""

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
from nba_impact.models.statistical_impact import _metrics
from run_aio_prior_bakeoff import _prior_frame

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "residual_box15_spm_v1"
CONTRACT = ROOT / "research/experiments/residual_box15_spm_v1.yml"
FEATURE_RUN = ROOT / "artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_8be676bd0f"
FEATURES = FEATURE_RUN / "five_year_features.parquet"
TARGETS = ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
MATRIX_ROOT = ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
WINDOWS = tuple(range(2018, 2027))
RATING_SEASONS = tuple(range(2022, 2027))
EVALUATED_RATING_SEASONS = RATING_SEASONS[:-1]
VARIANTS = ("box15", "box15_residual")
MODEL_ORDER = ("box15", "box15_residual", "zero_prior_rapm", "box15_aio", "box15_residual_aio")
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Residual SPM contract ID changed.")
    cutoff = contract["information_cutoff"]
    if tuple(cutoff["rating_seasons"]) != RATING_SEASONS:
        raise ValueError("Rating seasons differ from the frozen contract.")
    if tuple(cutoff["evaluated_rating_seasons"]) != EVALUATED_RATING_SEASONS:
        raise ValueError("Evaluated seasons differ from the frozen contract.")
    if cutoff["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return contract


def _load_panel() -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    features = pd.read_parquet(FEATURES)
    targets = pd.read_parquet(TARGETS)
    manifest = json.loads((FEATURE_RUN / "run.json").read_text())
    selected = {
        side: tuple(manifest["feature_contract"][side])
        for side in ("offense", "defense")
    }
    fields = ["PLAYER_ID", "Window_End", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]
    panel = features.merge(targets[fields], on=["PLAYER_ID", "Window_End"], how="inner", validate="one_to_one")
    panel = panel.loc[panel["Window_End"].isin(WINDOWS)].copy()
    panel["sample_weight"] = np.sqrt(np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1))
    required = list(dict.fromkeys((*selected["offense"], *selected["defense"])))
    if panel[required].isna().any().any() or not np.isfinite(panel[required].to_numpy(dtype=float)).all():
        raise ValueError("Residual SPM inputs are missing or nonfinite.")
    if panel["Window_End"].max() >= 2027:
        raise ValueError("Season 2027 entered the residual experiment.")
    return panel, selected


def _cross_fitted_base(train: pd.DataFrame, target: str) -> np.ndarray:
    prediction = pd.Series(index=train.index, dtype=float)
    for validation_season in sorted(train["Window_End"].unique()):
        inner = train.loc[train["Window_End"].ne(validation_season)].copy()
        validation = train.loc[train["Window_End"].eq(validation_season)]
        alpha = _select_alpha(
            inner.rename(columns={"Window_End": "Season"}),
            BOX_PIPM_STYLE_FEATURES,
            target,
            ALPHA_GRID,
        )
        prediction.loc[validation.index] = _fit(
            inner, BOX_PIPM_STYLE_FEATURES, target, alpha
        ).predict(validation.loc[:, BOX_PIPM_STYLE_FEATURES])
    if prediction.isna().any():
        raise ValueError("Cross-fitted Box15 did not cover every training row.")
    return prediction.loc[train.index].to_numpy(dtype=float)


def _select_residual(
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    base_oof: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    residual_panel = train.copy()
    residual_panel["residual_target"] = train[target].to_numpy(dtype=float) - base_oof
    rows = []
    best: tuple[float, float, float] | None = None
    for alpha in ALPHA_GRID:
        residual_prediction = pd.Series(index=train.index, dtype=float)
        for validation_season in sorted(train["Window_End"].unique()):
            inner = residual_panel.loc[residual_panel["Window_End"].ne(validation_season)]
            validation = residual_panel.loc[residual_panel["Window_End"].eq(validation_season)]
            residual_prediction.loc[validation.index] = _fit(
                inner, features, "residual_target", alpha
            ).predict(validation.loc[:, features])
        for gamma in GAMMA_GRID:
            prediction = base_oof + gamma * residual_prediction.loc[train.index].to_numpy(dtype=float)
            mse = float(np.average((train[target].to_numpy(dtype=float) - prediction) ** 2, weights=train["sample_weight"]))
            rows.append({"alpha": alpha, "gamma": gamma, "weighted_mse": mse})
            candidate = (mse, gamma, alpha)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    return best[2], best[1], pd.DataFrame(rows)


def _fit_priors(panel: pd.DataFrame, selected: dict[str, tuple[str, ...]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    selection_rows: list[pd.DataFrame] = []
    for season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 4 or test.empty:
            raise ValueError(f"Rating season {season} lacks residual training history.")
        outputs = {
            candidate: test[["PLAYER_ID", "Window_End"]].copy()
            for candidate in VARIANTS
        }
        for side in ("offense", "defense"):
            target = f"target_{side}"
            base_alpha = _select_alpha(
                train.rename(columns={"Window_End": "Season"}),
                BOX_PIPM_STYLE_FEATURES,
                target,
                ALPHA_GRID,
            )
            base_model = _fit(train, BOX_PIPM_STYLE_FEATURES, target, base_alpha)
            base_test = base_model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
            base_oof = _cross_fitted_base(train, target)
            residual_features = tuple(
                feature for feature in selected[side] if feature not in BOX_PIPM_STYLE_FEATURES
            )
            residual_alpha, gamma, grid = _select_residual(
                train, residual_features, target, base_oof
            )
            residual_panel = train.copy()
            residual_panel["residual_target"] = train[target].to_numpy(dtype=float) - base_oof
            residual_model = _fit(
                residual_panel, residual_features, "residual_target", residual_alpha
            )
            outputs["box15"][side] = base_test
            outputs["box15_residual"][side] = base_test + gamma * residual_model.predict(test.loc[:, residual_features])
            grid["rating_season"] = season
            grid["component"] = side
            grid["selected"] = grid["alpha"].eq(residual_alpha) & grid["gamma"].eq(gamma)
            grid["base_alpha"] = base_alpha
            grid["residual_feature_count"] = len(residual_features)
            selection_rows.append(grid)
        for candidate, prior in outputs.items():
            prior["net"] = prior["offense"] + prior["defense"]
            for component in ("offense", "defense", "net"):
                metric_rows.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "component": component,
                        **_metrics(
                            test[f"target_{component}"].to_numpy(dtype=float),
                            prior[component].to_numpy(dtype=float),
                            test["sample_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
            prior_rows.append(_prior_frame(prior, candidate))
    return pd.concat(prior_rows, ignore_index=True), pd.DataFrame(metric_rows), pd.concat(selection_rows, ignore_index=True)


def main() -> None:
    contract = _load_contract()
    panel, selected = _load_panel()
    priors, target_metrics, selection = _fit_priors(panel, selected)
    base.RATING_SEASONS = RATING_SEASONS
    base.EVALUATED_RATING_SEASONS = EVALUATED_RATING_SEASONS
    base.PRIOR_CANDIDATES = VARIANTS
    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = {
        frozenset(("box15", "box15_residual")),
        frozenset(("box15_aio", "box15_residual_aio")),
    }
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    ratings, games, coverage = base._score_models(priors, annual, MATRIX_ROOT)
    fold_metrics, summary = base._game_metrics_frames(games)
    bootstrap_models, bootstrap_pairs = base.paired_game_bootstrap(games, draws=5000, seed=20260829)
    sources = {
        "contract": CONTRACT,
        "features": FEATURES,
        "feature_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
        "runner": Path(__file__),
        **{f"matrix_{season}": MATRIX_ROOT / f"5y_end_{season}/manifest.json" for season in RATING_SEASONS},
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_seasons": list(RATING_SEASONS),
        "evaluated_rating_seasons": list(EVALUATED_RATING_SEASONS),
        "alpha_grid": list(ALPHA_GRID),
        "gamma_grid": list(GAMMA_GRID),
        "feature_counts": {side: len(selected[side]) - len(BOX_PIPM_STYLE_FEATURES) for side in selected},
        "sources": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for name, path in sources.items()},
        "bootstrap": {"draws": 5000, "seed": 20260829, "unit": "whole game within test season"},
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:10]
    output = ROOT / "artifacts/research/residual_box15_spm" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "selection.parquet": selection,
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
        bootstrap_pairs["candidate"].eq("box15_aio")
        & bootstrap_pairs["reference"].eq("box15_residual_aio")
    ].iloc[0]
    challenger_delta = -float(pair["mean_mse_delta"])
    challenger_low = -float(pair["bootstrap_95_high"])
    challenger_high = -float(pair["bootstrap_95_low"])
    status = "research_challenger" if challenger_high < 0 else "research_null"
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float((ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()),
            "season_2027_loaded": False,
        },
        "decision": {
            "challenger": "box15_residual_aio",
            "reference": "box15_aio",
            "mean_mse_delta": challenger_delta,
            "bootstrap_95_low": challenger_low,
            "bootstrap_95_high": challenger_high,
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
    print("\nSelected residual settings")
    print(selection.loc[selection["selected"]].to_string(index=False))


if __name__ == "__main__":
    main()
