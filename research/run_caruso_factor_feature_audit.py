#!/usr/bin/env python3
"""Test height and an observable CARUSO-style defense pack on 2026 factor RAPM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES
from nba_impact.models.factor_target_spm import (
    FACTORS,
    SIDES,
    _fit_factor_variant,
    _prepare_panel,
    build_related_feature_panel,
)


ROOT = Path(__file__).resolve().parents[1]
BRONZE = ROOT / "data/lake/bronze"
PLAYER_SHEETS = BRONZE / "gabriel_player_sheets/revision=54b57cf/year_totals"
SITE_DATA = BRONZE / "gabriel_site_data/revision=782ec8b"
FACTOR_PANEL = (
    ROOT
    / "research/rapm_lab/outputs/factor_reconstruction"
    / "factor_rapm_reconstruction_ts_v2_e8c10de3b2/model_panel.parquet"
)
OUTPUT_ROOT = ROOT / "artifacts/research/caruso_factor_feature_audit"
SEASONS = (2024, 2025, 2026)
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260829
POSITIONS = ("PG", "SG", "SF", "PF", "C")
RAW_CARUSO_FEATURES = (
    "caruso_rim_outcome_p100",
    "caruso_stop_p100",
    "caruso_rebound_oe_p100",
    "caruso_adjusted_deflections_p100",
)
CARUSO_FEATURES = (
    *RAW_CARUSO_FEATURES,
    *(f"{feature}_position_percentile" for feature in RAW_CARUSO_FEATURES),
    *(f"position_{position}" for position in POSITIONS),
)


def _season_residual(
    frame: pd.DataFrame,
    outcome: str,
    predictors: tuple[str, ...],
) -> pd.Series:
    values = frame[[outcome, *predictors]].apply(pd.to_numeric, errors="coerce")
    valid = values.notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.sum() <= len(predictors) + 1:
        return result
    design = np.column_stack(
        [np.ones(valid.sum()), *(values.loc[valid, feature] for feature in predictors)]
    )
    coefficients = np.linalg.lstsq(
        design, values.loc[valid, outcome].to_numpy(dtype=float), rcond=None
    )[0]
    result.loc[valid] = values.loc[valid, outcome] - design @ coefficients
    return result


def _add_caruso_features(features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    output = features.copy()
    bios = []
    for season in SEASONS:
        source = pd.read_parquet(
            PLAYER_SHEETS / f"{season}.parquet",
            columns=["PLAYER_ID", "PLAYER_HEIGHT_INCHES", "Pos"],
        )
        source["Season"] = season
        bios.append(source)
    bio = pd.concat(bios, ignore_index=True).drop_duplicates(["PLAYER_ID", "Season"])
    output = output.merge(bio, on=["PLAYER_ID", "Season"], how="left", validate="one_to_one")

    exposure = pd.to_numeric(output["DefPoss"], errors="coerce")
    for feature in (
        "event_stops_p100",
        "deflections_p100",
        "dreb_contests_p100",
    ):
        output.loc[output[feature].isna() & exposure.gt(0), feature] = 0.0

    output["caruso_rim_outcome_p100"] = output["rim_points_saved_p100"]
    output["caruso_stop_p100"] = output["event_stops_p100"]
    output["defended_rim_frequency"] = (
        pd.to_numeric(output["rim_dfga_p100"], errors="coerce")
        / pd.to_numeric(output["dfg_attempts_p100"], errors="coerce").replace(0, np.nan)
    ).fillna(0.0)
    output["caruso_rebound_oe_p100"] = output.groupby("Season", group_keys=False).apply(
        lambda group: _season_residual(
            group,
            "dreb_contests_p100",
            ("dfg_attempts_p100", "defended_rim_frequency"),
        ),
        include_groups=False,
    )
    output["caruso_adjusted_deflections_p100"] = output.groupby(
        "Season", group_keys=False
    ).apply(
        lambda group: _season_residual(group, "deflections_p100", ("STL_p100",)),
        include_groups=False,
    )
    for position in POSITIONS:
        output[f"position_{position}"] = output["Pos"].eq(position).astype(float)
    for feature in RAW_CARUSO_FEATURES:
        output[f"{feature}_position_percentile"] = output.groupby(
            ["Season", "Pos"]
        )[feature].rank(method="average", pct=True)

    quality = {
        "height_coverage": float(output["PLAYER_HEIGHT_INCHES"].notna().mean()),
        "position_coverage": float(output["Pos"].isin(POSITIONS).mean()),
        "caruso_feature_coverage": {
            feature: float(output[feature].notna().mean()) for feature in CARUSO_FEATURES
        },
        "rim_deterrence_available": False,
        "structural_zero_rule": (
            "Missing recorded event counts become zero only when defensive possession "
            "exposure is positive."
        ),
    }
    return output, quality


def _feature_map(extra_defense: tuple[str, ...]) -> dict[tuple[str, str], tuple[str, ...]]:
    return {
        (factor, side): (
            BOX_PIPM_STYLE_FEATURES + extra_defense
            if side == "defense"
            else BOX_PIPM_STYLE_FEATURES
        )
        for factor in FACTORS
        for side in SIDES
    }


def _paired_bootstrap(predictions: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(predictions), size=(BOOTSTRAP_DRAWS, len(predictions)))
    rows = []
    for factor in FACTORS:
        actual = predictions[f"target_{factor}_defense"].to_numpy(dtype=float)
        weight = predictions[f"weight_{factor}_defense"].to_numpy(dtype=float)
        baseline = predictions[f"box15_{factor}_defense"].to_numpy(dtype=float)
        for candidate in candidates:
            if candidate == "box15":
                continue
            challenger = predictions[f"{candidate}_{factor}_defense"].to_numpy(dtype=float)
            deltas = np.empty(BOOTSTRAP_DRAWS, dtype=float)
            for draw, index in enumerate(indices):
                denominator = weight[index].sum()
                deltas[draw] = (
                    np.sum(weight[index] * np.square(actual[index] - challenger[index]))
                    - np.sum(weight[index] * np.square(actual[index] - baseline[index]))
                ) / denominator
            rows.append(
                {
                    "candidate": candidate,
                    "target": factor,
                    "mse_delta_vs_box15_ci_low": float(np.quantile(deltas, 0.025)),
                    "mse_delta_vs_box15_ci_high": float(np.quantile(deltas, 0.975)),
                    "probability_better_than_box15": float(np.mean(deltas < 0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    sources = {
        "factor_panel": FACTOR_PANEL,
        "source_code": Path(__file__),
        **{f"player_{season}": PLAYER_SHEETS / f"{season}.parquet" for season in SEASONS},
        **{
            name: SITE_DATA / filename
            for name, filename in {
                "playtype": "playtype.csv",
                "dfg": "dfg.csv",
                "rim_dfg": "rimdfg.csv",
                "hustle": "hustle.csv",
            }.items()
        },
    }
    config = {
        "experiment_id": "caruso_factor_feature_audit_v1",
        "seasons": SEASONS,
        "minimum_possessions_per_side": 1000,
        "development_season": 2024,
        "selection_season": 2025,
        "diagnostic_season": 2026,
        "alphas": ALPHAS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "candidates": {
            "box15": [],
            "box15_height": ["PLAYER_HEIGHT_INCHES"],
            "box15_caruso": list(CARUSO_FEATURES),
            "box15_caruso_height": [*CARUSO_FEATURES, "PLAYER_HEIGHT_INCHES"],
        },
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
    }
    run_id = "caruso_factor_feature_audit_v1_" + hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / run_id
    if (output / "run.json").exists():
        print(output)
        return

    features, source_quality = build_related_feature_panel(
        PLAYER_SHEETS,
        playtype_path=SITE_DATA / "playtype.csv",
        dfg_path=SITE_DATA / "dfg.csv",
        rim_dfg_path=SITE_DATA / "rimdfg.csv",
        hustle_path=SITE_DATA / "hustle.csv",
        seasons=SEASONS,
    )
    features, caruso_quality = _add_caruso_features(features)
    targets = pd.read_parquet(FACTOR_PANEL).rename(
        columns={
            "target_offense": "target_normal_offense",
            "target_defense": "target_normal_defense",
            "target_net": "target_normal_net",
        }
    )
    panel = _prepare_panel(features, targets)
    panel = panel.loc[panel[["Poss_Off", "Poss_Def"]].min(axis=1).ge(1000)].copy()
    expected_rows = {2024: 365, 2025: 381, 2026: 387}
    if panel["Season"].value_counts().sort_index().to_dict() != expected_rows:
        raise ValueError("Eligible factor panel changed.")
    required = {feature for values in config["candidates"].values() for feature in values}
    if panel[list(required)].isna().any().any():
        missing = panel[list(required)].isna().sum()
        raise ValueError(f"Candidate features are incomplete: {missing[missing.gt(0)].to_dict()}")

    metrics = []
    selected = []
    predictions = panel.loc[panel["Season"].eq(2026), ["PLAYER_ID", "Season"]].copy()
    for candidate, extras in config["candidates"].items():
        candidate_predictions, candidate_metrics, _, candidate_selected = _fit_factor_variant(
            panel,
            candidate=candidate,
            include_context=False,
            feature_map=_feature_map(tuple(extras)),
            alphas=ALPHAS,
            development_season=2024,
            selection_season=2025,
            diagnostic_season=2026,
        )
        metrics.append(candidate_metrics)
        selected.extend(candidate_selected)
        for factor in FACTORS:
            predictions[f"target_{factor}_defense"] = panel.loc[
                panel["Season"].eq(2026), f"target_{factor}_defense"
            ].to_numpy(dtype=float)
            predictions[f"weight_{factor}_defense"] = panel.loc[
                panel["Season"].eq(2026), f"weight_{factor}_defense"
            ].to_numpy(dtype=float)
            predictions[f"{candidate}_{factor}_defense"] = candidate_predictions[
                f"predicted_{factor}_defense"
            ].to_numpy(dtype=float)

    metric_frame = pd.concat(metrics, ignore_index=True)
    diagnostic = metric_frame.loc[
        metric_frame["stage"].eq("diagnostic") & metric_frame["side"].eq("defense")
    ].copy()
    baseline = diagnostic.loc[diagnostic["candidate"].eq("box15"), [
        "target", "weighted_rmse", "weighted_r2"
    ]].rename(columns={
        "weighted_rmse": "box15_weighted_rmse",
        "weighted_r2": "box15_weighted_r2",
    })
    summary = diagnostic.merge(baseline, on="target", validate="many_to_one")
    summary["rmse_delta_vs_box15"] = summary["weighted_rmse"] - summary["box15_weighted_rmse"]
    summary["r2_delta_vs_box15"] = summary["weighted_r2"] - summary["box15_weighted_r2"]
    bootstrap = _paired_bootstrap(predictions, tuple(config["candidates"]))
    summary = summary.merge(bootstrap, on=["candidate", "target"], how="left", validate="one_to_one")

    output.mkdir(parents=True, exist_ok=False)
    metric_frame.to_parquet(output / "factor_metrics.parquet", index=False)
    summary.to_parquet(output / "defense_summary.parquet", index=False)
    bootstrap.to_parquet(output / "paired_bootstrap.parquet", index=False)
    predictions.to_parquet(output / "predictions_2026.parquet", index=False)
    pd.DataFrame(selected).to_parquet(output / "selected_models.parquet", index=False)
    write_json_atomic(
        {
            "run_id": run_id,
            "status": "reused_diagnostic_complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "quality": {
                "eligible_rows_by_season": expected_rows,
                "source": source_quality,
                "caruso": caruso_quality,
                "candidate_features_complete_on_eligible_rows": True,
                "season_2027_loaded": False,
            },
            "paths": {
                "factor_metrics": "factor_metrics.parquet",
                "defense_summary": "defense_summary.parquet",
                "paired_bootstrap": "paired_bootstrap.parquet",
                "predictions_2026": "predictions_2026.parquet",
                "selected_models": "selected_models.parquet",
            },
            "caveats": [
                "This is a same-learner feature audit against annual factor RAPM.",
                "Season 2026 is reused diagnostic evidence and cannot promote a model.",
                "The available source lacks CARUSO's on/off rim-deterrence component.",
                "Factor RAPM and the labels used here share lineup data.",
            ],
            "forbidden_interpretation": "CARUSO reproduction, untouched confirmation, or production promotion.",
        },
        output / "run.json",
    )
    print(output)
    print(summary[[
        "candidate", "target", "alpha", "weighted_rmse", "weighted_r2",
        "rmse_delta_vs_box15", "r2_delta_vs_box15",
        "mse_delta_vs_box15_ci_low", "mse_delta_vs_box15_ci_high",
        "probability_better_than_box15",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
