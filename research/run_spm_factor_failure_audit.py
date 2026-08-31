#!/usr/bin/env python3
"""Compare Box15 and full SPM feature banks against six factor RAPM targets."""

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
    _apply_reconstruction,
    _fit_factor_variant,
    _prepare_panel,
    _reconstruction_models,
    _score_normal,
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
FEATURE_MANIFEST = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79/run.json"
)
OUTPUT_ROOT = ROOT / "artifacts/research/spm_factor_failure_audit"
SEASONS = (2024, 2025, 2026)
ALPHAS = (1.0, 10.0, 100.0, 1000.0, 3000.0, 10000.0)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260829


def _weighted_mse(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average(np.square(actual - predicted), weights=weight))


def _bootstrap_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for factor in FACTORS:
        for side in SIDES:
            actual = frame[f"target_{factor}_{side}"].to_numpy(dtype=float)
            weight = frame[f"weight_{factor}_{side}"].to_numpy(dtype=float)
            box = frame[f"box15_{factor}_{side}"].to_numpy(dtype=float)
            full = frame[f"full_{factor}_{side}"].to_numpy(dtype=float)
            indices = rng.integers(0, len(frame), size=(BOOTSTRAP_DRAWS, len(frame)))
            deltas = np.array([
                _weighted_mse(actual[index], full[index], weight[index])
                - _weighted_mse(actual[index], box[index], weight[index])
                for index in indices
            ])
            rows.append(
                {
                    "target": factor,
                    "side": side,
                    "full_minus_box15_mse": _weighted_mse(actual, full, weight)
                    - _weighted_mse(actual, box, weight),
                    "ci_low": float(np.quantile(deltas, 0.025)),
                    "ci_high": float(np.quantile(deltas, 0.975)),
                    "probability_full_better": float(np.mean(deltas < 0)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    manifest = json.loads(FEATURE_MANIFEST.read_text())
    sources = {
        "factor_panel": FACTOR_PANEL,
        "feature_manifest": FEATURE_MANIFEST,
        "source_code": Path(__file__),
        "player_2022": PLAYER_SHEETS / "2022.parquet",
        "player_2023": PLAYER_SHEETS / "2023.parquet",
        "player_2024": PLAYER_SHEETS / "2024.parquet",
        "player_2025": PLAYER_SHEETS / "2025.parquet",
        "player_2026": PLAYER_SHEETS / "2026.parquet",
        "playtype": SITE_DATA / "playtype.csv",
        "dfg": SITE_DATA / "dfg.csv",
        "rim_dfg": SITE_DATA / "rimdfg.csv",
        "hustle": SITE_DATA / "hustle.csv",
    }
    source_hashes = {name: sha256_file(path) for name, path in sources.items()}
    config = {
        "experiment_id": "spm_factor_failure_audit_v1",
        "seasons": SEASONS,
        "minimum_possessions_per_side": 1000,
        "alphas": ALPHAS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "source_hashes": source_hashes,
    }
    run_id = "spm_factor_failure_audit_v1_" + hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / run_id
    if (output / "run.json").exists():
        print(output)
        return

    features, feature_quality = build_related_feature_panel(
        PLAYER_SHEETS,
        playtype_path=SITE_DATA / "playtype.csv",
        dfg_path=SITE_DATA / "dfg.csv",
        rim_dfg_path=SITE_DATA / "rimdfg.csv",
        hustle_path=SITE_DATA / "hustle.csv",
        seasons=SEASONS,
    )
    targets = pd.read_parquet(FACTOR_PANEL).rename(
        columns={
            "target_offense": "target_normal_offense",
            "target_defense": "target_normal_defense",
            "target_net": "target_normal_net",
        }
    )
    panel = _prepare_panel(features, targets)
    panel = panel.loc[panel[["Poss_Off", "Poss_Def"]].min(axis=1).ge(1000)].copy()
    if panel["Season"].value_counts().sort_index().to_dict() != {2024: 365, 2025: 381, 2026: 387}:
        raise ValueError("Eligible factor panel changed.")

    full_banks = {
        side: tuple(feature for feature in manifest["features"][side] if feature in panel)
        for side in SIDES
    }
    candidates = {
        "box15": {
            (factor, side): BOX_PIPM_STYLE_FEATURES
            for factor in FACTORS
            for side in SIDES
        },
        "full": {
            (factor, side): full_banks[side]
            for factor in FACTORS
            for side in SIDES
        },
    }
    runs = {}
    metric_frames = []
    model_rows = []
    for candidate, feature_map in candidates.items():
        predictions, metrics, _, selected = _fit_factor_variant(
            panel,
            candidate=candidate,
            include_context=False,
            feature_map=feature_map,
            alphas=ALPHAS,
            development_season=2024,
            selection_season=2025,
            diagnostic_season=2026,
        )
        runs[candidate] = predictions
        metric_frames.append(metrics)
        model_rows.extend(selected)

    diagnostic = panel.loc[panel["Season"].eq(2026)].copy()
    predictions = diagnostic[[
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ABBREVIATION",
        "Season",
        "target_normal_offense",
        "target_normal_defense",
        "target_normal_net",
        "weight_normal",
        *[f"target_{factor}_{side}" for factor in FACTORS for side in SIDES],
        *[f"weight_{factor}_{side}" for factor in FACTORS for side in SIDES],
    ]].copy()
    for candidate, frame in runs.items():
        for factor in FACTORS:
            for side in SIDES:
                predictions[f"{candidate}_{factor}_{side}"] = frame[
                    f"predicted_{factor}_{side}"
                ].to_numpy(dtype=float)

    factor_metrics = pd.concat(metric_frames, ignore_index=True)
    diagnostic_metrics = factor_metrics.loc[factor_metrics["stage"].eq("diagnostic")]
    summary = diagnostic_metrics.pivot(
        index=["target", "side"],
        columns="candidate",
        values=["weighted_rmse", "weighted_correlation", "weighted_r2"],
    )
    summary.columns = [f"{candidate}_{metric}" for metric, candidate in summary.columns]
    summary = summary.reset_index()
    summary["full_minus_box15_rmse"] = (
        summary["full_weighted_rmse"] - summary["box15_weighted_rmse"]
    )
    summary["full_minus_box15_r2"] = (
        summary["full_weighted_r2"] - summary["box15_weighted_r2"]
    )

    bootstrap = _bootstrap_deltas(predictions)
    summary = summary.merge(bootstrap, on=["target", "side"], validate="one_to_one")
    for row_index, row in summary.iterrows():
        target, side = row["target"], row["side"]
        box_error = predictions[f"box15_{target}_{side}"] - predictions[f"target_{target}_{side}"]
        full_error = predictions[f"full_{target}_{side}"] - predictions[f"target_{target}_{side}"]
        summary.loc[row_index, "residual_correlation"] = box_error.corr(full_error)
        summary.loc[row_index, "share_players_full_lower_abs_error"] = float(
            full_error.abs().lt(box_error.abs()).mean()
        )

    misses = []
    for candidate in candidates:
        for factor in FACTORS:
            for side in SIDES:
                block = predictions[[
                    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "Season",
                    f"target_{factor}_{side}", f"{candidate}_{factor}_{side}",
                    f"weight_{factor}_{side}",
                ]].copy()
                block["candidate"] = candidate
                block["target"] = factor
                block["side"] = side
                block["residual"] = (
                    block[f"{candidate}_{factor}_{side}"]
                    - block[f"target_{factor}_{side}"]
                )
                block["absolute_error"] = block["residual"].abs()
                block = block.nlargest(15, "absolute_error")
                misses.append(block.rename(columns={
                    f"target_{factor}_{side}": "actual",
                    f"{candidate}_{factor}_{side}": "predicted",
                    f"weight_{factor}_{side}": "factor_weight",
                })[[
                    "candidate", "target", "side", "PLAYER_ID", "PLAYER_NAME",
                    "TEAM_ABBREVIATION", "Season", "actual", "predicted",
                    "residual", "absolute_error", "factor_weight",
                ]])
    largest_misses = pd.concat(misses, ignore_index=True)

    reconstruction = _reconstruction_models(
        panel.loc[panel["Season"].isin((2024, 2025))], alpha=100.0
    )
    reconstructed = {}
    for candidate in candidates:
        candidate_frame = predictions.rename(columns={
            f"{candidate}_{factor}_{side}": f"predicted_{factor}_{side}"
            for factor in FACTORS
            for side in SIDES
        })
        reconstructed[candidate] = _apply_reconstruction(
            reconstruction, candidate_frame, predicted=True
        )
        for side in (*SIDES, "net"):
            predictions[f"{candidate}_reconstructed_{side}"] = reconstructed[candidate][side]
    normal_metrics = pd.DataFrame([
        row
        for candidate, values in reconstructed.items()
        for row in _score_normal(
            diagnostic, values, stage="diagnostic", candidate=candidate
        )
    ])

    output.mkdir(parents=True, exist_ok=False)
    factor_metrics.to_parquet(output / "factor_metrics.parquet", index=False)
    summary.to_parquet(output / "factor_summary.parquet", index=False)
    predictions.to_parquet(output / "predictions_2026.parquet", index=False)
    largest_misses.to_parquet(output / "largest_misses.parquet", index=False)
    normal_metrics.to_parquet(output / "normal_reconstruction_metrics.parquet", index=False)
    pd.DataFrame(model_rows).to_parquet(output / "selected_models.parquet", index=False)
    write_json_atomic(
        {
            "run_id": run_id,
            "status": "research_diagnostic",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "quality": {
                "eligible_rows_by_season": {"2024": 365, "2025": 381, "2026": 387},
                "feature_counts": {"box15": 15, **{f"full_{side}": len(bank) for side, bank in full_banks.items()}},
                "season_2027_loaded": False,
                "feature_source_quality": feature_quality,
            },
            "paths": {
                "factor_metrics": "factor_metrics.parquet",
                "factor_summary": "factor_summary.parquet",
                "predictions_2026": "predictions_2026.parquet",
                "largest_misses": "largest_misses.parquet",
                "normal_reconstruction_metrics": "normal_reconstruction_metrics.parquet",
                "selected_models": "selected_models.parquet",
            },
            "caveats": [
                "The audit uses annual factor RAPM targets, not the five-year SPM target.",
                "Season 2026 is reused diagnostic evidence and cannot promote a model.",
                "Factor and normal RAPM share the same lineup data, so reconstruction is descriptive.",
                "The 2026 defended-shot source is incomplete, which weakens shooting-defense conclusions.",
            ],
            "forbidden_interpretation": "Untouched confirmation, production promotion, or causal factor credit.",
        },
        output / "run.json",
    )
    print(output)


if __name__ == "__main__":
    main()
