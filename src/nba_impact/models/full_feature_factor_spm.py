"""Full-feature factor-target ceiling and overall-SPM context ablation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.factor_reconstruction import weighted_metrics
from nba_impact.models.factor_target_spm import (
    CONTEXT_SOURCES,
    FACTORS,
    INDIVIDUAL_FEATURES,
    SIDES,
    _apply_reconstruction,
    _fit_factor_variant,
    _metric_row,
    _prepare_panel,
    _reconstruction_models,
    _score_normal,
    build_related_feature_panel,
)
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_model_comparison import _fit_model


EXPERIMENT_ID = "factor_target_full_feature_spm_v1"
OFFENSE_CONTEXT = tuple(
    name for name in CONTEXT_SOURCES if name in {
        "teammate_spacing",
        "teammate_creation",
        "teammate_rim_pressure",
        "teammate_turnover_to_load",
        "teammate_offensive_load",
        "teammate_oreb",
    }
)
DEFENSE_CONTEXT = tuple(
    name for name in CONTEXT_SOURCES if name in {
        "teammate_dreb",
        "teammate_dreb_contests",
        "teammate_event_stops",
        "teammate_deflections",
        "teammate_rim_points_saved",
        "teammate_contested_shots",
    }
)


def _feature_banks(
    reference_manifest_path: str | Path,
    columns: set[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    manifest = json.loads(Path(reference_manifest_path).read_text())
    expected = {
        side: tuple(str(value) for value in manifest["features"][side])
        for side in SIDES
    }
    available = {
        side: tuple(value for value in expected[side] if value in columns)
        for side in SIDES
    }
    missing = {
        side: tuple(value for value in expected[side] if value not in columns)
        for side in SIDES
    }
    if not all(available.values()):
        raise ValueError("Full-feature diagnostic requires both feature banks.")
    return available, missing


def _fit_overall_variant(
    panel: pd.DataFrame,
    *,
    candidate: str,
    feature_banks: dict[str, tuple[str, ...]],
    context_banks: dict[str, tuple[str, ...]],
    include_context: bool,
    development_season: int,
    selection_season: int,
    diagnostic_season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    development = panel.loc[panel["Season"].eq(development_season)].copy()
    selection = panel.loc[panel["Season"].eq(selection_season)].copy()
    training = panel.loc[
        panel["Season"].isin((development_season, selection_season))
    ].copy()
    diagnostic = panel.loc[panel["Season"].eq(diagnostic_season)].copy()
    for frame in (development, selection, training, diagnostic):
        frame["sample_weight"] = frame["weight_normal"]

    selection_predictions = selection[["PLAYER_ID", "Season"]].copy()
    diagnostic_predictions = diagnostic[["PLAYER_ID", "Season"]].copy()
    metrics = []
    selected = []
    for side in SIDES:
        feature_names = feature_banks[side]
        if include_context:
            feature_names = tuple(
                dict.fromkeys((*feature_names, *context_banks[side]))
            )
        target = f"target_normal_{side}"
        selection_model = _fit_model(
            _frozen_model(side), development, feature_names, target
        )
        selection_prediction = selection_model.predict(
            selection.loc[:, feature_names]
        )
        selection_predictions[f"predicted_normal_{side}"] = selection_prediction
        metrics.append(
            _metric_row(
                selection,
                selection_prediction,
                stage="selection",
                candidate=candidate,
                target="normal",
                side=side,
                alpha=None,
                weight="sample_weight",
            )
        )

        diagnostic_model = _fit_model(
            _frozen_model(side), training, feature_names, target
        )
        diagnostic_prediction = diagnostic_model.predict(
            diagnostic.loc[:, feature_names]
        )
        diagnostic_predictions[f"predicted_normal_{side}"] = diagnostic_prediction
        metrics.append(
            _metric_row(
                diagnostic,
                diagnostic_prediction,
                stage="diagnostic",
                candidate=candidate,
                target="normal",
                side=side,
                alpha=None,
                weight="sample_weight",
            )
        )
        selected.append(
            {
                "candidate": candidate,
                "target": "normal",
                "side": side,
                "learner": "histogram_gbm" if side == "offense" else "ridge_3000",
                "features": list(feature_names),
            }
        )

    for predictions in (selection_predictions, diagnostic_predictions):
        predictions["predicted_normal_net"] = (
            predictions["predicted_normal_offense"]
            + predictions["predicted_normal_defense"]
        )
    for stage, frame, predictions in (
        ("selection", selection, selection_predictions),
        ("diagnostic", diagnostic, diagnostic_predictions),
    ):
        metrics.append(
            {
                "stage": stage,
                "candidate": candidate,
                "target": "normal",
                "side": "net",
                "alpha": None,
                "players": int(len(frame)),
                **weighted_metrics(
                    frame["target_normal_net"].to_numpy(dtype=float),
                    predictions["predicted_normal_net"].to_numpy(dtype=float),
                    frame["weight_normal"].to_numpy(dtype=float),
                ),
            }
        )
    return diagnostic_predictions, pd.DataFrame(metrics), selected


def run_full_feature_factor_spm(
    *,
    player_sheet_dir: str | Path,
    playtype_path: str | Path,
    dfg_path: str | Path,
    rim_dfg_path: str | Path,
    hustle_path: str | Path,
    factor_panel_path: str | Path,
    feature_reference_manifest_path: str | Path,
    contract_path: str | Path,
    artifact_root: str | Path,
) -> dict:
    contract = json.loads(Path(contract_path).read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Unexpected full-feature experiment ID.")
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Full-feature diagnostic requires a frozen contract.")
    if contract["untouched_confirmation_season"] in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched.")

    source_paths = {
        "playtype": Path(playtype_path),
        "dfg": Path(dfg_path),
        "rim_dfg": Path(rim_dfg_path),
        "hustle": Path(hustle_path),
        "factor_panel": Path(factor_panel_path),
        "feature_reference_manifest": Path(feature_reference_manifest_path),
        "contract": Path(contract_path),
        "source_code": Path(__file__),
        "factor_feature_builder": Path(build_related_feature_panel.__code__.co_filename),
        "frozen_spm_model": Path(_frozen_model.__code__.co_filename),
        "spm_fit_helper": Path(_fit_model.__code__.co_filename),
    }
    for season in range(min(contract["seasons"]) - 2, max(contract["seasons"]) + 1):
        source_paths[f"player_sheet_{season}"] = Path(player_sheet_dir) / f"{season}.parquet"
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}

    features, feature_quality = build_related_feature_panel(
        player_sheet_dir,
        playtype_path=playtype_path,
        dfg_path=dfg_path,
        rim_dfg_path=rim_dfg_path,
        hustle_path=hustle_path,
        seasons=tuple(contract["seasons"]),
    )
    feature_banks, missing_features = _feature_banks(
        feature_reference_manifest_path, set(features.columns)
    )
    expected_missing = {
        side: tuple(contract["expected_missing_features"].get(side, ()))
        for side in SIDES
    }
    if missing_features != expected_missing:
        raise ValueError(
            f"Feature availability changed: expected {expected_missing}, got {missing_features}."
        )
    context_banks = {"offense": OFFENSE_CONTEXT, "defense": DEFENSE_CONTEXT}
    full_factor_map = {
        (factor, side): feature_banks[side]
        for factor in FACTORS
        for side in SIDES
    }
    full_context_map = {
        (factor, side): context_banks[side]
        for factor in FACTORS
        for side in SIDES
    }
    config = {
        "contract": contract,
        "feature_banks": {key: list(value) for key, value in feature_banks.items()},
        "missing_features": {key: list(value) for key, value in missing_features.items()},
        "context_banks": {key: list(value) for key, value in context_banks.items()},
        "source_hashes": source_hashes,
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = (
        Path(artifact_root)
        / "research"
        / "factor_target_spm"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())

    targets = pd.read_parquet(factor_panel_path).rename(
        columns={
            "target_offense": "target_normal_offense",
            "target_defense": "target_normal_defense",
            "target_net": "target_normal_net",
        }
    )
    panel = _prepare_panel(features, targets)
    minimum = float(contract["minimum_normal_possessions_per_side"])
    panel = panel.loc[
        panel[["Poss_Off", "Poss_Def"]].min(axis=1).ge(minimum)
    ].copy()
    expected_rows = {
        int(key): int(value) for key, value in contract["expected_rows_by_season"].items()
    }
    actual_rows = panel["Season"].value_counts().sort_index().to_dict()
    if actual_rows != expected_rows:
        raise ValueError(f"Eligible row set changed: expected {expected_rows}, got {actual_rows}.")

    alphas = tuple(float(value) for value in contract["ridge_alphas"])
    development = int(contract["development_season"])
    selection = int(contract["selection_season"])
    diagnostic = int(contract["reused_diagnostic_season"])
    factor_runs = {}
    factor_metrics = []
    selected_models = []
    for candidate, feature_map, context_map, include_context in (
        ("factor_sparse", INDIVIDUAL_FEATURES, full_context_map, False),
        ("factor_sparse_context", INDIVIDUAL_FEATURES, full_context_map, True),
        ("factor_full", full_factor_map, full_context_map, False),
        ("factor_full_context", full_factor_map, full_context_map, True),
    ):
        predictions, metrics, _, selected = _fit_factor_variant(
            panel,
            candidate=candidate,
            include_context=include_context,
            feature_map=feature_map,
            context_map=context_map,
            alphas=alphas,
            development_season=development,
            selection_season=selection,
            diagnostic_season=diagnostic,
        )
        factor_runs[candidate] = predictions
        factor_metrics.append(metrics)
        selected_models.extend(selected)

    overall_runs = {}
    overall_metrics = []
    for candidate, include_context in (
        ("overall_full_spm", False),
        ("overall_full_spm_context", True),
    ):
        predictions, metrics, selected = _fit_overall_variant(
            panel,
            candidate=candidate,
            feature_banks=feature_banks,
            context_banks=context_banks,
            include_context=include_context,
            development_season=development,
            selection_season=selection,
            diagnostic_season=diagnostic,
        )
        overall_runs[candidate] = predictions
        overall_metrics.append(metrics)
        selected_models.extend(selected)

    diagnostic_rows = panel.loc[panel["Season"].eq(diagnostic)].copy()
    train_rows = panel.loc[panel["Season"].isin((development, selection))].copy()
    reconstruction = _reconstruction_models(
        train_rows, alpha=float(contract["factor_to_points_alpha"])
    )
    oracle = _apply_reconstruction(reconstruction, diagnostic_rows, predicted=False)
    normal_predictions = {"oracle_factor_reconstruction": oracle}
    for candidate, predictions in factor_runs.items():
        joined = diagnostic_rows.merge(
            predictions, on=["PLAYER_ID", "Season"], validate="one_to_one"
        )
        normal_predictions[candidate] = _apply_reconstruction(
            reconstruction, joined, predicted=True
        )
    for candidate, predictions in overall_runs.items():
        normal_predictions[candidate] = {
            side: predictions[f"predicted_normal_{side}"].to_numpy(dtype=float)
            for side in (*SIDES, "net")
        }
    mean = {
        side: np.full(
            len(diagnostic_rows),
            np.average(
                train_rows[f"target_normal_{side}"],
                weights=train_rows["weight_normal"],
            ),
        )
        for side in SIDES
    }
    mean["net"] = mean["offense"] + mean["defense"]
    normal_predictions["mean"] = mean
    normal_metrics = pd.DataFrame(
        [
            row
            for candidate, prediction in normal_predictions.items()
            for row in _score_normal(
                diagnostic_rows,
                prediction,
                stage="diagnostic",
                candidate=candidate,
            )
        ]
    )

    predictions = diagnostic_rows[
        [
            "PLAYER_ID",
            "PLAYER_NAME",
            "TEAM_ABBREVIATION",
            "Season",
            "target_normal_offense",
            "target_normal_defense",
            "target_normal_net",
            "weight_normal",
            *[f"target_{factor}_{side}" for factor in FACTORS for side in SIDES],
        ]
    ].copy()
    for candidate, values in normal_predictions.items():
        for side in (*SIDES, "net"):
            predictions[f"{candidate}_{side}"] = values[side]
    for candidate, frame in factor_runs.items():
        for factor in FACTORS:
            for side in SIDES:
                predictions[f"{candidate}_{factor}_{side}"] = frame[
                    f"predicted_{factor}_{side}"
                ].to_numpy(dtype=float)

    factor_metrics_frame = pd.concat(factor_metrics, ignore_index=True)
    overall_metrics_frame = pd.concat(overall_metrics, ignore_index=True)
    selected_models_frame = pd.DataFrame(selected_models)
    output.mkdir(parents=True, exist_ok=False)
    factor_metrics_frame.to_parquet(output / "factor_target_metrics.parquet", index=False)
    overall_metrics_frame.to_parquet(output / "overall_spm_metrics.parquet", index=False)
    normal_metrics.to_parquet(output / "normal_rapm_metrics.parquet", index=False)
    selected_models_frame.to_parquet(output / "selected_models.parquet", index=False)
    predictions.to_parquet(output / "predictions_2026.parquet", index=False)

    net_comparison = normal_metrics.loc[normal_metrics["side"].eq("net")].sort_values(
        "weighted_rmse", kind="stable"
    )
    overall_diagnostic = overall_metrics_frame.loc[
        overall_metrics_frame["stage"].eq("diagnostic")
    ]
    overall_pivot = overall_diagnostic.pivot(
        index="side", columns="candidate", values="weighted_rmse"
    )
    context_delta = {
        side: float(
            overall_pivot.loc[side, "overall_full_spm_context"]
            - overall_pivot.loc[side, "overall_full_spm"]
        )
        for side in (*SIDES, "net")
    }
    manifest = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            **feature_quality,
            "eligible_rows_by_season": {
                str(key): int(value) for key, value in actual_rows.items()
            },
            "feature_counts": {
                side: len(values) for side, values in feature_banks.items()
            },
            "missing_feature_counts": {
                side: len(values) for side, values in missing_features.items()
            },
            "season_2027_loaded": False,
            "net_identity_max_error": float(
                max(
                    np.max(np.abs(values["offense"] + values["defense"] - values["net"]))
                    for values in normal_predictions.values()
                )
            ),
        },
        "normal_rapm_net_comparison": net_comparison.to_dict("records"),
        "overall_context_weighted_rmse_delta": context_delta,
        "decision_rule": (
            "Retain teammate context only if it lowers selection and diagnostic net RMSE "
            "without a material side failure."
        ),
        "caveats": [
            "Season 2026 is reused diagnostic evidence; Season 2027 remains untouched.",
            "The current panel has all 127 offense fields but lacks the eight scorer-matchup defense fields, leaving 60 of 68 defense inputs.",
            "DFG and rim-DFG observations end in 2025, so the 2026 shooting-defense ceiling remains source-constrained.",
            "Same-season teammate context may absorb team or scheme strength and is not a player-skill estimate.",
            "Annual TEAM_ID makes teammate context approximate for traded players.",
            "Factor RAPM and normal RAPM share lineup data and are not independent validation targets.",
        ],
        "paths": {
            "factor_target_metrics": "factor_target_metrics.parquet",
            "overall_spm_metrics": "overall_spm_metrics.parquet",
            "normal_rapm_metrics": "normal_rapm_metrics.parquet",
            "selected_models": "selected_models.parquet",
            "predictions_2026": "predictions_2026.parquet",
        },
        "forbidden_interpretation": (
            "Production rating, untouched confirmation, causal factor credit, causal teammate effect, or complete shooting-defense ceiling."
        ),
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest
