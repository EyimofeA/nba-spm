"""Exact sample-weight ablation for the annual CourtSignal SPM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.single_season_spm import _selected_single_season_features
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_model_comparison import _fit_model


VARIANTS = ("sqrt_possessions", "unweighted")
COMPONENTS = ("offense", "defense", "net")


SPECIAL_DESCRIPTIONS = {
    "usage_events_p100": "Shots, free-throw trips and turnovers per 100 possessions.",
    "true_shooting_pct": "Scoring efficiency that values twos, threes and free throws.",
    "shot_quality_average": "Expected value of the player's shot mix before shot-making overperformance.",
    "shooting_proficiency_2017_eb": "Stabilized composite of shooting volume, location and accuracy.",
    "box_creation_2017_eb_p100": "Stabilized estimate of shots created for teammates per 100 possessions.",
    "offensive_load_2017_eb_p100": "Stabilized estimate of possessions a player finishes or creates per 100.",
    "assist_to_load_2017_eb": "Stabilized assists relative to offensive load.",
    "turnover_to_load_2017_eb": "Stabilized turnovers relative to offensive load.",
    "creation_to_load_2017_eb": "Stabilized teammate-shot creation relative to offensive load.",
    "behavioral_passer_score_v1": "Composite passing score from creation, assist quality and ball security.",
    "crafted_spacing_stable_v1": "Stabilized spacing proxy from three-point volume, accuracy and shot context.",
    "zts_pct_points": "True-shooting percentage points above expectation for the player's play-type mix.",
    "dfg_attempts_p100": "Shots defended as the nearest defender per 100 possessions.",
    "dfg_diff_pct_eb": "Stabilized opponent field-goal percentage difference on defended shots.",
    "rim_dfga_p100": "Shots defended at the rim per 100 possessions.",
    "rim_diff_pct_eb": "Stabilized opponent field-goal percentage difference at the rim.",
    "rim_points_saved_p100": "Empirical-Bayes stabilized rim points prevented relative to expected shooting per 100.",
    "rim_points_saved_p100_raw": "Raw rim points prevented per 100, before small-sample stabilization.",
    "self_oreb_adjusted_true_shooting_pct": "True shooting after subtracting a player's own offensive rebounds from true-shot attempts.",
    "recovered_blocks_p100": "Blocks recovered by the defender's team per 100 possessions.",
    "matchup_opponent_adjusted_points_saved_p100_eb": "Stabilized points saved after adjusting for the scorers guarded.",
    "matchup_fga_suppressed_vs_scorer_p100_eb": "Stabilized shot attempts suppressed relative to each scorer's norm.",
    "matchup_shotmaking_points_saved_vs_scorer_p100_eb": "Stabilized shot-making points prevented relative to scorer expectation.",
    "matchup_three_pa_suppressed_vs_scorer_p100_eb": "Stabilized three-point attempts suppressed relative to scorer tendency.",
    "matchup_turnovers_forced_vs_scorer_p100_eb": "Stabilized turnovers forced relative to scorer tendency.",
    "matchup_assists_suppressed_vs_scorer_p100_eb": "Stabilized assists suppressed relative to scorer tendency.",
    "matchup_shooting_fouls_prevented_vs_scorer_p100_eb": "Stabilized shooting fouls avoided relative to scorer tendency.",
    "matchup_blocks_p100": "Blocks recorded in tracked scorer-defender matchups per 100 possessions.",
}


TOKEN_LABELS = {
    "PTS": "Points",
    "AST": "Assists",
    "TOV": "Turnovers",
    "STL": "Steals",
    "BLK": "Blocks",
    "OREB": "Offensive rebounds",
    "DREB": "Defensive rebounds",
    "PF": "Personal fouls",
    "PFD": "Personal fouls drawn",
    "FTA": "Free-throw attempts",
    "FTM": "Free throws made",
    "FG2A": "Two-point attempts",
    "FG2M": "Two-pointers made",
    "FG3A": "Three-point attempts",
    "FG3M": "Three-pointers made",
}


def describe_feature(feature: str) -> str:
    """Return a short, public-facing definition for an SPM input."""
    if feature in SPECIAL_DESCRIPTIONS:
        return SPECIAL_DESCRIPTIONS[feature]
    if feature.endswith("_eb"):
        base = describe_feature(feature[:-3]).rstrip(".")
        return f"Empirical-Bayes stabilized {base[0].lower() + base[1:]}."
    if feature.endswith("_relative"):
        base = describe_feature(feature[: -len("_relative")]).rstrip(".")
        return f"{base} relative to that season's league average."
    if feature in TOKEN_LABELS:
        return TOKEN_LABELS[feature] + "."
    if feature.endswith("_p100"):
        stem = feature[:-5]
        if stem in TOKEN_LABELS:
            return TOKEN_LABELS[stem] + " per 100 possessions."
        return stem.replace("_", " ").capitalize() + " per 100 possessions."
    if feature.endswith("_pct"):
        return feature[:-4].replace("_", " ").upper() + " percentage."
    if feature.endswith("_accuracy"):
        return "Field-goal percentage for " + feature[:-9].replace("_", " ") + " shots."
    if feature.endswith("_frequency"):
        return "Share of shot attempts from " + feature[:-10].replace("_", " ") + "."
    if feature.endswith("_share"):
        return "Share of relevant events classified as " + feature[:-6].replace("_", " ") + "."
    if "_per_drive" in feature:
        return feature.replace("_per_drive", " per drive").replace("_", " ").capitalize() + "."
    if "_per_touch" in feature:
        return feature.replace("_per_touch", " per touch").replace("_", " ").capitalize() + "."
    return feature.replace("_", " ").capitalize() + "."


def build_feature_catalog(reference_run_path: str | Path) -> pd.DataFrame:
    selected = _selected_single_season_features(reference_run_path)
    rows = []
    all_features = dict.fromkeys((*selected["offense"], *selected["defense"]))
    for feature in all_features:
        sides = [side for side in ("offense", "defense") if feature in selected[side]]
        rows.append(
            {
                "feature": feature,
                "side": "both" if len(sides) == 2 else sides[0],
                "description": describe_feature(feature),
                "offense_input": feature in selected["offense"],
                "defense_input": feature in selected["defense"],
            }
        )
    return pd.DataFrame(rows)


def _score(actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if np.std(actual) > 0 and np.std(predicted) > 0
        else float("nan")
    )
    return {
        "rmse": float(mean_squared_error(actual, predicted, sample_weight=weight) ** 0.5),
        "correlation": correlation,
    }


def run_spm_weight_ablation(
    features_path: str | Path,
    reference_oof_path: str | Path,
    reference_run_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    """Compare the published square-root exposure weight with equal row weights."""
    features_path = Path(features_path)
    reference_oof_path = Path(reference_oof_path)
    reference_run_path = Path(reference_run_path)
    features = pd.read_parquet(features_path).rename(columns={"Window_End": "Season"})
    targets = pd.read_parquet(reference_oof_path)
    if features.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual SPM features contain duplicate player-season keys.")
    if targets.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual SPM target rows contain duplicate player-season keys.")
    target_fields = [
        "PLAYER_ID",
        "Season",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
        "sample_weight",
        "spm_offense",
        "spm_defense",
        "spm_net",
    ]
    panel = features.merge(
        targets[target_fields], on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    )
    if len(panel) != len(targets):
        raise ValueError("Feature join does not cover every reference SPM target row.")
    expected_weight = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    if not np.allclose(panel["sample_weight"], expected_weight):
        raise ValueError("Reference SPM sample weights do not match the published formula.")

    selected = _selected_single_season_features(reference_run_path)
    missing = sorted({field for fields in selected.values() for field in fields} - set(panel))
    if missing:
        raise ValueError(f"Annual SPM feature panel is missing {missing}.")

    seasons = tuple(sorted(int(value) for value in panel["Season"].unique()))
    predictions = []
    fold_rows = []
    for variant in VARIANTS:
        for test_season in seasons:
            train = panel.loc[panel["Season"].ne(test_season)].copy()
            test = panel.loc[panel["Season"].eq(test_season)].copy()
            train["sample_weight"] = (
                expected_weight.loc[train.index] if variant == "sqrt_possessions" else 1.0
            )
            output = test[
                [
                    "PLAYER_ID",
                    "Season",
                    "target_offense",
                    "target_defense",
                    "target_net",
                    "Poss_Off",
                    "Poss_Def",
                ]
            ].copy()
            output["variant"] = variant
            for side in ("offense", "defense"):
                model = _fit_model(
                    _frozen_model(side), train, selected[side], f"target_{side}"
                )
                output[f"spm_{side}"] = model.predict(test.loc[:, selected[side]])
            output["spm_net"] = output["spm_offense"] + output["spm_defense"]
            predictions.append(output)

            exposure = np.minimum(test["Poss_Off"], test["Poss_Def"]).to_numpy(dtype=float)
            for component in COMPONENTS:
                actual = test[f"target_{component}"].to_numpy(dtype=float)
                predicted = output[f"spm_{component}"].to_numpy(dtype=float)
                for evaluation, weight in (
                    ("sqrt_possessions", expected_weight.loc[test.index].to_numpy(dtype=float)),
                    ("equal_players", np.ones(len(test), dtype=float)),
                ):
                    fold_rows.append(
                        {
                            "variant": variant,
                            "test_season": test_season,
                            "component": component,
                            "evaluation": evaluation,
                            "rows": len(test),
                            "minimum_exposure": float(exposure.min()),
                            **_score(actual, predicted, weight),
                        }
                    )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby(["variant", "component", "evaluation"], as_index=False)
        .agg(folds=("test_season", "nunique"), mean_rmse=("rmse", "mean"), mean_correlation=("correlation", "mean"))
    )

    weighted = prediction_frame.loc[prediction_frame["variant"].eq("sqrt_possessions")]
    weighted = weighted.merge(
        targets[["PLAYER_ID", "Season", "spm_offense", "spm_defense", "spm_net"]],
        on=["PLAYER_ID", "Season"],
        suffixes=("_rerun", "_reference"),
        validate="one_to_one",
    )
    reproduction = {}
    for side in COMPONENTS:
        rerun = weighted[f"spm_{side}_rerun"].to_numpy(dtype=float)
        reference = weighted[f"spm_{side}_reference"].to_numpy(dtype=float)
        reproduction[side] = {
            "maximum_absolute_error": float(np.max(np.abs(rerun - reference))),
            "mean_absolute_error": float(np.mean(np.abs(rerun - reference))),
            "correlation": float(np.corrcoef(rerun, reference)[0, 1]),
        }

    source_hashes = {
        "features": sha256_file(features_path),
        "reference_oof": sha256_file(reference_oof_path),
        "reference_run": sha256_file(reference_run_path / "run.json"),
    }
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    run_id = f"spm_weight_ablation_v1_{identity}"
    output = Path(artifact_root) / "research" / "spm_weight_ablation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    prediction_frame.to_parquet(output / "oof_predictions.parquet", index=False)
    fold_metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    catalog = build_feature_catalog(reference_run_path)
    catalog.to_parquet(output / "feature_catalog.parquet", index=False)
    run = {
        "run_id": run_id,
        "status": "research_ablation_complete",
        "model_family": "annual_spm_sample_weight_ablation",
        "estimand": "single_season_normal_rapm_offense_defense_and_net",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "variants": list(VARIANTS),
            "evaluation_weights": ["sqrt_possessions", "equal_players"],
            "seasons": list(seasons),
            "offense_features": len(selected["offense"]),
            "defense_features": len(selected["defense"]),
            "source_hashes": source_hashes,
        },
        "quality": {
            "rows": len(panel),
            "duplicate_keys": 0,
            "weighted_reference_reproduction": reproduction,
            "feature_catalog_rows": len(catalog),
        },
        "paths": {
            "oof_predictions": "oof_predictions.parquet",
            "fold_metrics": "fold_metrics.parquet",
            "summary": "summary.parquet",
            "feature_catalog": "feature_catalog.parquet",
        },
        "caveats": [
            "This changes only SPM training weights; features, labels, learners and folds are fixed.",
            "Annual RAPM labels remain noisy and are not ground truth.",
            "Leave-one-season-out evaluation is retrospective and can train earlier ratings on later seasons.",
            "The current histogram-GBM runtime does not bitwise reproduce the saved website offense predictions; both ablation arms use the same current runtime.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
