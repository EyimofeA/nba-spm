"""Redundancy and next-season importance audit for the five-year SPM."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_interpretability import feature_mechanism
from nba_impact.models.statistical_model_comparison import _fit_model


EXPERIMENT_ID = "five_year_spm_feature_audit_v1"


def feature_encoding(feature: str) -> str:
    """Describe how a feature re-encodes its underlying basketball signal."""
    if feature.endswith("_relative"):
        return "era_relative"
    if "_eb" in feature:
        return "empirical_bayes"
    if feature.endswith(("_share", "_rate", "_frequency", "_accuracy", "_pct")):
        return "ratio_or_rate"
    if any(
        token in feature
        for token in (
            "proficiency", "creation_2017", "load_2017", "behavioral_passer",
            "crafted_spacing", "shot_quality", "zts_",
        )
    ):
        return "composite"
    return "direct"


def _score(actual: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    """Return complementary next-season scores; no single score is the gate."""
    error = actual - prediction
    actual_mean = float(np.average(actual, weights=weight))
    prediction_mean = float(np.average(prediction, weights=weight))
    covariance = float(
        np.average((actual - actual_mean) * (prediction - prediction_mean), weights=weight)
    )
    actual_variance = float(np.average((actual - actual_mean) ** 2, weights=weight))
    prediction_variance = float(
        np.average((prediction - prediction_mean) ** 2, weights=weight)
    )
    weighted_pearson = (
        covariance / np.sqrt(actual_variance * prediction_variance)
        if actual_variance > 0 and prediction_variance > 0
        else float("nan")
    )
    rank_actual = pd.Series(actual).rank(method="average").to_numpy()
    rank_prediction = pd.Series(prediction).rank(method="average").to_numpy()
    spearman = (
        float(np.corrcoef(rank_actual, rank_prediction)[0, 1])
        if np.std(rank_actual) > 0 and np.std(rank_prediction) > 0
        else float("nan")
    )
    return {
        "weighted_mae": float(np.average(np.abs(error), weights=weight)),
        "weighted_rmse": float(np.sqrt(np.average(error**2, weights=weight))),
        "weighted_pearson": float(weighted_pearson),
        "spearman": spearman,
    }


def redundancy_audit(
    features: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
    *,
    correlation_threshold: float = 0.95,
    min_pair_rows: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Find exact and near-duplicate selected signals without changing the model."""
    registry_rows: list[dict] = []
    pair_rows: list[dict] = []
    for side, requested in selected.items():
        available = tuple(feature for feature in requested if feature in features)
        numeric = features.loc[:, available].apply(pd.to_numeric, errors="coerce")
        correlation = numeric.corr(min_periods=min_pair_rows)
        best: dict[str, tuple[str | None, float]] = {
            feature: (None, float("nan")) for feature in requested
        }
        for left_index, left in enumerate(available):
            for right in available[left_index + 1 :]:
                value = float(correlation.loc[left, right])
                if not np.isfinite(value) or abs(value) < correlation_threshold:
                    continue
                rows = int(numeric[[left, right]].notna().all(axis=1).sum())
                pair_rows.append(
                    {
                        "side": side,
                        "feature_left": left,
                        "feature_right": right,
                        "correlation": value,
                        "absolute_correlation": abs(value),
                        "paired_rows": rows,
                        "left_encoding": feature_encoding(left),
                        "right_encoding": feature_encoding(right),
                    }
                )
                for feature, other in ((left, right), (right, left)):
                    if not np.isfinite(best[feature][1]) or abs(value) > abs(best[feature][1]):
                        best[feature] = (other, value)
        for feature in requested:
            present = feature in features
            series = numeric[feature] if present else pd.Series(dtype=float)
            partner, value = best[feature]
            registry_rows.append(
                {
                    "side": side,
                    "feature": feature,
                    "mechanism": feature_mechanism(feature),
                    "encoding": feature_encoding(feature),
                    "present_in_audit_panel": present,
                    "non_null_rows": int(series.notna().sum()) if present else 0,
                    "missing_fraction": float(series.isna().mean()) if present else 1.0,
                    "constant": bool(series.nunique(dropna=True) <= 1) if present else False,
                    "closest_feature": partner,
                    "closest_correlation": value,
                }
            )
    registry = pd.DataFrame(registry_rows)
    pairs = pd.DataFrame(pair_rows)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["side", "absolute_correlation", "feature_left", "feature_right"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
    summary = {
        "selected_feature_slots": int(len(registry)),
        "unique_selected_features": int(registry["feature"].nunique()),
        "missing_feature_slots": int((~registry["present_in_audit_panel"]).sum()),
        "constant_feature_slots": int(registry["constant"].sum()),
        "pairs_at_or_above_0_95": int(len(pairs)),
        "pairs_at_or_above_0_98": int((pairs["absolute_correlation"] >= 0.98).sum())
        if not pairs.empty
        else 0,
        "pairs_at_or_above_0_995": int((pairs["absolute_correlation"] >= 0.995).sum())
        if not pairs.empty
        else 0,
    }
    return registry, pairs, summary


def next_season_permutation_importance(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
    *,
    rating_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022),
    group_repeats: int = 5,
    individual_repeats: int = 1,
    seed: int = 20260826,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit chronologically and score permutations against next-season one-year RAPM."""
    five_targets = targets.loc[targets["horizon"].eq("5y")].copy()
    annual_targets = targets.loc[targets["horizon"].eq("1y")].copy()
    panel = features.merge(
        five_targets[
            [
                "PLAYER_ID", "Window_End", "target_offense", "target_defense",
                "Poss_Off", "Poss_Def",
            ]
        ],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    baseline_rows: list[dict] = []
    group_rows: list[dict] = []
    individual_rows: list[dict] = []
    for rating_season in rating_seasons:
        train = panel.loc[panel["Window_End"].lt(rating_season)].copy()
        scored = features.loc[features["Window_End"].eq(rating_season)].copy()
        target = annual_targets.loc[
            annual_targets["Window_End"].eq(rating_season + 1),
            [
                "PLAYER_ID", "target_offense", "target_defense", "Poss_Off", "Poss_Def",
            ],
        ].rename(
            columns={
                "Poss_Off": "next_Poss_Off",
                "Poss_Def": "next_Poss_Def",
                "target_offense": "next_target_offense",
                "target_defense": "next_target_defense",
            }
        )
        if train.empty or scored.empty or target.empty:
            raise ValueError(f"Empty chronological partition for rating season {rating_season}.")
        for side in ("offense", "defense"):
            columns = selected[side]
            if missing := sorted(set(columns) - set(features.columns)):
                raise ValueError(f"Audit panel is missing {side} features {missing}.")
            model = _fit_model(_frozen_model(side), train, columns, f"target_{side}")
            evaluation = scored[["PLAYER_ID", *columns]].merge(
                target, on="PLAYER_ID", how="inner", validate="one_to_one"
            )
            actual = evaluation[f"next_target_{side}"].to_numpy(dtype=float)
            weight = np.sqrt(
                np.minimum(evaluation["next_Poss_Off"], evaluation["next_Poss_Def"]).clip(lower=1)
            ).to_numpy(dtype=float)
            x_test = evaluation.loc[:, columns]
            baseline = _score(actual, model.predict(x_test), weight)
            baseline_rows.append(
                {
                    "rating_season": rating_season,
                    "test_season": rating_season + 1,
                    "side": side,
                    "train_window_end_min": int(train["Window_End"].min()),
                    "train_window_end_max": int(train["Window_End"].max()),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(evaluation)),
                    **baseline,
                }
            )
            groups: dict[str, list[str]] = {}
            for feature in columns:
                groups.setdefault(feature_mechanism(feature), []).append(feature)
            specs = (
                (groups, group_repeats, group_rows, "group"),
                ({feature: [feature] for feature in columns}, individual_repeats, individual_rows, "feature"),
            )
            for mapping, repeats, output, label in specs:
                for item_index, (name, fields) in enumerate(mapping.items()):
                    changes = []
                    for repeat in range(repeats):
                        rng = np.random.default_rng(
                            seed + rating_season * 10_000 + (0 if side == "offense" else 5_000)
                            + item_index * 101 + repeat
                        )
                        order = rng.permutation(len(x_test))
                        permuted = x_test.copy()
                        permuted.loc[:, fields] = x_test.iloc[order][fields].to_numpy()
                        score = _score(actual, model.predict(permuted), weight)
                        changes.append(
                            {
                                "weighted_mae_increase": score["weighted_mae"]
                                - baseline["weighted_mae"],
                                "weighted_rmse_increase": score["weighted_rmse"]
                                - baseline["weighted_rmse"],
                                "weighted_pearson_drop": baseline["weighted_pearson"]
                                - score["weighted_pearson"],
                                "spearman_drop": baseline["spearman"] - score["spearman"],
                            }
                        )
                    record = {
                        label: name,
                        "rating_season": rating_season,
                        "test_season": rating_season + 1,
                        "side": side,
                        "feature_count": len(fields),
                        "repeats": repeats,
                    }
                    for metric in changes[0]:
                        values = [change[metric] for change in changes]
                        record[f"{metric}_mean"] = float(np.mean(values))
                        record[f"{metric}_std"] = float(np.std(values, ddof=0))
                    output.append(record)
    return pd.DataFrame(baseline_rows), pd.DataFrame(group_rows), pd.DataFrame(individual_rows)


def redundancy_pruning_ablation(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
    pairs: pd.DataFrame,
    *,
    rating_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022),
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Refit two transparent alternatives for highly correlated era encodings."""
    offense_pairs = pairs.loc[
        pairs["side"].eq("offense")
        & pairs["absolute_correlation"].ge(0.95)
        & (
            pairs["feature_left"].str.endswith("_relative")
            | pairs["feature_right"].str.endswith("_relative")
        )
    ]
    relative = {
        feature
        for row in offense_pairs.itertuples()
        for feature in (row.feature_left, row.feature_right)
        if feature.endswith("_relative")
    }
    raw = {
        row.feature_right if row.feature_left.endswith("_relative") else row.feature_left
        for row in offense_pairs.itertuples()
    }
    variants = {
        "full": selected["offense"],
        "drop_redundant_relative": tuple(
            feature for feature in selected["offense"] if feature not in relative
        ),
        "drop_redundant_raw": tuple(
            feature for feature in selected["offense"] if feature not in raw
        ),
    }
    five_targets = targets.loc[targets["horizon"].eq("5y")].copy()
    annual_targets = targets.loc[targets["horizon"].eq("1y")].copy()
    panel = features.merge(
        five_targets[
            [
                "PLAYER_ID", "Window_End", "target_offense", "Poss_Off", "Poss_Def",
            ]
        ],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    rows: list[dict] = []
    for rating_season in rating_seasons:
        train = panel.loc[panel["Window_End"].lt(rating_season)].copy()
        target = annual_targets.loc[
            annual_targets["Window_End"].eq(rating_season + 1),
            ["PLAYER_ID", "target_offense", "Poss_Off", "Poss_Def"],
        ].rename(
            columns={
                "target_offense": "next_target_offense",
                "Poss_Off": "next_Poss_Off",
                "Poss_Def": "next_Poss_Def",
            }
        )
        scored = features.loc[features["Window_End"].eq(rating_season)].merge(
            target, on="PLAYER_ID", how="inner", validate="one_to_one"
        )
        actual = scored["next_target_offense"].to_numpy(dtype=float)
        weight = np.sqrt(
            np.minimum(scored["next_Poss_Off"], scored["next_Poss_Def"]).clip(lower=1)
        ).to_numpy(dtype=float)
        for variant, columns in variants.items():
            model = _fit_model(_frozen_model("offense"), train, columns, "target_offense")
            score = _score(actual, model.predict(scored.loc[:, columns]), weight)
            rows.append(
                {
                    "variant": variant,
                    "rating_season": rating_season,
                    "test_season": rating_season + 1,
                    "feature_count": len(columns),
                    "test_rows": len(scored),
                    **score,
                }
            )
    return pd.DataFrame(rows), variants


def run_five_year_spm_feature_audit(
    *,
    features_path: str | Path,
    targets_path: str | Path,
    reference_run_path: str | Path,
    artifact_root: str | Path,
    rating_seasons: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022),
    correlation_threshold: float = 0.95,
    min_pair_rows: int = 200,
    group_repeats: int = 5,
    individual_repeats: int = 1,
    seed: int = 20260826,
) -> dict:
    """Write a reproducible research artifact for the frozen five-year SPM."""
    reference_path = Path(reference_run_path)
    reference = json.loads(reference_path.read_text())
    raw_selected = reference.get("features") or reference.get("selected_features")
    if not isinstance(raw_selected, dict):
        raise ValueError("Reference run has no feature contract.")
    selected = {
        side: tuple(raw_selected[side]) for side in ("offense", "defense")
    }
    config = {
        "rating_seasons": list(rating_seasons),
        "test_seasons": [season + 1 for season in rating_seasons],
        "target": "next-season one-year zero-prior RAPM",
        "correlation_threshold": correlation_threshold,
        "min_pair_rows": min_pair_rows,
        "group_repeats": group_repeats,
        "individual_repeats": individual_repeats,
        "seed": seed,
        "source_hashes": {
            "features": sha256_file(features_path),
            "targets": sha256_file(targets_path),
            "reference_run": sha256_file(reference_run_path),
            "source_code": sha256_file(Path(__file__)),
        },
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"{EXPERIMENT_ID}_{identity}"
    output = Path(artifact_root) / "research" / "five_year_spm_feature_audit" / run_id
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())
    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(targets_path)
    registry, pairs, redundancy = redundancy_audit(
        features,
        selected,
        correlation_threshold=correlation_threshold,
        min_pair_rows=min_pair_rows,
    )
    baseline, grouped, individual = next_season_permutation_importance(
        features,
        targets,
        selected,
        rating_seasons=rating_seasons,
        group_repeats=group_repeats,
        individual_repeats=individual_repeats,
        seed=seed,
    )
    ablation, ablation_variants = redundancy_pruning_ablation(
        features,
        targets,
        selected,
        pairs,
        rating_seasons=rating_seasons,
    )
    output.mkdir(parents=True, exist_ok=True)
    registry.to_parquet(output / "feature_registry.parquet", index=False)
    pairs.to_parquet(output / "correlated_pairs.parquet", index=False)
    baseline.to_parquet(output / "next_season_baseline_metrics.parquet", index=False)
    grouped.to_parquet(output / "group_permutation_importance.parquet", index=False)
    individual.to_parquet(output / "feature_permutation_importance.parquet", index=False)
    ablation.to_parquet(output / "redundancy_pruning_ablation.parquet", index=False)
    run = {
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "features": {side: list(values) for side, values in selected.items()},
        "redundancy": redundancy,
        "baseline_equal_fold_mean": baseline.groupby("side")
        [["weighted_mae", "weighted_rmse", "weighted_pearson", "spearman"]]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
        "redundancy_pruning_variants": {
            variant: list(values) for variant, values in ablation_variants.items()
        },
        "redundancy_pruning_equal_fold_mean": ablation.groupby("variant")
        [["weighted_mae", "weighted_rmse", "weighted_pearson", "spearman"]]
        .mean()
        .reset_index()
        .to_dict(orient="records"),
        "artifact_path": str(output.relative_to(Path(artifact_root).parent)),
        "caveats": [
            "This is reused historical evidence; Season 2027 remains untouched.",
            "LOSO/LOOCV is a split design, while MAE, RMSE, Pearson, and Spearman are scores.",
            "Permutation importance measures model dependence, not causal player value.",
            "Correlated features can split individual importance; grouped results are primary.",
            "The next-season target is noisy one-year RAPM, so no single score is decisive.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
