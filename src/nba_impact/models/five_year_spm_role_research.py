"""Chronological role and zone-shotmaking research for the five-year SPM."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.five_year_spm_feature_audit import _score
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_model_comparison import _fit_model


EXPERIMENT_ID = "five_year_spm_role_research_v1"
ZONE_SPECS = (
    ("at_rim", 2.0),
    ("short_mid", 2.0),
    ("long_mid", 2.0),
    ("corner3", 3.0),
    ("arc3", 3.0),
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_role_context(role_dir: str | Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load the public annual role assignments into deterministic numeric fields."""
    root = Path(role_dir)
    merged: pd.DataFrame | None = None
    labels: dict[str, list[str]] = {}
    for side in ("offense", "defense"):
        files = sorted(root.glob(f"roles-{side}-*.json"))
        if not files:
            raise FileNotFoundError(f"No {side} role files found under {root}.")
        frame = pd.concat((pd.read_json(path) for path in files), ignore_index=True)
        frame = frame.rename(columns={"Season": "Window_End"})
        if frame.duplicated(["PLAYER_ID", "Window_End"]).any():
            raise ValueError(f"{side} roles contain duplicate player-season keys.")
        prefix = "off_role" if side == "offense" else "def_role"
        labels[side] = sorted(frame["raw_role"].dropna().astype(str).unique().tolist())
        output = frame[["PLAYER_ID", "Window_End", "raw_role", "x", "y"]].copy()
        output[f"{prefix}_known"] = output["raw_role"].notna().astype(float)
        output[f"{prefix}_x"] = pd.to_numeric(output["x"], errors="coerce")
        output[f"{prefix}_y"] = pd.to_numeric(output["y"], errors="coerce")
        output[f"{prefix}_label"] = output["raw_role"].astype("string")
        for label in labels[side]:
            output[f"{prefix}_{_slug(label)}"] = output["raw_role"].eq(label).astype(float)
        output = output.drop(columns=["raw_role", "x", "y"])
        merged = output if merged is None else merged.merge(
            output,
            on=["PLAYER_ID", "Window_End"],
            how="outer",
            validate="one_to_one",
        )
    assert merged is not None
    return merged, labels


def add_zone_shotmaking(features: pd.DataFrame, *, prior_attempts: float = 200.0) -> pd.DataFrame:
    """Add leave-one-player-out zone-adjusted points above expectation.

    The expectation is the same-window league accuracy in each shot zone after
    removing the player being estimated. The result is expressed per 100
    offensive possessions and then shrunk toward zero by shot attempts.
    """
    required = {"PLAYER_ID", "Window_End", "OffPoss"}
    for zone, _ in ZONE_SPECS:
        required.update((f"{zone}_fga_p100", f"{zone}_accuracy"))
    if missing := sorted(required - set(features.columns)):
        raise ValueError(f"Zone shotmaking is missing fields {missing}.")

    output = features.copy()
    off_poss = pd.to_numeric(output["OffPoss"], errors="coerce").clip(lower=0)
    total_attempts = pd.Series(0.0, index=output.index)
    total_actual_points = pd.Series(0.0, index=output.index)
    total_expected_points = pd.Series(0.0, index=output.index)
    for zone, point_value in ZONE_SPECS:
        attempts = (
            pd.to_numeric(output[f"{zone}_fga_p100"], errors="coerce").clip(lower=0)
            * off_poss
            / 100.0
        )
        accuracy = pd.to_numeric(output[f"{zone}_accuracy"], errors="coerce").clip(0, 1)
        makes = attempts * accuracy
        valid = attempts.notna() & accuracy.notna()
        attempts = attempts.where(valid, 0.0)
        makes = makes.where(valid, 0.0)
        group = output["Window_End"]
        league_attempts = attempts.groupby(group).transform("sum")
        league_makes = makes.groupby(group).transform("sum")
        other_attempts = league_attempts - attempts
        expected_accuracy = (league_makes - makes) / other_attempts.where(other_attempts.gt(0))
        total_attempts += attempts
        total_actual_points += point_value * makes
        total_expected_points += point_value * attempts * expected_accuracy.fillna(0.0)

    raw = 100.0 * (total_actual_points - total_expected_points) / off_poss.where(off_poss.gt(0))
    reliability = total_attempts / (total_attempts + float(prior_attempts))
    output["zone_shotmaking_p100_raw"] = raw.where(total_attempts.gt(0))
    output["zone_shotmaking_p100_eb"] = (
        reliability * output["zone_shotmaking_p100_raw"]
    ).where(total_attempts.gt(0))
    output["zone_shotmaking_attempts"] = total_attempts
    return output


def _role_columns(frame: pd.DataFrame, side: str) -> tuple[str, ...]:
    prefix = "off_role" if side == "offense" else "def_role"
    return tuple(
        column
        for column in frame.columns
        if column.startswith(f"{prefix}_") and not column.endswith("_label")
    )


def _expert_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    side: str,
    features: tuple[str, ...],
    minimum_expert_rows: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Fit hard role-specific experts with a global fallback for small roles."""
    target = f"target_{side}"
    label = "off_role_label" if side == "offense" else "def_role_label"
    global_model = _fit_model(_frozen_model(side), train, features, target)
    prediction = global_model.predict(test.loc[:, features])
    counts: dict[str, int] = {}
    for role in sorted(test[label].dropna().astype(str).unique()):
        role_train = train.loc[train[label].eq(role)]
        counts[role] = int(len(role_train))
        if len(role_train) < minimum_expert_rows:
            continue
        model = _fit_model(_frozen_model(side), role_train, features, target)
        mask = test[label].eq(role).to_numpy()
        prediction[mask] = model.predict(test.loc[mask, features])
    return prediction, counts


def evaluate_role_challengers(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
    *,
    rating_seasons: tuple[int, ...] = (2020, 2021, 2022),
    minimum_expert_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score role and shotmaking challengers on following-season annual RAPM."""
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
    role_features = {
        side: _role_columns(features, side) for side in ("offense", "defense")
    }
    variants = {
        "baseline": {
            "offense": selected["offense"],
            "defense": selected["defense"],
        },
        "zone_shotmaking": {
            "offense": (*selected["offense"], "zone_shotmaking_p100_eb"),
            "defense": selected["defense"],
        },
        "role_context": {
            "offense": (*selected["offense"], *role_features["offense"]),
            "defense": (*selected["defense"], *role_features["defense"]),
        },
        "role_context_plus_shotmaking": {
            "offense": (
                *selected["offense"], *role_features["offense"],
                "zone_shotmaking_p100_eb",
            ),
            "defense": (*selected["defense"], *role_features["defense"]),
        },
    }
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    expert_rows: list[dict] = []
    for rating_season in rating_seasons:
        train = panel.loc[panel["Window_End"].lt(rating_season)].copy()
        target = annual_targets.loc[
            annual_targets["Window_End"].eq(rating_season + 1),
            ["PLAYER_ID", "target_offense", "target_defense", "Poss_Off", "Poss_Def"],
        ].rename(
            columns={
                "target_offense": "next_target_offense",
                "target_defense": "next_target_defense",
                "Poss_Off": "next_Poss_Off",
                "Poss_Def": "next_Poss_Def",
            }
        )
        test = features.loc[features["Window_End"].eq(rating_season)].merge(
            target, on="PLAYER_ID", how="inner", validate="one_to_one"
        )
        # Keep the scored cohort identical across all variants and focus the
        # role test on players for whom both assignments actually exist.
        test = test.loc[
            test["off_role_known"].eq(1.0) & test["def_role_known"].eq(1.0)
        ].copy()
        if train.empty or test.empty:
            raise ValueError(f"Empty role fold for rating season {rating_season}.")
        fold = test[
            [
                "PLAYER_ID", "Window_End", "next_target_offense",
                "next_target_defense", "next_Poss_Off", "next_Poss_Def",
                "off_role_label", "def_role_label",
            ]
        ].copy()
        fold["next_target_net"] = (
            fold["next_target_offense"] + fold["next_target_defense"]
        )
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for variant, feature_map in variants.items():
            predictions[variant] = {}
            for side in ("offense", "defense"):
                columns = tuple(feature_map[side])
                if missing := sorted(set(columns) - set(features.columns)):
                    raise ValueError(f"{variant} is missing {side} fields {missing}.")
                model = _fit_model(_frozen_model(side), train, columns, f"target_{side}")
                predictions[variant][side] = model.predict(test.loc[:, columns])

        predictions["hard_role_experts"] = {}
        for side in ("offense", "defense"):
            prediction, counts = _expert_predictions(
                train,
                test,
                side=side,
                features=selected[side],
                minimum_expert_rows=minimum_expert_rows,
            )
            predictions["hard_role_experts"][side] = prediction
            for role, rows in counts.items():
                expert_rows.append(
                    {
                        "rating_season": rating_season,
                        "side": side,
                        "role": role,
                        "training_rows": rows,
                        "used_expert": rows >= minimum_expert_rows,
                    }
                )

        weight = np.sqrt(
            np.minimum(fold["next_Poss_Off"], fold["next_Poss_Def"]).clip(lower=1)
        ).to_numpy(dtype=float)
        for variant, side_predictions in predictions.items():
            for side in ("offense", "defense"):
                fold[f"prediction_{side}"] = side_predictions[side]
            fold["prediction_net"] = fold["prediction_offense"] + fold["prediction_defense"]
            for side in ("offense", "defense", "net"):
                metric_rows.append(
                    {
                        "variant": variant,
                        "rating_season": rating_season,
                        "test_season": rating_season + 1,
                        "side": side,
                        "test_rows": int(len(fold)),
                        **_score(
                            fold[f"next_target_{side}"].to_numpy(dtype=float),
                            fold[f"prediction_{side}"].to_numpy(dtype=float),
                            weight,
                        ),
                    }
                )
            saved = fold[
                [
                    "PLAYER_ID", "Window_End", "off_role_label", "def_role_label",
                    "next_target_offense", "next_target_defense", "next_target_net",
                    "prediction_offense", "prediction_defense", "prediction_net",
                ]
            ].copy()
            saved["variant"] = variant
            saved["test_season"] = rating_season + 1
            prediction_rows.append(saved)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(expert_rows),
    )


def _summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics.groupby(["variant", "side"], as_index=False)[
            ["weighted_mae", "weighted_rmse", "weighted_pearson", "spearman"]
        ]
        .mean()
    )
    baseline = metrics.loc[metrics["variant"].eq("baseline")].drop(columns="variant")
    comparison = metrics.loc[~metrics["variant"].eq("baseline")].merge(
        baseline,
        on=["rating_season", "test_season", "side", "test_rows"],
        suffixes=("_candidate", "_baseline"),
        validate="many_to_one",
    )
    for metric in ("weighted_mae", "weighted_rmse"):
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_candidate"] - comparison[f"{metric}_baseline"]
        )
    for metric in ("weighted_pearson", "spearman"):
        comparison[f"{metric}_delta"] = (
            comparison[f"{metric}_candidate"] - comparison[f"{metric}_baseline"]
        )
    deltas = (
        comparison.groupby(["variant", "side"], as_index=False)[
            [
                "weighted_mae_delta", "weighted_rmse_delta",
                "weighted_pearson_delta", "spearman_delta",
            ]
        ]
        .mean()
    )
    return summary, deltas


def run_five_year_spm_role_research(
    *,
    features_path: str | Path,
    targets_path: str | Path,
    reference_run_path: str | Path,
    role_dir: str | Path,
    existing_skill_run_path: str | Path,
    artifact_root: str | Path,
    rating_seasons: tuple[int, ...] = (2020, 2021, 2022),
    minimum_expert_rows: int = 100,
) -> dict:
    """Write a reproducible role-conditioned next-season research artifact."""
    reference = json.loads(Path(reference_run_path).read_text())
    raw_selected = reference.get("features") or reference.get("selected_features")
    if not isinstance(raw_selected, dict):
        raise ValueError("Reference run has no selected feature contract.")
    selected = {side: tuple(raw_selected[side]) for side in ("offense", "defense")}
    role_files = sorted(Path(role_dir).glob("roles-*.json"))
    if not role_files:
        raise FileNotFoundError("Role source directory contains no annual role files.")
    source_hashes = {
        "features": sha256_file(features_path),
        "targets": sha256_file(targets_path),
        "reference_run": sha256_file(reference_run_path),
        "existing_skill_run": sha256_file(existing_skill_run_path),
        "source_code": sha256_file(Path(__file__)),
        "roles": hashlib.sha256(
            "".join(sha256_file(path) for path in role_files).encode()
        ).hexdigest(),
    }
    config = {
        "rating_seasons": list(rating_seasons),
        "test_seasons": [season + 1 for season in rating_seasons],
        "primary_metric": "equal-fold mean next-season possession-weighted Pearson",
        "target": "following-season one-year zero-prior RAPM",
        "training_label": "five-year zero-prior RAPM ending in the rating season",
        "minimum_expert_rows": minimum_expert_rows,
        "zone_shotmaking_prior_attempts": 200.0,
        "source_hashes": source_hashes,
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    run_id = f"{EXPERIMENT_ID}_{identity}"
    output = Path(artifact_root) / "research" / "five_year_spm_role_research" / run_id
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())

    features = add_zone_shotmaking(pd.read_parquet(features_path))
    roles, labels = load_role_context(role_dir)
    features = features.merge(
        roles, on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one"
    )
    targets = pd.read_parquet(targets_path)
    metrics, predictions, experts = evaluate_role_challengers(
        features,
        targets,
        selected,
        rating_seasons=rating_seasons,
        minimum_expert_rows=minimum_expert_rows,
    )
    summary, deltas = _summarize(metrics)
    shotmaking = features[
        [
            "PLAYER_ID", "Window_End", "zone_shotmaking_p100_raw",
            "zone_shotmaking_p100_eb", "zone_shotmaking_attempts",
        ]
    ].copy()
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    deltas.to_parquet(output / "deltas_vs_baseline.parquet", index=False)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    experts.to_parquet(output / "expert_coverage.parquet", index=False)
    shotmaking.to_parquet(output / "zone_shotmaking.parquet", index=False)

    run = {
        "run_id": run_id,
        "experiment_id": EXPERIMENT_ID,
        "status": "exploratory_reused_history",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "role_labels": labels,
        "base_features": {side: list(values) for side, values in selected.items()},
        "summary": summary.to_dict(orient="records"),
        "deltas_vs_baseline": deltas.to_dict(orient="records"),
        "paths": {
            "fold_metrics": "fold_metrics.parquet",
            "summary": "summary.parquet",
            "deltas_vs_baseline": "deltas_vs_baseline.parquet",
            "predictions": "predictions.parquet",
            "expert_coverage": "expert_coverage.parquet",
            "zone_shotmaking": "zone_shotmaking.parquet",
        },
        "shotmaking_contract": {
            "name": "zone-adjusted shotmaking",
            "estimand": "points above a leave-one-player-out same-window league shooter with the same rim, short-mid, long-mid, corner-three, and arc-three attempt mix, per 100 offensive possessions",
            "shrinkage": "attempts / (attempts + 200), toward zero",
            "difference_from_existing": "The existing metric adjusts for defender distance and two-versus-three-point mix. This metric adjusts for five shot zones but cannot jointly condition on defender distance because the sources are separate aggregates.",
        },
        "caveats": [
            "Roles describe usage, not value or causal impact.",
            "The public defense role map was developed on 2018-21 data, so only the 2022-to-2023 fold is strictly post-map; earlier role folds are exploratory.",
            "The same player test rows are used for every variant within a fold.",
            "Hard role experts fall back to the global model when a role has fewer than the configured training rows.",
            "The next-season one-year RAPM target is noisy; Pearson is primary, with Spearman and MAE as guardrails.",
            "Season 2027 is untouched and absent.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
