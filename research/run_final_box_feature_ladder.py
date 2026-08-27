#!/usr/bin/env python3
"""Run the frozen final BoxPIPM feature ladder and interpretation audit."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit,
    _select_alpha,
)
from nba_impact.models.rapm_sufficient_statistics import (
    stored_evaluation_predictions,
)
from nba_impact.models.statistical_impact import _metrics
from run_aio_prior_bakeoff import _game_metrics, _prior_frame
from run_aio_prior_canonical_followup import _center, _remap_annual, _solve

import run_full_spm_history_ablation as base


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "final_box_feature_ladder_v1"
CONTRACT = ROOT / "research/experiments/final_box_feature_ladder_v1.yml"
FEATURE_RUN = (
    ROOT
    / "artifacts/research/complete_feature_coverage"
    / "semantically_complete_spm_features_v1_8be676bd0f"
)
TARGETS = (
    ROOT
    / "artifacts/models/five_year_target_spm"
    / "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
)
MATRIX_ROOT = (
    ROOT
    / "research/rapm_lab/outputs/rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
PLAYER_SHEET_2026 = (
    ROOT
    / "data/lake/bronze/gabriel_player_sheets"
    / "revision=54b57cf/year_totals/2026.parquet"
)
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260827
PERMUTATION_SEED = 20260828


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_reused_diagnostic",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must equal {value!r}.")
    cutoff = contract["information_cutoff"]
    if tuple(cutoff["rating_seasons"]) != base.RATING_SEASONS:
        raise ValueError("The rating-season contract changed.")
    if tuple(cutoff["evaluated_rating_seasons"]) != base.EVALUATED_RATING_SEASONS:
        raise ValueError("The evaluated rating-season contract changed.")
    if cutoff["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return contract


def _candidate_features(
    contract: dict,
    selected: dict[str, tuple[str, ...]],
) -> tuple[dict[str, dict[str, tuple[str, ...]]], dict[str, dict]]:
    current = {
        "offense": list(BOX_PIPM_STYLE_FEATURES),
        "defense": list(BOX_PIPM_STYLE_FEATURES),
    }
    candidates = {
        "box_15": {side: tuple(fields) for side, fields in current.items()}
    }
    family_by_name: dict[str, dict] = {}
    for family in contract["feature_families"]:
        name = family["family"]
        side = family["side"]
        fields = tuple(family["features"])
        missing = sorted(set(fields) - set(selected[side]))
        if missing:
            raise ValueError(f"Family {name} lacks selected {side} fields: {missing}")
        current[side].extend(fields)
        current[side] = list(dict.fromkeys(current[side]))
        candidate = f"box_plus_{name}"
        candidates[candidate] = {
            component: tuple(values) for component, values in current.items()
        }
        family_by_name[name] = {
            "side": side,
            "features": fields,
            "candidate": candidate,
        }
    candidates["completed_full_ridge_ceiling"] = selected
    expected_ladder = tuple(contract["ladder"]["eligible_steps"])
    actual_ladder = tuple(
        name
        for name in candidates
        if name not in {"box_15", "completed_full_ridge_ceiling"}
    )
    if actual_ladder != expected_ladder:
        raise ValueError("The constructed ladder differs from the frozen contract.")
    return candidates, family_by_name


def _fit_priors(
    panel: pd.DataFrame,
    candidates: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    prior_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    alpha_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    models: dict[tuple[int, str, str], object] = {}
    for season in base.RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 3 or test.empty:
            raise ValueError(f"Rating season {season} lacks chronological history.")
        for candidate, sides in candidates.items():
            prior = test[["PLAYER_ID", "Window_End"]].copy()
            for side in ("offense", "defense"):
                fields = sides[side]
                target = f"target_{side}"
                alpha = _select_alpha(
                    train.rename(columns={"Window_End": "Season"}),
                    fields,
                    target,
                    ALPHA_GRID,
                )
                model = _fit(train, fields, target, alpha)
                prediction = model.predict(test.loc[:, fields])
                prior[side] = prediction
                models[(season, candidate, side)] = model
                alpha_rows.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "side": side,
                        "selected_alpha": alpha,
                        "feature_count": len(fields),
                        "train_window_min": int(train["Window_End"].min()),
                        "train_window_max": int(train["Window_End"].max()),
                    }
                )
                metric_rows.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "component": side,
                        "players": len(test),
                        **_metrics(
                            test[target].to_numpy(dtype=float),
                            prediction,
                            test["sample_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
                indicator = model.named_steps["impute"].indicator_.features_
                if len(indicator):
                    raise ValueError("The completed panel created imputation indicators.")
                coefficients = model.named_steps["ridge"].coef_
                if len(coefficients) != len(fields):
                    raise ValueError("Ridge coefficient count differs from feature count.")
                coefficient_rows.extend(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "side": side,
                        "feature": feature,
                        "standardized_coefficient": float(coefficient),
                    }
                    for feature, coefficient in zip(fields, coefficients, strict=True)
                )
            prior["net"] = prior["offense"] + prior["defense"]
            metric_rows.append(
                {
                    "rating_season": season,
                    "candidate": candidate,
                    "component": "net",
                    "players": len(test),
                    **_metrics(
                        test["target_net"].to_numpy(dtype=float),
                        prior["net"].to_numpy(dtype=float),
                        test["sample_weight"].to_numpy(dtype=float),
                    ),
                }
            )
            prior_rows.append(_prior_frame(prior, candidate))
    return (
        pd.concat(prior_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(alpha_rows),
        pd.DataFrame(coefficient_rows),
        models,
    )


def _model_order(candidates: tuple[str, ...]) -> tuple[str, ...]:
    return (
        *candidates,
        "zero_prior_rapm",
        *(f"{candidate}_aio" for candidate in candidates),
    )


def _primary_pairs(candidates: tuple[str, ...]) -> set[frozenset[str]]:
    rows = {
        frozenset(("box_15", candidate))
        for candidate in candidates
        if candidate != "box_15"
    }
    rows.update(
        frozenset(("box_15_aio", f"{candidate}_aio"))
        for candidate in candidates
        if candidate != "box_15"
    )
    ladder = tuple(
        candidate for candidate in candidates if candidate != "completed_full_ridge_ceiling"
    )
    rows.update(frozenset(pair) for pair in itertools.pairwise(ladder))
    rows.update(
        frozenset((f"{left}_aio", f"{right}_aio"))
        for left, right in itertools.pairwise(ladder)
    )
    return rows


def _select_candidate(
    contract: dict,
    intervals: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[str, pd.DataFrame]:
    box_mse = float(
        intervals.loc[intervals["candidate"].eq("box_15_aio"), "equal_season_mse"].iloc[0]
    )
    box_correlation = float(
        summary.loc[
            summary["candidate"].eq("box_15_aio"), "mean_margin_correlation"
        ].iloc[0]
    )
    eligible = set(contract["ladder"]["eligible_steps"])
    rows = intervals.loc[
        intervals["candidate"].isin({f"{name}_aio" for name in eligible})
    ].copy()
    rows = rows.merge(
        summary[["candidate", "mean_margin_correlation"]],
        on="candidate",
        how="left",
        validate="one_to_one",
    )
    rows["beats_box_point_mse"] = rows["equal_season_mse"].lt(box_mse)
    rows["passes_correlation_guard"] = rows["mean_margin_correlation"].ge(
        box_correlation - 0.01
    )
    rows["eligible_for_selection"] = (
        rows["beats_box_point_mse"] & rows["passes_correlation_guard"]
    )
    valid = rows.loc[rows["eligible_for_selection"]].sort_values(
        ["equal_season_mse", "candidate"], kind="stable"
    )
    selected_aio = "box_15_aio" if valid.empty else str(valid.iloc[0]["candidate"])
    rows["selected"] = rows["candidate"].eq(selected_aio)
    return selected_aio.removesuffix("_aio"), rows


def _importance_groups(
    selected_candidate: str,
    candidates: dict[str, dict[str, tuple[str, ...]]],
    family_by_name: dict[str, dict],
) -> dict[str, dict[str, tuple[str, ...]]]:
    selected = candidates[selected_candidate]
    groups = {
        "traditional_box": {
            "offense": BOX_PIPM_STYLE_FEATURES,
            "defense": BOX_PIPM_STYLE_FEATURES,
        }
    }
    for name, spec in family_by_name.items():
        side = spec["side"]
        fields = tuple(feature for feature in spec["features"] if feature in selected[side])
        if fields:
            groups[name] = {
                "offense": fields if side == "offense" else (),
                "defense": fields if side == "defense" else (),
            }
    return groups


def _group_permutation_importance(
    *,
    selected_candidate: str,
    groups: dict[str, dict[str, tuple[str, ...]]],
    candidate_features: dict[str, tuple[str, ...]],
    panel: pd.DataFrame,
    models: dict,
    annual: dict,
    games: pd.DataFrame,
    repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    base_games = games.loc[
        games["candidate"].eq(f"{selected_candidate}_aio")
    ].copy()
    for group_index, (group, sides) in enumerate(groups.items()):
        for repeat in range(repeats):
            for season in base.EVALUATED_RATING_SEASONS:
                test = panel.loc[panel["Window_End"].eq(season)].copy()
                rng = np.random.default_rng(
                    PERMUTATION_SEED + group_index * 10_000 + repeat * 100 + season
                )
                order = rng.permutation(len(test))
                prior = test[["PLAYER_ID", "Window_End"]].copy()
                for side in ("offense", "defense"):
                    fields = candidate_features[side]
                    x = test.loc[:, fields].copy()
                    shuffled = sides[side]
                    if shuffled:
                        x.loc[:, shuffled] = x.iloc[order][list(shuffled)].to_numpy()
                    prior[side] = models[(season, selected_candidate, side)].predict(x)
                prior["net"] = prior["offense"] + prior["defense"]
                matrix_dir = MATRIX_ROOT / f"5y_end_{season}"
                players = np.load(matrix_dir / "player_ids.npy")
                bundle = _remap_annual(annual[season], players)
                center, _ = _center(_prior_frame(prior, selected_candidate), bundle)
                beta, intercept = _solve(bundle, center, scale=1.0)
                permuted_games = stored_evaluation_predictions(
                    matrix_dir, beta, intercept
                )
                baseline = base_games.loc[base_games["rating_season"].eq(season)]
                if set(permuted_games["game_id"]) != set(baseline["game_id"]):
                    raise ValueError("Permutation scored a different game set.")
                permuted_metric = _game_metrics(permuted_games)
                baseline_metric = _game_metrics(baseline)
                permuted_mse = permuted_metric["margin_rmse"] ** 2
                baseline_mse = baseline_metric["margin_rmse"] ** 2
                rows.append(
                    {
                        "group": group,
                        "repeat": repeat,
                        "rating_season": season,
                        "test_season": season + 1,
                        "feature_count": len(sides["offense"]) + len(sides["defense"]),
                        "permuted_mse": permuted_mse,
                        "baseline_mse": baseline_mse,
                        "mse_increase": permuted_mse - baseline_mse,
                        "correlation_drop": (
                            baseline_metric["margin_correlation"]
                            - permuted_metric["margin_correlation"]
                        ),
                    }
                )
    detail = pd.DataFrame(rows)
    repeat_summary = (
        detail.groupby(["group", "repeat", "feature_count"], as_index=False)
        .agg(
            equal_season_mse_increase=("mse_increase", "mean"),
            equal_season_correlation_drop=("correlation_drop", "mean"),
        )
    )
    summary = (
        repeat_summary.groupby(["group", "feature_count"], as_index=False)
        .agg(
            mean_mse_increase=("equal_season_mse_increase", "mean"),
            std_mse_increase=("equal_season_mse_increase", "std"),
            mean_correlation_drop=("equal_season_correlation_drop", "mean"),
            positive_mse_repeats=(
                "equal_season_mse_increase",
                lambda values: int(np.sum(np.asarray(values) > 0)),
            ),
            repeats=("repeat", "nunique"),
        )
        .sort_values("mean_mse_increase", ascending=False, kind="stable")
    )
    return detail, summary


def _coefficient_summary(
    coefficients: pd.DataFrame,
    selected_candidate: str,
    groups: dict[str, dict[str, tuple[str, ...]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = coefficients.loc[
        coefficients["candidate"].eq(selected_candidate)
    ].copy()
    feature_group = {}
    for group, sides in groups.items():
        for side, fields in sides.items():
            feature_group.update({(side, feature): group for feature in fields})
    selected["group"] = [
        feature_group[(side, feature)]
        for side, feature in selected[["side", "feature"]].itertuples(index=False)
    ]
    summary = (
        selected.groupby(["side", "group", "feature"], as_index=False)
        .agg(
            mean_standardized_coefficient=("standardized_coefficient", "mean"),
            mean_absolute_coefficient=(
                "standardized_coefficient",
                lambda values: float(np.mean(np.abs(values))),
            ),
            positive_folds=(
                "standardized_coefficient",
                lambda values: int(np.sum(np.asarray(values) > 0)),
            ),
            negative_folds=(
                "standardized_coefficient",
                lambda values: int(np.sum(np.asarray(values) < 0)),
            ),
            folds=("rating_season", "nunique"),
        )
        .sort_values(
            ["side", "mean_absolute_coefficient"],
            ascending=[True, False],
            kind="stable",
        )
    )
    return selected, summary


def _identity() -> pd.DataFrame:
    source = pd.read_parquet(
        PLAYER_SHEET_2026,
        columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"],
    ).drop_duplicates()
    if source["PLAYER_ID"].duplicated().any():
        raise ValueError("The 2026 player sheet has conflicting player identities.")
    return source


def main() -> None:
    contract = _load_contract()
    panel, selected = base._load_panel(
        FEATURE_RUN / "five_year_features.parquet",
        TARGETS,
        FEATURE_RUN / "run.json",
        (),
    )
    all_selected = list(dict.fromkeys((*selected["offense"], *selected["defense"])))
    if panel[all_selected].isna().any().any():
        raise ValueError("The completed panel contains missing selected inputs.")
    if not np.isfinite(panel[all_selected].to_numpy(dtype=float)).all():
        raise ValueError("The completed panel contains nonfinite selected inputs.")
    candidates, family_by_name = _candidate_features(contract, selected)
    candidate_names = tuple(candidates)
    priors, target_metrics, alpha_selection, coefficients, models = _fit_priors(
        panel, candidates
    )
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    model_order = _model_order(candidate_names)
    primary_pairs = _primary_pairs(candidate_names)
    base.PRIOR_CANDIDATES = candidate_names
    base.MODEL_ORDER = model_order
    base.PRIMARY_PAIRS = primary_pairs
    ratings, games, prior_coverage = base._score_models(
        priors, annual, MATRIX_ROOT
    )
    folds, summary = base._game_metrics_frames(games)
    intervals, paired = base.paired_game_bootstrap(
        games, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
    )
    selected_candidate, selection = _select_candidate(contract, intervals, summary)
    groups = _importance_groups(selected_candidate, candidates, family_by_name)
    importance_detail, importance_summary = _group_permutation_importance(
        selected_candidate=selected_candidate,
        groups=groups,
        candidate_features=candidates[selected_candidate],
        panel=panel,
        models=models,
        annual=annual,
        games=games,
        repeats=int(contract["interpretability"]["repeats"]),
    )
    selected_coefficients, coefficient_summary = _coefficient_summary(
        coefficients, selected_candidate, groups
    )

    identity = _identity()
    final_ratings = ratings.loc[
        ratings["rating_season"].eq(2026)
        & ratings["candidate"].isin(
            (selected_candidate, f"{selected_candidate}_aio")
        )
    ].merge(identity, on="PLAYER_ID", how="left", validate="many_to_one")
    final_ratings["rank"] = final_ratings.groupby("candidate")["net"].rank(
        method="min", ascending=False
    )
    final_ratings = final_ratings.sort_values(
        ["candidate", "rank", "PLAYER_ID"], kind="stable"
    )

    feature_rows = []
    for candidate, sides in candidates.items():
        for side, fields in sides.items():
            feature_rows.extend(
                {
                    "candidate": candidate,
                    "side": side,
                    "feature_order": index,
                    "feature": feature,
                }
                for index, feature in enumerate(fields, start=1)
            )
    candidate_feature_table = pd.DataFrame(feature_rows)

    source_paths = {
        "contract": CONTRACT,
        "features": FEATURE_RUN / "five_year_features.parquet",
        "feature_manifest": FEATURE_RUN / "run.json",
        "targets": TARGETS,
        "player_sheet_2026": PLAYER_SHEET_2026,
        "runner": Path(__file__),
        **{
            f"matrix_{season}": MATRIX_ROOT / f"5y_end_{season}/manifest.json"
            for season in base.RATING_SEASONS
        },
        **{
            f"possessions_{season}": POSSESSION_CACHE / f"matchups_{season}.parquet"
            for season in range(2020, 2024)
        },
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": sha256_file(CONTRACT),
        "candidate_order": list(candidate_names),
        "model_order": list(model_order),
        "alpha_grid": list(ALPHA_GRID),
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "whole game within test season",
            "aggregation": "equal-season mean MSE",
        },
        "permutation": {
            "seed": PERMUTATION_SEED,
            "repeats": int(contract["interpretability"]["repeats"]),
            "unit": "player row within rating season",
            "score": "downstream next-season AIO game-margin MSE",
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        },
    }
    run_identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = (
        ROOT
        / "artifacts/research/final_box_feature_ladder"
        / f"{EXPERIMENT_ID}_{run_identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "candidate_features.parquet": candidate_feature_table,
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "alpha_selection.parquet": alpha_selection,
        "ratings.parquet": ratings,
        "final_2026_leaderboards.parquet": final_ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": intervals,
        "paired_bootstrap.parquet": paired,
        "selection.parquet": selection,
        "prior_coverage.parquet": prior_coverage,
        "matrix_reconstruction.parquet": reconstruction,
        "group_permutation_detail.parquet": importance_detail,
        "group_permutation_summary.parquet": importance_summary,
        "selected_standardized_coefficients.parquet": selected_coefficients,
        "coefficient_summary.parquet": coefficient_summary,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    model_root = output / "models"
    model_root.mkdir()
    for (season, candidate, side), model in models.items():
        joblib.dump(model, model_root / f"{season}_{candidate}_{side}.joblib")
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "next-season game margin from a five-year statistical prior",
        "selected_candidate": selected_candidate,
        "selected_aio": f"{selected_candidate}_aio",
        "contract": contract,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "selected_input_missing_values": 0,
            "selected_input_nonfinite_values": 0,
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
            "final_leaderboard_name_coverage": float(
                final_ratings["PLAYER_NAME"].notna().mean()
            ),
            "season_2027_loaded": False,
        },
        "files": {},
        "models": {
            "path": "models",
            "count": len(models),
        },
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(f"selected={selected_candidate}")
    print(
        summary.loc[summary["candidate"].str.endswith("_aio")]
        .sort_values("mean_margin_rmse", kind="stable")
        .to_string(index=False)
    )
    print("\nGrouped downstream permutation importance")
    print(importance_summary.to_string(index=False))


if __name__ == "__main__":
    main()
