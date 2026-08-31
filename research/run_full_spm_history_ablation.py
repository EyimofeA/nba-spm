#!/usr/bin/env python3
"""Refit full and history-complete five-year SPMs against BoxPIPM-style."""

from __future__ import annotations

import argparse
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
    _fit as _fit_box,
)
from nba_impact.models.rapm import load_legacy_possessions
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_impact import _metrics
from nba_impact.models.statistical_model_comparison import _fit_model
from run_aio_prior_bakeoff import _game_metrics, _prior_frame
from run_aio_prior_canonical_followup import (
    _annual_from_frame,
    _center,
    _recover_annual,
    _remap_annual,
    _solve,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "full_spm_history_ablation_v2"
RATING_SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
EVALUATED_RATING_SEASONS = RATING_SEASONS[:-1]
PRIOR_CANDIDATES = ("full_spm", "history_complete_spm", "box_pipm")
MODEL_ORDER = (
    "full_spm",
    "history_complete_spm",
    "box_pipm",
    "zero_prior_rapm",
    "full_spm_aio",
    "history_complete_spm_aio",
    "box_pipm_aio",
)
PRIMARY_PAIRS = {
    frozenset(("full_spm", "history_complete_spm")),
    frozenset(("full_spm_aio", "history_complete_spm_aio")),
    frozenset(("full_spm", "box_pipm")),
    frozenset(("full_spm_aio", "box_pipm_aio")),
    frozenset(("history_complete_spm", "box_pipm")),
    frozenset(("history_complete_spm_aio", "box_pipm_aio")),
}
ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


def _select_box_alpha_rolling_origin(
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    alpha_grid: tuple[float, ...],
) -> float:
    """Select Box15 ridge alpha without using later rolling windows."""
    seasons = tuple(sorted(int(value) for value in train["Window_End"].unique()))
    if len(seasons) < 3:
        raise ValueError("Rolling-origin alpha selection requires three window ends.")
    scores: list[tuple[float, float]] = []
    for alpha in alpha_grid:
        fold_scores = []
        for validation_end in seasons[2:]:
            inner_train = train.loc[train["Window_End"].lt(validation_end)]
            validation = train.loc[train["Window_End"].eq(validation_end)]
            prediction = _fit_box(inner_train, features, target, alpha).predict(
                validation.loc[:, features]
            )
            fold_scores.append(
                _metrics(
                    validation[target].to_numpy(dtype=float),
                    prediction,
                    validation["sample_weight"].to_numpy(dtype=float),
                )["weighted_rmse"]
            )
        scores.append((float(np.mean(fold_scores)), alpha))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def _load_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "frozen_reused_diagnostic",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must equal {value!r}.")
    rating_seasons = tuple(contract["information_cutoff"]["rating_seasons"])
    evaluated = tuple(contract["information_cutoff"]["evaluated_rating_seasons"])
    test_seasons = tuple(contract["information_cutoff"]["test_seasons"])
    if rating_seasons != RATING_SEASONS:
        raise ValueError("The rating-season contract changed.")
    if evaluated != EVALUATED_RATING_SEASONS:
        raise ValueError("The evaluated rating-season contract changed.")
    if test_seasons != tuple(season + 1 for season in EVALUATED_RATING_SEASONS):
        raise ValueError("The test-season contract changed.")
    if int(contract["information_cutoff"]["final_unscored_rating_season"]) != 2026:
        raise ValueError("The final current rating must remain 2026 and unscored.")
    if contract["information_cutoff"]["season_2027"] != "forbidden":
        raise ValueError("Season 2027 must remain forbidden.")
    return json.loads(json.dumps(contract, default=str))


def _load_panel(
    features_path: Path,
    targets_path: Path,
    feature_manifest_path: Path,
    removed: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    features = pd.read_parquet(features_path)
    targets = pd.read_parquet(targets_path)
    manifest = json.loads(feature_manifest_path.read_text())
    selected = {
        side: tuple(manifest["feature_contract"][side])
        for side in ("offense", "defense")
    }
    if missing := sorted(set(removed) - set(selected["defense"])):
        raise ValueError(f"Declared removals are not in the defense contract: {missing}")
    required = {
        "PLAYER_ID",
        "Window_End",
        *selected["offense"],
        *selected["defense"],
    }
    if missing := sorted(required - set(features.columns)):
        raise ValueError(f"Feature panel lacks {missing}.")
    target_columns = {
        "PLAYER_ID",
        "Window_End",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
    }
    if missing := sorted(target_columns - set(targets.columns)):
        raise ValueError(f"Target panel lacks {missing}.")
    panel = features.merge(
        targets[list(target_columns)],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    if panel["Window_End"].max() >= 2027:
        raise ValueError("Season 2027 entered the feature or target panel.")
    if panel.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("The refit panel has duplicate player-window keys.")
    return panel, selected


def _fit_priors(
    panel: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
    removed: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    history_defense = tuple(
        feature for feature in selected["defense"] if feature not in set(removed)
    )
    rows: list[pd.DataFrame] = []
    target_rows: list[dict] = []
    selection_rows: list[dict] = []
    models: dict[tuple[int, str, str], object] = {}
    features_by_candidate = {
        "full_spm": selected,
        "history_complete_spm": {
            "offense": selected["offense"],
            "defense": history_defense,
        },
        "box_pipm": {
            "offense": BOX_PIPM_STYLE_FEATURES,
            "defense": BOX_PIPM_STYLE_FEATURES,
        },
    }
    for season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 3 or test.empty:
            raise ValueError(f"Rating season {season} lacks chronological history.")
        for candidate in PRIOR_CANDIDATES:
            prior = test[["PLAYER_ID", "Window_End"]].copy()
            for side in ("offense", "defense"):
                fields = features_by_candidate[candidate][side]
                target = f"target_{side}"
                if candidate == "box_pipm":
                    alpha = _select_box_alpha_rolling_origin(
                        train,
                        fields,
                        target,
                        ALPHA_GRID,
                    )
                    model = _fit_box(train, fields, target, alpha)
                    model_description = {"family": "ridge", "alpha": alpha}
                else:
                    model = _fit_model(
                        _frozen_model(side), train, fields, target
                    )
                    model_description = {
                        "family": "histogram_gbm" if side == "offense" else "ridge",
                        "frozen": True,
                    }
                prediction = model.predict(test.loc[:, fields])
                prior[side] = prediction
                models[(season, candidate, side)] = model
                selection_rows.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "component": side,
                        "feature_count": len(fields),
                        **model_description,
                    }
                )
                target_rows.append(
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
            prior["net"] = prior["offense"] + prior["defense"]
            target_rows.append(
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
            rows.append(_prior_frame(prior, candidate))
    return (
        pd.concat(rows, ignore_index=True),
        pd.DataFrame(target_rows),
        pd.DataFrame(selection_rows),
        models,
    )


def _annual_bundles(cache_root: Path, matrix_root: Path) -> tuple[dict, pd.DataFrame]:
    annual = {}
    for season in range(2020, 2024):
        frame = load_legacy_possessions(cache_root, (season,), game_types=("regular",))
        annual[season] = _annual_from_frame(frame, season)
    reconstruction = []
    for season in (2024, 2025, 2026):
        bundle, quality = _recover_annual(
            matrix_root / f"5y_end_{season}", season, annual
        )
        annual[season] = bundle
        reconstruction.append(quality)
    return annual, pd.DataFrame(reconstruction)


def _score_models(
    priors: pd.DataFrame,
    annual: dict,
    matrix_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rating_rows = []
    game_rows = []
    coverage_rows = []
    for season in RATING_SEASONS:
        matrix_dir = matrix_root / f"5y_end_{season}"
        matrix_players = np.load(matrix_dir / "player_ids.npy")
        bundle = _remap_annual(annual[season], matrix_players)
        zero_center = np.zeros(2 * len(bundle.players) + 1)
        zero_beta, zero_intercept = _solve(bundle, zero_center, scale=0.0)
        season_models = [("zero_prior_rapm", zero_beta, zero_intercept)]
        for candidate in PRIOR_CANDIDATES:
            prior = priors.loc[
                priors["candidate"].eq(candidate)
                & priors["Window_End"].eq(season)
            ]
            center, coverage = _center(prior, bundle)
            standalone = center.copy()
            standalone[-1] = zero_beta[-1]
            posterior, posterior_intercept = _solve(bundle, center, scale=1.0)
            season_models.extend(
                [
                    (candidate, standalone, zero_intercept),
                    (f"{candidate}_aio", posterior, posterior_intercept),
                ]
            )
            coverage_rows.append(
                {"candidate": candidate, "rating_season": season, **coverage}
            )
        for candidate, beta, intercept in season_models:
            if season in EVALUATED_RATING_SEASONS:
                game = stored_evaluation_predictions(matrix_dir, beta, intercept)
                game["candidate"] = candidate
                game["rating_season"] = season
                game["test_season"] = season + 1
                game["squared_error"] = (
                    game["actual_margin"] - game["predicted_margin"]
                ) ** 2
                game_rows.append(game)
            n = len(bundle.players)
            rating = pd.DataFrame(
                {
                    "PLAYER_ID": bundle.players,
                    "offense": 100.0 * beta[:n],
                    "defense": -100.0 * beta[n : 2 * n],
                    "Poss_Off": bundle.off_possessions,
                    "Poss_Def": bundle.def_possessions,
                }
            )
            rating["net"] = rating["offense"] + rating["defense"]
            rating["candidate"] = candidate
            rating["rating_season"] = season
            rating_rows.append(rating)
    games = pd.concat(game_rows, ignore_index=True)
    for season, frame in games.groupby("test_season"):
        counts = frame.groupby("candidate")["game_id"].nunique()
        if set(counts.index) != set(MODEL_ORDER) or counts.nunique() != 1:
            raise ValueError(f"Candidates do not score identical {season} games.")
        outcome_hashes = frame.groupby("candidate").apply(
            lambda group: hashlib.sha256(
                "|".join(
                    sorted(
                        group["game_id"].astype(str)
                        + ":"
                        + group["actual_margin"].astype(str)
                    )
                ).encode()
            ).hexdigest(),
            include_groups=False,
        )
        if outcome_hashes.nunique() != 1:
            raise ValueError(f"Candidates do not share {season} outcomes.")
    return pd.concat(rating_rows, ignore_index=True), games, pd.DataFrame(coverage_rows)


def _game_metrics_frames(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (season, candidate), frame in games.groupby(["test_season", "candidate"]):
        rows.append(
            {
                "test_season": int(season),
                "rating_season": int(season) - 1,
                "candidate": candidate,
                **_game_metrics(frame),
            }
        )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values("mean_margin_rmse", kind="stable")
    )
    return folds, summary


def paired_game_bootstrap(
    games: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_errors = []
    for _, frame in games.groupby("test_season", sort=True):
        wide = frame.pivot(
            index="game_id", columns="candidate", values=["actual_margin", "predicted_margin"]
        )
        actual = wide["actual_margin"]
        if actual.isna().any().any() or not actual.nunique(axis=1).eq(1).all():
            raise ValueError("Candidates must score identical complete outcomes.")
        predictions = wide["predicted_margin"].reindex(columns=MODEL_ORDER)
        if predictions.isna().any().any():
            raise ValueError("A candidate missed a scored game.")
        actual_values = actual.iloc[:, 0].to_numpy(dtype=float)
        season_errors.append(
            (actual_values[:, None] - predictions.to_numpy(dtype=float)) ** 2
        )
    point_mse = np.mean([values.mean(axis=0) for values in season_errors], axis=0)
    season_mse = np.asarray([values.mean(axis=0) for values in season_errors])
    rng = np.random.default_rng(seed)
    draw_mse = np.empty((draws, len(MODEL_ORDER)), dtype=float)
    for draw in range(draws):
        draw_mse[draw] = np.mean(
            [
                values[rng.integers(0, len(values), len(values))].mean(axis=0)
                for values in season_errors
            ],
            axis=0,
        )
    ranks = np.argsort(np.argsort(draw_mse, axis=1), axis=1) + 1
    model_rows = []
    for index, candidate in enumerate(MODEL_ORDER):
        low, high = np.quantile(draw_mse[:, index], [0.025, 0.975])
        model_rows.append(
            {
                "candidate": candidate,
                "folds": len(season_errors),
                "equal_season_mse": float(point_mse[index]),
                "equal_season_rmse": float(np.sqrt(point_mse[index])),
                "bootstrap_rmse_95_low": float(np.sqrt(low)),
                "bootstrap_rmse_95_high": float(np.sqrt(high)),
                "probability_best": float(np.mean(ranks[:, index] == 1)),
                "mean_bootstrap_rank": float(ranks[:, index].mean()),
                "bootstrap_draws": draws,
            }
        )
    pair_rows = []
    for left, right in itertools.combinations(MODEL_ORDER, 2):
        left_index = MODEL_ORDER.index(left)
        right_index = MODEL_ORDER.index(right)
        delta = draw_mse[:, left_index] - draw_mse[:, right_index]
        low, high = np.quantile(delta, [0.025, 0.975])
        pair_rows.append(
            {
                "candidate": left,
                "reference": right,
                "primary_comparison": frozenset((left, right)) in PRIMARY_PAIRS,
                "mean_mse_delta": float(point_mse[left_index] - point_mse[right_index]),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_candidate_lower_mse": float(np.mean(delta < 0)),
                "candidate_fold_wins": int(
                    np.sum(season_mse[:, left_index] < season_mse[:, right_index])
                ),
                "reference_fold_wins": int(
                    np.sum(season_mse[:, right_index] < season_mse[:, left_index])
                ),
                "bootstrap_draws": draws,
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(pair_rows)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/experiments/full_spm_history_ablation_v2.yml",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "artifacts/research/full_feature_2014_2026/panels/full_spm_features_2014_2026_v1_4c77ae6acc/five_year_features.parquet",
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=ROOT / "artifacts/research/full_feature_2014_2026/panels/full_spm_features_2014_2026_v1_4c77ae6acc/run.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet",
    )
    parser.add_argument(
        "--possession-cache",
        type=Path,
        default=ROOT / "rapm/data/possession_cache",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices",
    )
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    contract = _load_contract(args.contract)
    removed = tuple(contract["removed_defense_features"])
    source_paths = {
        "contract": args.contract,
        "features": args.features,
        "feature_manifest": args.feature_manifest,
        "targets": args.targets,
        "runner": Path(__file__),
        **{
            f"matrix_{season}": args.matrix_root / f"5y_end_{season}/manifest.json"
            for season in RATING_SEASONS
        },
        **{
            f"possessions_{season}": args.possession_cache / f"matchups_{season}.parquet"
            for season in range(2020, 2024)
        },
    }
    if missing := [str(path) for path in source_paths.values() if not path.exists()]:
        raise FileNotFoundError(f"Missing refit inputs: {missing}")
    panel, selected = _load_panel(
        args.features, args.targets, args.feature_manifest, removed
    )
    priors, target_metrics, selections, models = _fit_priors(
        panel, selected, removed
    )
    annual, reconstruction = _annual_bundles(
        args.possession_cache, args.matrix_root
    )
    ratings, games, coverage = _score_models(priors, annual, args.matrix_root)
    fold_metrics, summary = _game_metrics_frames(games)
    bootstrap_models, bootstrap_pairs = paired_game_bootstrap(
        games, draws=args.draws, seed=args.seed
    )

    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_seasons": list(RATING_SEASONS),
        "evaluated_rating_seasons": list(EVALUATED_RATING_SEASONS),
        "test_seasons": [season + 1 for season in EVALUATED_RATING_SEASONS],
        "final_unscored_rating_season": 2026,
        "removed_defense_features": list(removed),
        "feature_counts": {
            "full_offense": len(selected["offense"]),
            "full_defense": len(selected["defense"]),
            "history_complete_offense": len(selected["offense"]),
            "history_complete_defense": len(selected["defense"]) - len(removed),
            "box": len(BOX_PIPM_STYLE_FEATURES),
        },
        "rapm_penalties": {"offense": 3000.0, "defense": 3000.0, "home": 300.0},
        "center_scale": 1.0,
        "bootstrap": {
            "draws": args.draws,
            "seed": args.seed,
            "unit": "whole game within test season",
            "aggregation": "equal-season mean MSE",
        },
        "sources": {
            name: {"path": _relative(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = (
        args.artifact_root
        / "research"
        / "full_spm_history_ablation"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "priors.parquet": priors,
        "target_metrics.parquet": target_metrics,
        "model_selection.parquet": selections,
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
    model_root = output / "models"
    model_root.mkdir()
    for (season, candidate, side), model in models.items():
        joblib.dump(model, model_root / f"{season}_{candidate}_{side}.joblib")
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "next-season game margin from prior-season player ratings",
        "contract": contract,
        "config": config,
        "quality": {
            "panel_rows": len(panel),
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
            "season_2027_loaded": False,
            "final_rating_season": int(ratings["rating_season"].max()),
        },
        "files": {},
        "forbidden_interpretation": (
            "All five test seasons are reused evidence. This run cannot promote a public model."
        ),
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "sha256": sha256_file(output / name),
            "rows": len(frame),
        }
    run["files"]["models"] = {
        "path": "models",
        "files": len(models),
    }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print("\nPrimary paired comparisons")
    print(
        bootstrap_pairs.loc[bootstrap_pairs["primary_comparison"]]
        .sort_values("mean_mse_delta")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
