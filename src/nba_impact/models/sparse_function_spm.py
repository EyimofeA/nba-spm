"""Sparse, function-first five-year statistical plus-minus challenger.

The model deliberately uses one observed feature for each declared basketball
function. Separate ridge regressions predict offense and defense five-year RAPM.
Every historical prediction is trained only on earlier complete five-year
windows. The experiment is a challenger and does not alter the public SPM.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features import _aggregate_window, _load_source
from nba_impact.data.statistical_features_v2 import _engineer_window
from nba_impact.models.public_aio_benchmark import build_team_win_benchmark
from nba_impact.models.spm_role_team_win_benchmark import (
    _load_team_games,
    load_bbref_player_team_minutes,
)


EXPERIMENT_ID = "sparse_function_spm_v1"
RIDGE_ALPHA = 3000.0
RATING_SEASONS = (2021, 2022, 2023, 2024, 2025, 2026)
PLAYER_TEST_SEASONS = (2022, 2023, 2024, 2025, 2026)
TEAM_WIN_RATING_SEASONS = (2021, 2022)

FEATURE_SPECS = {
    "offense": (
        (
            "scoring_load",
            "offensive_load_2017_eb_p100",
            "Shrunk shooting, passing, free-throw and turnover workload per 100.",
        ),
        (
            "shotmaking",
            "effective_fg_pct",
            "Field-goal value per attempt, with threes worth 1.5 makes.",
        ),
        (
            "passing_creation",
            "potential_assists_p100_relative",
            "Potential assists per 100 relative to the same-window league mean.",
        ),
        (
            "ball_security",
            "turnover_to_load_2017_eb",
            "Shrunk turnovers divided by offensive load.",
        ),
        (
            "spacing",
            "crafted_spacing_stable_v1",
            "Shrunk three-point volume and accuracy relative to league shot value.",
        ),
        (
            "foul_pressure",
            "shooting_fouls_drawn_p100",
            "Shooting fouls drawn per 100 offensive possessions.",
        ),
        (
            "offensive_rebounding",
            "OREB_p100",
            "Offensive rebounds per 100 offensive possessions.",
        ),
    ),
    "defense": (
        (
            "rim_protection_proxy",
            "BLK_p100_relative",
            "Blocks per 100 relative to the same-window league mean; not rim deterrence.",
        ),
        (
            "disruption",
            "STL_p100_relative",
            "Steals per 100 relative to the same-window league mean.",
        ),
        (
            "contest_involvement",
            "rebound_contest_share",
            "Share of rebound chances on which the player records a contest.",
        ),
        (
            "defensive_rebounding",
            "DREB_p100_relative",
            "Defensive rebounds per 100 relative to the same-window league mean.",
        ),
        (
            "foul_discipline",
            "PF_p100_relative",
            "Personal fouls per 100 relative to the same-window league mean.",
        ),
    ),
}


def selected_features() -> dict[str, tuple[str, ...]]:
    """Return the frozen one-feature-per-function model inputs."""
    return {
        side: tuple(feature for _, feature, _ in specs)
        for side, specs in FEATURE_SPECS.items()
    }


def feature_registry() -> pd.DataFrame:
    """Return the human-readable model contract."""
    rows = []
    for side, specs in FEATURE_SPECS.items():
        for function, feature, description in specs:
            rows.append(
                {
                    "side": side,
                    "function": function,
                    "feature": feature,
                    "description": description,
                }
            )
    return pd.DataFrame(rows)


def build_five_year_features(
    player_sheet_dir: str | Path,
    *,
    window_ends: tuple[int, ...] = tuple(range(2018, 2027)),
) -> pd.DataFrame:
    """Build complete five-year feature windows from one canonical sheet source."""
    root = Path(player_sheet_dir)
    minimum = min(window_ends) - 4
    maximum = max(window_ends)
    if maximum >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    loaded = {
        season: _load_source(root / f"{season}.parquet", season)[0]
        for season in range(minimum, maximum + 1)
    }
    outputs = []
    for end in window_ends:
        frames = [loaded[season] for season in range(end - 4, end + 1)]
        temporal = [
            _aggregate_window([loaded[season]], season)
            for season in range(end - 2, end + 1)
        ]
        outputs.append(
            _engineer_window(_aggregate_window(frames, end), frames, temporal)
        )
    features = pd.concat(outputs, ignore_index=True)
    if features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Five-year feature keys are not unique.")
    expected = set((*selected_features()["offense"], *selected_features()["defense"]))
    if missing := sorted(expected - set(features.columns)):
        raise ValueError(f"Five-year features are missing {missing}.")
    return features


def standardize_within_window(features: pd.DataFrame) -> pd.DataFrame:
    """Convert each input to a same-window player z-score without target data."""
    output = features.copy()
    for feature in (*selected_features()["offense"], *selected_features()["defense"]):
        values = pd.to_numeric(output[feature], errors="coerce")
        center = values.groupby(output["Window_End"]).transform("mean")
        scale = values.groupby(output["Window_End"]).transform("std").replace(0.0, np.nan)
        output[feature] = (values - center) / scale
    return output


def _model(alpha: float = RIDGE_ALPHA) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _fit_side(
    train: pd.DataFrame,
    score: pd.DataFrame,
    *,
    side: str,
    alpha: float,
) -> tuple[np.ndarray, Pipeline]:
    columns = selected_features()[side]
    model = _model(alpha)
    model.fit(
        train.loc[:, columns],
        train[f"target_{side}"],
        model__sample_weight=train["sample_weight"],
    )
    return model.predict(score.loc[:, columns]), model


def _coefficient_rows(
    model: Pipeline,
    *,
    side: str,
    rating_season: int,
) -> list[dict]:
    inputs = np.asarray(selected_features()[side], dtype=object)
    transformed = model.named_steps["impute"].get_feature_names_out(inputs)
    coefficients = model.named_steps["model"].coef_
    rows = [
        {
            "rating_season": rating_season,
            "side": side,
            "term": "intercept",
            "coefficient": float(model.named_steps["model"].intercept_),
        }
    ]
    rows.extend(
        {
            "rating_season": rating_season,
            "side": side,
            "term": str(term),
            "coefficient": float(value),
        }
        for term, value in zip(transformed, coefficients, strict=True)
    )
    return rows


def fit_historical_predictions(
    features: pd.DataFrame,
    five_year_targets: pd.DataFrame,
    *,
    alpha: float = RIDGE_ALPHA,
    rating_seasons: tuple[int, ...] = RATING_SEASONS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    """Fit expanding chronological folds and return the final fitted models."""
    target = five_year_targets.copy()
    if "window_end" in target and "Window_End" not in target:
        target = target.rename(columns={"window_end": "Window_End"})
    if "offense" in target and "target_offense" not in target:
        target = target.rename(
            columns={
                "offense": "target_offense",
                "defense": "target_defense",
                "net": "target_net",
            }
        )
    target = target.loc[target["Window_End"].between(2018, 2026)].copy()
    panel = features.merge(
        target[
            [
                "PLAYER_ID",
                "Window_End",
                "target_offense",
                "target_defense",
                "Poss_Off",
                "Poss_Def",
            ]
        ],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    prediction_rows = []
    coefficient_rows = []
    final_models: dict[str, Pipeline] = {}
    for rating_season in rating_seasons:
        train = panel.loc[panel["Window_End"].lt(rating_season)].copy()
        score = features.loc[features["Window_End"].eq(rating_season)].copy()
        training_ends = tuple(sorted(train["Window_End"].unique()))
        if len(training_ends) < 3:
            raise ValueError(f"Rating season {rating_season} has fewer than three train ends.")
        fold = score[["PLAYER_ID", "Window_End"]].copy()
        for side in ("offense", "defense"):
            fold[f"prediction_{side}"] , model = _fit_side(
                train, score, side=side, alpha=alpha
            )
            coefficient_rows.extend(
                _coefficient_rows(model, side=side, rating_season=rating_season)
            )
            if rating_season == max(rating_seasons):
                final_models[side] = model
        fold["prediction_net"] = (
            fold["prediction_offense"] + fold["prediction_defense"]
        )
        fold["training_window_start"] = training_ends[0]
        fold["training_window_end"] = training_ends[-1]
        fold["training_window_count"] = len(training_ends)
        prediction_rows.append(fold)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    if predictions.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Sparse SPM predictions are not unique.")
    identity_error = (
        predictions["prediction_offense"]
        + predictions["prediction_defense"]
        - predictions["prediction_net"]
    ).abs().max()
    if identity_error > 1e-12:
        raise ValueError("Sparse SPM offense plus defense does not equal net.")
    return predictions, pd.DataFrame(coefficient_rows), final_models


def _weighted_metrics(
    actual: np.ndarray, prediction: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
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
    correlation = (
        covariance / np.sqrt(actual_variance * prediction_variance)
        if actual_variance > 0 and prediction_variance > 0
        else np.nan
    )
    return {
        "weighted_mae": float(np.average(np.abs(error), weights=weight)),
        "weighted_rmse": float(np.sqrt(np.average(error**2, weights=weight))),
        "weighted_pearson": float(correlation),
        "spearman": float(spearmanr(actual, prediction).statistic),
    }


def evaluate_next_season_players(
    sparse_predictions: pd.DataFrame,
    full_predictions: pd.DataFrame,
    annual_ratings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare sparse and full SPM on identical following-season RAPM rows."""
    full = full_predictions.loc[full_predictions["target_kind"].eq("five_year")].copy()
    annual = annual_ratings.loc[annual_ratings["candidate"].eq("zero_prior_rapm")].copy()
    annual = annual.loc[np.minimum(annual["Poss_Off"], annual["Poss_Def"]).gt(0)]
    rows = []
    matched_rows = []
    for test_season in PLAYER_TEST_SEASONS:
        rating_season = test_season - 1
        sparse = sparse_predictions.loc[
            sparse_predictions["Window_End"].eq(rating_season)
        ].rename(
            columns={
                "prediction_offense": "offense_sparse",
                "prediction_defense": "defense_sparse",
                "prediction_net": "net_sparse",
            }
        )
        baseline = full.loc[full["Window_End"].eq(rating_season)].rename(
            columns={
                "prior_offense_per_100": "offense_full",
                "prior_defense_per_100": "defense_full",
                "prior_net_per_100": "net_full",
            }
        )
        target = annual.loc[annual["rating_season"].eq(test_season)].rename(
            columns={
                "offense": "target_offense",
                "defense": "target_defense",
                "net": "target_net",
            }
        )
        matched = sparse.merge(
            baseline[
                ["PLAYER_ID", "offense_full", "defense_full", "net_full"]
            ],
            on="PLAYER_ID",
            how="inner",
            validate="one_to_one",
        ).merge(
            target[
                [
                    "PLAYER_ID",
                    "target_offense",
                    "target_defense",
                    "target_net",
                    "Poss_Off",
                    "Poss_Def",
                ]
            ],
            on="PLAYER_ID",
            how="inner",
            validate="one_to_one",
        )
        matched["rating_season"] = rating_season
        matched["test_season"] = test_season
        matched["sample_weight"] = np.sqrt(
            np.minimum(matched["Poss_Off"], matched["Poss_Def"]).clip(lower=1)
        )
        for candidate in ("sparse_function_spm", "full_five_year_spm"):
            suffix = "sparse" if candidate == "sparse_function_spm" else "full"
            for side in ("offense", "defense", "net"):
                rows.append(
                    {
                        "rating_season": rating_season,
                        "test_season": test_season,
                        "candidate": candidate,
                        "side": side,
                        "rows": int(len(matched)),
                        **_weighted_metrics(
                            matched[f"target_{side}"].to_numpy(dtype=float),
                            matched[f"{side}_{suffix}"].to_numpy(dtype=float),
                            matched["sample_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
        matched_rows.append(matched)
    return pd.DataFrame(rows), pd.concat(matched_rows, ignore_index=True)


def summarize_player_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["candidate", "side"], as_index=False)[
            ["weighted_mae", "weighted_rmse", "weighted_pearson", "spearman"]
        ]
        .mean()
        .sort_values(["side", "candidate"], kind="stable")
    )


def evaluate_team_wins(
    sparse_predictions: pd.DataFrame,
    full_predictions: pd.DataFrame,
    *,
    html_root: str | Path,
    identity_root: str | Path,
    schedule_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the existing oracle-minutes team-win diagnostic on identical ratings."""
    full = full_predictions.loc[full_predictions["target_kind"].eq("five_year")].copy()
    rating_rows = []
    for season in TEAM_WIN_RATING_SEASONS:
        sparse = sparse_predictions.loc[sparse_predictions["Window_End"].eq(season)]
        baseline = full.loc[full["Window_End"].eq(season)]
        common = sparse.merge(
            baseline,
            on="PLAYER_ID",
            suffixes=("_sparse", "_full"),
            validate="one_to_one",
        )
        for metric, suffix, label in (
            ("sparse_function_spm", "sparse", "Sparse function SPM"),
            ("full_five_year_spm", "full", "Full five-year SPM"),
        ):
            rating_rows.append(
                pd.DataFrame(
                    {
                        "PLAYER_ID": common["PLAYER_ID"].astype(int),
                        "Season": season,
                        "metric": metric,
                        "metric_label": label,
                        "offense": common[
                            "prediction_offense"
                            if suffix == "sparse"
                            else "prior_offense_per_100"
                        ],
                        "defense": common[
                            "prediction_defense"
                            if suffix == "sparse"
                            else "prior_defense_per_100"
                        ],
                        "net": common[
                            "prediction_net"
                            if suffix == "sparse"
                            else "prior_net_per_100"
                        ],
                    }
                )
            )
    ratings = pd.concat(rating_rows, ignore_index=True)
    all_seasons = tuple(range(min(TEAM_WIN_RATING_SEASONS), max(TEAM_WIN_RATING_SEASONS) + 2))
    team_games = _load_team_games(Path(schedule_root), all_seasons)
    identity_paths = {
        season: Path(identity_root) / f"{season}.parquet" for season in all_seasons
    }
    minutes, _, source_coverage = load_bbref_player_team_minutes(
        html_root, identity_paths, team_games
    )
    folds, summary, metric_coverage = build_team_win_benchmark(
        ratings,
        minutes,
        team_games,
        rating_seasons=TEAM_WIN_RATING_SEASONS,
        minimum_metric_minutes=250.0,
        replacement_values=(-2.0,),
    )
    return folds, summary, metric_coverage, source_coverage


def run_sparse_function_spm(
    *,
    player_sheet_dir: str | Path,
    five_year_targets_path: str | Path,
    reference_features_path: str | Path,
    full_predictions_path: str | Path,
    annual_ratings_path: str | Path,
    html_root: str | Path,
    identity_root: str | Path,
    schedule_root: str | Path,
    contract_path: str | Path,
    artifact_root: str | Path,
    alpha: float = RIDGE_ALPHA,
) -> dict:
    """Build, evaluate and persist the sparse SPM challenger."""
    source_paths = {
        "five_year_targets": Path(five_year_targets_path),
        "reference_features": Path(reference_features_path),
        "full_predictions": Path(full_predictions_path),
        "annual_ratings": Path(annual_ratings_path),
        "contract": Path(contract_path),
        "statistical_features_builder": Path(_aggregate_window.__code__.co_filename),
        "statistical_features_v2_builder": Path(_engineer_window.__code__.co_filename),
        "team_win_benchmark_builder": Path(build_team_win_benchmark.__code__.co_filename),
        "team_minute_loader": Path(load_bbref_player_team_minutes.__code__.co_filename),
    }
    for season in range(2014, 2027):
        source_paths[f"player_sheet_{season}"] = Path(player_sheet_dir) / f"{season}.parquet"
    for season in range(min(TEAM_WIN_RATING_SEASONS), max(TEAM_WIN_RATING_SEASONS) + 2):
        source_paths[f"bbref_totals_{season}"] = Path(html_root) / f"nba_{season}_totals.html"
        source_paths[f"team_identity_{season}"] = Path(identity_root) / f"{season}.parquet"
        source_paths[f"team_schedule_{season}"] = (
            Path(schedule_root) / f"leaguegamelog_{season}.json.gz"
        )
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    config = {
        "alpha": alpha,
        "rating_seasons": list(RATING_SEASONS),
        "player_test_seasons": list(PLAYER_TEST_SEASONS),
        "team_win_rating_seasons": list(TEAM_WIN_RATING_SEASONS),
        "features": {side: list(values) for side, values in selected_features().items()},
        "source_hashes": source_hashes,
        "source_code": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = Path(artifact_root) / "research" / "sparse_function_spm" / f"{EXPERIMENT_ID}_{identity}"
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())

    raw_features = build_five_year_features(player_sheet_dir)
    features = standardize_within_window(raw_features)
    targets = pd.read_parquet(five_year_targets_path)
    predictions, coefficients, models = fit_historical_predictions(
        features, targets, alpha=alpha
    )
    full_predictions = pd.read_parquet(full_predictions_path)
    annual_ratings = pd.read_parquet(annual_ratings_path)
    player_metrics, matched_players = evaluate_next_season_players(
        predictions, full_predictions, annual_ratings
    )
    player_summary = summarize_player_metrics(player_metrics)
    team_folds, team_summary, team_coverage, team_source_coverage = evaluate_team_wins(
        predictions,
        full_predictions,
        html_root=html_root,
        identity_root=identity_root,
        schedule_root=schedule_root,
    )

    reference = pd.read_parquet(reference_features_path)
    overlap = reference.loc[reference["Window_End"].between(2018, 2023)].merge(
        raw_features,
        on=["PLAYER_ID", "Window_End"],
        suffixes=("_reference", "_rebuilt"),
        validate="one_to_one",
    )
    overlap_rows = []
    for side, values in selected_features().items():
        for feature in values:
            left = pd.to_numeric(overlap[f"{feature}_reference"], errors="coerce")
            right = pd.to_numeric(overlap[f"{feature}_rebuilt"], errors="coerce")
            valid = left.notna() & right.notna()
            overlap_rows.append(
                {
                    "side": side,
                    "feature": feature,
                    "rows": int(valid.sum()),
                    "pearson": float(left[valid].corr(right[valid])),
                    "maximum_absolute_difference": float((left[valid] - right[valid]).abs().max()),
                }
            )
    overlap_qa = pd.DataFrame(overlap_rows)
    if overlap_qa["pearson"].min() < 0.999:
        raise ValueError("Rebuilt sparse features do not reproduce the reference panel.")

    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    predictions.to_parquet(output / "predictions.parquet", index=False)
    coefficients.to_parquet(output / "coefficients.parquet", index=False)
    feature_registry().to_parquet(output / "feature_registry.parquet", index=False)
    overlap_qa.to_parquet(output / "feature_overlap_qa.parquet", index=False)
    player_metrics.to_parquet(output / "player_fold_metrics.parquet", index=False)
    player_summary.to_parquet(output / "player_summary.parquet", index=False)
    matched_players.to_parquet(output / "matched_player_predictions.parquet", index=False)
    team_folds.to_parquet(output / "team_win_folds.parquet", index=False)
    team_summary.to_parquet(output / "team_win_summary.parquet", index=False)
    team_coverage.to_parquet(output / "team_win_coverage.parquet", index=False)
    team_source_coverage.to_parquet(output / "team_win_source_coverage.parquet", index=False)
    for side, model in models.items():
        joblib.dump(model, output / "models" / f"{side}.joblib")

    sparse_net = player_summary.loc[
        player_summary["candidate"].eq("sparse_function_spm")
        & player_summary["side"].eq("net")
    ].iloc[0]
    full_net = player_summary.loc[
        player_summary["candidate"].eq("full_five_year_spm")
        & player_summary["side"].eq("net")
    ].iloc[0]
    sparse_team = team_summary.loc[team_summary["metric"].eq("sparse_function_spm")].iloc[0]
    full_team = team_summary.loc[team_summary["metric"].eq("full_five_year_spm")].iloc[0]
    status = (
        "research_challenger_retained"
        if float(sparse_team["mean_r_squared"]) >= float(full_team["mean_r_squared"])
        else "research_null"
    )
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": json.loads(Path(contract_path).read_text()),
        "config": config,
        "quality": {
            "feature_rows": int(len(features)),
            "prediction_rows": int(len(predictions)),
            "player_test_folds": int(player_metrics["test_season"].nunique()),
            "team_win_folds": int(team_folds["rating_season"].nunique()),
            "minimum_reference_feature_correlation": float(overlap_qa["pearson"].min()),
            "component_identity_max_error": float(
                (
                    predictions["prediction_offense"]
                    + predictions["prediction_defense"]
                    - predictions["prediction_net"]
                ).abs().max()
            ),
            "season_2027_loaded": False,
        },
        "result": {
            "sparse_next_season_net_pearson": float(sparse_net["weighted_pearson"]),
            "full_next_season_net_pearson": float(full_net["weighted_pearson"]),
            "sparse_next_season_net_rmse": float(sparse_net["weighted_rmse"]),
            "full_next_season_net_rmse": float(full_net["weighted_rmse"]),
            "sparse_team_win_mean_r_squared": float(sparse_team["mean_r_squared"]),
            "full_team_win_mean_r_squared": float(full_team["mean_r_squared"]),
        },
        "decision": (
            "Retain as a sparse research challenger because it matches or improves the two-fold team-win diagnostic."
            if status == "research_challenger_retained"
            else "Keep the full five-year SPM; the sparse model did not improve the primary team-win diagnostic."
        ),
        "caveats": [
            "The team-win benchmark uses observed next-season minutes and is not a preseason forecast.",
            "Only two team-win folds are available from the local source bundle.",
            "All evaluated seasons are reused historical diagnostics; Season 2027 remains untouched.",
            "Blocks are a rim-protection proxy, not a measure of rim deterrence or defended-shot quality.",
            "The fixed ridge alpha was not tuned on these outcomes.",
        ],
        "paths": {
            "predictions": "predictions.parquet",
            "coefficients": "coefficients.parquet",
            "feature_registry": "feature_registry.parquet",
            "feature_overlap_qa": "feature_overlap_qa.parquet",
            "player_fold_metrics": "player_fold_metrics.parquet",
            "player_summary": "player_summary.parquet",
            "matched_player_predictions": "matched_player_predictions.parquet",
            "team_win_folds": "team_win_folds.parquet",
            "team_win_summary": "team_win_summary.parquet",
            "team_win_coverage": "team_win_coverage.parquet",
            "team_win_source_coverage": "team_win_source_coverage.parquet",
            "models": "models",
        },
        "forbidden_interpretation": "Public SPM promotion, causal skill value, or untouched confirmation.",
    }
    write_json_atomic(run, output / "run.json")
    return run
