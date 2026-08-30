"""Frozen same-season blocked-game validation for retrospective impact."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_impact.data.blocked_game_box_features import aggregate_box15_features
from nba_impact.data.full_spm_features import build_rolling_five_year_features
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit as fit_box_model,
)
from nba_impact.models.impact_validation_suite import (
    prior_center,
)
from nba_impact.models.rapm import (
    RapmConfig,
    RapmDesign,
    _game_margin_frame,
    _game_margin_metrics,
    fit_coefficients,
    fit_coefficients_with_center,
)
from nba_impact.models.statistical_impact import _metrics


CANDIDATES = ("zero_prior_rapm", "box15_aio")
DEFAULT_ALPHA_GRID = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)


@dataclass(frozen=True)
class Box15Models:
    offense: object
    defense: object
    offense_alpha: float
    defense_alpha: float


def deterministic_game_folds(
    possessions: pd.DataFrame, *, folds: int = 5
) -> pd.DataFrame:
    """Assign chronological whole games round-robin to deterministic folds."""
    if folds < 2:
        raise ValueError("At least two whole-game folds are required.")
    required = {"gameid", "date"}
    if missing := sorted(required - set(possessions.columns)):
        raise ValueError(f"Possession frame is missing {missing}.")
    games = possessions[["gameid", "date"]].copy()
    games["game_id"] = games.pop("gameid").astype(str)
    games["date"] = pd.to_datetime(games["date"], errors="raise")
    if games.groupby("game_id")["date"].nunique().gt(1).any():
        raise ValueError("A game cannot have more than one date.")
    games = (
        games.drop_duplicates("game_id")
        .sort_values(["date", "game_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(games) < folds:
        raise ValueError("The game universe is smaller than the fold count.")
    games["fold"] = np.arange(len(games), dtype=np.int16) % folds
    if games["game_id"].duplicated().any():
        raise AssertionError("Fold assignment game IDs must be unique.")
    return games[["game_id", "date", "fold"]]


def select_box_alpha_rolling_origin(
    train: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> float:
    """Tune ridge strength with strictly earlier rolling-window folds."""
    seasons = tuple(sorted(int(value) for value in train["Window_End"].unique()))
    if len(seasons) < 3:
        raise ValueError("Rolling-origin alpha selection requires three window ends.")
    scores: list[tuple[float, float]] = []
    for alpha in alpha_grid:
        fold_scores = []
        for validation_end in seasons[2:]:
            inner_train = train.loc[train["Window_End"].lt(validation_end)]
            validation = train.loc[train["Window_End"].eq(validation_end)]
            if inner_train.empty or validation.empty:
                raise ValueError("Rolling-origin alpha fold is empty.")
            prediction = fit_box_model(
                inner_train, features, target, alpha
            ).predict(validation.loc[:, features])
            fold_scores.append(
                _metrics(
                    validation[target].to_numpy(dtype=float),
                    prediction,
                    validation["sample_weight"].to_numpy(dtype=float),
                )["weighted_rmse"]
            )
        scores.append((float(np.mean(fold_scores)), float(alpha)))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def historical_box15_panel(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    rating_season: int,
) -> pd.DataFrame:
    """Merge earlier five-year Box15 inputs and five-year RAPM labels."""
    feature_required = {"PLAYER_ID", "Window_End", *BOX_PIPM_STYLE_FEATURES}
    target_required = {
        "PLAYER_ID",
        "Window_End",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
    }
    if missing := sorted(feature_required - set(features.columns)):
        raise ValueError(f"Historical Box15 features are missing {missing}.")
    if missing := sorted(target_required - set(targets.columns)):
        raise ValueError(f"Historical Box15 targets are missing {missing}.")
    if features.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Historical Box15 feature keys must be unique.")
    if targets.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Historical Box15 target keys must be unique.")
    panel = features[list(feature_required)].merge(
        targets[list(target_required)],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel = panel.loc[panel["Window_End"].lt(rating_season)].copy()
    if panel.empty or panel["Window_End"].max() >= rating_season:
        raise ValueError("Historical training must end before the rating season.")
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    return panel.sort_values(["Window_End", "PLAYER_ID"]).reset_index(drop=True)


def fit_historical_box15_models(
    panel: pd.DataFrame,
    *,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> Box15Models:
    """Fit the frozen offense and defense Box15 mappings once."""
    alphas = {
        side: select_box_alpha_rolling_origin(
            panel,
            BOX_PIPM_STYLE_FEATURES,
            f"target_{side}",
            alpha_grid,
        )
        for side in ("offense", "defense")
    }
    return Box15Models(
        offense=fit_box_model(
            panel, BOX_PIPM_STYLE_FEATURES, "target_offense", alphas["offense"]
        ),
        defense=fit_box_model(
            panel, BOX_PIPM_STYLE_FEATURES, "target_defense", alphas["defense"]
        ),
        offense_alpha=alphas["offense"],
        defense_alpha=alphas["defense"],
    )


def fold_local_box15_prior(
    *,
    historical_annual: pd.DataFrame,
    player_game_ledger: pd.DataFrame,
    held_game_ids: tuple[str, ...],
    rating_season: int,
    models: Box15Models,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a five-year prior after removing one fold from the current season."""
    earlier = historical_annual.loc[
        historical_annual["Window_End"].between(rating_season - 4, rating_season - 1)
    ].copy()
    expected = list(range(rating_season - 4, rating_season))
    if sorted(earlier["Window_End"].astype(int).unique()) != expected:
        raise ValueError(f"Fold-local prior requires complete annual history {expected}.")
    current = aggregate_box15_features(
        player_game_ledger,
        season=rating_season,
        excluded_game_ids=held_game_ids,
    )
    annual = pd.concat([earlier, current], ignore_index=True, sort=False)
    selected = {
        "offense": BOX_PIPM_STYLE_FEATURES,
        "defense": BOX_PIPM_STYLE_FEATURES,
    }
    pooled = build_rolling_five_year_features(
        annual,
        player_sheet_dir=None,
        selected=selected,
        window_ends=(rating_season,),
    )
    prior = pooled[["PLAYER_ID", "Window_End"]].copy()
    prior["offense"] = models.offense.predict(
        pooled.loc[:, BOX_PIPM_STYLE_FEATURES]
    )
    prior["defense"] = models.defense.predict(
        pooled.loc[:, BOX_PIPM_STYLE_FEATURES]
    )
    prior["net"] = prior["offense"] + prior["defense"]
    if not np.allclose(prior["offense"] + prior["defense"], prior["net"]):
        raise AssertionError("Fold-local prior components must add to net.")
    return prior, pooled


def run_gate_a(
    *,
    design: RapmDesign,
    possessions: pd.DataFrame,
    player_game_ledger: pd.DataFrame,
    historical_annual: pd.DataFrame,
    historical_five_year: pd.DataFrame,
    historical_targets: pd.DataFrame,
    rating_season: int,
    folds: int = 5,
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID,
) -> dict[str, pd.DataFrame | dict[str, float]]:
    """Compare zero-prior RAPM and Box15 AIO on identical held games."""
    if set(np.unique(design.seasons)) != {rating_season}:
        raise ValueError("Gate A design must contain exactly the rating season.")
    assignment = deterministic_game_folds(possessions, folds=folds)
    if set(assignment["game_id"]) != set(design.game_ids.astype(str)):
        raise ValueError("Fold assignment and RAPM design game universes differ.")
    if set(player_game_ledger["game_id"]) != set(assignment["game_id"]):
        raise ValueError("Box15 ledger and fold game universes differ.")

    panel = historical_box15_panel(
        historical_five_year, historical_targets, rating_season=rating_season
    )
    models = fit_historical_box15_models(panel, alpha_grid=alpha_grid)
    config = RapmConfig(
        seasons=(rating_season,),
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_home=300.0,
    )
    game_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    prior_rows: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    pooled_rows: list[pd.DataFrame] = []
    for fold in range(folds):
        held_games = tuple(
            assignment.loc[assignment["fold"].eq(fold), "game_id"].astype(str)
        )
        test_mask = np.isin(design.game_ids.astype(str), held_games)
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise ValueError(f"Fold {fold} has an empty RAPM partition.")
        prior, pooled = fold_local_box15_prior(
            historical_annual=historical_annual,
            player_game_ledger=player_game_ledger,
            held_game_ids=held_games,
            rating_season=rating_season,
            models=models,
        )
        prior["fold"] = fold
        prior_rows.append(prior)
        pooled["fold"] = fold
        pooled_rows.append(pooled)

        zero_beta, zero_intercept = fit_coefficients(
            design, config, row_mask=train_mask
        )
        center, coverage = prior_center(design, prior, row_mask=train_mask)
        box_beta, box_intercept = fit_coefficients_with_center(
            design, config, center, row_mask=train_mask
        )
        coverage_rows.append({"fold": fold, **coverage})
        fold_candidates = (
            ("zero_prior_rapm", zero_beta, zero_intercept),
            ("box15_aio", box_beta, box_intercept),
        )
        expected_hash: str | None = None
        for candidate, beta, intercept in fold_candidates:
            games = _game_margin_frame(
                design, beta, intercept, test_mask, train_mask
            )
            possession_rows = pd.Series(design.game_ids[test_mask].astype(str)).value_counts()
            games["possession_rows"] = games["game_id"].astype(str).map(possession_rows)
            games["candidate"] = candidate
            games["season"] = rating_season
            games["fold"] = fold
            outcome_hash = hashlib.sha256(
                "|".join(
                    games.sort_values("game_id").apply(
                        lambda row: f"{row['game_id']}:{row['actual_margin']}", axis=1
                    )
                ).encode()
            ).hexdigest()
            if expected_hash is None:
                expected_hash = outcome_hash
            elif outcome_hash != expected_hash:
                raise AssertionError("Candidates must score identical held-game outcomes.")
            metric_rows.append(
                {
                    "season": rating_season,
                    "fold": fold,
                    "candidate": candidate,
                    **_game_margin_metrics(
                        design, beta, intercept, test_mask, train_mask
                    ),
                }
            )
            game_rows.append(games)

    games = pd.concat(game_rows, ignore_index=True)
    counts = games.groupby("candidate")["game_id"].nunique()
    if set(counts.index) != set(CANDIDATES) or counts.nunique() != 1:
        raise AssertionError("Gate A candidates must score identical game counts.")
    if games.groupby(["candidate", "game_id"]).size().ne(1).any():
        raise AssertionError("Each candidate must score every game exactly once.")
    return {
        "fold_assignments": assignment,
        "game_predictions": games,
        "fold_metrics": pd.DataFrame(metric_rows),
        "fold_priors": pd.concat(prior_rows, ignore_index=True),
        "fold_features": pd.concat(pooled_rows, ignore_index=True),
        "prior_coverage": pd.DataFrame(coverage_rows),
        "model_selection": pd.DataFrame(
            [
                {
                    "component": "offense",
                    "selected_alpha": models.offense_alpha,
                    "training_window_end_max": int(panel["Window_End"].max()),
                    "training_window_ends": int(panel["Window_End"].nunique()),
                },
                {
                    "component": "defense",
                    "selected_alpha": models.defense_alpha,
                    "training_window_end_max": int(panel["Window_End"].max()),
                    "training_window_ends": int(panel["Window_End"].nunique()),
                },
            ]
        ),
    }


def paired_whole_game_mse_bootstrap(
    predictions: pd.DataFrame,
    *,
    candidate: str = "box15_aio",
    reference: str = "zero_prior_rapm",
    draws: int = 5_000,
    seed: int = 20260830,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample paired whole games within season and retain every joint draw."""
    required = {
        "season",
        "game_id",
        "candidate",
        "actual_margin",
        "predicted_margin",
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Game predictions are missing {missing}.")
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required.")
    expected_candidates = {candidate, reference}
    if set(predictions["candidate"].unique()) != expected_candidates:
        raise ValueError(f"Predictions must contain exactly {sorted(expected_candidates)}.")
    if predictions.duplicated(["season", "game_id", "candidate"]).any():
        raise ValueError("Every candidate must score each season-game exactly once.")

    season_errors: list[tuple[np.ndarray, np.ndarray]] = []
    for _, season_frame in predictions.groupby("season", sort=True):
        wide = season_frame.pivot(
            index="game_id",
            columns="candidate",
            values=["actual_margin", "predicted_margin"],
        )
        actual = wide["actual_margin"].reindex(columns=[candidate, reference])
        forecast = wide["predicted_margin"].reindex(columns=[candidate, reference])
        if actual.isna().any().any() or forecast.isna().any().any():
            raise ValueError("Both candidates must score every whole game.")
        if not np.allclose(actual[candidate], actual[reference]):
            raise ValueError("Paired candidates must share identical actual margins.")
        truth = actual[candidate].to_numpy(dtype=float)
        season_errors.append(
            (
                (truth - forecast[candidate].to_numpy(dtype=float)) ** 2,
                (truth - forecast[reference].to_numpy(dtype=float)) ** 2,
            )
        )
    if not season_errors:
        raise ValueError("No complete season-game pairs were available.")

    point_candidate = float(np.mean([left.mean() for left, _ in season_errors]))
    point_reference = float(np.mean([right.mean() for _, right in season_errors]))
    rng = np.random.default_rng(seed)
    candidate_draws = np.empty(draws, dtype=float)
    reference_draws = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = []
        for left, right in season_errors:
            indices = rng.integers(0, len(left), len(left))
            sampled.append((left[indices].mean(), right[indices].mean()))
        candidate_draws[draw] = np.mean([values[0] for values in sampled])
        reference_draws[draw] = np.mean([values[1] for values in sampled])
    delta = candidate_draws - reference_draws
    low, high = np.quantile(delta, [0.025, 0.975])
    summary = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "reference": reference,
                "seasons": len(season_errors),
                "games": int(predictions["game_id"].nunique()),
                "candidate_mse": point_candidate,
                "reference_mse": point_reference,
                "candidate_rmse": float(np.sqrt(point_candidate)),
                "reference_rmse": float(np.sqrt(point_reference)),
                "rmse_improvement_reference_minus_candidate": float(
                    np.sqrt(point_reference) - np.sqrt(point_candidate)
                ),
                "mean_mse_delta_candidate_minus_reference": (
                    point_candidate - point_reference
                ),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_candidate_lower_mse": float(np.mean(delta < 0)),
                "bootstrap_draws": draws,
                "seed": seed,
            }
        ]
    )
    draw_frame = pd.DataFrame(
        {
            "draw": np.arange(draws, dtype=np.int32),
            "candidate_mse": candidate_draws,
            "reference_mse": reference_draws,
            "mse_delta_candidate_minus_reference": delta,
        }
    )
    return summary, draw_frame


def pooled_game_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize pooled whole-game accuracy and calibration by candidate."""
    required = {"candidate", "game_id", "actual_margin", "predicted_margin"}
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Game predictions are missing {missing}.")
    rows = []
    for candidate, frame in predictions.groupby("candidate", sort=True):
        actual = frame["actual_margin"].to_numpy(dtype=float)
        forecast = frame["predicted_margin"].to_numpy(dtype=float)
        error = actual - forecast
        variance = float(np.var(forecast, ddof=0))
        slope = (
            float(np.cov(actual, forecast, ddof=0)[0, 1] / variance)
            if variance > 0
            else float("nan")
        )
        rows.append(
            {
                "candidate": candidate,
                "games": len(frame),
                "mse": float(np.mean(error**2)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mae": float(np.mean(np.abs(error))),
                "correlation": float(pd.Series(actual).corr(pd.Series(forecast))),
                "calibration_intercept": float(actual.mean() - slope * forecast.mean()),
                "calibration_slope": slope,
                "actual_margin_sd": float(np.std(actual, ddof=0)),
                "predicted_margin_sd": float(np.std(forecast, ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)
