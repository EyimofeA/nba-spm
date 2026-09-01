#!/usr/bin/env python3
"""Compare 5/7/9-year normal and age-adjusted RAPM targets downstream."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.age_adjusted_rapm import (
    build_age_design,
    current_age_player_coefficients,
    fit_age_adjusted_rapm,
)
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit as fit_box
from nba_impact.models.prior_informed_rapm import build_prior_center
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    build_design,
    fit_coefficient_center_path,
    fit_coefficients,
    load_unified_terminal_possessions,
)
from nba_impact.models.statistical_impact import _metrics as player_metrics

try:
    from run_annual_spm_learner_screen import (
        ModelSpec,
        _feature_arms,
        _fit as fit_rich,
        _prune_features,
        _weighted_metrics,
    )
except ModuleNotFoundError:  # Imported as research.run_* by tests.
    from research.run_annual_spm_learner_screen import (
        ModelSpec,
        _feature_arms,
        _fit as fit_rich,
        _prune_features,
        _weighted_metrics,
    )


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "target_window_spm_aio_v1"
CONTRACT = ROOT / "research/experiments/target_window_spm_aio_v1.yml"
OUTPUT_ROOT = ROOT / "artifacts/research/target_window_spm_aio"
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
SILVER_POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SILVER_LINEUPS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
HISTORICAL_BOX = ROOT / (
    "artifacts/research/historical_box15_extension/"
    "historical_box15_extension_v1_08ff4c34ff/annual_box15_features.parquet"
)
RICH_RUN = ROOT / (
    "artifacts/research/complete_feature_coverage/"
    "semantically_complete_spm_features_v1_fdee01ec4e"
)
RICH_ANNUAL = RICH_RUN / "annual_features.parquet"
MECHANISM_ANNUAL = ROOT / (
    "artifacts/research/mechanism_feature_panel/"
    "mechanism_feature_panel_v1_9224606a01/annual_features.parquet"
)
ATLAS = ROOT / (
    "artifacts/research/spm_feature_atlas/"
    "spm_feature_atlas_v1_6949ad7b60/feature_atlas.parquet"
)
HISTORICAL_AGES = ROOT / "data/lake/bronze/historical_player_sheets/year_totals"
CURRENT_AGES = ROOT / (
    "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals"
)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment contract changed.")
    rating = tuple(int(value) for value in contract["rating_seasons"])
    tests = tuple(int(value) for value in contract["test_seasons"])
    if rating != tuple(range(2014, 2026)) or tests != tuple(range(2015, 2027)):
        raise ValueError("The requested 2014-to-2025 rating folds changed.")
    if tests != tuple(value + 1 for value in rating):
        raise ValueError("Every rating season must predict the next season.")
    if tuple(contract["target_horizons"]) != (5, 7, 9):
        raise ValueError("Target horizons must remain 5, 7, and 9 seasons.")
    if tuple(contract["target_variants"]) != ("normal", "age_adjusted"):
        raise ValueError("Target variants must remain normal and age-adjusted.")
    return contract


def _load_ages(seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        path = (
            HISTORICAL_AGES / f"{season}.csv"
            if season <= 2013
            else CURRENT_AGES / f"{season}.parquet"
        )
        frame = (
            pd.read_csv(path, usecols=["PLAYER_ID", "AGE"])
            if path.suffix == ".csv"
            else pd.read_parquet(path, columns=["PLAYER_ID", "AGE"])
        )
        frame["Season"] = season
        rows.append(frame)
    panel = pd.concat(rows, ignore_index=True)
    panel["PLAYER_ID"] = pd.to_numeric(panel["PLAYER_ID"], errors="raise").astype(int)
    panel["AGE"] = pd.to_numeric(panel["AGE"], errors="coerce")
    conflicts = panel.groupby(["Season", "PLAYER_ID"])["AGE"].nunique(dropna=True)
    if conflicts.gt(1).any():
        raise ValueError("Age source has conflicting player-season values.")
    return panel.drop_duplicates(["Season", "PLAYER_ID"], keep="first")


def _annual_features(
    annual: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Return one row of contemporaneous inputs per player-season."""
    fields = tuple(dict.fromkeys((*selected["offense"], *selected["defense"])))
    required = {"PLAYER_ID", "Window_End", "OffPoss", "DefPoss", *fields}
    if missing := sorted(required - set(annual.columns)):
        raise ValueError(f"Annual feature panel misses {missing}.")
    result = annual.loc[
        :, ["PLAYER_ID", "Window_End", "OffPoss", "DefPoss", *fields]
    ].copy()
    if result.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Annual feature keys are not unique.")
    return result


def _rating_frame(
    players: np.ndarray,
    coefficients: np.ndarray,
    off_possessions: np.ndarray,
    def_possessions: np.ndarray,
    *,
    horizon: int,
    window_end: int,
    target_variant: str,
) -> pd.DataFrame:
    n = len(players)
    active = np.minimum(off_possessions, def_possessions) > 0
    frame = pd.DataFrame(
        {
            "PLAYER_ID": players[active],
            "Window_End": int(window_end),
            "horizon": int(horizon),
            "target_variant": target_variant,
            "target_offense": 100.0 * coefficients[:n][active],
            "target_defense": -100.0 * coefficients[n : 2 * n][active],
            "Poss_Off": off_possessions[active],
            "Poss_Def": def_possessions[active],
        }
    )
    frame["target_net"] = frame["target_offense"] + frame["target_defense"]
    return frame


def _build_targets(
    frame: pd.DataFrame,
    ages: pd.DataFrame,
    contract: dict,
    checkpoint: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every target once and checkpoint each horizon-end pair."""
    original = build_design(frame, include_home=True)
    season_mean = frame.groupby("season")["pts"].mean()
    centered = frame["pts"].to_numpy(dtype=float) - frame["season"].map(season_mean).to_numpy(dtype=float)
    design = replace(original, y=centered)
    age_design = build_age_design(
        frame,
        ages,
        reference_age=int(contract["target_rapm"]["age_reference"]),
    )
    config = RapmConfig(
        seasons=tuple(range(1997, 2027)),
        lambda_off=float(contract["target_rapm"]["lambda_off"]),
        lambda_def=float(contract["target_rapm"]["lambda_def"]),
        lambda_home=float(contract["target_rapm"]["lambda_home"]),
        data_scope="target_window_spm_aio_labels",
    )
    outputs = []
    quality = []
    for horizon in contract["target_horizons"]:
        for end in range(1997 + int(horizon) - 1, 2026):
            path = checkpoint / f"targets_{horizon}y_end_{end}.parquet"
            quality_path = checkpoint / f"targets_{horizon}y_end_{end}.json"
            if path.exists() and quality_path.exists():
                outputs.append(pd.read_parquet(path))
                quality.append(json.loads(quality_path.read_text()))
                continue
            mask = design.seasons >= end - int(horizon) + 1
            mask &= design.seasons <= end
            X = design.X[mask]
            n = len(design.players)
            off = np.asarray(X[:, :n].sum(axis=0)).ravel()
            deff = np.asarray(X[:, n : 2 * n].sum(axis=0)).ravel()
            normal, _ = fit_coefficients(design, config, row_mask=mask)
            age_fit = fit_age_adjusted_rapm(
                design,
                age_design,
                config,
                age_penalty=float(contract["target_rapm"]["age_penalty"]),
                row_mask=mask,
            )
            current_age, age_coverage = current_age_player_coefficients(
                age_fit,
                design,
                ages,
                season=end,
            )
            target = pd.concat(
                [
                    _rating_frame(
                        design.players,
                        normal,
                        off,
                        deff,
                        horizon=int(horizon),
                        window_end=end,
                        target_variant="normal",
                    ),
                    _rating_frame(
                        design.players,
                        current_age,
                        off,
                        deff,
                        horizon=int(horizon),
                        window_end=end,
                        target_variant="age_adjusted",
                    ),
                ],
                ignore_index=True,
            )
            row = {
                "horizon": int(horizon),
                "window_end": end,
                "possession_rows": int(mask.sum()),
                "players": int((np.minimum(off, deff) > 0).sum()),
                "current_age_player_coverage": age_coverage,
            }
            _atomic_parquet(target, path)
            write_json_atomic(row, quality_path)
            outputs.append(target)
            quality.append(row)
            print(f"target {horizon}y end {end}: complete", flush=True)
    targets = pd.concat(outputs, ignore_index=True)
    if targets.duplicated(["PLAYER_ID", "Window_End", "horizon", "target_variant"]).any():
        raise ValueError("Target keys are not unique.")
    return targets, pd.DataFrame(quality)


def _load_complete_target_checkpoints(
    contract: dict, checkpoint: Path
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    target_paths = []
    quality_paths = []
    for horizon in contract["target_horizons"]:
        for end in range(1997 + int(horizon) - 1, 2026):
            target_paths.append(checkpoint / f"targets_{horizon}y_end_{end}.parquet")
            quality_paths.append(checkpoint / f"targets_{horizon}y_end_{end}.json")
    if not all(path.exists() for path in (*target_paths, *quality_paths)):
        return None
    targets = pd.concat((pd.read_parquet(path) for path in target_paths), ignore_index=True)
    quality = pd.DataFrame(json.loads(path.read_text()) for path in quality_paths)
    return targets, quality


def _panel(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    panel = features.merge(
        targets,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    return panel


def _fit_box_priors(
    panel: pd.DataFrame,
    rating_seasons: tuple[int, ...],
    alphas: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selections = []
    for season in rating_seasons:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train.empty or test.empty:
            continue
        prior = test[["PLAYER_ID", "Window_End"]].copy()
        for side in ("offense", "defense"):
            target = f"target_{side}"
            alpha = float(alphas[side])
            model = fit_box(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
            prediction = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
            prior[f"prior_{side}_per_100"] = prediction
            selections.append(
                {
                    "learner": "box15",
                    "rating_season": season,
                    "side": side,
                    "selected_params": json.dumps({"alpha": alpha}),
                    "features": len(BOX_PIPM_STYLE_FEATURES),
                    **player_metrics(
                        test[target].to_numpy(dtype=float),
                        prediction,
                        test["sample_weight"].to_numpy(dtype=float),
                    ),
                }
            )
        prior["prior_net_per_100"] = (
            prior["prior_offense_per_100"] + prior["prior_defense_per_100"]
        )
        rows.append(prior)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(selections)


def _fit_rich_priors(
    panel: pd.DataFrame,
    atlas: pd.DataFrame,
    rating_seasons: tuple[int, ...],
    *,
    specs: dict[str, ModelSpec],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    renamed = panel.rename(columns={"Window_End": "Season"})
    arms = {
        side: _feature_arms(atlas, renamed, side)["audited_all"]
        for side in ("offense", "defense")
    }
    rows = []
    selections = []
    for season in rating_seasons:
        train = renamed.loc[renamed["Season"].lt(season)].copy()
        test = renamed.loc[renamed["Season"].eq(season)].copy()
        if train.empty or test.empty:
            continue
        side_predictions = []
        for side in ("offense", "defense"):
            target = f"target_{side}"
            features = _prune_features(train, arms[side], threshold=threshold)
            model = fit_rich(specs[side], train, features, target)
            predicted = model.predict(test.loc[:, features])
            metrics = _weighted_metrics(
                test[target].to_numpy(dtype=float),
                predicted,
                test["sample_weight"].to_numpy(dtype=float),
            )
            side_predictions.append(
                test[["PLAYER_ID", "Season"]]
                .assign(**{f"prior_{side}_per_100": predicted})
                .rename(columns={"Season": "Window_End"})
            )
            selections.append(
                {
                    "learner": "rich_spm",
                    "rating_season": season,
                    "side": side,
                    "selected_params": json.dumps(specs[side].params, sort_keys=True),
                    "features": len(features),
                    "weighted_rmse": metrics["weighted_rmse"],
                    "correlation": metrics["weighted_correlation"],
                    "calibration_slope": metrics["calibration_slope"],
                }
            )
        prior = side_predictions[0].merge(
            side_predictions[1],
            on=["PLAYER_ID", "Window_End"],
            validate="one_to_one",
        )
        prior["prior_net_per_100"] = (
            prior["prior_offense_per_100"] + prior["prior_defense_per_100"]
        )
        rows.append(prior)
    if not rows:
        return pd.DataFrame(), pd.DataFrame(selections)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(selections)


def _rating_table(design, beta: np.ndarray, train_mask: np.ndarray) -> pd.DataFrame:
    n = len(design.players)
    X = design.X[train_mask]
    return pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "offense": 100.0 * beta[:n],
            "defense": -100.0 * beta[n : 2 * n],
            "Poss_Off": np.asarray(X[:, :n].sum(axis=0)).ravel(),
            "Poss_Def": np.asarray(X[:, n : 2 * n].sum(axis=0)).ravel(),
        }
    ).assign(net=lambda value: value["offense"] + value["defense"])


def _evaluate_priors(
    priors: pd.DataFrame,
    contract: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = []
    ratings = []
    coverage = []
    for rating_season in contract["rating_seasons"]:
        available = priors.loc[priors["Window_End"].eq(rating_season)]
        if available.empty:
            continue
        test_season = int(rating_season) + 1
        frame = load_unified_terminal_possessions(
            POSSESSION_CACHE,
            SILVER_POSSESSIONS,
            SILVER_LINEUPS,
            (int(rating_season), test_season),
            transition_season=2024,
            game_types=("regular",),
        )
        design = build_design(frame, include_home=True)
        train_mask = design.seasons == int(rating_season)
        test_mask = design.seasons == test_season
        config = RapmConfig(
            seasons=(int(rating_season), test_season),
            lambda_off=float(contract["aio"]["lambda_off"]),
            lambda_def=float(contract["aio"]["lambda_def"]),
            lambda_home=float(contract["aio"]["lambda_home"]),
            data_scope="target_window_spm_aio_one_season_update",
        )
        zero_center = np.zeros(design.X.shape[1], dtype=float)
        zero_beta, zero_intercept = fit_coefficient_center_path(
            design,
            config,
            zero_center,
            center_scales=(0.0,),
            row_mask=train_mask,
        )[0.0]
        zero_game = _game_margin_frame(
            design, zero_beta, zero_intercept, test_mask, train_mask
        )
        zero_game["candidate"] = "zero_prior_rapm"
        zero_game["rating_season"] = int(rating_season)
        zero_game["test_season"] = test_season
        zero_game["squared_error"] = (
            zero_game["actual_margin"] - zero_game["predicted_margin"]
        ) ** 2
        games.append(zero_game)
        zero_rating = _rating_table(design, zero_beta, train_mask)
        zero_rating["candidate"] = "zero_prior_rapm"
        zero_rating["rating_season"] = int(rating_season)
        ratings.append(zero_rating)
        for candidate, prior in available.groupby("candidate", sort=True):
            center, candidate_coverage = build_prior_center(
                design,
                prior,
                prior_window_end=int(rating_season),
                train_mask=train_mask,
                test_mask=test_mask,
            )
            standalone = center.copy()
            standalone[-1] = zero_beta[-1]
            posterior, posterior_intercept = fit_coefficient_center_path(
                design,
                config,
                center,
                center_scales=(1.0,),
                row_mask=train_mask,
            )[1.0]
            for model, beta, intercept in (
                (candidate, standalone, zero_intercept),
                (f"{candidate}_aio", posterior, posterior_intercept),
            ):
                game = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
                game["candidate"] = model
                game["rating_season"] = int(rating_season)
                game["test_season"] = test_season
                game["squared_error"] = (
                    game["actual_margin"] - game["predicted_margin"]
                ) ** 2
                games.append(game)
                rating = _rating_table(design, beta, train_mask)
                rating["candidate"] = model
                rating["rating_season"] = int(rating_season)
                ratings.append(rating)
            coverage.append(
                {"candidate": candidate, "rating_season": int(rating_season), **candidate_coverage}
            )
        print(f"AIO rating {rating_season} -> games {test_season}: complete", flush=True)
    return (
        pd.concat(games, ignore_index=True),
        pd.concat(ratings, ignore_index=True),
        pd.DataFrame(coverage),
    )


def _game_metrics(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (candidate, test_season), frame in games.groupby(["candidate", "test_season"]):
        actual = frame["actual_margin"].to_numpy(dtype=float)
        predicted = frame["predicted_margin"].to_numpy(dtype=float)
        variance = float(np.var(predicted))
        rows.append(
            {
                "candidate": candidate,
                "test_season": int(test_season),
                "games": len(frame),
                "mse": float(np.mean((actual - predicted) ** 2)),
                "rmse": float(np.sqrt(np.mean((actual - predicted) ** 2))),
                "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
                "calibration_slope": float(np.cov(actual, predicted, ddof=0)[0, 1] / variance) if variance > 0 else np.nan,
            }
        )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            first_test_season=("test_season", "min"),
            last_test_season=("test_season", "max"),
            equal_season_mse=("mse", "mean"),
            mean_correlation=("correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values(["folds", "equal_season_mse"], ascending=[False, True])
    )
    summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
    return folds, summary


def _paired_age_bootstrap(games: pd.DataFrame, *, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for learner in ("box15", "rich_spm"):
        for horizon in (5, 7, 9):
            for suffix in ("", "_aio"):
                normal = f"{learner}_{horizon}y_normal{suffix}"
                age = f"{learner}_{horizon}y_age_adjusted{suffix}"
                left = games.loc[games["candidate"].eq(normal)]
                right = games.loc[games["candidate"].eq(age)]
                seasons = sorted(set(left["test_season"]) & set(right["test_season"]))
                season_arrays = []
                for season in seasons:
                    a = left.loc[left["test_season"].eq(season)].set_index("game_id")
                    b = right.loc[right["test_season"].eq(season)].set_index("game_id")
                    if not a.index.equals(b.index):
                        raise ValueError(f"{normal} and {age} do not score identical games.")
                    season_arrays.append(
                        np.column_stack([a["squared_error"].to_numpy(), b["squared_error"].to_numpy()])
                    )
                if not season_arrays:
                    continue
                observed = float(np.mean([np.mean(v[:, 0] - v[:, 1]) for v in season_arrays]))
                samples = np.empty(draws, dtype=float)
                for draw in range(draws):
                    season_means = []
                    for values in season_arrays:
                        take = rng.integers(0, len(values), len(values))
                        season_means.append(
                            float(np.mean(values[take, 0] - values[take, 1]))
                        )
                    samples[draw] = float(np.mean(season_means))
                low, high = np.quantile(samples, [0.025, 0.975])
                rows.append(
                    {
                        "learner": learner,
                        "horizon": horizon,
                        "stage": "aio" if suffix else "prior",
                        "folds": len(seasons),
                        "normal_minus_age_mse": observed,
                        "lower_95": float(low),
                        "upper_95": float(high),
                        "probability_age_better": float(np.mean(samples > 0)),
                    }
                )
    return pd.DataFrame(rows)


def _key_comparison_bootstrap(
    games: pd.DataFrame, *, draws: int, seed: int
) -> pd.DataFrame:
    """Paired whole-game intervals for the decisions this experiment can make."""
    comparisons = []
    for learner in ("box15", "rich_spm"):
        comparisons.extend(
            (
                f"{learner}_{left}y_normal_aio",
                f"{learner}_{right}y_normal_aio",
            )
            for left, right in ((7, 5), (9, 5), (9, 7))
        )
    for horizon in (5, 7, 9):
        for suffix in ("", "_aio"):
            comparisons.append(
                (
                    f"box15_{horizon}y_normal{suffix}",
                    f"rich_spm_{horizon}y_normal{suffix}",
                )
            )
    rng = np.random.default_rng(seed + 1)
    rows = []
    for candidate_a, candidate_b in comparisons:
        left = games.loc[games["candidate"].eq(candidate_a)]
        right = games.loc[games["candidate"].eq(candidate_b)]
        seasons = sorted(set(left["test_season"]) & set(right["test_season"]))
        differences = []
        for season in seasons:
            a = left.loc[left["test_season"].eq(season)].set_index("game_id").sort_index()
            b = right.loc[right["test_season"].eq(season)].set_index("game_id").sort_index()
            if not a.index.equals(b.index):
                raise ValueError(f"{candidate_a} and {candidate_b} do not score identical games.")
            differences.append(
                a["squared_error"].to_numpy(dtype=float)
                - b["squared_error"].to_numpy(dtype=float)
            )
        samples = np.empty(draws, dtype=float)
        for draw in range(draws):
            samples[draw] = float(
                np.mean(
                    [
                        values[rng.integers(0, len(values), len(values))].mean()
                        for values in differences
                    ]
                )
            )
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "folds": len(seasons),
                "a_minus_b_mse": float(np.mean([values.mean() for values in differences])),
                "lower_95": float(low),
                "upper_95": float(high),
                "probability_b_better": float(np.mean(samples > 0)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    contract = _load_contract()
    source_paths = {
        "contract": CONTRACT,
        "runner": Path(__file__),
        "historical_box_features": HISTORICAL_BOX,
        "rich_annual_features": RICH_ANNUAL,
        "mechanism_annual_features": MECHANISM_ANNUAL,
        "feature_atlas": ATLAS,
        "silver_possessions": SILVER_POSSESSIONS,
        "silver_lineups": SILVER_LINEUPS,
        **{
            f"legacy_possessions_{season}": POSSESSION_CACHE / f"matchups_{season}.parquet"
            for season in range(1997, 2024)
        },
        **{
            f"age_source_{season}": (
                HISTORICAL_AGES / f"{season}.csv"
                if season <= 2013
                else CURRENT_AGES / f"{season}.parquet"
            )
            for season in range(1997, 2027)
        },
    }
    hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"{EXPERIMENT_ID}_{identity}"
    if (output / "run.json").exists():
        print(output)
        return
    checkpoint = OUTPUT_ROOT / "target_checkpoints_v1"
    checkpoint.mkdir(parents=True, exist_ok=True)

    checkpointed = _load_complete_target_checkpoints(contract, checkpoint)
    if checkpointed is None:
        seasons = tuple(range(1997, 2027))
        possessions = load_unified_terminal_possessions(
            POSSESSION_CACHE,
            SILVER_POSSESSIONS,
            SILVER_LINEUPS,
            seasons,
            transition_season=2024,
            game_types=("regular",),
        )
        ages = _load_ages(seasons)
        targets, target_quality = _build_targets(possessions, ages, contract, checkpoint)
    else:
        targets, target_quality = checkpointed

    atlas = pd.read_parquet(ATLAS)
    box_annual = pd.read_parquet(HISTORICAL_BOX)
    rich_annual = pd.read_parquet(RICH_ANNUAL)
    mechanism = pd.read_parquet(MECHANISM_ANNUAL)
    extra = [column for column in mechanism.columns if column not in rich_annual.columns]
    rich_annual = rich_annual.merge(
        mechanism[["PLAYER_ID", "Window_End", *extra]],
        on=["PLAYER_ID", "Window_End"],
        how="left",
        validate="one_to_one",
    )
    rich_candidates = {
        side: _feature_arms(atlas, rich_annual.rename(columns={"Window_End": "Season"}), side)["audited_all"]
        for side in ("offense", "defense")
    }
    box_selected = {"offense": BOX_PIPM_STYLE_FEATURES, "defense": BOX_PIPM_STYLE_FEATURES}
    prior_rows = []
    selection_rows = []
    coverage_rows = []
    box_features = _annual_features(box_annual, box_selected)
    rich_features = _annual_features(rich_annual, rich_candidates)
    rich_specs = {
        "offense": ModelSpec(
            "elastic_net",
            {
                "alpha": float(contract["spm"]["rich"]["offense_alpha"]),
                "l1_ratio": float(contract["spm"]["rich"]["offense_l1_ratio"]),
            },
        ),
        "defense": ModelSpec(
            "ridge",
            {"alpha": float(contract["spm"]["rich"]["defense_alpha"])},
        ),
    }
    for horizon in contract["target_horizons"]:
        for target_variant in contract["target_variants"]:
            variant_targets = targets.loc[
                targets["horizon"].eq(horizon) & targets["target_variant"].eq(target_variant)
            ]
            box_panel = _panel(box_features, variant_targets)
            box_priors, box_selection = _fit_box_priors(
                box_panel,
                tuple(contract["rating_seasons"]),
                {
                    "offense": float(contract["spm"]["box15"]["offense_alpha"]),
                    "defense": float(contract["spm"]["box15"]["defense_alpha"]),
                },
            )
            box_priors["candidate"] = f"box15_{horizon}y_{target_variant}"
            box_selection["horizon"] = int(horizon)
            box_selection["target_variant"] = target_variant
            prior_rows.append(box_priors)
            selection_rows.append(box_selection)

            rich_panel = _panel(rich_features, variant_targets)
            rich_priors, rich_selection = _fit_rich_priors(
                rich_panel,
                atlas,
                tuple(contract["rating_seasons"]),
                specs=rich_specs,
                threshold=float(contract["spm"]["rich"]["correlation_threshold"]),
            )
            if not rich_priors.empty:
                rich_priors["candidate"] = f"rich_spm_{horizon}y_{target_variant}"
                rich_selection["horizon"] = int(horizon)
                rich_selection["target_variant"] = target_variant
                prior_rows.append(rich_priors)
                selection_rows.append(rich_selection)
            coverage_rows.append(
                {
                    "horizon": int(horizon),
                    "target_variant": target_variant,
                    "box_first_rating_season": int(box_priors["Window_End"].min()),
                    "box_last_rating_season": int(box_priors["Window_End"].max()),
                    "rich_first_rating_season": int(rich_priors["Window_End"].min()) if not rich_priors.empty else None,
                    "rich_last_rating_season": int(rich_priors["Window_End"].max()) if not rich_priors.empty else None,
                }
            )
            print(f"SPM {horizon}y {target_variant}: complete", flush=True)

    priors = pd.concat(prior_rows, ignore_index=True)
    selections = pd.concat(selection_rows, ignore_index=True)
    games, ratings, prior_coverage = _evaluate_priors(priors, contract)
    folds, summary = _game_metrics(games)
    bootstrap = _paired_age_bootstrap(
        games,
        draws=int(contract["evaluation"]["bootstrap_draws"]),
        seed=int(contract["evaluation"]["bootstrap_seed"]),
    )
    key_bootstrap = _key_comparison_bootstrap(
        games,
        draws=int(contract["evaluation"]["bootstrap_draws"]),
        seed=int(contract["evaluation"]["bootstrap_seed"]),
    )

    outputs = {
        "targets.parquet": targets,
        "target_quality.parquet": target_quality,
        "priors.parquet": priors,
        "spm_fold_metrics.parquet": selections,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "paired_age_bootstrap.parquet": bootstrap,
        "paired_key_comparisons.parquet": key_bootstrap,
        "source_coverage.parquet": pd.DataFrame(coverage_rows),
        "prior_coverage.parquet": prior_coverage,
    }
    for name, value in outputs.items():
        _atomic_parquet(value, output / name)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "contract": contract,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": hashes[name]}
            for name, path in source_paths.items()
        },
        "quality": {
            "target_rows": len(targets),
            "prior_rows": len(priors),
            "game_prediction_rows": len(games),
            "target_component_identity_error": float(
                (targets["target_offense"] + targets["target_defense"] - targets["target_net"]).abs().max()
            ),
            "rating_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, value in outputs.items():
        run["files"][name] = {"rows": len(value), "sha256": sha256_file(output / name)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print("\nNormal minus age-adjusted paired MSE")
    print(bootstrap.to_string(index=False))


if __name__ == "__main__":
    main()
