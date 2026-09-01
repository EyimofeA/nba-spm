#!/usr/bin/env python3
"""Diagnose and redesign statistical priors for a one-season RAPM update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, diags, load_npz
from scipy.sparse.linalg import splu
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES, _fit as fit_box
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_unified_terminal_possessions,
)
from nba_impact.models.rapm_sufficient_statistics import (
    store_lambda_research_matrices,
)

try:
    from run_annual_spm_learner_screen import (
        ModelSpec,
        _feature_arms,
        _fit as fit_rich,
        _prune_features,
        _weighted_metrics,
    )
except ModuleNotFoundError:
    from research.run_annual_spm_learner_screen import (
        ModelSpec,
        _feature_arms,
        _fit as fit_rich,
        _prune_features,
        _weighted_metrics,
    )


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "aio_prior_complementarity_v1"
CONTRACT = ROOT / "research/experiments/aio_prior_complementarity_v1.yml"
OUTPUT_ROOT = ROOT / "artifacts/research/aio_prior_complementarity"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints_v1"
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
SILVER_POSSESSIONS = ROOT / "data/lake/silver/possessions.parquet"
SILVER_LINEUPS = ROOT / "data/lake/silver/possession_lineup_segments.parquet"
ANNUAL_TARGETS = ROOT / (
    "artifacts/models/canonical_annual_target_panel/"
    "canonical_annual_target_panel_v1_2d9ff74ca3/targets.parquet"
)
TARGET_WINDOW_RUN = ROOT / (
    "artifacts/research/target_window_spm_aio/"
    "target_window_spm_aio_v1_8e028133cb"
)
TARGET_WINDOWS = TARGET_WINDOW_RUN / "targets.parquet"
BOX_ANNUAL = ROOT / (
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
SIDES = ("offense", "defense")


@dataclass(frozen=True)
class AnnualMatrix:
    season: int
    players: np.ndarray
    xtx: csr_matrix
    xty: np.ndarray
    off_exposure: np.ndarray
    def_exposure: np.ndarray
    base_intercept: float
    game_design: csr_matrix
    game_ids: np.ndarray
    actual_margin: np.ndarray
    intercept_multiplier: np.ndarray
    unknown_slots: np.ndarray


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def checkpoint_frame(path: Path, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Load a complete stage or build it once with an atomic rename."""
    if path.exists():
        return pd.read_parquet(path)
    frame = builder()
    _atomic_parquet(frame, path)
    return frame


def canonical_frame_hash(frame: pd.DataFrame, keys: Iterable[str]) -> str:
    ordered = frame.sort_values(list(keys), kind="stable").reset_index(drop=True)
    payload = pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _load_contract(path: Path = CONTRACT) -> dict:
    contract = yaml.safe_load(path.read_text())
    _validate_contract(contract)
    return contract


def _validate_contract(contract: dict) -> None:
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment ID changed.")
    rating = tuple(map(int, contract["seasons"]["rating"]))
    outcome = tuple(map(int, contract["seasons"]["outcome"]))
    if rating != tuple(range(2016, 2026)) or outcome != tuple(range(2017, 2027)):
        raise ValueError("The ten requested rating folds changed.")
    if outcome != tuple(season + 1 for season in rating):
        raise ValueError("Each rating season must predict the next season.")
    if tuple(contract["seasons"]["design_selection_outcomes"]) != tuple(range(2017, 2022)):
        raise ValueError("Design-selection seasons changed.")
    if tuple(contract["seasons"]["later_diagnostic_outcomes"]) != tuple(range(2022, 2027)):
        raise ValueError("Later diagnostic seasons changed.")
    if int(contract["evaluation"]["bootstrap_draws"]) != 5000:
        raise ValueError("The publication bootstrap must use 5,000 draws.")


def past_reference(current: pd.DataFrame) -> pd.DataFrame:
    """R9 current ending at t-1 is exactly R9 past for rating season t."""
    output = current.copy()
    output["rating_season"] = output["rating_season"].astype(int) + 1
    output["reference"] = "nine_year_past"
    return output


def fully_lagged_pairs(
    features: pd.DataFrame,
    current_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Pair X_s with R9 current_s and label its application as rating s+1."""
    panel = features.merge(
        current_targets,
        left_on=["PLAYER_ID", "Season"],
        right_on=["PLAYER_ID", "rating_season"],
        how="inner",
        validate="one_to_one",
    )
    panel["application_rating_season"] = panel["Season"].astype(int) + 1
    return panel


def lambda_components(total: np.ndarray | float, trust: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    total_values = np.asarray(total, dtype=float)
    trust_values = np.asarray(trust, dtype=float)
    if (
        not np.isfinite(total_values).all()
        or not np.isfinite(trust_values).all()
        or (total_values < 0).any()
        or (trust_values < 0).any()
        or (trust_values > 1).any()
    ):
        raise ValueError("Penalty totals must be nonnegative and trust must lie in [0, 1].")
    return total_values * (1.0 - trust_values), total_values * trust_values


def select_from_history(
    rows: pd.DataFrame,
    parameter_columns: list[str],
    defaults: dict[str, float | str],
    *,
    current_outcome_season: int,
    selection_outcomes: tuple[int, ...],
    minimum_folds: int,
) -> dict[str, float | str]:
    """Select equal-season MSE from earlier folds and freeze after development."""
    if current_outcome_season in selection_outcomes:
        eligible = rows.loc[rows["outcome_season"].lt(current_outcome_season)]
    else:
        eligible = rows.loc[rows["outcome_season"].isin(selection_outcomes)]
    if eligible["outcome_season"].nunique() < minimum_folds:
        return defaults.copy()
    grouped = (
        eligible.groupby([*parameter_columns, "outcome_season"], as_index=False)
        .agg(mse=("mse", "mean"))
        .groupby(parameter_columns, as_index=False)
        .agg(equal_season_mse=("mse", "mean"))
    )
    winner = grouped.sort_values(["equal_season_mse", *parameter_columns], kind="stable").iloc[0]
    return {column: winner[column].item() if hasattr(winner[column], "item") else winner[column] for column in parameter_columns}


def assert_identical_games(games: pd.DataFrame) -> None:
    for season, frame in games.groupby("outcome_season", sort=True):
        wide = frame.pivot(index="game_id", columns="candidate", values="actual_margin")
        if wide.isna().any().any() or not wide.nunique(axis=1).eq(1).all():
            raise ValueError(f"Candidates do not score identical games in {season}.")


def assert_component_identity(frame: pd.DataFrame, *, tolerance: float = 1e-10) -> None:
    error = (frame["offense"] + frame["defense"] - frame["net"]).abs().max()
    if pd.notna(error) and float(error) > tolerance:
        raise ValueError(f"Offense plus defense differs from net by {error}.")


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    left_mean = float(np.average(left, weights=weight))
    right_mean = float(np.average(right, weights=weight))
    covariance = float(np.average((left - left_mean) * (right - right_mean), weights=weight))
    left_variance = float(np.average((left - left_mean) ** 2, weights=weight))
    right_variance = float(np.average((right - right_mean) ** 2, weights=weight))
    if left_variance <= 0 or right_variance <= 0:
        return math.nan
    return covariance / math.sqrt(left_variance * right_variance)


def _metric_row(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    variance = float(np.var(predicted))
    return {
        "mse": float(np.mean(error**2)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "calibration_slope": float(np.cov(actual, predicted, ddof=0)[0, 1] / variance) if variance > 0 else math.nan,
    }


def _matrix_dir(season: int) -> Path:
    return CHECKPOINT_ROOT / "annual_matrices" / f"season_{season}"


def _build_annual_matrices(contract: dict) -> pd.DataFrame:
    rows = []
    for season in map(int, contract["seasons"]["rating"]):
        directory = _matrix_dir(season)
        if not (directory / "manifest.json").exists():
            frame = load_unified_terminal_possessions(
                POSSESSION_CACHE,
                SILVER_POSSESSIONS,
                SILVER_LINEUPS,
                (season, season + 1),
                transition_season=2024,
                game_types=("regular",),
            )
            train = frame.loc[frame["season"].eq(season)].copy()
            evaluate = frame.loc[frame["season"].eq(season + 1)].copy()
            store_lambda_research_matrices(
                train,
                directory,
                evaluation_frame=evaluate,
                metadata={"rating_season": season, "outcome_season": season + 1},
            )
        manifest = json.loads((directory / "manifest.json").read_text())
        rows.append(
            {
                "rating_season": season,
                "outcome_season": season + 1,
                "training_rows": manifest["train"]["possession_rows"],
                "players": manifest["train"]["players_in_train_or_evaluation"],
                "games": manifest["evaluation"]["games"],
                "manifest_sha256": sha256_file(directory / "manifest.json"),
            }
        )
    return pd.DataFrame(rows)


def _load_matrix(season: int) -> AnnualMatrix:
    root = _matrix_dir(season)
    manifest = json.loads((root / "manifest.json").read_text())
    return AnnualMatrix(
        season=season,
        players=np.load(root / "player_ids.npy"),
        xtx=load_npz(root / "train_xtx.npz").tocsr(),
        xty=np.load(root / "train_xty_centered.npy"),
        off_exposure=np.load(root / "train_off_possessions.npy"),
        def_exposure=np.load(root / "train_def_possessions.npy"),
        base_intercept=float(manifest["train"]["base_intercept"]),
        game_design=load_npz(root / "evaluation_game_design.npz").tocsr(),
        game_ids=np.load(root / "evaluation_game_ids.npy").astype(str),
        actual_margin=np.load(root / "evaluation_actual_margin.npy"),
        intercept_multiplier=np.load(root / "evaluation_intercept_multiplier.npy"),
        unknown_slots=np.load(root / "evaluation_unknown_player_slots.npy"),
    )


def coefficient_center(prior: pd.DataFrame, matrix: AnnualMatrix) -> tuple[np.ndarray, np.ndarray]:
    indexed = prior.set_index("PLAYER_ID")
    offense = pd.to_numeric(indexed["prior_offense"], errors="raise").reindex(matrix.players)
    defense = pd.to_numeric(indexed["prior_defense"], errors="raise").reindex(matrix.players)
    present = (offense.notna() & defense.notna()).to_numpy()
    off = offense.fillna(0.0).to_numpy(dtype=float) / 100.0
    deff = -defense.fillna(0.0).to_numpy(dtype=float) / 100.0
    if matrix.off_exposure.sum() > 0:
        off -= np.average(off, weights=matrix.off_exposure)
    if matrix.def_exposure.sum() > 0:
        deff -= np.average(deff, weights=matrix.def_exposure)
    return np.concatenate([off, deff, np.zeros(1)]), present


def _recenter(matrix: AnnualMatrix, raw: np.ndarray, *, include_base: bool) -> tuple[np.ndarray, float]:
    n = len(matrix.players)
    beta = np.asarray(raw, dtype=float).copy()
    off_mean = float(np.average(beta[:n], weights=matrix.off_exposure))
    def_mean = float(np.average(beta[n : 2 * n], weights=matrix.def_exposure))
    beta[:n] -= off_mean
    beta[n : 2 * n] -= def_mean
    intercept = (matrix.base_intercept if include_base else 0.0) + 5.0 * (off_mean + def_mean)
    return beta, intercept


def _game_prediction(matrix: AnnualMatrix, beta: np.ndarray, intercept: float) -> np.ndarray:
    return np.asarray(matrix.game_design @ beta).ravel() + intercept * matrix.intercept_multiplier


def _uniform_basis(
    matrix: AnnualMatrix,
    center: np.ndarray,
    offense_total: float,
    defense_total: float,
    home_penalty: float,
    factor_cache: dict[tuple[int, float, float], Callable[[np.ndarray], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    key = (matrix.season, float(offense_total), float(defense_total))
    if key not in factor_cache:
        n = len(matrix.players)
        penalty = np.concatenate(
            [
                np.full(n, offense_total),
                np.full(n, defense_total),
                np.asarray([home_penalty]),
            ]
        )
        factor_cache[key] = splu((matrix.xtx + diags(penalty, format="csr")).tocsc()).solve
    solve = factor_cache[key]
    n = len(matrix.players)
    base_beta, base_intercept = _recenter(matrix, solve(matrix.xty), include_base=True)
    off_rhs = np.zeros(2 * n + 1)
    off_rhs[:n] = offense_total * center[:n]
    def_rhs = np.zeros(2 * n + 1)
    def_rhs[n : 2 * n] = defense_total * center[n : 2 * n]
    off_beta, off_intercept = _recenter(matrix, solve(off_rhs), include_base=False)
    def_beta, def_intercept = _recenter(matrix, solve(def_rhs), include_base=False)
    return (
        _game_prediction(matrix, base_beta, base_intercept),
        _game_prediction(matrix, off_beta, off_intercept),
        _game_prediction(matrix, def_beta, def_intercept),
    )


def _solve_heterogeneous(
    matrix: AnnualMatrix,
    center: np.ndarray,
    *,
    offense_total: float,
    defense_total: float,
    offense_trust: float,
    defense_trust: float,
    offense_multiplier: np.ndarray,
    defense_multiplier: np.ndarray,
    home_penalty: float,
) -> tuple[np.ndarray, float]:
    n = len(matrix.players)
    off_zero, off_prior = lambda_components(offense_total, offense_trust)
    def_zero, def_prior = lambda_components(defense_total, defense_trust)
    off_prior = np.broadcast_to(off_prior, (n,)) * offense_multiplier
    def_prior = np.broadcast_to(def_prior, (n,)) * defense_multiplier
    off_total = np.broadcast_to(off_zero, (n,)) + off_prior
    def_total = np.broadcast_to(def_zero, (n,)) + def_prior
    penalty = np.concatenate([off_total, def_total, np.asarray([home_penalty])])
    rhs = matrix.xty.copy()
    rhs[:n] += off_prior * center[:n]
    rhs[n : 2 * n] += def_prior * center[n : 2 * n]
    raw = splu((matrix.xtx + diags(penalty, format="csr")).tocsc()).solve(rhs)
    return _recenter(matrix, raw, include_base=True)


def _standalone_prediction(matrix: AnnualMatrix, center: np.ndarray) -> np.ndarray:
    beta, intercept = _recenter(matrix, center, include_base=True)
    return _game_prediction(matrix, beta, intercept)


def _rating_frame(matrix: AnnualMatrix, beta: np.ndarray, candidate: str) -> pd.DataFrame:
    n = len(matrix.players)
    return pd.DataFrame(
        {
            "PLAYER_ID": matrix.players,
            "rating_season": matrix.season,
            "candidate": candidate,
            "offense": 100.0 * beta[:n],
            "defense": -100.0 * beta[n : 2 * n],
            "Poss_Off": matrix.off_exposure,
            "Poss_Def": matrix.def_exposure,
        }
    ).assign(net=lambda value: value["offense"] + value["defense"])


def _future_reference(contract: dict) -> pd.DataFrame:
    rows = []
    config = RapmConfig(
        seasons=tuple(range(2017, 2027)),
        lambda_off=float(contract["rapm"]["target"]["lambda_off"]),
        lambda_def=float(contract["rapm"]["target"]["lambda_def"]),
        lambda_home=float(contract["rapm"]["target"]["lambda_home"]),
        data_scope="aio_prior_complementarity_future_reference",
    )
    frame = load_unified_terminal_possessions(
        POSSESSION_CACHE,
        SILVER_POSSESSIONS,
        SILVER_LINEUPS,
        tuple(range(2017, 2027)),
        transition_season=2024,
        game_types=("regular",),
    )
    original = build_design(frame, include_home=True)
    season_mean = frame.groupby("season")["pts"].mean()
    design = replace(
        original,
        y=frame["pts"].to_numpy(dtype=float)
        - frame["season"].map(season_mean).to_numpy(dtype=float),
    )
    for end in range(2019, 2027):
        mask = (design.seasons >= end - 2) & (design.seasons <= end)
        beta, _ = fit_coefficients(design, config, row_mask=mask)
        n = len(design.players)
        X = design.X[mask]
        off = np.asarray(X[:, :n].sum(axis=0)).ravel()
        deff = np.asarray(X[:, n : 2 * n].sum(axis=0)).ravel()
        active = np.minimum(off, deff) > 0
        output = pd.DataFrame(
            {
                "PLAYER_ID": design.players[active],
                "rating_season": end - 3,
                "reference": "three_year_future",
                "target_offense": 100.0 * beta[:n][active],
                "target_defense": -100.0 * beta[n : 2 * n][active],
                "Poss_Off": off[active],
                "Poss_Def": deff[active],
            }
        )
        output["target_net"] = output["target_offense"] + output["target_defense"]
        rows.append(output)
        print(f"future reference {end - 3} -> {end}: complete", flush=True)
    return pd.concat(rows, ignore_index=True)


def _reference_panels(contract: dict) -> pd.DataFrame:
    annual = pd.read_parquet(ANNUAL_TARGETS).rename(columns={"Season": "rating_season"})
    annual["reference"] = "one_year"
    current = pd.read_parquet(TARGET_WINDOWS)
    current = current.loc[current["horizon"].eq(9) & current["target_variant"].eq("normal")].copy()
    current = current.rename(columns={"Window_End": "rating_season"})
    current["reference"] = "nine_year_current"
    current = current[
        ["PLAYER_ID", "rating_season", "reference", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]
    ]
    future = checkpoint_frame(CHECKPOINT_ROOT / "references" / "three_year_future.parquet", lambda: _future_reference(contract))
    output = pd.concat(
        [
            annual[["PLAYER_ID", "rating_season", "reference", "target_offense", "target_defense", "target_net", "Poss_Off", "Poss_Def"]],
            current,
            past_reference(current),
            future,
        ],
        ignore_index=True,
    )
    if output.duplicated(["PLAYER_ID", "rating_season", "reference"]).any():
        raise ValueError("Reference keys are not unique.")
    return output


def _load_features(contract: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    box = pd.read_parquet(BOX_ANNUAL).rename(columns={"Window_End": "Season"})
    rich = pd.read_parquet(RICH_ANNUAL).rename(columns={"Window_End": "Season"})
    mechanism = pd.read_parquet(MECHANISM_ANNUAL).rename(columns={"Window_End": "Season"})
    extra = [column for column in mechanism.columns if column not in rich.columns]
    rich = rich.merge(
        mechanism[["PLAYER_ID", "Season", *extra]],
        on=["PLAYER_ID", "Season"],
        how="left",
        validate="one_to_one",
    )
    atlas = pd.read_parquet(ATLAS)
    rich_arms = {
        side: _feature_arms(atlas, rich, side)["audited_all"]
        for side in SIDES
    }
    exact = set(contract["outcome_censor"]["exact"])
    prefixes = tuple(contract["outcome_censor"]["scorer_matchup_result_prefixes"])
    censored_arms = {
        side: outcome_censored_features(rich_arms[side], exact, prefixes)
        for side in SIDES
    }
    required_box = {"PLAYER_ID", "Season", *BOX_PIPM_STYLE_FEATURES}
    if missing := sorted(required_box - set(box.columns)):
        raise ValueError(f"Box feature panel misses {missing}.")
    required_rich = {
        "PLAYER_ID",
        "Season",
        "OffPoss",
        "DefPoss",
        *rich_arms["offense"],
        *rich_arms["defense"],
        *contract["defense_residual"]["activity_only"],
        *contract["defense_residual"]["outcome_augmented"],
    }
    if missing := sorted(required_rich - set(rich.columns)):
        raise ValueError(f"Rich feature panel misses {missing}.")
    if box.duplicated(["PLAYER_ID", "Season"]).any() or rich.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Annual feature keys are not unique.")
    return box, rich, atlas, rich_arms, censored_arms


def outcome_censored_features(
    features: Iterable[str], exact: set[str], prefixes: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in features
        if feature not in exact and not feature.startswith(prefixes)
    )


def _learning_panel(
    features: pd.DataFrame,
    references: pd.DataFrame,
    design: str,
) -> pd.DataFrame:
    reference_name = {
        "current_control": "nine_year_current",
        "target_excluded": "nine_year_past",
        "fully_lagged": "nine_year_current",
    }[design]
    target = references.loc[references["reference"].eq(reference_name)].copy()
    if design == "fully_lagged":
        panel = fully_lagged_pairs(features, target)
    else:
        panel = features.merge(
            target,
            left_on=["PLAYER_ID", "Season"],
            right_on=["PLAYER_ID", "rating_season"],
            how="inner",
            validate="one_to_one",
        )
        panel["application_rating_season"] = panel["Season"].astype(int)
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    return panel


def _fit_prior_side(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    side: str,
    learner: str,
    features: tuple[str, ...],
    contract: dict,
) -> tuple[np.ndarray, tuple[str, ...]]:
    target = f"target_{side}"
    if learner == "box15":
        alpha = float(contract["spm"]["box15"][f"{side}_alpha"])
        model = fit_box(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
        return model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES]), BOX_PIPM_STYLE_FEATURES
    selected = _prune_features(
        train,
        features,
        threshold=float(contract["spm"]["correlation_prune_threshold"]),
    )
    if side == "offense":
        spec = ModelSpec(
            "elastic_net",
            {
                "alpha": float(contract["spm"]["rich"]["offense_alpha"]),
                "l1_ratio": float(contract["spm"]["rich"]["offense_l1_ratio"]),
            },
        )
    else:
        spec = ModelSpec(
            "ridge",
            {"alpha": float(contract["spm"]["rich"]["defense_alpha"])},
        )
    model = fit_rich(spec, train, selected, target)
    return model.predict(test.loc[:, selected]), selected


def _base_priors(
    contract: dict,
    references: pd.DataFrame,
    box_features: pd.DataFrame,
    rich_features: pd.DataFrame,
    rich_arms: dict[str, tuple[str, ...]],
    censored_arms: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    selections = []
    rating_seasons = tuple(map(int, contract["seasons"]["rating"]))
    for design in contract["designs"]:
        panels = {
            "box15": _learning_panel(box_features, references, design),
            "rich": _learning_panel(rich_features, references, design),
            "rich_censored": _learning_panel(rich_features, references, design),
        }
        for rating_season in rating_seasons:
            input_season = rating_season - int(contract["designs"][design]["rating_input_lag"])
            candidates = {}
            for learner, panel in panels.items():
                train = panel.loc[panel["Season"].lt(input_season)].copy()
                test = panel.loc[panel["Season"].eq(input_season)].copy()
                if train.empty or test.empty:
                    continue
                prior = test[["PLAYER_ID"]].copy()
                feature_map = censored_arms if learner == "rich_censored" else rich_arms
                for side in SIDES:
                    predicted, selected = _fit_prior_side(
                        train,
                        test,
                        side=side,
                        learner="box15" if learner == "box15" else "rich",
                        features=feature_map[side],
                        contract=contract,
                    )
                    prior[f"prior_{side}"] = predicted
                    selections.append(
                        {
                            "design": design,
                            "candidate": learner,
                            "rating_season": rating_season,
                            "input_season": input_season,
                            "side": side,
                            "training_start": int(train["Season"].min()),
                            "training_end": int(train["Season"].max()),
                            "training_seasons": int(train["Season"].nunique()),
                            "selected_features": len(selected),
                            "selected_feature_names": json.dumps(selected),
                        }
                    )
                prior["prior_net"] = prior["prior_offense"] + prior["prior_defense"]
                candidates[learner] = prior
            if set(candidates) != {"box15", "rich", "rich_censored"}:
                continue
            common = set.intersection(*(set(frame["PLAYER_ID"]) for frame in candidates.values()))
            for learner, prior in candidates.items():
                prior = prior.loc[prior["PLAYER_ID"].isin(common)].copy()
                prior["design"] = design
                prior["candidate"] = learner
                prior["rating_season"] = rating_season
                prior["input_season"] = input_season
                rows.append(prior)
            print(f"base priors {design} {rating_season}: complete", flush=True)
    output = pd.concat(rows, ignore_index=True)
    assert_component_identity(
        output.rename(columns={"prior_offense": "offense", "prior_defense": "defense", "prior_net": "net"})
    )
    return output, pd.DataFrame(selections)


def _control_games(
    matrix: AnnualMatrix,
    prior: pd.DataFrame,
    contract: dict,
    factor_cache: dict,
    *,
    candidate: str,
) -> pd.DataFrame:
    center, _ = coefficient_center(prior, matrix)
    control = contract["rapm"]["aio_control"]
    base, off, deff = _uniform_basis(
        matrix,
        center,
        float(control["lambda_off"]),
        float(control["lambda_def"]),
        float(control["lambda_home"]),
        factor_cache,
    )
    predicted = base + off + deff
    return pd.DataFrame(
        {
            "game_id": matrix.game_ids,
            "rating_season": matrix.season,
            "outcome_season": matrix.season + 1,
            "candidate": candidate,
            "actual_margin": matrix.actual_margin,
            "predicted_margin": predicted,
            "mse": (matrix.actual_margin - predicted) ** 2,
        }
    )


def _direct_blends(
    contract: dict,
    base_priors: pd.DataFrame,
    matrices: dict[int, AnnualMatrix],
    factor_cache: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gamma_grid = tuple(map(float, contract["direct_blend"]["gamma"]))
    selection_outcomes = tuple(map(int, contract["seasons"]["design_selection_outcomes"]))
    output = []
    grid_rows = []
    for design in ("target_excluded", "fully_lagged"):
        candidates = base_priors.loc[base_priors["design"].eq(design)]
        for rating_season in map(int, contract["seasons"]["rating"]):
            box = candidates.loc[candidates["candidate"].eq("box15") & candidates["rating_season"].eq(rating_season)].set_index("PLAYER_ID")
            rich = candidates.loc[candidates["candidate"].eq("rich") & candidates["rating_season"].eq(rating_season)].set_index("PLAYER_ID")
            if box.empty or rich.empty:
                continue
            if not box.index.equals(rich.index):
                rich = rich.reindex(box.index)
            for gamma_off in gamma_grid:
                for gamma_def in gamma_grid:
                    prior = pd.DataFrame(
                        {
                            "PLAYER_ID": box.index,
                            "prior_offense": box["prior_offense"] + gamma_off * (rich["prior_offense"] - box["prior_offense"]),
                            "prior_defense": box["prior_defense"] + gamma_def * (rich["prior_defense"] - box["prior_defense"]),
                        }
                    )
                    prior["prior_net"] = prior["prior_offense"] + prior["prior_defense"]
                    game = _control_games(
                        matrices[rating_season],
                        prior,
                        contract,
                        factor_cache,
                        candidate=f"blend_{gamma_off:g}_{gamma_def:g}",
                    )
                    grid_rows.append(
                        {
                            "design": design,
                            "rating_season": rating_season,
                            "outcome_season": rating_season + 1,
                            "gamma_offense": gamma_off,
                            "gamma_defense": gamma_def,
                            "mse": float(game["mse"].mean()),
                        }
                    )
            grid = pd.DataFrame(grid_rows)
            history = grid.loc[grid["design"].eq(design)]
            chosen = select_from_history(
                history,
                ["gamma_offense", "gamma_defense"],
                {"gamma_offense": 0.0, "gamma_defense": 0.0},
                current_outcome_season=rating_season + 1,
                selection_outcomes=selection_outcomes,
                minimum_folds=int(contract["direct_blend"]["minimum_prior_folds"]),
            )
            prior = pd.DataFrame(
                {
                    "PLAYER_ID": box.index,
                    "prior_offense": box["prior_offense"] + float(chosen["gamma_offense"]) * (rich["prior_offense"] - box["prior_offense"]),
                    "prior_defense": box["prior_defense"] + float(chosen["gamma_defense"]) * (rich["prior_defense"] - box["prior_defense"]),
                }
            )
            prior["prior_net"] = prior["prior_offense"] + prior["prior_defense"]
            prior["design"] = design
            prior["candidate"] = "direct_blend"
            prior["rating_season"] = rating_season
            prior["input_season"] = rating_season - int(contract["designs"][design]["rating_input_lag"])
            prior["gamma_offense"] = float(chosen["gamma_offense"])
            prior["gamma_defense"] = float(chosen["gamma_defense"])
            output.append(prior.reset_index(drop=True))
    return pd.concat(output, ignore_index=True), pd.DataFrame(grid_rows)


def _ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _cross_fitted_box_defense(train: pd.DataFrame, contract: dict) -> np.ndarray:
    prediction = pd.Series(index=train.index, dtype=float)
    seasons = tuple(sorted(map(int, train["Season"].unique())))
    for held_season in seasons:
        inner = train.loc[train["Season"].ne(held_season)]
        held = train.loc[train["Season"].eq(held_season)]
        if inner.empty:
            raise ValueError("Defense residual needs at least two training seasons.")
        model = fit_box(
            inner,
            BOX_PIPM_STYLE_FEATURES,
            "target_defense",
            float(contract["spm"]["box15"]["defense_alpha"]),
        )
        prediction.loc[held.index] = model.predict(held.loc[:, BOX_PIPM_STYLE_FEATURES])
    if prediction.isna().any():
        raise ValueError("Cross-fitted Box15 labels have missing rows.")
    return prediction.loc[train.index].to_numpy(dtype=float)


def _defense_residuals(
    contract: dict,
    references: pd.DataFrame,
    rich_features: pd.DataFrame,
    base_priors: pd.DataFrame,
    matrices: dict[int, AnnualMatrix],
    factor_cache: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alpha_grid = tuple(map(float, contract["defense_residual"]["alpha"]))
    gamma_grid = tuple(map(float, contract["defense_residual"]["gamma"]))
    selection_outcomes = tuple(map(int, contract["seasons"]["design_selection_outcomes"]))
    pool_fields = {
        "activity": tuple(contract["defense_residual"]["activity_only"]),
        "outcome": tuple(
            dict.fromkeys(
                [
                    *contract["defense_residual"]["activity_only"],
                    *contract["defense_residual"]["outcome_augmented"],
                ]
            )
        ),
    }
    output = []
    grid_rows = []
    for design in ("target_excluded", "fully_lagged"):
        panel = _learning_panel(rich_features, references, design)
        for rating_season in map(int, contract["seasons"]["rating"]):
            input_season = rating_season - int(contract["designs"][design]["rating_input_lag"])
            train = panel.loc[panel["Season"].lt(input_season)].copy()
            test = panel.loc[panel["Season"].eq(input_season)].copy()
            if test.empty:
                continue
            base = base_priors.loc[
                base_priors["design"].eq(design)
                & base_priors["candidate"].eq("box15")
                & base_priors["rating_season"].eq(rating_season)
            ].set_index("PLAYER_ID")
            test = test.loc[test["PLAYER_ID"].isin(base.index)].copy()
            base = base.reindex(test["PLAYER_ID"])
            if train["Season"].nunique() < 2:
                for pool in pool_fields:
                    prior = pd.DataFrame(
                        {
                            "PLAYER_ID": test["PLAYER_ID"].to_numpy(),
                            "prior_offense": base["prior_offense"].to_numpy(dtype=float),
                            "prior_defense": base["prior_defense"].to_numpy(dtype=float),
                        }
                    )
                    prior["prior_net"] = prior["prior_offense"] + prior["prior_defense"]
                    prior["design"] = design
                    prior["candidate"] = f"def_residual_{pool}"
                    prior["rating_season"] = rating_season
                    prior["input_season"] = input_season
                    prior["residual_alpha"] = 300.0
                    prior["residual_gamma"] = 0.0
                    output.append(prior)
                continue
            base_oof = _cross_fitted_box_defense(train, contract)
            residual_train = train.copy()
            residual_train["residual_target"] = train["target_defense"].to_numpy(dtype=float) - base_oof
            predicted_by_pool_alpha = {}
            for pool, fields in pool_fields.items():
                for alpha in alpha_grid:
                    model = _ridge_pipeline(alpha)
                    model.fit(
                        residual_train.loc[:, fields],
                        residual_train["residual_target"],
                        model__sample_weight=residual_train["sample_weight"],
                    )
                    predicted_by_pool_alpha[(pool, alpha)] = model.predict(test.loc[:, fields])
                    for gamma in gamma_grid:
                        prior = pd.DataFrame(
                            {
                                "PLAYER_ID": test["PLAYER_ID"].to_numpy(),
                                "prior_offense": base["prior_offense"].to_numpy(dtype=float),
                                "prior_defense": base["prior_defense"].to_numpy(dtype=float) + gamma * predicted_by_pool_alpha[(pool, alpha)],
                            }
                        )
                        prior["prior_net"] = prior["prior_offense"] + prior["prior_defense"]
                        game = _control_games(
                            matrices[rating_season],
                            prior,
                            contract,
                            factor_cache,
                            candidate=f"def_residual_{pool}_{alpha:g}_{gamma:g}",
                        )
                        grid_rows.append(
                            {
                                "design": design,
                                "pool": pool,
                                "rating_season": rating_season,
                                "outcome_season": rating_season + 1,
                                "alpha": alpha,
                                "gamma": gamma,
                                "mse": float(game["mse"].mean()),
                            }
                        )
            grid = pd.DataFrame(grid_rows)
            for pool in pool_fields:
                history = grid.loc[grid["design"].eq(design) & grid["pool"].eq(pool)]
                chosen = select_from_history(
                    history,
                    ["alpha", "gamma"],
                    {"alpha": 300.0, "gamma": 0.0},
                    current_outcome_season=rating_season + 1,
                    selection_outcomes=selection_outcomes,
                    minimum_folds=int(contract["defense_residual"]["minimum_prior_folds"]),
                )
                prior = pd.DataFrame(
                    {
                        "PLAYER_ID": test["PLAYER_ID"].to_numpy(),
                        "prior_offense": base["prior_offense"].to_numpy(dtype=float),
                        "prior_defense": base["prior_defense"].to_numpy(dtype=float)
                        + float(chosen["gamma"]) * predicted_by_pool_alpha[(pool, float(chosen["alpha"]))],
                    }
                )
                prior["prior_net"] = prior["prior_offense"] + prior["prior_defense"]
                prior["design"] = design
                prior["candidate"] = f"def_residual_{pool}"
                prior["rating_season"] = rating_season
                prior["input_season"] = input_season
                prior["residual_alpha"] = float(chosen["alpha"])
                prior["residual_gamma"] = float(chosen["gamma"])
                output.append(prior)
            print(f"defense residuals {design} {rating_season}: complete", flush=True)
    return pd.concat(output, ignore_index=True), pd.DataFrame(grid_rows)


def _common_prior_coverage(priors: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected = {
        "current_control": {"box15", "rich"},
        "target_excluded": {"box15", "rich", "rich_censored", "direct_blend", "def_residual_activity", "def_residual_outcome"},
        "fully_lagged": {"box15", "rich", "rich_censored", "direct_blend", "def_residual_activity", "def_residual_outcome"},
    }
    global_common: dict[int, set[int]] = {}
    for rating_season in map(int, contract["seasons"]["rating"]):
        season = priors.loc[priors["rating_season"].eq(rating_season)]
        sets = []
        for design, candidates in expected.items():
            for candidate in candidates:
                frame = season.loc[season["design"].eq(design) & season["candidate"].eq(candidate)]
                if frame.empty:
                    raise ValueError(f"Missing prior {design}/{candidate}/{rating_season}.")
                sets.append(set(map(int, frame["PLAYER_ID"])))
        global_common[rating_season] = set.intersection(*sets)
    kept = []
    coverage = []
    for rating_season in map(int, contract["seasons"]["rating"]):
        for design, candidates in expected.items():
            for candidate in sorted(candidates):
                frame = priors.loc[
                    priors["design"].eq(design)
                    & priors["candidate"].eq(candidate)
                    & priors["rating_season"].eq(rating_season)
                ]
                common = global_common[rating_season]
                selected = frame.loc[frame["PLAYER_ID"].isin(common)].copy()
                kept.append(selected)
                coverage.append(
                    {
                        "design": design,
                        "candidate": candidate,
                        "rating_season": rating_season,
                        "available_players": len(frame),
                        "common_prior_players": len(selected),
                        "coverage_fraction": len(selected) / len(frame),
                    }
                )
    output = pd.concat(kept, ignore_index=True)
    if output.duplicated(["design", "candidate", "rating_season", "PLAYER_ID"]).any():
        raise ValueError("Prior keys are not unique after coverage matching.")
    return output, pd.DataFrame(coverage)


def _candidate_name(design: str, candidate: str) -> str:
    return f"{design}__{candidate}"


def _prior_target_rows(priors: pd.DataFrame, references: pd.DataFrame, rich_features: pd.DataFrame) -> pd.DataFrame:
    target_names = {
        "current_control": "nine_year_current",
        "target_excluded": "nine_year_past",
        "fully_lagged": "nine_year_past",
    }
    availability = rich_features[[
        "PLAYER_ID",
        "Season",
        "OffPoss",
        "DefPoss",
        "has_hustle_tracking",
        "has_matchup_tracking",
        "has_dfg_tracking",
        "has_rim_defense_tracking",
    ]].copy()
    availability["unavailable_source_family_count"] = 4.0 - availability[
        ["has_hustle_tracking", "has_matchup_tracking", "has_dfg_tracking", "has_rim_defense_tracking"]
    ].sum(axis=1)
    rows = []
    for design, frame in priors.groupby("design"):
        target = references.loc[references["reference"].eq(target_names[design])]
        joined = frame.merge(
            target,
            on=["PLAYER_ID", "rating_season"],
            how="inner",
            validate="many_to_one",
        )
        input_meta = availability.rename(columns={"Season": "input_season"})
        joined = joined.merge(
            input_meta[["PLAYER_ID", "input_season", "unavailable_source_family_count"]],
            on=["PLAYER_ID", "input_season"],
            how="left",
            validate="many_to_one",
        )
        joined["reference_exposure"] = np.minimum(joined["Poss_Off"], joined["Poss_Def"])
        joined["sample_weight"] = np.sqrt(joined["reference_exposure"].clip(lower=1))
        joined["candidate_full"] = [
            _candidate_name(design, candidate) for candidate in joined["candidate"]
        ]
        rows.append(joined)
    output = pd.concat(rows, ignore_index=True)
    box = output.loc[output["candidate"].eq("box15"), ["design", "rating_season", "PLAYER_ID", "prior_offense", "prior_defense"]].rename(
        columns={"prior_offense": "box_offense", "prior_defense": "box_defense"}
    )
    rich = output.loc[output["candidate"].eq("rich"), ["design", "rating_season", "PLAYER_ID", "prior_offense", "prior_defense"]].rename(
        columns={"prior_offense": "rich_offense", "prior_defense": "rich_defense"}
    )
    disagreement = box.merge(rich, on=["design", "rating_season", "PLAYER_ID"], validate="one_to_one")
    for side in SIDES:
        disagreement[f"absolute_{side}_disagreement"] = (
            disagreement[f"box_{side}"] - disagreement[f"rich_{side}"]
        ).abs()
    output = output.merge(
        disagreement[["design", "rating_season", "PLAYER_ID", "absolute_offense_disagreement", "absolute_defense_disagreement"]],
        on=["design", "rating_season", "PLAYER_ID"],
        validate="many_to_one",
    )
    return output


def _shared_error_diagnostics(
    contract: dict,
    prior_targets: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = prior_targets.loc[
        prior_targets["design"].eq("current_control")
        & prior_targets["candidate"].isin(("box15", "rich"))
        & prior_targets["rating_season"].le(int(contract["seasons"]["future_reference_rating_end"]))
    ].copy()
    future = references.loc[references["reference"].eq("three_year_future")].copy()
    one = references.loc[references["reference"].eq("one_year")].copy()
    future = future.merge(
        one[["PLAYER_ID", "rating_season", "target_offense", "target_defense", "Poss_Off", "Poss_Def"]],
        on=["PLAYER_ID", "rating_season"],
        suffixes=("_future", "_one"),
        validate="one_to_one",
    )
    matched = current.merge(
        future,
        on=["PLAYER_ID", "rating_season"],
        how="inner",
        validate="many_to_one",
    )
    rows = []
    for (candidate, season), frame in matched.groupby(["candidate", "rating_season"], sort=True):
        for side in SIDES:
            prior_error = frame[f"prior_{side}"] - frame[f"target_{side}_future"]
            rapm_error = frame[f"target_{side}_one"] - frame[f"target_{side}_future"]
            weight = np.minimum(frame["Poss_Off_future"], frame["Poss_Def_future"]).to_numpy(dtype=float)
            for exposure_cut in (0, 1000):
                keep = np.minimum(
                    np.minimum(frame["Poss_Off_one"], frame["Poss_Def_one"]),
                    np.minimum(frame["Poss_Off_future"], frame["Poss_Def_future"]),
                ).to_numpy(dtype=float) >= exposure_cut
                if keep.sum() < 3:
                    continue
                rows.append(
                    {
                        "candidate": candidate,
                        "rating_season": int(season),
                        "side": side,
                        "exposure_cut": exposure_cut,
                        "players": int(keep.sum()),
                        "weighted_error_correlation": _weighted_correlation(
                            prior_error.to_numpy(dtype=float)[keep],
                            rapm_error.to_numpy(dtype=float)[keep],
                            weight[keep],
                        ),
                        "prior_error_bias": float(np.average(prior_error.to_numpy(dtype=float)[keep], weights=weight[keep])),
                        "prior_error_variance": float(np.average((prior_error.to_numpy(dtype=float)[keep] - np.average(prior_error.to_numpy(dtype=float)[keep], weights=weight[keep])) ** 2, weights=weight[keep])),
                    }
                )
    folds = pd.DataFrame(rows)
    summaries = []
    bootstrap_rows = []
    rng = np.random.default_rng(int(contract["evaluation"]["bootstrap_seed"]) + 11)
    draws = int(contract["evaluation"]["shared_error_bootstrap_draws"])
    for (side, exposure_cut), frame in folds.groupby(["side", "exposure_cut"]):
        wide = frame.pivot(index="rating_season", columns="candidate", values="weighted_error_correlation").dropna()
        for candidate in ("box15", "rich"):
            values = np.clip(wide[candidate].to_numpy(dtype=float), -0.999999, 0.999999)
            summaries.append(
                {
                    "side": side,
                    "exposure_cut": exposure_cut,
                    "candidate": candidate,
                    "seasons": len(values),
                    "equal_season_fisher_correlation": float(np.tanh(np.mean(np.arctanh(values)))),
                }
            )
        delta = np.arctanh(np.clip(wide["rich"].to_numpy(), -0.999999, 0.999999)) - np.arctanh(np.clip(wide["box15"].to_numpy(), -0.999999, 0.999999))
        samples = np.empty(draws)
        for draw in range(draws):
            samples[draw] = float(delta[rng.integers(0, len(delta), len(delta))].mean())
        low80, high80 = np.quantile(samples, [0.1, 0.9])
        correlation_delta = float(
            np.tanh(np.mean(np.arctanh(np.clip(wide["rich"], -0.999999, 0.999999))))
            - np.tanh(np.mean(np.arctanh(np.clip(wide["box15"], -0.999999, 0.999999))))
        )
        bootstrap_rows.append(
            {
                "side": side,
                "exposure_cut": exposure_cut,
                "seasons": len(delta),
                "rich_minus_box_correlation": correlation_delta,
                "fisher_delta_80_low": float(low80),
                "fisher_delta_80_high": float(high80),
                "positive_season_fraction": float(np.mean(delta > 0)),
                "materially_greater_shared_error": bool(
                    correlation_delta >= float(contract["evaluation"]["shared_error_material_delta"])
                    and low80 > 0
                    and np.mean(delta > 0) >= 2 / 3
                ),
            }
        )
    return folds, pd.DataFrame(summaries), pd.DataFrame(bootstrap_rows)


def _falsification_controls(
    contract: dict,
    prior_targets: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(contract["evaluation"]["bootstrap_seed"]) + 29)
    current = prior_targets.loc[
        prior_targets["design"].eq("current_control")
        & prior_targets["candidate"].isin(("box15", "rich"))
        & prior_targets["rating_season"].le(int(contract["seasons"]["future_reference_rating_end"]))
    ].copy()
    future = references.loc[references["reference"].eq("three_year_future")]
    one = references.loc[references["reference"].eq("one_year")]
    frame = current.merge(future, on=["PLAYER_ID", "rating_season"], suffixes=("", "_future"), validate="many_to_one")
    frame = frame.merge(
        one[["PLAYER_ID", "rating_season", "target_offense", "target_defense"]],
        on=["PLAYER_ID", "rating_season"],
        suffixes=("", "_one"),
        validate="many_to_one",
    )
    rows = []
    for season, season_frame in frame.groupby("rating_season", sort=True):
        for component in SIDES:
            reference_rows = season_frame.loc[season_frame["candidate"].eq("box15")]
            permuted = reference_rows[f"target_{component}_future"].to_numpy(dtype=float).copy()
            rng.shuffle(permuted)
            for candidate, group in season_frame.groupby("candidate", sort=True):
                group = group.sort_values("PLAYER_ID")
                ordered_reference = reference_rows.sort_values("PLAYER_ID")
                if not np.array_equal(group["PLAYER_ID"], ordered_reference["PLAYER_ID"]):
                    raise ValueError("Falsification candidates do not share players.")
                prior_error = group[f"prior_{component}"].to_numpy(dtype=float) - permuted
                rapm_error = group[f"target_{component}_one"].to_numpy(dtype=float) - permuted
                rows.append(
                    {
                        "control": "within_season_permuted_reference",
                        "candidate": candidate,
                        "rating_season": int(season),
                        "side": component,
                        "replicate": 0,
                        "correlation": float(np.corrcoef(prior_error, rapm_error)[0, 1]),
                    }
                )
            centered_target = reference_rows[f"target_{component}_future"].to_numpy(dtype=float).copy()
            centered_target -= centered_target.mean()
            for replicate in range(20):
                noise = rng.normal(size=len(reference_rows))
                rows.append(
                    {
                        "control": "gaussian_noise_feature",
                        "candidate": "noise",
                        "rating_season": int(season),
                        "side": component,
                        "replicate": replicate,
                        "correlation": float(np.corrcoef(noise, centered_target)[0, 1]),
                    }
                )
    return pd.DataFrame(rows)


def _falsification_summary(controls: pd.DataFrame) -> pd.DataFrame:
    permuted = controls.loc[controls["control"].eq("within_season_permuted_reference")]
    rows = []
    for side, frame in permuted.groupby("side", sort=True):
        wide = frame.pivot(index="rating_season", columns="candidate", values="correlation")
        delta = np.arctanh(np.clip(wide["rich"], -0.999999, 0.999999)) - np.arctanh(np.clip(wide["box15"], -0.999999, 0.999999))
        rows.append(
            {
                "control": "within_season_permuted_reference",
                "side": side,
                "seasons": len(wide),
                "box_fisher_correlation": float(np.tanh(np.arctanh(np.clip(wide["box15"], -0.999999, 0.999999)).mean())),
                "rich_fisher_correlation": float(np.tanh(np.arctanh(np.clip(wide["rich"], -0.999999, 0.999999)).mean())),
                "mean_fisher_rich_minus_box": float(delta.mean()),
            }
        )
    noise = controls.loc[controls["control"].eq("gaussian_noise_feature")]
    for side, frame in noise.groupby("side", sort=True):
        rows.append(
            {
                "control": "gaussian_noise_feature",
                "side": side,
                "seasons": int(frame["rating_season"].nunique()),
                "box_fisher_correlation": math.nan,
                "rich_fisher_correlation": math.nan,
                "mean_fisher_rich_minus_box": float(frame["correlation"].abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _prior_metrics(prior_targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, season), frame in prior_targets.groupby(["candidate_full", "rating_season"], sort=True):
        for side in (*SIDES, "net"):
            metrics = _weighted_metrics(
                frame[f"target_{side}"].to_numpy(dtype=float),
                frame[f"prior_{side}"].to_numpy(dtype=float),
                frame["sample_weight"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "candidate": candidate,
                    "rating_season": int(season),
                    "side": side,
                    "players": len(frame),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def _uniform_grid(
    contract: dict,
    priors: pd.DataFrame,
    matrices: dict[int, AnnualMatrix],
    factor_cache: dict,
) -> pd.DataFrame:
    design = contract["prior_design"]
    rows = []
    for (prior_design, candidate, rating_season), prior in priors.groupby(
        ["design", "candidate", "rating_season"], sort=True
    ):
        matrix = matrices[int(rating_season)]
        center, _ = coefficient_center(prior, matrix)
        for off_total in map(float, design["offense_total_penalty"]):
            for def_total in map(float, design["defense_total_penalty"]):
                base, off, deff = _uniform_basis(
                    matrix,
                    center,
                    off_total,
                    def_total,
                    float(design["home_penalty"]),
                    factor_cache,
                )
                for off_trust in map(float, design["prior_trust"]):
                    for def_trust in map(float, design["prior_trust"]):
                        predicted = base + off_trust * off + def_trust * deff
                        metrics = _metric_row(matrix.actual_margin, predicted)
                        rows.append(
                            {
                                "candidate": _candidate_name(prior_design, candidate),
                                "rating_season": int(rating_season),
                                "outcome_season": int(rating_season) + 1,
                                "offense_total_penalty": off_total,
                                "defense_total_penalty": def_total,
                                "offense_trust": off_trust,
                                "defense_trust": def_trust,
                                **metrics,
                            }
                        )
        print(f"uniform grid {prior_design}/{candidate}/{rating_season}: complete", flush=True)
    return pd.DataFrame(rows)


def _selected_uniform_parameters(contract: dict, grid: pd.DataFrame) -> pd.DataFrame:
    parameter_columns = [
        "offense_total_penalty",
        "defense_total_penalty",
        "offense_trust",
        "defense_trust",
    ]
    defaults = {
        "offense_total_penalty": float(contract["prior_design"]["default_offense_total_penalty"]),
        "defense_total_penalty": float(contract["prior_design"]["default_defense_total_penalty"]),
        "offense_trust": float(contract["prior_design"]["default_prior_trust"]),
        "defense_trust": float(contract["prior_design"]["default_prior_trust"]),
    }
    selection_outcomes = tuple(map(int, contract["seasons"]["design_selection_outcomes"]))
    rows = []
    for candidate, history in grid.groupby("candidate", sort=True):
        for rating_season in map(int, contract["seasons"]["rating"]):
            chosen = select_from_history(
                history,
                parameter_columns,
                defaults,
                current_outcome_season=rating_season + 1,
                selection_outcomes=selection_outcomes,
                minimum_folds=int(contract["prior_design"]["minimum_prior_folds"]),
            )
            rows.append(
                {
                    "candidate": candidate,
                    "rating_season": rating_season,
                    "outcome_season": rating_season + 1,
                    **chosen,
                }
            )
    return pd.DataFrame(rows)


def _variance_model_multipliers(
    history: pd.DataFrame,
    current: pd.DataFrame,
    *,
    side: str,
    contract: dict,
) -> np.ndarray:
    if history["rating_season"].nunique() < 2:
        return np.ones(len(current), dtype=float)
    predictors = pd.DataFrame(
        {
            "log_exposure": np.log1p(history["reference_exposure"].to_numpy(dtype=float)),
            "absolute_box_rich_disagreement": history[f"absolute_{side}_disagreement"].to_numpy(dtype=float),
            "unavailable_source_family_count": history["unavailable_source_family_count"].fillna(4).to_numpy(dtype=float),
        }
    )
    target = np.log(
        (history[f"prior_{side}"] - history[f"target_{side}"]).to_numpy(dtype=float) ** 2
        + 1e-4
    )
    model = _ridge_pipeline(float(contract["player_precision"]["alpha"]))
    model.fit(predictors, target, model__sample_weight=history["sample_weight"])
    current_predictors = pd.DataFrame(
        {
            "log_exposure": np.log1p(current["reference_exposure"].to_numpy(dtype=float)),
            "absolute_box_rich_disagreement": current[f"absolute_{side}_disagreement"].to_numpy(dtype=float),
            "unavailable_source_family_count": current["unavailable_source_family_count"].fillna(4).to_numpy(dtype=float),
        }
    )
    log_variance = model.predict(current_predictors)
    low_q, high_q = map(float, contract["player_precision"]["variance_clip_quantiles"])
    training_prediction = model.predict(predictors)
    low, high = np.quantile(training_prediction, [low_q, high_q])
    variance = np.exp(np.clip(log_variance, low, high))
    precision = 1.0 / variance
    precision /= np.median(precision)
    lower, upper = map(float, contract["player_precision"]["precision_multiplier_clip"])
    return np.clip(precision, lower, upper)


def _map_multiplier(
    current: pd.DataFrame,
    matrix: AnnualMatrix,
    values: np.ndarray,
) -> np.ndarray:
    mapped = pd.Series(values, index=current["PLAYER_ID"]).reindex(matrix.players)
    return mapped.fillna(1.0).to_numpy(dtype=float)


def _precision_mode_games(
    contract: dict,
    priors: pd.DataFrame,
    prior_targets: pd.DataFrame,
    matrices: dict[int, AnnualMatrix],
    selected_uniform: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = {
        "uniform": (False, False),
        "heterogeneous_offense": (True, False),
        "heterogeneous_defense": (False, True),
        "heterogeneous_both": (True, True),
    }
    game_rows = []
    multiplier_rows = []
    for (design, candidate, rating_season), prior in priors.groupby(
        ["design", "candidate", "rating_season"], sort=True
    ):
        candidate_full = _candidate_name(design, candidate)
        parameters = selected_uniform.loc[
            selected_uniform["candidate"].eq(candidate_full)
            & selected_uniform["rating_season"].eq(rating_season)
        ].iloc[0]
        matrix = matrices[int(rating_season)]
        center, present = coefficient_center(prior, matrix)
        current = prior_targets.loc[
            prior_targets["candidate_full"].eq(candidate_full)
            & prior_targets["rating_season"].eq(rating_season)
        ].copy()
        history = prior_targets.loc[
            prior_targets["candidate_full"].eq(candidate_full)
            & prior_targets["rating_season"].lt(rating_season)
        ].copy()
        side_multiplier = {
            side: _variance_model_multipliers(
                history,
                current,
                side=side,
                contract=contract,
            )
            for side in SIDES
        }
        mapped = {
            side: _map_multiplier(current, matrix, side_multiplier[side])
            for side in SIDES
        }
        for side in SIDES:
            multiplier_rows.extend(
                {
                    "candidate": candidate_full,
                    "rating_season": int(rating_season),
                    "PLAYER_ID": int(player),
                    "side": side,
                    "multiplier": float(value),
                    "has_prior": bool(has_prior),
                }
                for player, value, has_prior in zip(matrix.players, mapped[side], present)
            )
        for mode, (hetero_off, hetero_def) in modes.items():
            beta, intercept = _solve_heterogeneous(
                matrix,
                center,
                offense_total=float(parameters["offense_total_penalty"]),
                defense_total=float(parameters["defense_total_penalty"]),
                offense_trust=float(parameters["offense_trust"]),
                defense_trust=float(parameters["defense_trust"]),
                offense_multiplier=mapped["offense"] if hetero_off else np.ones(len(matrix.players)),
                defense_multiplier=mapped["defense"] if hetero_def else np.ones(len(matrix.players)),
                home_penalty=float(contract["prior_design"]["home_penalty"]),
            )
            predicted = _game_prediction(matrix, beta, intercept)
            game_rows.append(
                pd.DataFrame(
                    {
                        "game_id": matrix.game_ids,
                        "candidate": candidate_full,
                        "precision_mode": mode,
                        "rating_season": int(rating_season),
                        "outcome_season": int(rating_season) + 1,
                        "actual_margin": matrix.actual_margin,
                        "predicted_margin": predicted,
                        "squared_error": (matrix.actual_margin - predicted) ** 2,
                    }
                )
            )
    return pd.concat(game_rows, ignore_index=True), pd.DataFrame(multiplier_rows)


def _select_precision_modes(contract: dict, games: pd.DataFrame) -> pd.DataFrame:
    summary = (
        games.groupby(["candidate", "precision_mode", "outcome_season"], as_index=False)
        .agg(mse=("squared_error", "mean"))
    )
    selection_outcomes = tuple(map(int, contract["seasons"]["design_selection_outcomes"]))
    rows = []
    for candidate, history in summary.groupby("candidate", sort=True):
        for rating_season in map(int, contract["seasons"]["rating"]):
            chosen = select_from_history(
                history,
                ["precision_mode"],
                {"precision_mode": "uniform"},
                current_outcome_season=rating_season + 1,
                selection_outcomes=selection_outcomes,
                minimum_folds=int(contract["prior_design"]["minimum_prior_folds"]),
            )
            rows.append(
                {
                    "candidate": candidate,
                    "rating_season": rating_season,
                    "outcome_season": rating_season + 1,
                    "precision_mode": chosen["precision_mode"],
                }
            )
    return pd.DataFrame(rows)


def _selected_predictions(
    contract: dict,
    priors: pd.DataFrame,
    matrices: dict[int, AnnualMatrix],
    precision_games: pd.DataFrame,
    precision_selection: pd.DataFrame,
    selected_uniform: pd.DataFrame,
    prior_targets: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aio = precision_games.merge(
        precision_selection,
        on=["candidate", "rating_season", "outcome_season", "precision_mode"],
        how="inner",
        validate="many_to_one",
    )
    aio["stage"] = "aio"
    prior_games = []
    rating_rows = []
    for (design, candidate, rating_season), prior in priors.groupby(
        ["design", "candidate", "rating_season"], sort=True
    ):
        candidate_full = _candidate_name(design, candidate)
        matrix = matrices[int(rating_season)]
        center, _ = coefficient_center(prior, matrix)
        prediction = _standalone_prediction(matrix, center)
        prior_games.append(
            pd.DataFrame(
                {
                    "game_id": matrix.game_ids,
                    "candidate": candidate_full,
                    "precision_mode": "not_applicable",
                    "rating_season": int(rating_season),
                    "outcome_season": int(rating_season) + 1,
                    "actual_margin": matrix.actual_margin,
                    "predicted_margin": prediction,
                    "squared_error": (matrix.actual_margin - prediction) ** 2,
                    "stage": "prior",
                }
            )
        )
        rating_rows.append(_rating_frame(matrix, center, f"{candidate_full}__spm_prior"))
        selection = precision_selection.loc[
            precision_selection["candidate"].eq(candidate_full)
            & precision_selection["rating_season"].eq(rating_season)
        ].iloc[0]
        parameters = selected_uniform.loc[
            selected_uniform["candidate"].eq(candidate_full)
            & selected_uniform["rating_season"].eq(rating_season)
        ].iloc[0]
        current = prior_targets.loc[
            prior_targets["candidate_full"].eq(candidate_full)
            & prior_targets["rating_season"].eq(rating_season)
        ]
        history = prior_targets.loc[
            prior_targets["candidate_full"].eq(candidate_full)
            & prior_targets["rating_season"].lt(rating_season)
        ]
        mapped = {}
        for side in SIDES:
            values = _variance_model_multipliers(history, current, side=side, contract=contract)
            mapped[side] = _map_multiplier(current, matrix, values)
        mode = selection["precision_mode"]
        beta, _ = _solve_heterogeneous(
            matrix,
            center,
            offense_total=float(parameters["offense_total_penalty"]),
            defense_total=float(parameters["defense_total_penalty"]),
            offense_trust=float(parameters["offense_trust"]),
            defense_trust=float(parameters["defense_trust"]),
            offense_multiplier=mapped["offense"] if mode in {"heterogeneous_offense", "heterogeneous_both"} else np.ones(len(matrix.players)),
            defense_multiplier=mapped["defense"] if mode in {"heterogeneous_defense", "heterogeneous_both"} else np.ones(len(matrix.players)),
            home_penalty=float(contract["prior_design"]["home_penalty"]),
        )
        rating_rows.append(_rating_frame(matrix, beta, f"{candidate_full}__aio"))
    games = pd.concat([pd.concat(prior_games, ignore_index=True), aio], ignore_index=True)
    assert_identical_games(games.loc[games["stage"].eq("aio")])
    ratings = pd.concat(rating_rows, ignore_index=True)
    assert_component_identity(ratings)
    folds = []
    for (candidate, stage, outcome_season), frame in games.groupby(["candidate", "stage", "outcome_season"], sort=True):
        folds.append(
            {
                "candidate": candidate,
                "stage": stage,
                "rating_season": int(outcome_season) - 1,
                "outcome_season": int(outcome_season),
                "games": len(frame),
                **_metric_row(
                    frame["actual_margin"].to_numpy(dtype=float),
                    frame["predicted_margin"].to_numpy(dtype=float),
                ),
            }
        )
    return games, ratings, pd.DataFrame(folds)


def _mse_decomposition(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, frame in games.groupby("candidate", sort=True):
        prior = frame.loc[frame["stage"].eq("prior")].set_index(["outcome_season", "game_id"])
        aio = frame.loc[frame["stage"].eq("aio")].set_index(["outcome_season", "game_id"])
        if not prior.index.equals(aio.index):
            aio = aio.reindex(prior.index)
        for outcome_season, index in prior.groupby(level="outcome_season").groups.items():
            p = prior.loc[index]
            a = aio.loc[index]
            actual = p["actual_margin"].to_numpy(dtype=float)
            prior_prediction = p["predicted_margin"].to_numpy(dtype=float)
            update = a["predicted_margin"].to_numpy(dtype=float) - prior_prediction
            error = actual - prior_prediction
            prior_mse = float(np.mean(error**2))
            update_cost = float(np.mean(update**2))
            correction_benefit = float(2.0 * np.mean(error * update))
            aio_mse = float(np.mean((actual - a["predicted_margin"].to_numpy(dtype=float)) ** 2))
            rows.append(
                {
                    "candidate": candidate,
                    "outcome_season": int(outcome_season),
                    "prior_mse": prior_mse,
                    "update_cost": update_cost,
                    "correction_benefit": correction_benefit,
                    "reconstructed_aio_mse": prior_mse + update_cost - correction_benefit,
                    "aio_mse": aio_mse,
                    "identity_error": abs(aio_mse - (prior_mse + update_cost - correction_benefit)),
                    "aio_gain": prior_mse - aio_mse,
                    "prior_error_update_correlation": float(np.corrcoef(error, update)[0, 1]),
                }
            )
    output = pd.DataFrame(rows)
    if output["identity_error"].max() > 1e-9:
        raise ValueError("MSE decomposition failed.")
    return output


def _paired_bootstrap(
    games: pd.DataFrame,
    *,
    reference: str,
    stage: str,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    selected = games.loc[games["stage"].eq(stage)].copy()
    candidates = tuple(sorted(selected["candidate"].unique()))
    rng = np.random.default_rng(seed)
    rows = []
    reference_frame = selected.loc[selected["candidate"].eq(reference)]
    for candidate in candidates:
        if candidate == reference:
            continue
        challenger = selected.loc[selected["candidate"].eq(candidate)]
        season_deltas = []
        wins = 0
        for season in sorted(set(reference_frame["outcome_season"]) & set(challenger["outcome_season"])):
            left = reference_frame.loc[reference_frame["outcome_season"].eq(season)].set_index("game_id").sort_index()
            right = challenger.loc[challenger["outcome_season"].eq(season)].set_index("game_id").sort_index()
            if not left.index.equals(right.index):
                raise ValueError(f"{candidate} and {reference} do not score identical games.")
            delta = right["squared_error"].to_numpy(dtype=float) - left["squared_error"].to_numpy(dtype=float)
            season_deltas.append((int(season), delta))
            wins += int(delta.mean() < 0)
        samples = np.empty(draws)
        for draw in range(draws):
            samples[draw] = float(
                np.mean(
                    [
                        delta[rng.integers(0, len(delta), len(delta))].mean()
                        for _, delta in season_deltas
                    ]
                )
            )
        low, high = np.quantile(samples, [0.025, 0.975])
        point = float(np.mean([delta.mean() for _, delta in season_deltas]))
        reference_mse = float(
            np.mean(
                [
                    reference_frame.loc[reference_frame["outcome_season"].eq(season), "squared_error"].mean()
                    for season, _ in season_deltas
                ]
            )
        )
        challenger_mse = reference_mse + point
        rows.append(
            {
                "reference": reference,
                "candidate": candidate,
                "stage": stage,
                "folds": len(season_deltas),
                "candidate_minus_reference_mse": point,
                "candidate_minus_reference_rmse": math.sqrt(challenger_mse) - math.sqrt(reference_mse),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_candidate_better": float(np.mean(samples < 0)),
                "fold_wins": wins,
            }
        )
    return pd.DataFrame(rows)


def _leave_one_season_out_sensitivity(games: pd.DataFrame, reference: str) -> pd.DataFrame:
    selected = games.loc[games["stage"].eq("aio")]
    rows = []
    for candidate in sorted(selected["candidate"].unique()):
        if candidate == reference:
            continue
        for omitted in sorted(selected["outcome_season"].unique()):
            keep = selected["outcome_season"].ne(omitted)
            frame = selected.loc[keep & selected["candidate"].isin((reference, candidate))]
            fold = frame.groupby(["candidate", "outcome_season"], as_index=False).agg(mse=("squared_error", "mean"))
            means = fold.groupby("candidate")["mse"].mean()
            rows.append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "omitted_outcome_season": int(omitted),
                    "candidate_minus_reference_mse": float(means[candidate] - means[reference]),
                }
            )
    return pd.DataFrame(rows)


def _segmented_diagnostics(
    games: pd.DataFrame,
    prior_targets: pd.DataFrame,
    reference: str,
) -> pd.DataFrame:
    rows = []
    game_frame = games.loc[games["stage"].eq("aio")].copy()
    game_frame["segment"] = np.select(
        [
            game_frame["outcome_season"].le(2021),
            game_frame["outcome_season"].between(2022, 2023),
        ],
        ["design_selection", "legacy_later"],
        default="canonical_transition",
    )
    fold = game_frame.groupby(["candidate", "segment", "outcome_season"], as_index=False).agg(mse=("squared_error", "mean"))
    reference_values = fold.loc[fold["candidate"].eq(reference)].rename(columns={"mse": "reference_mse"})
    fold = fold.merge(reference_values[["segment", "outcome_season", "reference_mse"]], on=["segment", "outcome_season"], validate="many_to_one")
    for (candidate, segment), frame in fold.groupby(["candidate", "segment"], sort=True):
        rows.append(
            {
                "diagnostic": "game_source_era",
                "candidate": candidate,
                "side": "net",
                "segment": segment,
                "folds": int(frame["outcome_season"].nunique()),
                "metric": "candidate_minus_reference_mse",
                "value": float((frame["mse"] - frame["reference_mse"]).mean()),
            }
        )
    player = prior_targets.copy()
    player["segment"] = np.where(player["reference_exposure"].lt(1000), "under_1000", "at_least_1000")
    for (candidate, segment), frame in player.groupby(["candidate_full", "segment"], sort=True):
        for component in SIDES:
            error = frame[f"target_{component}"] - frame[f"prior_{component}"]
            rows.append(
                {
                    "diagnostic": "player_exposure",
                    "candidate": candidate,
                    "side": component,
                    "segment": segment,
                    "folds": int(frame["rating_season"].nunique()),
                    "metric": "weighted_target_rmse",
                    "value": float(np.sqrt(np.average(error**2, weights=frame["sample_weight"]))),
                }
            )
    return pd.DataFrame(rows)


def _fold_deltas(folds: pd.DataFrame, reference: str) -> pd.DataFrame:
    baseline = folds.loc[folds["candidate"].eq(reference)].rename(
        columns={
            "mse": "reference_mse",
            "rmse": "reference_rmse",
            "correlation": "reference_correlation",
        }
    )
    output = folds.merge(
        baseline[["stage", "outcome_season", "reference_mse", "reference_rmse", "reference_correlation"]],
        on=["stage", "outcome_season"],
        validate="many_to_one",
    )
    output["mse_delta"] = output["mse"] - output["reference_mse"]
    output["rmse_delta"] = output["rmse"] - output["reference_rmse"]
    output["correlation_delta"] = output["correlation"] - output["reference_correlation"]
    return output


def _dual_head_spm(priors: pd.DataFrame, decision: dict) -> pd.DataFrame:
    def split(candidate: str) -> tuple[str, str]:
        design, name = candidate.split("__", maxsplit=1)
        return design, name

    impact_design, impact_candidate = split(decision["selected_spm_impact"])
    prior_design, prior_candidate = split(decision["selected_spm_prior"])
    impact = priors.loc[
        priors["design"].eq(impact_design) & priors["candidate"].eq(impact_candidate)
    ][["PLAYER_ID", "rating_season", "prior_offense", "prior_defense", "prior_net"]].rename(
        columns={
            "prior_offense": "spm_impact_offense",
            "prior_defense": "spm_impact_defense",
            "prior_net": "spm_impact_net",
        }
    )
    complementary = priors.loc[
        priors["design"].eq(prior_design) & priors["candidate"].eq(prior_candidate)
    ][["PLAYER_ID", "rating_season", "prior_offense", "prior_defense", "prior_net"]].rename(
        columns={
            "prior_offense": "spm_prior_offense",
            "prior_defense": "spm_prior_defense",
            "prior_net": "spm_prior_net",
        }
    )
    output = impact.merge(complementary, on=["PLAYER_ID", "rating_season"], validate="one_to_one")
    if not np.allclose(output["spm_impact_offense"] + output["spm_impact_defense"], output["spm_impact_net"]):
        raise ValueError("SPM impact sides do not sum to net.")
    if not np.allclose(output["spm_prior_offense"] + output["spm_prior_defense"], output["spm_prior_net"]):
        raise ValueError("SPM prior sides do not sum to net.")
    return output


def _candidate_decision(contract: dict, games: pd.DataFrame, folds: pd.DataFrame, bootstrap: pd.DataFrame) -> dict:
    reference = "current_control__box15"
    selection_outcomes = set(map(int, contract["seasons"]["design_selection_outcomes"]))
    later_outcomes = set(map(int, contract["seasons"]["later_diagnostic_outcomes"]))
    early = folds.loc[folds["stage"].eq("aio") & folds["outcome_season"].isin(selection_outcomes)]
    early_summary = early.groupby("candidate", as_index=False).agg(mse=("mse", "mean"))
    challenger = early_summary.sort_values(["mse", "candidate"], kind="stable").iloc[0]["candidate"]
    pair = bootstrap.loc[bootstrap["candidate"].eq(challenger)].iloc[0] if challenger != reference else None
    late = folds.loc[folds["stage"].eq("aio") & folds["outcome_season"].isin(later_outcomes)]
    wide = late.pivot(index="outcome_season", columns="candidate", values=["mse", "correlation"])
    if challenger == reference:
        passes = False
        rmse_gain = 0.0
        correlation_loss = 0.0
        wins = 0
    else:
        reference_mse = float(wide["mse"][reference].mean())
        challenger_mse = float(wide["mse"][challenger].mean())
        rmse_gain = math.sqrt(reference_mse) - math.sqrt(challenger_mse)
        correlation_loss = float(wide["correlation"][reference].mean() - wide["correlation"][challenger].mean())
        wins = int((wide["mse"][challenger] < wide["mse"][reference]).sum())
        passes = bool(
            rmse_gain >= float(contract["evaluation"]["practical_rmse_gate"])
            and float(pair["bootstrap_95_high"]) < 0
            and correlation_loss <= float(contract["evaluation"]["maximum_correlation_loss"])
            and wins >= int(contract["evaluation"]["minimum_later_fold_wins"])
        )
    if passes and ("rich" in challenger or "blend" in challenger or "residual" in challenger):
        classification = "hybrid prior selected"
    elif passes and not challenger.startswith("current_control"):
        classification = "target-overlap correction selected"
    else:
        classification = "Box15 retained"
        challenger = reference
    rich_impact = folds.loc[
        folds["stage"].eq("prior") & folds["candidate"].str.contains("__rich$")
    ]
    impact_choice = (
        rich_impact.loc[rich_impact["outcome_season"].isin(selection_outcomes)]
        .groupby("candidate", as_index=False)
        .agg(mse=("mse", "mean"))
        .sort_values(["mse", "candidate"], kind="stable")
        .iloc[0]["candidate"]
    )
    return {
        "classification": classification,
        "reference": reference,
        "selected_spm_prior": challenger,
        "selected_spm_impact": impact_choice,
        "later_rmse_gain_points_per_game": rmse_gain,
        "later_mean_correlation_loss": correlation_loss,
        "later_fold_wins": wins,
        "passes_numeric_gate": passes,
    }


def _write_outputs(output: Path, frames: dict[str, pd.DataFrame], run: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in frames.items():
        _atomic_parquet(frame, output / name)
        run["files"][name] = {
            "path": name,
            "rows": len(frame),
            "sha256": sha256_file(output / name),
            "row_set_sha256": canonical_frame_hash(frame, frame.columns[: min(4, len(frame.columns))]),
        }
    write_json_atomic(run, output / "run.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "matrices", "references", "priors", "score"), default="all")
    args = parser.parse_args()
    started = time.perf_counter()
    contract = _load_contract()
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    matrix_quality = checkpoint_frame(CHECKPOINT_ROOT / "annual_matrix_quality.parquet", lambda: _build_annual_matrices(contract))
    if args.stage == "matrices":
        print(CHECKPOINT_ROOT)
        return
    references = checkpoint_frame(CHECKPOINT_ROOT / "references.parquet", lambda: _reference_panels(contract))
    if args.stage == "references":
        print(CHECKPOINT_ROOT)
        return
    box_features, rich_features, _, rich_arms, censored_arms = _load_features(contract)
    base_priors_path = CHECKPOINT_ROOT / "base_priors.parquet"
    base_selections_path = CHECKPOINT_ROOT / "base_prior_selections.parquet"
    if base_priors_path.exists() and base_selections_path.exists():
        base_priors = pd.read_parquet(base_priors_path)
        base_selections = pd.read_parquet(base_selections_path)
    else:
        base_priors, base_selections = _base_priors(
            contract,
            references,
            box_features,
            rich_features,
            rich_arms,
            censored_arms,
        )
        _atomic_parquet(base_priors, base_priors_path)
        _atomic_parquet(base_selections, base_selections_path)
    matrices = {season: _load_matrix(season) for season in map(int, contract["seasons"]["rating"])}
    factor_cache: dict = {}
    direct_path = CHECKPOINT_ROOT / "direct_blend_priors.parquet"
    direct_grid_path = CHECKPOINT_ROOT / "direct_blend_grid.parquet"
    if direct_path.exists() and direct_grid_path.exists():
        direct, direct_grid = pd.read_parquet(direct_path), pd.read_parquet(direct_grid_path)
    else:
        direct, direct_grid = _direct_blends(contract, base_priors, matrices, factor_cache)
        _atomic_parquet(direct, direct_path)
        _atomic_parquet(direct_grid, direct_grid_path)
    residual_path = CHECKPOINT_ROOT / "defense_residual_priors.parquet"
    residual_grid_path = CHECKPOINT_ROOT / "defense_residual_grid.parquet"
    if residual_path.exists() and residual_grid_path.exists():
        residual, residual_grid = pd.read_parquet(residual_path), pd.read_parquet(residual_grid_path)
    else:
        residual, residual_grid = pd.DataFrame(), pd.DataFrame()
    expected_residual_keys = {
        (design, pool, season)
        for design in ("target_excluded", "fully_lagged")
        for pool in ("def_residual_activity", "def_residual_outcome")
        for season in map(int, contract["seasons"]["rating"])
    }
    observed_residual_keys = set(
        residual[["design", "candidate", "rating_season"]].drop_duplicates().itertuples(index=False, name=None)
    ) if not residual.empty else set()
    if observed_residual_keys != expected_residual_keys:
        residual, residual_grid = _defense_residuals(
            contract,
            references,
            rich_features,
            base_priors,
            matrices,
            factor_cache,
        )
        _atomic_parquet(residual, residual_path)
        _atomic_parquet(residual_grid, residual_grid_path)
    priors, coverage = _common_prior_coverage(
        pd.concat([base_priors, direct, residual], ignore_index=True, sort=False),
        contract,
    )
    _atomic_parquet(priors, CHECKPOINT_ROOT / "matched_priors.parquet")
    _atomic_parquet(coverage, CHECKPOINT_ROOT / "prior_coverage.parquet")
    if args.stage == "priors":
        print(CHECKPOINT_ROOT)
        return

    prior_targets = _prior_target_rows(priors, references, rich_features)
    _atomic_parquet(prior_targets, CHECKPOINT_ROOT / "prior_target_rows.parquet")
    prior_metrics = _prior_metrics(prior_targets)
    shared_folds, shared_summary, shared_bootstrap = _shared_error_diagnostics(contract, prior_targets, references)
    falsification = _falsification_controls(contract, prior_targets, references)
    falsification_summary = _falsification_summary(falsification)
    uniform_path = CHECKPOINT_ROOT / "uniform_penalty_grid.parquet"
    uniform_grid = pd.read_parquet(uniform_path) if uniform_path.exists() else pd.DataFrame()
    expected_candidates = {
        _candidate_name(design, candidate)
        for design, candidate in priors[["design", "candidate"]].drop_duplicates().itertuples(index=False, name=None)
    }
    observed_uniform = set(uniform_grid.get("candidate", pd.Series(dtype=str)).unique())
    if expected_candidates.issubset(observed_uniform):
        uniform_grid = uniform_grid.loc[uniform_grid["candidate"].isin(expected_candidates)].copy()
        _atomic_parquet(uniform_grid, uniform_path)
    else:
        uniform_grid = _uniform_grid(contract, priors, matrices, factor_cache)
        _atomic_parquet(uniform_grid, uniform_path)
    selected_uniform = _selected_uniform_parameters(contract, uniform_grid)
    precision_games_path = CHECKPOINT_ROOT / "precision_mode_games.parquet"
    multipliers_path = CHECKPOINT_ROOT / "precision_multipliers.parquet"
    if precision_games_path.exists() and multipliers_path.exists():
        precision_games = pd.read_parquet(precision_games_path)
        multipliers = pd.read_parquet(multipliers_path)
    else:
        precision_games, multipliers = pd.DataFrame(), pd.DataFrame()
    observed_precision = set(precision_games.get("candidate", pd.Series(dtype=str)).unique())
    if expected_candidates.issubset(observed_precision):
        precision_games = precision_games.loc[precision_games["candidate"].isin(expected_candidates)].copy()
        multipliers = multipliers.loc[multipliers["candidate"].isin(expected_candidates)].copy()
        _atomic_parquet(precision_games, precision_games_path)
        _atomic_parquet(multipliers, multipliers_path)
    else:
        precision_games, multipliers = _precision_mode_games(
            contract,
            priors,
            prior_targets,
            matrices,
            selected_uniform,
        )
        _atomic_parquet(precision_games, precision_games_path)
        _atomic_parquet(multipliers, multipliers_path)
    precision_selection = _select_precision_modes(contract, precision_games)
    games, ratings, folds = _selected_predictions(
        contract,
        priors,
        matrices,
        precision_games,
        precision_selection,
        selected_uniform,
        prior_targets,
    )
    fold_deltas = _fold_deltas(folds, "current_control__box15")
    decomposition = _mse_decomposition(games)
    reference = "current_control__box15"
    bootstrap = _paired_bootstrap(
        games,
        reference=reference,
        stage="aio",
        draws=int(contract["evaluation"]["bootstrap_draws"]),
        seed=int(contract["evaluation"]["bootstrap_seed"]),
    )
    sensitivity = _leave_one_season_out_sensitivity(games, reference)
    segmented = _segmented_diagnostics(games, prior_targets, reference)
    decision = _candidate_decision(contract, games, folds, bootstrap)
    dual_head = _dual_head_spm(priors, decision)

    source_paths = {
        "contract": CONTRACT,
        "runner": Path(__file__),
        "annual_targets": ANNUAL_TARGETS,
        "target_windows": TARGET_WINDOWS,
        "target_window_manifest": TARGET_WINDOW_RUN / "run.json",
        "box_annual": BOX_ANNUAL,
        "rich_annual": RICH_ANNUAL,
        "rich_manifest": RICH_RUN / "run.json",
        "mechanism_annual": MECHANISM_ANNUAL,
        "atlas": ATLAS,
        "silver_possessions": SILVER_POSSESSIONS,
        "silver_lineups": SILVER_LINEUPS,
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    identity = hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"{EXPERIMENT_ID}_{identity}"
    if (output / "run.json").exists():
        print(output)
        return
    frames = {
        "matrix_quality.parquet": matrix_quality,
        "references.parquet": references,
        "base_prior_selections.parquet": base_selections,
        "priors.parquet": priors,
        "prior_coverage.parquet": coverage,
        "prior_target_metrics.parquet": prior_metrics,
        "shared_error_folds.parquet": shared_folds,
        "shared_error_summary.parquet": shared_summary,
        "shared_error_bootstrap.parquet": shared_bootstrap,
        "falsification_controls.parquet": falsification,
        "falsification_summary.parquet": falsification_summary,
        "direct_blend_grid.parquet": direct_grid,
        "defense_residual_grid.parquet": residual_grid,
        "uniform_penalty_grid.parquet": uniform_grid,
        "uniform_parameter_selections.parquet": selected_uniform,
        "precision_mode_selections.parquet": precision_selection,
        "precision_multipliers.parquet": multipliers,
        "game_predictions.parquet": games,
        "ratings.parquet": ratings,
        "fold_metrics.parquet": folds,
        "fold_deltas.parquet": fold_deltas,
        "mse_decomposition.parquet": decomposition,
        "paired_bootstrap.parquet": bootstrap,
        "leave_one_season_out_sensitivity.parquet": sensitivity,
        "segmented_diagnostics.parquet": segmented,
        "dual_head_spm.parquet": dual_head,
    }
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "contract": contract,
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": source_hashes[name]}
            for name, path in source_paths.items()
        },
        "quality": {
            "rating_seasons": sorted(map(int, games["rating_season"].unique())),
            "outcome_seasons": sorted(map(int, games["outcome_season"].unique())),
            "identical_games": True,
            "maximum_component_identity_error": float((ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()),
            "maximum_mse_decomposition_error": float(decomposition["identity_error"].max()),
            "checkpoint_root": str(CHECKPOINT_ROOT.relative_to(ROOT)),
        },
        "decision": decision,
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    _write_outputs(output, frames, run)
    print(output)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
