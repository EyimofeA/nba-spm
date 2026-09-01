#!/usr/bin/env python3
"""Select stable rich SPM features and test complementarity with RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    build_design,
    fit_coefficients,
    fit_coefficients_with_center,
    load_unified_terminal_possessions,
)

try:
    from run_aio_prior_complementarity import (
        ATLAS,
        POSSESSION_CACHE,
        RICH_ANNUAL,
        SILVER_LINEUPS,
        SILVER_POSSESSIONS,
        _control_games,
        _learning_panel,
        _load_contract as load_source_contract,
        _load_features,
        _load_matrix,
        _metric_row,
        _prune_features,
        _reference_panels,
        coefficient_center,
    )
    from run_annual_spm_learner_screen import (
        ModelSpec,
        _fit as fit_rich,
        _weighted_metrics,
    )
except ModuleNotFoundError:
    from research.run_aio_prior_complementarity import (
        ATLAS,
        POSSESSION_CACHE,
        RICH_ANNUAL,
        SILVER_LINEUPS,
        SILVER_POSSESSIONS,
        _control_games,
        _learning_panel,
        _load_contract as load_source_contract,
        _load_features,
        _load_matrix,
        _metric_row,
        _prune_features,
        _reference_panels,
        coefficient_center,
    )
    from research.run_annual_spm_learner_screen import (
        ModelSpec,
        _fit as fit_rich,
        _weighted_metrics,
    )


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "spm_consensus_complementarity_v1"
CONTRACT = ROOT / "research/experiments/spm_consensus_complementarity_v1.yml"
OUTPUT_ROOT = ROOT / "artifacts/research/spm_consensus_complementarity"
CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints_v4"
SOURCE_RUN = ROOT / (
    "artifacts/research/aio_prior_complementarity/"
    "aio_prior_complementarity_v1_4d83e381af"
)
SIDES = ("offense", "defense")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _load_contract(path: Path = CONTRACT) -> dict:
    contract = yaml.safe_load(path.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment ID changed.")
    if tuple(contract["seasons"]["rating"]) != tuple(range(2016, 2026)):
        raise ValueError("Rating folds changed.")
    if int(contract["selector"]["gaussian_noise_features"]) != 20:
        raise ValueError("The frozen selector requires 20 noise features.")
    if int(contract["diagnostics"]["bootstrap_draws"]) != 5000:
        raise ValueError("The diagnostic bootstrap requires 5,000 draws.")
    if float(contract["diagnostics"]["split_penalty_scale"]) != 0.5:
        raise ValueError("Disjoint half-sample fits require half-scaled penalties.")
    return contract


def stable_game_partition(game_ids: Iterable[str]) -> np.ndarray:
    """Assign every game to one deterministic disjoint half."""
    return np.asarray(
        [int(hashlib.sha256(str(game_id).encode()).hexdigest()[-1], 16) % 2 for game_id in game_ids],
        dtype=np.int8,
    )


def recurrence_threshold(
    minimum: float,
    null_frequencies: np.ndarray,
    quantile: float,
) -> float:
    finite = np.asarray(null_frequencies, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(max(minimum, np.quantile(finite, quantile) if len(finite) else minimum))


def _noise_columns(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    ordered = frame[["PLAYER_ID", "Season"]].astype("int64")
    row_seed = pd.util.hash_pandas_object(ordered, index=False).to_numpy(dtype=np.uint64)
    output = {}
    for index in range(count):
        rng = np.random.default_rng(seed + index)
        base = rng.standard_normal(len(frame))
        order = np.argsort(row_seed, kind="stable")
        values = np.empty(len(frame), dtype=float)
        values[order] = base
        output[f"__noise_{index:02d}"] = values
    return pd.DataFrame(output, index=frame.index)


def _player_bootstrap(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    players = frame["PLAYER_ID"].drop_duplicates().to_numpy()
    sampled = rng.choice(players, size=len(players), replace=True)
    counts = pd.Series(sampled).value_counts(sort=False)
    parts = []
    for player_id, count in counts.items():
        rows = frame.loc[frame["PLAYER_ID"].eq(player_id)]
        parts.extend([rows] * int(count))
    return pd.concat(parts, ignore_index=True)


def _permuted_within_season(
    frame: pd.DataFrame,
    target: str,
    rng: np.random.Generator,
) -> np.ndarray:
    output = frame[target].to_numpy(dtype=float).copy()
    for indices in frame.groupby("Season", sort=False).indices.values():
        output[indices] = rng.permutation(output[indices])
    return output


def _selector_model(contract: dict) -> Pipeline:
    selector = contract["selector"]
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    alpha=float(selector["alpha"]),
                    l1_ratio=float(selector["l1_ratio"]),
                    max_iter=20_000,
                    tol=1e-5,
                    random_state=int(selector["seed"]),
                ),
            ),
        ]
    )


def _selected_support(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    target: str,
    contract: dict,
) -> tuple[pd.DataFrame, tuple[str, ...], dict]:
    selector = contract["selector"]
    noise_count = int(selector["gaussian_noise_features"])
    noise = _noise_columns(frame, noise_count, int(selector["seed"]))
    augmented = pd.concat([frame.reset_index(drop=True), noise.reset_index(drop=True)], axis=1)
    all_features = (*features, *noise.columns)
    real_counts = pd.Series(0, index=all_features, dtype=float)
    null_counts = pd.Series(0, index=all_features, dtype=float)
    resamples = int(selector["resamples"])
    for draw in range(resamples):
        rng = np.random.default_rng(int(selector["seed"]) + draw)
        sample = _player_bootstrap(augmented, rng)
        model = _selector_model(contract)
        model.fit(
            sample.loc[:, all_features],
            sample[target],
            model__sample_weight=sample["sample_weight"],
        )
        real_counts += np.abs(model.named_steps["model"].coef_) > 1e-10

        null_sample = sample.copy()
        null_sample[target] = _permuted_within_season(null_sample, target, rng)
        null_model = _selector_model(contract)
        null_model.fit(
            null_sample.loc[:, all_features],
            null_sample[target],
            model__sample_weight=null_sample["sample_weight"],
        )
        null_counts += np.abs(null_model.named_steps["model"].coef_) > 1e-10

    real_frequency = real_counts / resamples
    null_frequency = null_counts / resamples
    calibration = np.concatenate(
        [
            null_frequency.loc[list(features)].to_numpy(dtype=float),
            real_frequency.loc[list(noise.columns)].to_numpy(dtype=float),
        ]
    )
    threshold = recurrence_threshold(
        float(selector["minimum_recurrence"]),
        calibration,
        float(selector["null_quantile"]),
    )
    mandatory = tuple(feature for feature in BOX_PIPM_STYLE_FEATURES if feature in features)
    selected = tuple(
        feature
        for feature in features
        if feature in mandatory or float(real_frequency[feature]) >= threshold
    )
    rows = pd.DataFrame(
        {
            "feature": all_features,
            "real_selection_frequency": real_frequency.to_numpy(),
            "null_selection_frequency": null_frequency.to_numpy(),
            "is_noise": [feature.startswith("__noise_") for feature in all_features],
            "mandatory": [feature in mandatory for feature in all_features],
            "selected": [feature in selected for feature in all_features],
            "threshold": threshold,
        }
    )
    quality = {
        "threshold": threshold,
        "selected": len(selected),
        "candidate_features": len(features),
        "maximum_noise_frequency": float(real_frequency.loc[list(noise.columns)].max()),
        "maximum_permuted_target_frequency": float(null_frequency.loc[list(features)].max()),
    }
    return rows, selected, quality


def _fit_consensus_priors(
    contract: dict,
    references: pd.DataFrame,
    rich: pd.DataFrame,
    rich_arms: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = _learning_panel(rich, references, "target_excluded")
    prior_rows = []
    frequency_rows = []
    quality_rows = []
    for rating_season in map(int, contract["seasons"]["rating"]):
        train = panel.loc[panel["Season"].lt(rating_season)].copy()
        test = panel.loc[panel["Season"].eq(rating_season)].copy()
        if train.empty or test.empty:
            continue
        prior = test[["PLAYER_ID"]].copy()
        for side in SIDES:
            candidates = _prune_features(
                train,
                rich_arms[side],
                threshold=float(contract["selector"]["correlation_prune_threshold"]),
            )
            frequencies, selected, quality = _selected_support(
                train.reset_index(drop=True),
                candidates,
                f"target_{side}",
                contract,
            )
            if not selected:
                raise ValueError(f"Consensus selector retained no {side} features for {rating_season}.")
            spec = (
                ModelSpec(
                    "elastic_net",
                    {
                        "alpha": float(contract["final_spm"]["offense_alpha"]),
                        "l1_ratio": float(contract["final_spm"]["offense_l1_ratio"]),
                    },
                )
                if side == "offense"
                else ModelSpec("ridge", {"alpha": float(contract["final_spm"]["defense_alpha"])})
            )
            model = fit_rich(spec, train, selected, f"target_{side}")
            prior[f"prior_{side}"] = model.predict(test.loc[:, selected])
            frequencies.insert(0, "side", side)
            frequencies.insert(0, "rating_season", rating_season)
            frequency_rows.append(frequencies)
            quality_rows.append(
                {
                    "rating_season": rating_season,
                    "side": side,
                    "training_start": int(train["Season"].min()),
                    "training_end": int(train["Season"].max()),
                    "training_rows": len(train),
                    **quality,
                    "selected_feature_names": json.dumps(selected),
                }
            )
        prior["prior_net"] = prior["prior_offense"] + prior["prior_defense"]
        prior["rating_season"] = rating_season
        prior["design"] = "target_excluded"
        prior["candidate"] = "box15_consensus"
        prior_rows.append(prior)
        print(f"consensus prior {rating_season}: complete", flush=True)
    return (
        pd.concat(prior_rows, ignore_index=True),
        pd.concat(frequency_rows, ignore_index=True),
        pd.DataFrame(quality_rows),
    )


def _matched_priors(consensus: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_parquet(SOURCE_RUN / "priors.parquet")
    source = source.loc[
        source["candidate"].isin(("box15", "rich"))
        & source["design"].isin(("current_control", "target_excluded", "fully_lagged"))
    ].copy()
    all_rows = pd.concat([source, consensus], ignore_index=True, sort=False)
    required = {
        ("current_control", "box15"),
        ("current_control", "rich"),
        ("target_excluded", "box15"),
        ("target_excluded", "rich"),
        ("target_excluded", "box15_consensus"),
        ("fully_lagged", "box15"),
        ("fully_lagged", "rich"),
    }
    rows = []
    for season in sorted(consensus["rating_season"].unique()):
        season_rows = all_rows.loc[all_rows["rating_season"].eq(season)]
        groups = {
            key: frame
            for key, frame in season_rows.groupby(["design", "candidate"], sort=False)
            if key in required
        }
        if set(groups) != required:
            raise ValueError(f"Prior candidates are incomplete for {season}.")
        common = set.intersection(*(set(frame["PLAYER_ID"]) for frame in groups.values()))
        for frame in groups.values():
            rows.append(frame.loc[frame["PLAYER_ID"].isin(common)])
    output = pd.concat(rows, ignore_index=True)
    if output.duplicated(["design", "candidate", "rating_season", "PLAYER_ID"]).any():
        raise ValueError("Matched prior keys are not unique.")
    return output


def _next_season_games(contract: dict, priors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_contract = load_source_contract()
    factor_cache = {}
    game_rows = []
    rating_rows = []
    candidates = (
        ("current_control", "box15"),
        ("target_excluded", "box15"),
        ("target_excluded", "rich"),
        ("target_excluded", "box15_consensus"),
    )
    for rating_season in map(int, contract["seasons"]["rating"]):
        matrix = _load_matrix(rating_season)
        for design, candidate in candidates:
            prior = priors.loc[
                priors["rating_season"].eq(rating_season)
                & priors["design"].eq(design)
                & priors["candidate"].eq(candidate)
            ]
            label = f"{design}__{candidate}"
            game_rows.append(_control_games(matrix, prior, source_contract, factor_cache, candidate=label))
            center, _ = coefficient_center(prior, matrix)
            n = len(matrix.players)
            rating_rows.append(
                pd.DataFrame(
                    {
                        "PLAYER_ID": matrix.players,
                        "rating_season": rating_season,
                        "candidate": label,
                        "prior_offense": 100 * center[:n],
                        "prior_defense": -100 * center[n : 2 * n],
                    }
                ).assign(prior_net=lambda frame: frame["prior_offense"] + frame["prior_defense"])
            )
    return pd.concat(game_rows, ignore_index=True), pd.concat(rating_rows, ignore_index=True)


def _fold_metrics(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, outcome_season), frame in games.groupby(["candidate", "outcome_season"], sort=True):
        rows.append(
            {
                "candidate": candidate,
                "outcome_season": outcome_season,
                "games": len(frame),
                **_metric_row(frame["actual_margin"].to_numpy(), frame["predicted_margin"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def _prior_target_metrics(priors: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    past = references.loc[references["reference"].eq("nine_year_past")]
    candidates = priors.loc[
        priors["design"].eq("target_excluded")
        & priors["candidate"].isin(("box15", "rich", "box15_consensus"))
    ]
    matched = candidates.merge(
        past,
        on=["PLAYER_ID", "rating_season"],
        how="inner",
        validate="many_to_one",
    )
    matched["sample_weight"] = np.sqrt(
        np.minimum(matched["Poss_Off"], matched["Poss_Def"]).clip(lower=1)
    )
    rows = []
    for (candidate, season), frame in matched.groupby(["candidate", "rating_season"], sort=True):
        for side in (*SIDES, "net"):
            rows.append(
                {
                    "candidate": candidate,
                    "rating_season": season,
                    "side": side,
                    "players": len(frame),
                    **_weighted_metrics(
                        frame[f"target_{side}"].to_numpy(dtype=float),
                        frame[f"prior_{side}"].to_numpy(dtype=float),
                        frame["sample_weight"].to_numpy(dtype=float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_summary(games: pd.DataFrame, reference: str, contract: dict) -> pd.DataFrame:
    reference_rows = games.loc[games["candidate"].eq(reference)]
    rows = []
    for candidate in sorted(set(games["candidate"]) - {reference}):
        candidate_rows = games.loc[games["candidate"].eq(candidate)]
        joined = reference_rows.merge(
            candidate_rows,
            on=["game_id", "rating_season", "outcome_season"],
            suffixes=("_reference", "_candidate"),
            validate="one_to_one",
        )
        if len(joined) != len(reference_rows) or len(joined) != len(candidate_rows):
            raise ValueError(f"Game mismatch for {candidate}.")
        deltas = []
        rng = np.random.default_rng(int(contract["selector"]["seed"]))
        season_groups = [frame for _, frame in joined.groupby("outcome_season", sort=True)]
        for _ in range(int(contract["diagnostics"]["bootstrap_draws"])):
            season_values = []
            for frame in season_groups:
                indices = rng.integers(0, len(frame), len(frame))
                season_values.append(
                    float(
                        np.mean(
                            frame["mse_candidate"].to_numpy()[indices]
                            - frame["mse_reference"].to_numpy()[indices]
                        )
                    )
                )
            deltas.append(float(np.mean(season_values)))
        by_season = joined.groupby("outcome_season", sort=True).agg(
            reference_mse=("mse_reference", "mean"),
            candidate_mse=("mse_candidate", "mean"),
        )
        reference_mse = float(by_season["reference_mse"].mean())
        candidate_mse = float(by_season["candidate_mse"].mean())
        delta = candidate_mse - reference_mse
        rows.append(
            {
                "reference": reference,
                "candidate": candidate,
                "folds": len(by_season),
                "games": len(joined),
                "reference_mse": reference_mse,
                "candidate_mse": candidate_mse,
                "candidate_minus_reference_mse": delta,
                "reference_rmse": math.sqrt(reference_mse),
                "candidate_rmse": math.sqrt(candidate_mse),
                "reference_minus_candidate_rmse": math.sqrt(reference_mse) - math.sqrt(candidate_mse),
                "wins": int((by_season["candidate_mse"] < by_season["reference_mse"]).sum()),
                "lower_95": float(np.quantile(deltas, 0.025)),
                "upper_95": float(np.quantile(deltas, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def _future_split_references(contract: dict) -> pd.DataFrame:
    cache = CHECKPOINT_ROOT / "future_split_references.parquet"
    ledger_path = CHECKPOINT_ROOT / "future_split_game_ledger.parquet"
    if cache.exists() and ledger_path.exists():
        return pd.read_parquet(cache)
    frame = load_unified_terminal_possessions(
        POSSESSION_CACHE,
        SILVER_POSSESSIONS,
        SILVER_LINEUPS,
        tuple(range(2017, 2027)),
        transition_season=2024,
        game_types=("regular",),
    )
    design = build_design(frame, include_home=True)
    game_ledger = frame[["gameid", "season"]].drop_duplicates().rename(
        columns={"gameid": "game_id"}
    )
    game_ledger["split"] = stable_game_partition(game_ledger["game_id"])
    if game_ledger.duplicated(["game_id", "season"]).any():
        raise ValueError("Future-reference game keys are not unique.")
    _atomic_parquet(game_ledger, ledger_path)
    season_mean = frame.groupby("season")["pts"].mean()
    design = replace(
        design,
        y=frame["pts"].to_numpy(dtype=float) - frame["season"].map(season_mean).to_numpy(dtype=float),
    )
    partition = stable_game_partition(design.game_ids)
    penalty_scale = float(contract["diagnostics"]["split_penalty_scale"])
    config = RapmConfig(
        seasons=tuple(range(2017, 2027)),
        lambda_off=penalty_scale * float(contract["rapm"]["target"]["lambda_off"]),
        lambda_def=penalty_scale * float(contract["rapm"]["target"]["lambda_def"]),
        lambda_home=penalty_scale * float(contract["rapm"]["target"]["lambda_home"]),
        data_scope="spm_consensus_disjoint_future_reference",
    )
    rows = []
    for rating_season in range(2016, int(contract["seasons"]["disjoint_reference_rating_end"]) + 1):
        seasons = (rating_season + 1, rating_season + 2, rating_season + 3)
        for split in (0, 1):
            mask = np.isin(design.seasons, seasons) & (partition == split)
            beta, _ = fit_coefficients(design, config, row_mask=mask)
            n = len(design.players)
            X = design.X[mask]
            off = np.asarray(X[:, :n].sum(axis=0)).ravel()
            deff = np.asarray(X[:, n : 2 * n].sum(axis=0)).ravel()
            active = np.minimum(off, deff) > 0
            rows.append(
                pd.DataFrame(
                    {
                        "PLAYER_ID": design.players[active],
                        "rating_season": rating_season,
                        "split": split,
                        "target_offense": 100 * beta[:n][active],
                        "target_defense": -100 * beta[n : 2 * n][active],
                        "Poss_Off": off[active],
                        "Poss_Def": deff[active],
                    }
                )
            )
        print(f"disjoint future reference {rating_season}: complete", flush=True)
    output = pd.concat(rows, ignore_index=True)
    _atomic_parquet(output, cache)
    return output


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=float)
    weight = weight / weight.sum()
    left_centered = left - np.sum(weight * left)
    right_centered = right - np.sum(weight * right)
    covariance = np.sum(weight * left_centered * right_centered)
    denominator = math.sqrt(np.sum(weight * left_centered**2) * np.sum(weight * right_centered**2))
    return float(covariance / denominator) if denominator > 0 else math.nan


def cross_reference_correlations(
    priors: pd.DataFrame,
    annual: pd.DataFrame,
    split_references: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    current = priors.loc[
        priors["design"].eq("current_control") & priors["candidate"].isin(("box15", "rich"))
    ]
    for season in sorted(split_references["rating_season"].unique()):
        one = annual.loc[
            annual["rating_season"].eq(season) & annual["reference"].eq("one_year")
        ]
        left = split_references.loc[
            split_references["rating_season"].eq(season) & split_references["split"].eq(0)
        ].rename(
            columns={
                column: f"{column}_a"
                for column in ("target_offense", "target_defense", "Poss_Off", "Poss_Def")
            }
        )
        right = split_references.loc[
            split_references["rating_season"].eq(season) & split_references["split"].eq(1)
        ].rename(
            columns={
                column: f"{column}_b"
                for column in ("target_offense", "target_defense", "Poss_Off", "Poss_Def")
            }
        )
        one = one.rename(
            columns={
                column: f"{column}_one"
                for column in ("target_offense", "target_defense", "Poss_Off", "Poss_Def")
            }
        )
        base = one.merge(left, on=["PLAYER_ID", "rating_season"], validate="one_to_one")
        base = base.merge(right, on=["PLAYER_ID", "rating_season"], validate="one_to_one")
        for candidate in ("box15", "rich"):
            prior = current.loc[current["rating_season"].eq(season) & current["candidate"].eq(candidate)]
            matched = prior.merge(base, on=["PLAYER_ID", "rating_season"], validate="one_to_one")
            weight = np.sqrt(
                np.minimum.reduce(
                    [
                        matched["Poss_Off_a"].to_numpy(),
                        matched["Poss_Def_a"].to_numpy(),
                        matched["Poss_Off_b"].to_numpy(),
                        matched["Poss_Def_b"].to_numpy(),
                    ]
                ).clip(min=1)
            )
            for side in SIDES:
                correlations = []
                for prior_reference, rapm_reference in (("a", "b"), ("b", "a")):
                    prior_error = matched[f"prior_{side}"] - matched[f"target_{side}_{prior_reference}"]
                    rapm_error = matched[f"target_{side}_one"] - matched[f"target_{side}_{rapm_reference}"]
                    correlations.append(_weighted_correlation(prior_error.to_numpy(), rapm_error.to_numpy(), weight))
                rows.append(
                    {
                        "rating_season": season,
                        "candidate": candidate,
                        "side": side,
                        "players": len(matched),
                        "cross_reference_correlation": float(np.mean(correlations)),
                        "correlation_a_b": correlations[0],
                        "correlation_b_a": correlations[1],
                    }
                )
    return pd.DataFrame(rows)


def _cross_reference_summary(frame: pd.DataFrame, contract: dict) -> pd.DataFrame:
    pivot = frame.pivot(index=["rating_season", "side"], columns="candidate", values="cross_reference_correlation").reset_index()
    pivot["rich_minus_box15"] = pivot["rich"] - pivot["box15"]
    rows = []
    rng = np.random.default_rng(int(contract["selector"]["seed"]))
    for side, side_rows in pivot.groupby("side", sort=True):
        values = side_rows["rich_minus_box15"].dropna().to_numpy(dtype=float)
        draws = np.asarray(
            [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(int(contract["diagnostics"]["bootstrap_draws"]))]
        )
        rows.append(
            {
                "side": side,
                "seasons": len(values),
                "mean_rich_minus_box15_correlation": float(values.mean()),
                "mean_fisher_z_delta": float(
                    np.mean(
                        np.arctanh(np.clip(side_rows["rich"].to_numpy(dtype=float), -0.999999, 0.999999))
                        - np.arctanh(np.clip(side_rows["box15"].to_numpy(dtype=float), -0.999999, 0.999999))
                    )
                ),
                "lower_80": float(np.quantile(draws, 0.10)),
                "upper_80": float(np.quantile(draws, 0.90)),
                "lower_95": float(np.quantile(draws, 0.025)),
                "upper_95": float(np.quantile(draws, 0.975)),
                "positive_seasons": int((values > 0).sum()),
                "materially_greater_shared_error": bool(
                    values.mean() >= float(contract["diagnostics"]["material_shared_error_delta"])
                    and np.quantile(draws, 0.10) > 0
                    and (values > 0).sum() >= math.ceil(2 * len(values) / 3)
                ),
            }
        )
    return pd.DataFrame(rows)


def _center_for_design(prior: pd.DataFrame, design, train_mask: np.ndarray) -> np.ndarray:
    indexed = prior.set_index("PLAYER_ID")
    offense = indexed["prior_offense"].reindex(design.players).fillna(0).to_numpy(dtype=float) / 100
    defense = -indexed["prior_defense"].reindex(design.players).fillna(0).to_numpy(dtype=float) / 100
    n = len(design.players)
    X = design.X[train_mask]
    off_weight = np.asarray(X[:, :n].sum(axis=0)).ravel()
    def_weight = np.asarray(X[:, n : 2 * n].sum(axis=0)).ravel()
    if off_weight.sum() > 0:
        offense -= np.average(offense, weights=off_weight)
    if def_weight.sum() > 0:
        defense -= np.average(defense, weights=def_weight)
    return np.concatenate([offense, defense, np.zeros(1)])


def _heldout_likelihood(contract: dict, priors: pd.DataFrame) -> pd.DataFrame:
    cache = CHECKPOINT_ROOT / "heldout_likelihood_games.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frame = load_unified_terminal_possessions(
        POSSESSION_CACHE,
        SILVER_POSSESSIONS,
        SILVER_LINEUPS,
        tuple(map(int, contract["seasons"]["rating"])),
        transition_season=2024,
        game_types=("regular",),
    )
    penalty_scale = float(contract["diagnostics"]["split_penalty_scale"])
    config = RapmConfig(
        seasons=tuple(map(int, contract["seasons"]["rating"])),
        lambda_off=penalty_scale * float(contract["rapm"]["aio"]["lambda_off"]),
        lambda_def=penalty_scale * float(contract["rapm"]["aio"]["lambda_def"]),
        lambda_home=penalty_scale * float(contract["rapm"]["aio"]["lambda_home"]),
        data_scope="spm_consensus_disjoint_likelihood",
    )
    rows = []
    for season in map(int, contract["seasons"]["rating"]):
        season_frame = frame.loc[frame["season"].eq(season)].copy()
        design = build_design(season_frame, include_home=True)
        partition = stable_game_partition(design.game_ids)
        for candidate in ("box15", "rich"):
            prior = priors.loc[
                priors["design"].eq("fully_lagged")
                & priors["candidate"].eq(candidate)
                & priors["rating_season"].eq(season)
            ]
            for train_split in (0, 1):
                train_mask = partition == train_split
                test_mask = ~train_mask
                center = _center_for_design(prior, design, train_mask)
                beta, intercept = fit_coefficients_with_center(design, config, center, row_mask=train_mask)
                games = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
                games["rating_season"] = season
                games["candidate"] = candidate
                games["train_split"] = train_split
                games["squared_error"] = (games["actual_margin"] - games["predicted_margin"]) ** 2
                rows.append(games)
        print(f"held-out likelihood {season}: complete", flush=True)
    output = pd.concat(rows, ignore_index=True)
    _atomic_parquet(output, cache)
    return output


def _heldout_summary(games: pd.DataFrame) -> pd.DataFrame:
    by_season = games.groupby(["candidate", "rating_season"], sort=True)["squared_error"].mean().reset_index()
    pivot = by_season.pivot(index="rating_season", columns="candidate", values="squared_error")
    rows = []
    for candidate in ("box15", "rich"):
        values = pivot[candidate]
        rows.append(
            {
                "candidate": candidate,
                "folds": len(values),
                "equal_season_mse": float(values.mean()),
                "equal_season_rmse": float(math.sqrt(values.mean())),
                "wins_vs_other": int((values < pivot["rich" if candidate == "box15" else "box15"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _artifact_hashes(frames: dict[str, pd.DataFrame], output_root: Path) -> dict[str, dict]:
    output = {}
    for name, frame in frames.items():
        path = Path(name)
        keys = [column for column in ("rating_season", "outcome_season", "candidate", "side", "feature", "game_id") if column in frame]
        ordered = frame.sort_values(keys, kind="stable") if keys else frame
        row_hash = hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()).hexdigest()
        output[path.name] = {
            "path": path.name,
            "rows": len(frame),
            "sha256": sha256_file(output_root / path.name),
            "row_set_sha256": row_hash,
        }
    return output


def _possession_hashes() -> dict[str, str]:
    return {
        **{
            f"legacy_possessions_{season}": sha256_file(
                POSSESSION_CACHE / f"matchups_{season}.parquet"
            )
            for season in range(2017, 2024)
        },
        "silver_possessions": sha256_file(SILVER_POSSESSIONS),
        "silver_lineups": sha256_file(SILVER_LINEUPS),
    }


def _validate_outputs(frames: dict[str, pd.DataFrame]) -> dict:
    consensus = frames["consensus_priors.parquet"]
    if consensus.duplicated(["PLAYER_ID", "rating_season"]).any():
        raise ValueError("Consensus prior keys are not unique.")
    component_error = float(
        np.max(np.abs(consensus["prior_offense"] + consensus["prior_defense"] - consensus["prior_net"]))
    )
    if component_error > 1e-10:
        raise ValueError("Consensus offense plus defense does not equal net.")

    frequencies = frames["selection_frequencies.parquet"]
    invalid_selected = frequencies.loc[
        frequencies["selected"]
        & ~frequencies["mandatory"]
        & frequencies["real_selection_frequency"].lt(frequencies["threshold"])
    ]
    if not invalid_selected.empty or frequencies.loc[frequencies["is_noise"], "selected"].any():
        raise ValueError("The consensus recurrence rule was violated.")

    next_games = frames["next_season_games.parquet"]
    next_keys = None
    for _, group in next_games.groupby("candidate", sort=False):
        keys = set(zip(group["game_id"], group["rating_season"], group["outcome_season"]))
        next_keys = keys if next_keys is None else next_keys
        if keys != next_keys:
            raise ValueError("Next-season candidates do not score identical games.")

    heldout = frames["heldout_likelihood_games.parquet"]
    heldout_keys = None
    for _, group in heldout.groupby("candidate", sort=False):
        keys = set(zip(group["game_id"], group["rating_season"]))
        heldout_keys = keys if heldout_keys is None else heldout_keys
        if keys != heldout_keys or group.duplicated(["game_id", "rating_season"]).any():
            raise ValueError("Held-out likelihood candidates do not score identical games once.")

    ledger = frames["future_split_game_ledger.parquet"]
    if ledger.duplicated(["game_id", "season"]).any() or not set(ledger["split"]).issubset({0, 1}):
        raise ValueError("Future-reference game partitions are invalid.")
    cross = frames["cross_reference_correlations.parquet"]
    if cross["players"].le(0).any() or not np.isfinite(cross["cross_reference_correlation"]).all():
        raise ValueError("Cross-reference correlations are incomplete.")
    return {
        "maximum_component_identity_error": component_error,
        "identical_next_season_games": True,
        "identical_heldout_likelihood_games": True,
        "disjoint_reference_game_keys": True,
        "minimum_cross_reference_players": int(cross["players"].min()),
    }


def run(contract_path: Path = CONTRACT) -> dict:
    contract = _load_contract(contract_path)
    source_contract = load_source_contract()
    references = _reference_panels(source_contract)
    _, rich, _, rich_arms, _ = _load_features(source_contract)
    selector_paths = {
        "priors": CHECKPOINT_ROOT / "consensus_priors.parquet",
        "frequencies": CHECKPOINT_ROOT / "selection_frequencies.parquet",
        "quality": CHECKPOINT_ROOT / "selector_quality.parquet",
    }
    if all(path.exists() for path in selector_paths.values()):
        consensus = pd.read_parquet(selector_paths["priors"])
        frequencies = pd.read_parquet(selector_paths["frequencies"])
        selector_quality = pd.read_parquet(selector_paths["quality"])
    else:
        consensus, frequencies, selector_quality = _fit_consensus_priors(
            contract, references, rich, rich_arms
        )
        _atomic_parquet(consensus, selector_paths["priors"])
        _atomic_parquet(frequencies, selector_paths["frequencies"])
        _atomic_parquet(selector_quality, selector_paths["quality"])
    priors = _matched_priors(consensus)
    next_games, prior_ratings = _next_season_games(contract, priors)
    fold_metrics = _fold_metrics(next_games)
    paired = _paired_summary(next_games, "current_control__box15", contract)
    target_excluded_paired = _paired_summary(next_games, "target_excluded__box15", contract)
    later_paired = _paired_summary(
        next_games.loc[next_games["outcome_season"].ge(2022)],
        "current_control__box15",
        contract,
    )
    split_references = _future_split_references(contract)
    split_game_ledger = pd.read_parquet(CHECKPOINT_ROOT / "future_split_game_ledger.parquet")
    cross_reference = cross_reference_correlations(priors, references, split_references)
    heldout_games = _heldout_likelihood(contract, priors)
    heldout_summary = _heldout_summary(heldout_games)
    heldout_for_pair = heldout_games.rename(
        columns={"squared_error": "mse", "rating_season": "outcome_season"}
    )
    heldout_for_pair["rating_season"] = heldout_for_pair["outcome_season"] - 1
    heldout_paired = _paired_summary(heldout_for_pair, "box15", contract)
    prior_target_metrics = _prior_target_metrics(priors, references)
    cross_reference_summary = _cross_reference_summary(cross_reference, contract)

    frames = {
        "consensus_priors.parquet": consensus,
        "matched_priors.parquet": priors,
        "selection_frequencies.parquet": frequencies,
        "selector_quality.parquet": selector_quality,
        "prior_ratings.parquet": prior_ratings,
        "next_season_games.parquet": next_games,
        "fold_metrics.parquet": fold_metrics,
        "prior_target_metrics.parquet": prior_target_metrics,
        "paired_summary.parquet": paired,
        "target_excluded_paired_summary.parquet": target_excluded_paired,
        "later_paired_summary.parquet": later_paired,
        "future_split_references.parquet": split_references,
        "future_split_game_ledger.parquet": split_game_ledger,
        "cross_reference_correlations.parquet": cross_reference,
        "cross_reference_summary.parquet": cross_reference_summary,
        "heldout_likelihood_games.parquet": heldout_games,
        "heldout_likelihood_summary.parquet": heldout_summary,
        "heldout_likelihood_paired.parquet": heldout_paired,
    }
    possession_hashes = _possession_hashes()
    content_hash = hashlib.sha256(
        json.dumps(
            {
                "contract": sha256_file(contract_path),
                "runner": sha256_file(Path(__file__)),
                "source_run": sha256_file(SOURCE_RUN / "run.json"),
                "rich": sha256_file(RICH_ANNUAL),
                "atlas": sha256_file(ATLAS),
                "possessions": possession_hashes,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"{EXPERIMENT_ID}_{content_hash}"
    output = OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        _atomic_parquet(frame, output / name)
    quality = _validate_outputs(frames)
    manifest = {
        "run_id": run_id,
        "status": "research_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand_id": contract["estimand_id"],
        "config_sha256": sha256_file(contract_path),
        "code_sha256": sha256_file(Path(__file__)),
        "source_run_id": contract["source_experiment"]["run_id"],
        "source_run_sha256": sha256_file(SOURCE_RUN / "run.json"),
        "input_hashes": {
            "rich_annual": sha256_file(RICH_ANNUAL),
            "feature_atlas": sha256_file(ATLAS),
            **possession_hashes,
        },
        "quality": {
            "rating_seasons": list(map(int, contract["seasons"]["rating"])),
            **quality,
        },
        "artifacts": _artifact_hashes(frames, output),
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(manifest, output / "run.json")
    print(json.dumps({"run_id": run_id, "artifact_path": str(output)}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    args = parser.parse_args()
    run(args.contract)


if __name__ == "__main__":
    main()
