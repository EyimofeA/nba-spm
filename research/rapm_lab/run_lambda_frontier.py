"""Compare broad, bivariate, and adaptive-EB rolling RAPM penalties."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import (
    bivariate_penalty_matrix,
    diagonal_penalty_matrix,
    solve_stored_generalized_ridge,
    stored_evaluation_predictions,
    stored_training_diagnostics,
    store_lambda_research_matrices,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "research" / "experiments" / "rolling_5y_lambda_frontier_v1.json"
)
DEFAULT_MATRIX_RUN = (
    REPO_ROOT
    / "research"
    / "rapm_lab"
    / "outputs"
    / "rolling_5y_2014_2026"
    / "rolling_5y_rapm_2014_2026_a7754bfb77"
)
SIDES = ("offense", "defense")


def _candidate_id(candidate: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return f"{candidate['family']}_{digest}"


def _deduplicate(candidates: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for candidate in candidates:
        unique[_candidate_id(candidate)] = candidate
    return list(unique.values())


def _scalar_stage_one(contract: dict) -> list[dict]:
    spec = contract["scalar_search"]
    sampler = qmc.Sobol(d=2, scramble=True, seed=int(spec["sobol_seed"]))
    power = int(math.ceil(math.log2(int(spec["sobol_candidates"]))))
    unit = sampler.random_base2(power)[: int(spec["sobol_candidates"])]
    lower = np.log10(
        [spec["lambda_off_bounds"][0], spec["lambda_def_bounds"][0]]
    )
    upper = np.log10(
        [spec["lambda_off_bounds"][1], spec["lambda_def_bounds"][1]]
    )
    values = 10.0 ** (lower + unit * (upper - lower))
    candidates = [
        {
            "family": "scalar",
            "lambda_off": float(row[0]),
            "lambda_def": float(row[1]),
            "lambda_home": float(spec["lambda_home"]),
            "stage": "sobol",
        }
        for row in values
    ]
    candidates.extend(
        {
            "family": "scalar",
            "lambda_off": float(offense),
            "lambda_def": float(defense),
            "lambda_home": float(spec["lambda_home"]),
            "stage": "anchor",
        }
        for offense, defense in spec["anchors"]
    )
    return _deduplicate(candidates)


def _penalty_scale(candidate: dict) -> float:
    if candidate["family"] == "empirical_bayes":
        return float("nan")
    return float(math.sqrt(candidate["lambda_off"] * candidate["lambda_def"]))


def _select(summary: pd.DataFrame, method: str) -> pd.Series:
    if method == "selection_correlation":
        maximum = float(summary["mean_correlation"].max())
        eligible = summary.loc[summary["mean_correlation"] >= maximum - 0.0005]
        return eligible.sort_values(
            ["mean_rmse", "penalty_scale"], ascending=[True, False], kind="stable"
        ).iloc[0]
    if method == "selection_rmse":
        minimum = float(summary["mean_rmse"].min())
        eligible = summary.loc[summary["mean_rmse"] <= minimum + 0.01]
        return eligible.sort_values(
            ["penalty_scale", "mean_correlation"],
            ascending=[False, False],
            kind="stable",
        ).iloc[0]
    if method == "training_gcv":
        return summary.sort_values(
            ["mean_log_gcv", "penalty_scale"], ascending=[True, False], kind="stable"
        ).iloc[0]
    raise ValueError(f"Unknown selection method: {method}")


def _summarize(folds: pd.DataFrame, candidates: dict[str, dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for candidate_id, frame in folds.groupby("candidate_id", sort=False):
        candidate = candidates[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "candidate_json": json.dumps(candidate, sort_keys=True),
                "mean_correlation": float(frame["margin_correlation"].mean()),
                "mean_mae": float(frame["margin_mae"].mean()),
                "mean_rmse": float(frame["margin_rmse"].mean()),
                "mean_log_gcv": float(np.log(frame["training_gcv"]).mean()),
                "penalty_scale": _penalty_scale(candidate),
            }
        )
    return pd.DataFrame(rows)


def _metrics(predictions: pd.DataFrame) -> dict[str, float | int]:
    actual = predictions["actual_margin"].to_numpy(dtype=np.float64)
    predicted = predictions["predicted_margin"].to_numpy(dtype=np.float64)
    error = actual - predicted
    return {
        "games": int(len(predictions)),
        "margin_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "margin_mae": float(np.mean(np.abs(error))),
        "margin_rmse": float(np.sqrt(np.mean(error**2))),
    }


def _adaptive_eb_penalty(
    matrix_dir: Path,
    contract: dict,
    adaptation_strength: float,
    cache: dict[str, dict],
) -> tuple[object, dict]:
    spec = contract["empirical_bayes"]
    key = str(matrix_dir.resolve())
    if key not in cache:
        players = np.load(matrix_dir / "player_ids.npy")
        pilot = spec["pilot"]
        pilot_penalty = diagonal_penalty_matrix(
            len(players),
            lambda_off=float(pilot["lambda_off"]),
            lambda_def=float(pilot["lambda_def"]),
            lambda_home=float(pilot["lambda_home"]),
        )
        pilot_solution = solve_stored_generalized_ridge(matrix_dir, pilot_penalty)
        diagnostics = stored_training_diagnostics(
            matrix_dir,
            pilot_solution,
            pilot_penalty,
            probes=int(spec["posterior_diagonal_probes"]),
            seed=int(spec["posterior_seed"]),
            return_inverse_diagonal=True,
        )
        cache[key] = {
            "players": players,
            "raw_beta": pilot_solution.raw_beta,
            "inverse_diagonal": diagnostics["inverse_diagonal"],
            "residual_variance": float(diagnostics["residual_variance"]),
            "off_counts": np.load(matrix_dir / "train_off_possessions.npy"),
            "def_counts": np.load(matrix_dir / "train_def_possessions.npy"),
        }
    pilot_state = cache[key]
    n_players = len(pilot_state["players"])
    residual_variance = pilot_state["residual_variance"]
    posterior_variance = residual_variance * np.maximum(
        np.asarray(pilot_state["inverse_diagonal"]), 0.0
    )
    lower_global, upper_global = (
        float(value) for value in spec["global_precision_bounds"]
    )
    lower_multiplier, upper_multiplier = (
        float(value) for value in spec["precision_multiplier_bounds"]
    )
    side_precisions: list[np.ndarray] = []
    metadata: dict[str, float] = {"adaptation_strength": adaptation_strength}
    for side_index, side in enumerate(SIDES):
        start = side_index * n_players
        stop = start + n_players
        beta = np.asarray(pilot_state["raw_beta"])[start:stop]
        variance = posterior_variance[start:stop]
        counts = np.asarray(pilot_state[f"{'off' if side == 'offense' else 'def'}_counts"])
        known = counts > 0
        squared = beta[known] ** 2
        winsor_limit = float(np.quantile(squared, 0.95))
        global_prior_variance = max(
            # One Gaussian empirical-Bayes EM update uses the posterior second
            # moment E[beta^2 | y] = posterior_mean^2 + posterior_variance.
            float(np.mean(np.minimum(squared, winsor_limit)) + np.mean(variance[known])),
            float(spec["minimum_prior_variance"]),
        )
        global_precision = float(
            np.clip(residual_variance / global_prior_variance, lower_global, upper_global)
        )
        local_floor = float(spec["local_signal_floor_fraction"]) * global_prior_variance
        local_variance = np.maximum(beta**2 + variance, local_floor)
        blended_variance = (
            (1.0 - adaptation_strength) * global_prior_variance
            + adaptation_strength * local_variance
        )
        multiplier = np.clip(
            global_prior_variance / blended_variance,
            lower_multiplier,
            upper_multiplier,
        )
        multiplier[~known] = upper_multiplier
        precision = global_precision * multiplier
        side_precisions.append(precision)
        metadata.update(
            {
                f"{side}_global_prior_variance": global_prior_variance,
                f"{side}_global_precision": global_precision,
                f"{side}_precision_p10": float(np.quantile(precision[known], 0.10)),
                f"{side}_precision_median": float(np.median(precision[known])),
                f"{side}_precision_p90": float(np.quantile(precision[known], 0.90)),
            }
        )
    penalty = diagonal_penalty_matrix(
        n_players,
        lambda_off=side_precisions[0],
        lambda_def=side_precisions[1],
        lambda_home=float(spec["pilot"]["lambda_home"]),
    )
    return penalty, metadata


def _build_penalty(
    matrix_dir: Path,
    candidate: dict,
    contract: dict,
    eb_cache: dict[str, dict],
):
    n_players = len(np.load(matrix_dir / "player_ids.npy"))
    if candidate["family"] == "scalar":
        return (
            diagonal_penalty_matrix(
                n_players,
                lambda_off=float(candidate["lambda_off"]),
                lambda_def=float(candidate["lambda_def"]),
                lambda_home=float(candidate["lambda_home"]),
            ),
            {},
        )
    if candidate["family"] == "bivariate":
        return (
            bivariate_penalty_matrix(
                n_players,
                lambda_off=float(candidate["lambda_off"]),
                lambda_def=float(candidate["lambda_def"]),
                lambda_home=float(candidate["lambda_home"]),
                published_prior_correlation=float(candidate["prior_correlation"]),
            ),
            {},
        )
    if candidate["family"] == "empirical_bayes":
        return _adaptive_eb_penalty(
            matrix_dir,
            contract,
            float(candidate["adaptation_strength"]),
            eb_cache,
        )
    raise ValueError(f"Unknown candidate family: {candidate['family']}")


def _score_candidates(
    candidates: list[dict],
    seasons: list[int],
    matrix_dirs: dict[int, Path],
    contract: dict,
    checkpoint_root: Path,
    *,
    calculate_gcv: bool,
    eb_cache: dict[str, dict],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    gcv_spec = contract["training_gcv"]
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = _candidate_id(candidate)
        scope = "selection" if calculate_gcv else "diagnostic"
        checkpoint = checkpoint_root / f"{scope}__{candidate_id}.parquet"
        if checkpoint.exists():
            frame = pd.read_parquet(checkpoint)
        else:
            rows: list[dict] = []
            for season in seasons:
                matrix_dir = matrix_dirs[int(season)]
                penalty, penalty_metadata = _build_penalty(
                    matrix_dir, candidate, contract, eb_cache
                )
                solution = solve_stored_generalized_ridge(matrix_dir, penalty)
                predictions = stored_evaluation_predictions(
                    matrix_dir, solution.beta, solution.intercept
                )
                metrics = _metrics(predictions)
                row = {
                    "candidate_id": candidate_id,
                    "family": candidate["family"],
                    "evaluation_season": int(season),
                    **metrics,
                    **penalty_metadata,
                }
                if calculate_gcv:
                    diagnostic = stored_training_diagnostics(
                        matrix_dir,
                        solution,
                        penalty,
                        probes=int(gcv_spec["trace_probes"]),
                        seed=int(gcv_spec["seed"]) + int(season),
                    )
                    row.update(
                        {
                            "training_gcv": float(diagnostic["gcv"]),
                            "effective_df": float(diagnostic["effective_df"]),
                            "training_residual_variance": float(
                                diagnostic["residual_variance"]
                            ),
                        }
                    )
                rows.append(row)
            frame = pd.DataFrame(rows)
            frame.to_parquet(checkpoint, index=False)
        frames.append(frame)
        if index == 1 or index % 10 == 0 or index == len(candidates):
            print(
                f"{scope} {index}/{len(candidates)}: {candidate_id}", flush=True
            )
    return pd.concat(frames, ignore_index=True)


def _local_scalar_candidates(parents: list[dict], contract: dict) -> list[dict]:
    multipliers = [float(value) for value in contract["scalar_search"]["local_multipliers"]]
    candidates: list[dict] = []
    for parent in parents:
        for off_multiplier in multipliers:
            for def_multiplier in multipliers:
                candidates.append(
                    {
                        "family": "scalar",
                        "lambda_off": float(parent["lambda_off"] * off_multiplier),
                        "lambda_def": float(parent["lambda_def"] * def_multiplier),
                        "lambda_home": float(parent["lambda_home"]),
                        "stage": "local_refinement",
                    }
                )
    return _deduplicate(candidates)


def _family_winners(
    summary: pd.DataFrame,
    candidates: dict[str, dict],
    methods: tuple[str, ...] = (
        "selection_correlation",
        "selection_rmse",
        "training_gcv",
    ),
) -> dict[str, dict]:
    return {
        method: candidates[str(_select(summary, method)["candidate_id"])]
        for method in methods
    }


def _candidate_ratings(
    matrix_dir: Path,
    candidate: dict,
    contract: dict,
    eb_cache: dict[str, dict],
) -> pd.DataFrame:
    penalty, _ = _build_penalty(matrix_dir, candidate, contract, eb_cache)
    solution = solve_stored_generalized_ridge(matrix_dir, penalty)
    n_players = len(solution.players)
    offense = 100.0 * solution.beta[:n_players]
    defense = -100.0 * solution.beta[n_players : 2 * n_players]
    return pd.DataFrame(
        {
            "PLAYER_ID": solution.players.astype(np.int64),
            "offense": offense,
            "defense": defense,
            "net": offense + defense,
            "Poss_Off": np.load(matrix_dir / "train_off_possessions.npy"),
            "Poss_Def": np.load(matrix_dir / "train_def_possessions.npy"),
        }
    )


def _stability_and_exposure_audit(
    finalists: dict[str, dict],
    matrix_by_end: dict[int, Path],
    contract: dict,
    eb_cache: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stability_rows: list[dict] = []
    exposure_rows: list[dict] = []
    rating_frames: list[pd.DataFrame] = []
    window_a, window_b = (int(value) for value in contract["validation"]["stability_windows"])
    high_floor = int(contract["validation"]["high_exposure_floor_per_side"])
    bands = [float(value) for value in contract["validation"]["exposure_bands"]]
    for label, candidate in finalists.items():
        first = _candidate_ratings(matrix_by_end[window_a], candidate, contract, eb_cache)
        second = _candidate_ratings(matrix_by_end[window_b], candidate, contract, eb_cache)
        first["window_end"] = window_a
        second["window_end"] = window_b
        first["finalist"] = label
        second["finalist"] = label
        rating_frames.extend([first, second])
        matched = first.merge(second, on="PLAYER_ID", suffixes=("_a", "_b"))
        matched = matched.loc[
            (matched[["Poss_Off_a", "Poss_Def_a", "Poss_Off_b", "Poss_Def_b"]] >= high_floor).all(axis=1)
        ]
        for component in ("offense", "defense", "net"):
            stability_rows.append(
                {
                    "finalist": label,
                    "component": component,
                    "matched_players": int(len(matched)),
                    "adjacent_window_correlation": float(
                        matched[[f"{component}_a", f"{component}_b"]].corr().iloc[0, 1]
                    ),
                    "adjacent_window_rmse": float(
                        np.sqrt(
                            np.mean(
                                (matched[f"{component}_a"] - matched[f"{component}_b"]) ** 2
                            )
                        )
                    ),
                }
            )
        latest = second.copy()
        latest["exposure"] = latest[["Poss_Off", "Poss_Def"]].min(axis=1)
        latest["exposure_band"] = pd.cut(
            latest["exposure"], bins=bands, right=False, include_lowest=True
        ).astype(str)
        for exposure_band, group in latest.groupby("exposure_band", observed=True):
            for component in ("offense", "defense", "net"):
                values = group[component].to_numpy(dtype=np.float64)
                exposure_rows.append(
                    {
                        "finalist": label,
                        "exposure_band": exposure_band,
                        "component": component,
                        "players": int(len(group)),
                        "p01": float(np.quantile(values, 0.01)),
                        "p99": float(np.quantile(values, 0.99)),
                        "max_abs": float(np.max(np.abs(values))),
                    }
                )
    return (
        pd.DataFrame(stability_rows),
        pd.DataFrame(exposure_rows),
        pd.concat(rating_frames, ignore_index=True),
    )


def _synthetic_frame(seed: int, rows: int, players: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    player_ids = np.arange(1, players + 1, dtype=np.int64)
    latent_group = rng.choice(np.asarray([0, 1]), size=players, p=[0.75, 0.25])
    scales = np.where(latent_group == 1, 0.035, 0.012)
    correlation = (-0.5, 0.0, 0.5)[seed % 3]
    shared = rng.normal(size=players)
    independent = rng.normal(size=players)
    offense = scales * shared
    defense = scales * (
        correlation * shared + math.sqrt(1.0 - correlation**2) * independent
    )
    offense -= offense.mean()
    defense -= defense.mean()
    records: list[dict] = []
    for row in range(rows):
        selected = rng.choice(player_ids, size=10, replace=False)
        away = selected[:5]
        home = selected[5:]
        home_poss = bool(rng.integers(0, 2))
        attacking = home if home_poss else away
        defending = away if home_poss else home
        mean = (
            1.14
            + offense[attacking - 1].sum()
            - defense[defending - 1].sum()
            + (0.012 if home_poss else -0.012)
        )
        records.append(
            {
                "home_poss": home_poss,
                "pts": float(mean + rng.normal(0.0, 0.92)),
                **{f"a{i + 1}": int(value) for i, value in enumerate(away)},
                **{f"h{i + 1}": int(value) for i, value in enumerate(home)},
                "season": 2020,
                "date": "2019-11-01",
                "period": 1,
                "num": row + 1,
                "gameid": f"S{seed:03d}{row // 200:04d}",
            }
        )
    truth = pd.DataFrame(
        {
            "PLAYER_ID": player_ids,
            "true_offense": 100.0 * offense,
            "true_defense": 100.0 * defense,
        }
    )
    truth["true_net"] = truth["true_offense"] + truth["true_defense"]
    return pd.DataFrame(records), truth


def _synthetic_recovery(
    finalists: dict[str, dict],
    contract: dict,
) -> pd.DataFrame:
    spec = contract["validation"]
    rows: list[dict] = []
    for seed_index in range(int(spec["synthetic_seeds"])):
        seed = 20260830 + seed_index
        frame, truth = _synthetic_frame(
            seed, int(spec["synthetic_rows"]), int(spec["synthetic_players"])
        )
        with tempfile.TemporaryDirectory(prefix="courtsignal_lambda_synth_") as directory:
            matrix_dir = Path(directory)
            store_lambda_research_matrices(frame, matrix_dir)
            eb_cache: dict[str, dict] = {}
            for label, candidate in finalists.items():
                ratings = _candidate_ratings(matrix_dir, candidate, contract, eb_cache)
                matched = ratings.merge(truth, on="PLAYER_ID", validate="one_to_one")
                for component in ("offense", "defense", "net"):
                    error = matched[component] - matched[f"true_{component}"]
                    rows.append(
                        {
                            "seed": seed,
                            "finalist": label,
                            "component": component,
                            "recovery_correlation": float(
                                matched[[component, f"true_{component}"]].corr().iloc[0, 1]
                            ),
                            "recovery_rmse": float(np.sqrt(np.mean(error**2))),
                        }
                    )
    return pd.DataFrame(rows)


def _synthetic_relabeling_invariance(contract: dict) -> dict[str, float]:
    frame, _ = _synthetic_frame(20260930, rows=6000, players=60)
    rng = np.random.default_rng(20260930)
    old_ids = np.arange(1, 61, dtype=np.int64)
    new_ids = rng.permutation(old_ids) + 1000
    forward = dict(zip(old_ids.tolist(), new_ids.tolist(), strict=True))
    inverse = {value: key for key, value in forward.items()}
    relabeled = frame.copy()
    player_columns = [*(f"a{i}" for i in range(1, 6)), *(f"h{i}" for i in range(1, 6))]
    for column in player_columns:
        relabeled[column] = relabeled[column].map(forward).astype(np.int64)
    baseline = {
        "family": "scalar",
        **{key: float(value) for key, value in contract["baseline"].items()},
        "stage": "invariance",
    }
    with tempfile.TemporaryDirectory(prefix="courtsignal_lambda_invariance_a_") as a_dir:
        with tempfile.TemporaryDirectory(prefix="courtsignal_lambda_invariance_b_") as b_dir:
            a_path = Path(a_dir)
            b_path = Path(b_dir)
            store_lambda_research_matrices(frame, a_path)
            store_lambda_research_matrices(relabeled, b_path)
            first = _candidate_ratings(a_path, baseline, contract, {})
            second = _candidate_ratings(b_path, baseline, contract, {})
            second["PLAYER_ID"] = second["PLAYER_ID"].map(inverse).astype(np.int64)
            matched = first.merge(second, on="PLAYER_ID", suffixes=("_a", "_b"))
    result = {
        f"{component}_max_abs_error": float(
            np.max(np.abs(matched[f"{component}_a"] - matched[f"{component}_b"]))
        )
        for component in ("offense", "defense", "net")
    }
    tolerance = float(
        contract["validation"]["offense_plus_defense_identity_tolerance"]
    )
    if max(result.values()) > tolerance:
        raise ValueError(f"Player relabeling invariance failed: {result}")
    return result


def run_frontier(contract_path: Path, matrix_run: Path) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Lambda frontier requires a frozen contract.")
    if matrix_run.name != contract["matrix_run_id"]:
        raise ValueError("Matrix run does not match the contract.")
    catalog = pd.read_parquet(matrix_run / "lambda_matrix_catalog.parquet")
    complete = catalog.loc[catalog["evaluation_status"].eq("complete")].copy()
    complete["evaluation_season"] = complete["evaluation_season"].astype(int)
    expected = sorted(contract["selection_seasons"] + contract["reused_diagnostic_seasons"])
    if sorted(complete["evaluation_season"].tolist()) != expected or 2027 in expected:
        raise ValueError("Evaluation season contract mismatch or Season 2027 exposure.")
    matrix_dirs = {
        int(row.evaluation_season): REPO_ROOT / Path(row.manifest).parent
        for row in complete.itertuples(index=False)
    }
    matrix_by_end = {
        int(row.window_end): REPO_ROOT / Path(row.manifest).parent
        for row in catalog.itertuples(index=False)
    }
    identity_payload = {
        "contract_hash": sha256_file(contract_path),
        "script_hash": sha256_file(Path(__file__)),
        "solver_hash": sha256_file(
            REPO_ROOT / "src" / "nba_impact" / "models" / "rapm_sufficient_statistics.py"
        ),
        "matrix_run_id": matrix_run.name,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = matrix_run / f"lambda_frontier_v1_{identity}"
    completed_path = output / "run.json"
    if completed_path.exists():
        return json.loads(completed_path.read_text())
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    eb_cache: dict[str, dict] = {}

    stage_one = _scalar_stage_one(contract)
    stage_one_map = {_candidate_id(candidate): candidate for candidate in stage_one}
    stage_one_folds = _score_candidates(
        stage_one,
        contract["selection_seasons"],
        matrix_dirs,
        contract,
        checkpoints,
        calculate_gcv=True,
        eb_cache=eb_cache,
    )
    stage_one_summary = _summarize(stage_one_folds, stage_one_map)
    stage_one_winners = _family_winners(stage_one_summary, stage_one_map)
    local = _local_scalar_candidates(list(stage_one_winners.values()), contract)
    existing = set(stage_one_map)
    local = [candidate for candidate in local if _candidate_id(candidate) not in existing]
    local_map = {_candidate_id(candidate): candidate for candidate in local}
    local_folds = _score_candidates(
        local,
        contract["selection_seasons"],
        matrix_dirs,
        contract,
        checkpoints,
        calculate_gcv=True,
        eb_cache=eb_cache,
    )
    scalar_candidates = {**stage_one_map, **local_map}
    scalar_folds = pd.concat([stage_one_folds, local_folds], ignore_index=True)
    scalar_summary = _summarize(scalar_folds, scalar_candidates)
    scalar_winners = _family_winners(scalar_summary, scalar_candidates)

    bivariate = _deduplicate(
        [
            {
                "family": "bivariate",
                "lambda_off": float(parent["lambda_off"]),
                "lambda_def": float(parent["lambda_def"]),
                "lambda_home": float(parent["lambda_home"]),
                "prior_correlation": float(correlation),
                "stage": "bivariate",
            }
            for parent in scalar_winners.values()
            for correlation in contract["bivariate_search"]["published_prior_correlations"]
        ]
    )
    bivariate_map = {_candidate_id(candidate): candidate for candidate in bivariate}
    bivariate_folds = _score_candidates(
        bivariate,
        contract["selection_seasons"],
        matrix_dirs,
        contract,
        checkpoints,
        calculate_gcv=True,
        eb_cache=eb_cache,
    )
    bivariate_summary = _summarize(bivariate_folds, bivariate_map)
    bivariate_winners = _family_winners(bivariate_summary, bivariate_map)

    empirical_bayes = [
        {
            "family": "empirical_bayes",
            "adaptation_strength": float(strength),
            "stage": "adaptive_type_ii_moment",
        }
        for strength in contract["empirical_bayes"]["adaptation_strengths"]
    ]
    empirical_bayes_map = {
        _candidate_id(candidate): candidate for candidate in empirical_bayes
    }
    empirical_bayes_folds = _score_candidates(
        empirical_bayes,
        contract["selection_seasons"],
        matrix_dirs,
        contract,
        checkpoints,
        calculate_gcv=True,
        eb_cache=eb_cache,
    )
    empirical_bayes_summary = _summarize(empirical_bayes_folds, empirical_bayes_map)
    # Adaptive candidates have fold-varying scales; use metrics directly for ties.
    empirical_bayes_summary["penalty_scale"] = 0.0
    empirical_bayes_winners = _family_winners(
        empirical_bayes_summary, empirical_bayes_map
    )

    all_candidates = {**scalar_candidates, **bivariate_map, **empirical_bayes_map}
    all_selection_folds = pd.concat(
        [scalar_folds, bivariate_folds, empirical_bayes_folds], ignore_index=True
    )
    all_summary = pd.concat(
        [scalar_summary, bivariate_summary, empirical_bayes_summary], ignore_index=True
    )
    all_summary.to_parquet(output / "selection_summary.parquet", index=False)
    all_selection_folds.to_parquet(output / "selection_folds.parquet", index=False)

    finalists: dict[str, dict] = {
        "baseline": {
            "family": "scalar",
            **{key: float(value) for key, value in contract["baseline"].items()},
            "stage": "baseline",
        }
    }
    for family, winners in (
        ("scalar", scalar_winners),
        ("bivariate", bivariate_winners),
        ("empirical_bayes", empirical_bayes_winners),
    ):
        for method, candidate in winners.items():
            finalists[f"{family}__{method}"] = candidate
    diagnostic_candidates = _deduplicate(list(finalists.values()))
    diagnostic_folds = _score_candidates(
        diagnostic_candidates,
        contract["reused_diagnostic_seasons"],
        matrix_dirs,
        contract,
        checkpoints,
        calculate_gcv=False,
        eb_cache=eb_cache,
    )
    diagnostic_folds.to_parquet(output / "diagnostic_folds.parquet", index=False)
    diagnostic_rows: list[dict] = []
    for candidate_id, frame in diagnostic_folds.groupby("candidate_id"):
        candidate = all_candidates.get(candidate_id)
        if candidate is None:
            candidate = next(
                value for value in finalists.values() if _candidate_id(value) == candidate_id
            )
        diagnostic_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_json": json.dumps(candidate, sort_keys=True),
                "family": candidate["family"],
                "mean_correlation": float(frame["margin_correlation"].mean()),
                "mean_mae": float(frame["margin_mae"].mean()),
                "mean_rmse": float(frame["margin_rmse"].mean()),
            }
        )
    diagnostic_summary = pd.DataFrame(diagnostic_rows)
    diagnostic_summary.to_parquet(output / "diagnostic_summary.parquet", index=False)

    stability, exposure, finalist_ratings = _stability_and_exposure_audit(
        finalists, matrix_by_end, contract, eb_cache
    )
    stability.to_parquet(output / "stability_audit.parquet", index=False)
    exposure.to_parquet(output / "exposure_tail_audit.parquet", index=False)
    finalist_ratings.to_parquet(output / "finalist_ratings_2025_2026.parquet", index=False)
    synthetic = _synthetic_recovery(finalists, contract)
    synthetic.to_parquet(output / "synthetic_recovery.parquet", index=False)
    relabeling_invariance = _synthetic_relabeling_invariance(contract)

    selection_winner_rows = []
    for label, candidate in finalists.items():
        candidate_id = _candidate_id(candidate)
        selection_match = all_summary.loc[all_summary["candidate_id"].eq(candidate_id)]
        diagnostic_match = diagnostic_summary.loc[
            diagnostic_summary["candidate_id"].eq(candidate_id)
        ]
        selection_winner_rows.append(
            {
                "label": label,
                "candidate_id": candidate_id,
                "candidate": candidate,
                "selection": (
                    selection_match.iloc[0][
                        ["mean_correlation", "mean_mae", "mean_rmse", "mean_log_gcv"]
                    ].to_dict()
                    if not selection_match.empty
                    else None
                ),
                "diagnostic": diagnostic_match.iloc[0][
                    ["mean_correlation", "mean_mae", "mean_rmse"]
                ].to_dict(),
            }
        )
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_comparison_complete",
        **identity_payload,
        "selection_seasons": contract["selection_seasons"],
        "reused_diagnostic_seasons": contract["reused_diagnostic_seasons"],
        "untouched_confirmation_season": contract["untouched_confirmation_season"],
        "season_2027_loaded": False,
        "candidate_counts": {
            "scalar_stage_one": len(stage_one),
            "scalar_local": len(local),
            "scalar_total": len(scalar_candidates),
            "bivariate": len(bivariate),
            "empirical_bayes": len(empirical_bayes),
        },
        "finalists": selection_winner_rows,
        "player_relabeling_invariance": relabeling_invariance,
        "artifacts": {
            "selection_folds": "selection_folds.parquet",
            "selection_summary": "selection_summary.parquet",
            "diagnostic_folds": "diagnostic_folds.parquet",
            "diagnostic_summary": "diagnostic_summary.parquet",
            "stability_audit": "stability_audit.parquet",
            "exposure_tail_audit": "exposure_tail_audit.parquet",
            "synthetic_recovery": "synthetic_recovery.parquet",
        },
        "promotion_attempted": False,
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, completed_path)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--matrix-run", type=Path, default=DEFAULT_MATRIX_RUN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_frontier(args.contract.resolve(), args.matrix_run.resolve())
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
