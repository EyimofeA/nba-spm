"""Matched-game comparison of statistical-prior and zero-prior RAPM."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmConfig,
    RapmDesign,
    _game_margin_frame,
    _game_margin_metrics,
    build_design,
    fit_coefficient_center_path,
)


PRIOR_COLUMNS = (
    "prior_offense_per_100",
    "prior_defense_per_100",
    "prior_net_per_100",
)


def _candidate_name(scale: float) -> str:
    return f"prior_scale_{scale:g}".replace(".", "p")


def build_prior_center(
    design: RapmDesign,
    priors: pd.DataFrame,
    *,
    prior_window_end: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Map positive-good statistical ratings to RAPM coefficient signs."""
    window = priors.loc[priors["Window_End"].eq(prior_window_end)].copy()
    if window.empty:
        raise ValueError(f"No statistical priors exist for window {prior_window_end}.")
    if window["PLAYER_ID"].duplicated().any():
        raise ValueError(f"Prior window {prior_window_end} has duplicate player IDs.")
    window = window.set_index("PLAYER_ID")
    player_index = pd.Index(design.players)
    offense = pd.to_numeric(
        window["prior_offense_per_100"], errors="raise"
    ).reindex(player_index)
    defense = pd.to_numeric(
        window["prior_defense_per_100"], errors="raise"
    ).reindex(player_index)
    has_prior = offense.notna() & defense.notna()
    offense_coefficients = offense.fillna(0.0).to_numpy(dtype=np.float64) / 100.0
    defense_coefficients = -defense.fillna(0.0).to_numpy(dtype=np.float64) / 100.0

    n_players = len(design.players)
    train_X = design.X[train_mask]
    off_counts = np.asarray(train_X[:, :n_players].sum(axis=0)).ravel()
    def_counts = np.asarray(train_X[:, n_players : 2 * n_players].sum(axis=0)).ravel()
    if off_counts.sum() <= 0 or def_counts.sum() <= 0:
        raise ValueError("Prior centering requires positive training possessions.")
    offense_coefficients -= np.average(offense_coefficients, weights=off_counts)
    defense_coefficients -= np.average(defense_coefficients, weights=def_counts)

    center = np.concatenate([offense_coefficients, defense_coefficients])
    if design.X.shape[1] == 2 * n_players + 1:
        center = np.append(center, 0.0)
    test_columns = design.X[test_mask, : 2 * n_players].indices % n_players
    coverage = {
        "prior_window_end": prior_window_end,
        "players_in_design": n_players,
        "players_with_prior": int(has_prior.sum()),
        "train_off_possession_coverage": float(
            np.average(has_prior.to_numpy(dtype=float), weights=off_counts)
        ),
        "train_def_possession_coverage": float(
            np.average(has_prior.to_numpy(dtype=float), weights=def_counts)
        ),
        "test_lineup_slot_coverage": float(
            has_prior.to_numpy(dtype=float)[test_columns].mean()
        ),
        "centered_offense_mean": float(
            np.average(center[:n_players], weights=off_counts)
        ),
        "centered_defense_mean": float(
            np.average(center[n_players : 2 * n_players], weights=def_counts)
        ),
    }
    return center, coverage


def _fit_prior_only_nuisance(
    design: RapmDesign,
    center: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit only intercept and home court while player coefficients stay fixed."""
    X = design.X[train_mask]
    residual = design.y[train_mask] - np.asarray(X @ center).ravel()
    beta = center.copy()
    if design.X.shape[1] == 2 * len(design.players) + 1:
        home = X[:, -1].toarray().ravel()
        nuisance = np.column_stack([np.ones(len(residual)), home])
        intercept, home_coefficient = np.linalg.lstsq(
            nuisance, residual, rcond=None
        )[0]
        beta[-1] = home_coefficient
        return beta, float(intercept)
    return beta, float(residual.mean())


def _record_candidate(
    design: RapmDesign,
    beta: np.ndarray,
    intercept: float,
    *,
    candidate: str,
    test_season: int,
    train_seasons: tuple[int, ...],
    prior_window_end: int,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    metrics = {
        "candidate": candidate,
        "train_start": train_seasons[0],
        "train_end": train_seasons[-1],
        "prior_window_end": prior_window_end,
        "test_season": test_season,
        **_game_margin_metrics(design, beta, intercept, test_mask, train_mask),
    }
    games = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
    games["candidate"] = candidate
    games["test_season"] = test_season
    games["squared_error"] = (
        games["actual_margin"] - games["predicted_margin"]
    ) ** 2
    return metrics, games


def paired_confirmation_bootstrap(
    games: pd.DataFrame,
    *,
    selected_candidate: str,
    confirmation_test_seasons: tuple[int, ...],
    repetitions: int,
    seed: int,
) -> dict:
    """Bootstrap paired game squared-error deltas within each test season."""
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive.")
    candidates = ["zero_prior", selected_candidate]
    matched = games.loc[
        games["test_season"].isin(confirmation_test_seasons)
        & games["candidate"].isin(candidates),
        ["test_season", "game_id", "candidate", "squared_error"],
    ]
    if matched.duplicated(["test_season", "game_id", "candidate"]).any():
        raise ValueError("Confirmation game predictions must have unique candidate keys.")
    wide = matched.pivot(
        index=["test_season", "game_id"],
        columns="candidate",
        values="squared_error",
    )
    if wide[candidates].isna().any().any():
        raise ValueError("Confirmation candidates must use exactly matched games.")
    wide["mse_delta_selected_minus_zero"] = (
        wide[selected_candidate] - wide["zero_prior"]
    )
    season_deltas = [
        group["mse_delta_selected_minus_zero"].to_numpy(dtype=np.float64)
        for _, group in wide.groupby(level="test_season", sort=True)
    ]
    if len(season_deltas) != len(confirmation_test_seasons):
        raise ValueError("Every confirmation season must contain matched games.")
    observed = float(np.mean([values.mean() for values in season_deltas]))
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        samples[index] = np.mean(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for values in season_deltas
            ]
        )
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "estimand": "equal-season-weighted game MSE selected minus zero prior",
        "selected_candidate": selected_candidate,
        "observed_mse_delta": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "probability_selected_better": float((samples < 0).mean()),
        "matched_games": int(len(wide)),
        "confirmation_seasons": [int(value) for value in confirmation_test_seasons],
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


def run_prior_informed_rapm_comparison(
    frame: pd.DataFrame,
    priors_path: str | Path,
    config: RapmConfig,
    *,
    artifact_root: str | Path,
    test_seasons: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024),
    train_window: int = 3,
    prior_scales: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    selection_test_seasons: tuple[int, ...] = (2020, 2021, 2022),
    confirmation_test_seasons: tuple[int, ...] = (2023, 2024),
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 20260810,
) -> dict:
    """Select prior scale on old folds and check it on later matched games."""
    if train_window < 1:
        raise ValueError("train_window must be positive.")
    if not prior_scales or any(not 0.0 < scale <= 1.0 for scale in prior_scales):
        raise ValueError("prior_scales must contain values in (0, 1].")
    if len(set(prior_scales)) != len(prior_scales):
        raise ValueError("prior_scales must be unique.")
    if set(selection_test_seasons) & set(confirmation_test_seasons):
        raise ValueError("Selection and confirmation seasons must be disjoint.")
    if set((*selection_test_seasons, *confirmation_test_seasons)) != set(test_seasons):
        raise ValueError("Selection and confirmation seasons must partition test_seasons.")
    if max(selection_test_seasons) >= min(confirmation_test_seasons):
        raise ValueError("Selection seasons must precede confirmation seasons.")

    priors = pd.read_parquet(priors_path)
    required_prior_columns = {"PLAYER_ID", "Window_End", *PRIOR_COLUMNS}
    if missing := sorted(required_prior_columns - set(priors.columns)):
        raise ValueError(f"Prior-informed RAPM is missing prior columns {missing}.")
    if priors.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Prior-informed RAPM prior keys must be unique.")
    if not np.isfinite(priors[list(PRIOR_COLUMNS)].to_numpy()).all():
        raise ValueError("Prior-informed RAPM priors must be finite.")

    design = build_design(frame)
    available_seasons = set(int(value) for value in np.unique(design.seasons))
    required_seasons = {
        season
        for test_season in test_seasons
        for season in range(test_season - train_window, test_season + 1)
    }
    if missing := sorted(required_seasons - available_seasons):
        raise ValueError(f"Prior-informed RAPM is missing possession seasons {missing}.")

    fold_rows: list[dict] = []
    game_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    scales_with_zero = (0.0, *prior_scales)
    for test_season in test_seasons:
        train_seasons = tuple(range(test_season - train_window, test_season))
        prior_window_end = train_seasons[-1]
        train_mask = np.isin(design.seasons, train_seasons)
        test_mask = design.seasons == test_season
        center, coverage = build_prior_center(
            design,
            priors,
            prior_window_end=prior_window_end,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        coverage["test_season"] = test_season
        coverage_rows.append(coverage)
        fitted = fit_coefficient_center_path(
            design,
            config,
            center,
            center_scales=scales_with_zero,
            row_mask=train_mask,
        )
        for scale in scales_with_zero:
            candidate = "zero_prior" if scale == 0.0 else _candidate_name(scale)
            beta, intercept = fitted[scale]
            metrics, games = _record_candidate(
                design,
                beta,
                intercept,
                candidate=candidate,
                test_season=test_season,
                train_seasons=train_seasons,
                prior_window_end=prior_window_end,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            metrics["prior_scale"] = scale
            fold_rows.append(metrics)
            game_frames.append(games)
        prior_beta, prior_intercept = _fit_prior_only_nuisance(
            design, center, train_mask
        )
        metrics, games = _record_candidate(
            design,
            prior_beta,
            prior_intercept,
            candidate="prior_only",
            test_season=test_season,
            train_seasons=train_seasons,
            prior_window_end=prior_window_end,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        metrics["prior_scale"] = 1.0
        fold_rows.append(metrics)
        game_frames.append(games)

    folds = pd.DataFrame(fold_rows)
    games = pd.concat(game_frames, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    eligible_candidates = ["zero_prior", *(_candidate_name(scale) for scale in prior_scales)]
    selection = (
        folds.loc[
            folds["test_season"].isin(selection_test_seasons)
            & folds["candidate"].isin(eligible_candidates)
        ]
        .groupby(["candidate", "prior_scale"], as_index=False)
        .agg(
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            folds=("test_season", "nunique"),
        )
        .sort_values(["mean_margin_rmse", "prior_scale"], kind="stable")
    )
    winner = selection.iloc[0]
    selected_candidate = str(winner["candidate"])
    selected_scale = float(winner["prior_scale"])
    confirmation_candidates = {"zero_prior", "prior_only", selected_candidate}
    confirmation = (
        folds.loc[
            folds["test_season"].isin(confirmation_test_seasons)
            & folds["candidate"].isin(confirmation_candidates)
        ]
        .groupby(["candidate", "prior_scale"], as_index=False)
        .agg(
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            folds=("test_season", "nunique"),
        )
    )
    zero_by_season = folds.loc[
        folds["candidate"].eq("zero_prior"), ["test_season", "margin_rmse"]
    ].rename(columns={"margin_rmse": "zero_margin_rmse"})
    confirmation_fold_comparison = folds.loc[
        folds["test_season"].isin(confirmation_test_seasons)
        & folds["candidate"].isin(confirmation_candidates)
    ].merge(zero_by_season, on="test_season", validate="many_to_one")
    confirmation_fold_comparison["rmse_delta_vs_zero"] = (
        confirmation_fold_comparison["margin_rmse"]
        - confirmation_fold_comparison["zero_margin_rmse"]
    )
    confirmation_wins = (
        confirmation_fold_comparison.groupby("candidate")["rmse_delta_vs_zero"]
        .apply(lambda values: int((values < 0).sum()))
        .to_dict()
    )
    confirmation["fold_wins_vs_zero"] = confirmation["candidate"].map(
        confirmation_wins
    )
    confirmation = confirmation.sort_values("mean_margin_rmse", kind="stable")
    confirmation_lookup = confirmation.set_index("candidate")
    selected_beats_zero = bool(
        selected_candidate != "zero_prior"
        and confirmation_lookup.loc[selected_candidate, "mean_margin_rmse"]
        < confirmation_lookup.loc["zero_prior", "mean_margin_rmse"]
    )
    bootstrap_candidate = (
        selected_candidate
        if selected_candidate != "zero_prior"
        else _candidate_name(max(prior_scales))
    )
    paired_bootstrap = paired_confirmation_bootstrap(
        games,
        selected_candidate=bootstrap_candidate,
        confirmation_test_seasons=confirmation_test_seasons,
        repetitions=bootstrap_repetitions,
        seed=bootstrap_seed,
    )
    selected_fold_wins = int(confirmation_wins.get(selected_candidate, 0))
    improvement_demonstrated = bool(
        selected_candidate != "zero_prior"
        and selected_beats_zero
        and selected_fold_wins == len(confirmation_test_seasons)
        and paired_bootstrap["ci_95_upper"] < 0
    )
    if selected_candidate == "zero_prior":
        status = "prior_center_rejected_in_selection"
    elif improvement_demonstrated:
        status = "research_challenger_reused_confirmation"
    else:
        status = "prior_center_improvement_not_demonstrated"

    run_id = f"prior_informed_rapm_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "prior_informed_rapm" / run_id
    output.mkdir(parents=True, exist_ok=False)
    folds.to_parquet(output / "fold_results.parquet", index=False)
    games.to_parquet(output / "game_predictions.parquet", index=False)
    coverage.to_parquet(output / "prior_coverage.parquet", index=False)
    selection.to_parquet(output / "scale_selection.parquet", index=False)
    confirmation.to_parquet(output / "confirmation_summary.parquet", index=False)
    confirmation_fold_comparison.to_parquet(
        output / "confirmation_fold_comparison.parquet", index=False
    )
    source_hashes = {
        str(path): sha256_file(path)
        for path in frame.attrs.get("source_paths", [])
        if Path(path).exists()
    }
    run = {
        "run_id": run_id,
        "model_family": "prior_informed_normal_rapm",
        "estimand": (
            f"{train_window}_season_lineup_adjusted_descriptive_points_per_100_"
            "evaluated_on_next_season_game_margins"
        ),
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "train_window": train_window,
            "test_seasons": list(test_seasons),
            "selection_test_seasons": list(selection_test_seasons),
            "confirmation_test_seasons": list(confirmation_test_seasons),
            "prior_scales": list(prior_scales),
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": bootstrap_seed,
            "prior_mapping": {
                "offense": "prior_offense_per_100 / 100",
                "defense": "-prior_defense_per_100 / 100",
                "missing_player": "zero",
                "block_centering": "training-possession-weighted mean zero",
            },
            "prior_only_nuisance_fit": "training intercept and home coefficient only",
            "priors_path": str(Path(priors_path).resolve()),
            "source_hashes": {
                "priors": sha256_file(priors_path),
                "source_code": sha256_file(Path(__file__)),
                "rapm_solver": sha256_file(Path(__file__).with_name("rapm.py")),
                "possession_files": source_hashes,
            },
        },
        "quality": {
            "folds": len(test_seasons),
            "games": int(
                games[["test_season", "game_id"]].drop_duplicates().shape[0]
            ),
            "duplicate_prior_keys": 0,
            "minimum_train_off_possession_prior_coverage": float(
                coverage["train_off_possession_coverage"].min()
            ),
            "minimum_train_def_possession_prior_coverage": float(
                coverage["train_def_possession_coverage"].min()
            ),
            "minimum_test_lineup_slot_prior_coverage": float(
                coverage["test_lineup_slot_coverage"].min()
            ),
            "maximum_centered_block_mean": float(
                coverage[["centered_offense_mean", "centered_defense_mean"]]
                .abs()
                .to_numpy()
                .max()
            ),
        },
        "metrics": {
            "selected_candidate": selected_candidate,
            "selected_prior_scale": selected_scale,
            "bootstrap_candidate": bootstrap_candidate,
            "selection": selection.to_dict(orient="records"),
            "confirmation": confirmation.to_dict(orient="records"),
            "selected_beats_zero_on_reused_confirmation": selected_beats_zero,
            "paired_confirmation_bootstrap": paired_bootstrap,
            "improvement_demonstrated": improvement_demonstrated,
        },
        "caveats": [
            "Evaluation uses observed test-season lineups; this is retrodiction, not a deployable forecast.",
            "The 2022-2024 seasons informed earlier feature research and cannot support a clean production promotion claim.",
            "The statistical prior and RAPM evidence use overlapping same-window games and are not independent measurements.",
            "Legacy possessions end in 2024; the empty 2025 legacy cache is excluded.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
