"""Five-year predictive AIO with a frozen time-decay and SPM-prior test.

The model uses five trailing seasons of possession evidence.  A time-decay grid
is selected only on 2020-24 future games.  The frozen predictive SPM supplies a
coefficient-space prior for one named challenger.  Seasons 2025-26 are reused
diagnostics and cannot alter the selected settings.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, spsolve

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import (
    RapmDesign,
    _game_margin_frame,
    build_design,
    load_current_player_names,
    load_unified_terminal_possessions,
)


ARMS = (
    "five_year_zero_prior",
    "selected_decay_zero_prior",
    "five_year_spm_prior_aio",
    "selected_decay_spm_prior_aio",
)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    frame.to_parquet(partial, index=False)
    partial.replace(path)


def _load_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": "predictive_current_aio_2026_v1",
        "status": "preregistered_reused_diagnostic",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must be {value!r}.")
    development = tuple(int(value) for value in contract["development_seasons"])
    diagnostics = tuple(int(value) for value in contract["reused_diagnostic_seasons"])
    if development != (2020, 2021, 2022, 2023, 2024):
        raise ValueError("Development seasons must remain 2020 through 2024.")
    if diagnostics != (2025, 2026):
        raise ValueError("Reused diagnostics must remain 2025 and 2026.")
    if max((*development, *diagnostics)) > 2026:
        raise ValueError(
            "Season 2027 must be rejected before possession data are read."
        )
    expected_half_lives = (0.5, 1.0, 2.0, 3.0, 5.0, None)
    if tuple(contract["half_lives_years"]) != expected_half_lives:
        raise ValueError(f"half_lives_years must be exactly {expected_half_lives}.")
    if int(contract["window_seasons"]) != 5:
        raise ValueError("The predictive backbone must use five trailing seasons.")
    if float(contract["spm_center_scale"]) != 1.0:
        raise ValueError("The only SPM prior arm must use center scale 1.0.")
    return contract


def _half_life_label(value: float | None) -> str:
    return "none" if value is None else f"{value:g}"


def _season_weight(season: int, window_end: int, half_life: float | None) -> float:
    if half_life is None:
        return 1.0
    if half_life <= 0:
        raise ValueError("half-life must be positive.")
    return float(2.0 ** ((season - window_end) / half_life))


def _penalty_vector(
    n_players: int, *, lambda_off: float, lambda_def: float, lambda_home: float
) -> np.ndarray:
    return np.concatenate(
        [
            np.full(n_players, lambda_off, dtype=float),
            np.full(n_players, lambda_def, dtype=float),
            np.asarray([lambda_home], dtype=float),
        ]
    )


def build_season_statistics(design: RapmDesign) -> dict[int, dict[str, object]]:
    """Compute reusable per-season cross-products in the global player order."""
    output: dict[int, dict[str, object]] = {}
    for season in sorted(int(value) for value in np.unique(design.seasons)):
        mask = design.seasons == season
        x = design.X[mask]
        y = design.y[mask]
        output[season] = {
            "xtx": (x.T @ x).tocsr(),
            "xty": np.asarray(x.T @ y).ravel(),
            "xsum": np.asarray(x.sum(axis=0)).ravel(),
            "ysum": float(y.sum()),
            "rows": int(mask.sum()),
        }
    return output


def fit_from_season_statistics(
    statistics: dict[int, dict[str, object]],
    train_seasons: tuple[int, ...],
    *,
    n_players: int,
    lambda_off: float,
    lambda_def: float,
    lambda_home: float,
    half_life: float | None,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    missing = sorted(set(train_seasons) - set(statistics))
    if missing:
        raise ValueError(f"Missing season statistics: {missing}.")
    window_end = max(train_seasons)
    weighted_rows = 0.0
    weighted_y = 0.0
    xtx: csr_matrix | None = None
    xty = np.zeros(2 * n_players + 1, dtype=float)
    xsum = np.zeros(2 * n_players + 1, dtype=float)
    for season in train_seasons:
        weight = _season_weight(season, window_end, half_life)
        item = statistics[season]
        xtx = item["xtx"] * weight if xtx is None else xtx + item["xtx"] * weight
        xty += np.asarray(item["xty"], dtype=float) * weight
        xsum += np.asarray(item["xsum"], dtype=float) * weight
        weighted_y += float(item["ysum"]) * weight
        weighted_rows += int(item["rows"]) * weight
    if xtx is None or weighted_rows <= 0:
        raise ValueError("The weighted training window is empty.")
    intercept = weighted_y / weighted_rows
    penalty = _penalty_vector(
        n_players,
        lambda_off=lambda_off,
        lambda_def=lambda_def,
        lambda_home=lambda_home,
    )
    prior = (
        np.zeros_like(penalty) if center is None else np.asarray(center, dtype=float)
    )
    if prior.shape != penalty.shape or not np.isfinite(prior).all():
        raise ValueError("The predictive SPM center must match the RAPM coefficients.")
    lhs = xtx + diags(penalty, format="csr")
    rhs = xty - intercept * xsum + penalty * prior
    try:
        beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:  # pragma: no cover - old scipy compatibility.
        beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        beta = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(beta, dtype=float)
    off_count = xsum[:n_players]
    def_count = xsum[n_players : 2 * n_players]
    off_mean = float(np.average(beta[:n_players], weights=off_count))
    def_mean = float(np.average(beta[n_players : 2 * n_players], weights=def_count))
    beta[:n_players] -= off_mean
    beta[n_players : 2 * n_players] -= def_mean
    intercept += 5.0 * (off_mean + def_mean)
    return beta, float(intercept), off_count, def_count


def build_spm_center(
    design: RapmDesign,
    predictions: pd.DataFrame,
    *,
    target_season: int,
    off_exposure: np.ndarray,
    def_exposure: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    required = {
        "PLAYER_ID",
        "Target_Season",
        "predicted_offense",
        "predicted_defense",
        "predicted_net",
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Selected predictive SPM rows are missing {missing}.")
    window = predictions.loc[predictions["Target_Season"].eq(target_season)].copy()
    if window.empty or window["PLAYER_ID"].duplicated().any():
        raise ValueError(
            f"Predictive SPM center for {target_season} is empty or duplicated."
        )
    indexed = window.set_index("PLAYER_ID")
    players = pd.Index(design.players)
    offense = pd.to_numeric(indexed["predicted_offense"], errors="raise").reindex(
        players
    )
    defense = pd.to_numeric(indexed["predicted_defense"], errors="raise").reindex(
        players
    )
    available = offense.notna() & defense.notna()
    off = offense.fillna(0.0).to_numpy(dtype=float) / 100.0
    deff = -defense.fillna(0.0).to_numpy(dtype=float) / 100.0
    off -= np.average(off, weights=off_exposure)
    deff -= np.average(deff, weights=def_exposure)
    center = np.concatenate([off, deff, np.asarray([0.0])])
    test_columns = design.X[test_mask, : 2 * len(players)].indices % len(players)
    return center, {
        "target_season": int(target_season),
        "players_with_prior": int(available.sum()),
        "train_off_possession_coverage": float(
            np.average(available.to_numpy(dtype=float), weights=off_exposure)
        ),
        "train_def_possession_coverage": float(
            np.average(available.to_numpy(dtype=float), weights=def_exposure)
        ),
        "test_lineup_slot_coverage": float(
            available.to_numpy(dtype=float)[test_columns].mean()
        ),
    }


def _game_metrics(games: pd.DataFrame) -> dict:
    error = games["actual_margin"] - games["predicted_margin"]
    actual = games["actual_margin"].to_numpy(dtype=float)
    predicted = games["predicted_margin"].to_numpy(dtype=float)
    variance = float(np.var(predicted, ddof=0))
    slope = (
        float(np.cov(actual, predicted, ddof=0)[0, 1] / variance)
        if variance > 0
        else np.nan
    )
    return {
        "games": int(len(games)),
        "margin_rmse": float(np.sqrt(np.mean(error**2))),
        "margin_mae": float(np.mean(np.abs(error))),
        "margin_correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "calibration_slope": slope,
        "calibration_intercept": float(actual.mean() - slope * predicted.mean()),
        "predicted_margin_sd": float(predicted.std(ddof=0)),
    }


def _rating_table(
    design: RapmDesign,
    beta: np.ndarray,
    off_exposure: np.ndarray,
    def_exposure: np.ndarray,
    *,
    names: pd.DataFrame,
    target_season: int,
    arm: str,
) -> pd.DataFrame:
    n = len(design.players)
    output = pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "offense": 100.0 * beta[:n],
            "defense": -100.0 * beta[n : 2 * n],
            "Poss_Off": off_exposure,
            "Poss_Def": def_exposure,
            "target_season": int(target_season),
            "arm": arm,
        }
    )
    output["net"] = output["offense"] + output["defense"]
    return output.merge(names, on="PLAYER_ID", how="left", validate="one_to_one")


def _paired_bootstrap(
    games: pd.DataFrame,
    *,
    selected_arm: str,
    baseline_arm: str,
    seasons: tuple[int, ...],
    draws: int,
    seed: int,
) -> dict:
    scoped = games.loc[
        games["test_season"].isin(seasons)
        & games["arm"].isin([selected_arm, baseline_arm])
    ]
    wide = scoped.pivot(
        index=["test_season", "game_id"], columns="arm", values="squared_error"
    )
    if wide[[selected_arm, baseline_arm]].isna().any().any():
        raise ValueError("Paired bootstrap arms do not share identical games.")
    deltas = [
        group[selected_arm].to_numpy(dtype=float)
        - group[baseline_arm].to_numpy(dtype=float)
        for _, group in wide.groupby(level="test_season", sort=True)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for index in range(draws):
        samples[index] = np.mean(
            [
                rng.choice(values, size=len(values), replace=True).mean()
                for values in deltas
            ]
        )
    observed = float(np.mean([values.mean() for values in deltas]))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "selected_minus_baseline_game_mse": observed,
        "ci_95_lower": float(lower),
        "ci_95_upper": float(upper),
        "probability_selected_better": float(np.mean(samples < 0)),
        "draws": int(draws),
        "seed": int(seed),
        "seasons": list(seasons),
        "matched_games": int(len(wide)),
    }


def build_predictive_current_aio(
    contract_path: str | Path,
    spm_predictions_path: str | Path,
    legacy_cache_dir: str | Path,
    current_possessions_path: str | Path,
    current_segments_path: str | Path,
    names_path: str | Path,
    player_games_path: str | Path,
    *,
    artifact_root: str | Path,
) -> dict:
    started = time.perf_counter()
    contract_file = Path(contract_path)
    contract = _load_contract(contract_file)
    development = tuple(int(value) for value in contract["development_seasons"])
    diagnostics = tuple(int(value) for value in contract["reused_diagnostic_seasons"])
    test_seasons = development + diagnostics
    window = int(contract["window_seasons"])
    all_seasons = tuple(range(min(test_seasons) - window, max(test_seasons) + 1))
    if max(all_seasons) > 2026:
        raise ValueError("Season 2027 is forbidden.")

    spm_predictions = pd.read_parquet(spm_predictions_path)
    selected_methods = set(spm_predictions.get("method", pd.Series(dtype=str)).dropna())
    if selected_methods != {"raw"}:
        raise ValueError(
            f"The frozen SPM trajectory winner must be raw; found {selected_methods}."
        )
    frame = load_unified_terminal_possessions(
        legacy_cache_dir,
        current_possessions_path,
        current_segments_path,
        all_seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    design = build_design(frame, include_home=True)
    statistics = build_season_statistics(design)
    names = load_current_player_names(names_path, player_games_path)
    n_players = len(design.players)
    lambda_off = float(contract["rapm"]["lambda_off"])
    lambda_def = float(contract["rapm"]["lambda_def"])
    lambda_home = float(contract["rapm"]["lambda_home"])

    source_hashes = {
        "contract": sha256_file(contract_file),
        "spm_predictions": sha256_file(spm_predictions_path),
        "source_code": sha256_file(Path(__file__)),
        "current_possessions": sha256_file(current_possessions_path),
        "current_segments": sha256_file(current_segments_path),
        "legacy_possessions": {
            str(season): sha256_file(
                Path(legacy_cache_dir) / f"matchups_{season}.parquet"
            )
            for season in all_seasons
            if season < 2024
        },
    }
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    run_id = f"predictive_current_aio_2026_v1_{identity}"
    output = Path(artifact_root) / "models" / "predictive_current_aio" / run_id
    checkpoint = output / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)

    half_lives = tuple(contract["half_lives_years"])
    decay_metric_rows: list[dict] = []
    for test_season in development:
        path = checkpoint / f"decay_{test_season}.parquet"
        if path.exists():
            decay_metric_rows.extend(pd.read_parquet(path).to_dict("records"))
            continue
        train_seasons = tuple(range(test_season - window, test_season))
        train_mask = np.isin(design.seasons, train_seasons)
        test_mask = design.seasons == test_season
        fold_rows = []
        for half_life in half_lives:
            beta, intercept, _, _ = fit_from_season_statistics(
                statistics,
                train_seasons,
                n_players=n_players,
                lambda_off=lambda_off,
                lambda_def=lambda_def,
                lambda_home=lambda_home,
                half_life=half_life,
            )
            games = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
            fold_rows.append(
                {
                    "test_season": int(test_season),
                    "half_life_years": _half_life_label(half_life),
                    **_game_metrics(games),
                }
            )
        table = pd.DataFrame(fold_rows)
        _atomic_parquet(table, path)
        decay_metric_rows.extend(fold_rows)
        print(f"time decay {test_season}: checkpointed", flush=True)
    decay_metrics = pd.DataFrame(decay_metric_rows)
    decay_summary = (
        decay_metrics.groupby("half_life_years", as_index=False)
        .agg(
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            folds=("test_season", "nunique"),
        )
        .sort_values(["mean_margin_rmse", "half_life_years"], kind="stable")
        .reset_index(drop=True)
    )
    selected_label = str(decay_summary.iloc[0]["half_life_years"])
    selected_half_life = None if selected_label == "none" else float(selected_label)

    metric_rows: list[dict] = []
    game_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    rowset_hashes: dict[str, str] = {}
    for test_season in test_seasons:
        paths = {
            name: checkpoint / f"arms_{test_season}_{name}.parquet"
            for name in ("metrics", "games", "ratings", "coverage")
        }
        if all(path.exists() for path in paths.values()):
            metric_rows.extend(pd.read_parquet(paths["metrics"]).to_dict("records"))
            games = pd.read_parquet(paths["games"])
            game_frames.append(games)
            rating_frames.append(pd.read_parquet(paths["ratings"]))
            coverage_rows.extend(pd.read_parquet(paths["coverage"]).to_dict("records"))
            rowset_hashes[str(test_season)] = hashlib.sha256(
                "\n".join(sorted(games["game_id"].astype(str).unique())).encode()
            ).hexdigest()
            continue
        train_seasons = tuple(range(test_season - window, test_season))
        train_mask = np.isin(design.seasons, train_seasons)
        test_mask = design.seasons == test_season
        fits: dict[str, tuple[np.ndarray, float, np.ndarray, np.ndarray]] = {}
        for label, half_life in (
            ("five_year_zero_prior", None),
            ("selected_decay_zero_prior", selected_half_life),
        ):
            fits[label] = fit_from_season_statistics(
                statistics,
                train_seasons,
                n_players=n_players,
                lambda_off=lambda_off,
                lambda_def=lambda_def,
                lambda_home=lambda_home,
                half_life=half_life,
            )
        for label, half_life, exposure_source in (
            ("five_year_spm_prior_aio", None, "five_year_zero_prior"),
            (
                "selected_decay_spm_prior_aio",
                selected_half_life,
                "selected_decay_zero_prior",
            ),
        ):
            _, _, off_exposure, def_exposure = fits[exposure_source]
            center, coverage = build_spm_center(
                design,
                spm_predictions,
                target_season=test_season,
                off_exposure=off_exposure,
                def_exposure=def_exposure,
                test_mask=test_mask,
            )
            coverage["arm"] = label
            coverage_rows.append(coverage)
            fits[label] = fit_from_season_statistics(
                statistics,
                train_seasons,
                n_players=n_players,
                lambda_off=lambda_off,
                lambda_def=lambda_def,
                lambda_home=lambda_home,
                half_life=half_life,
                center=center,
            )
        fold_metrics = []
        fold_games = []
        fold_ratings = []
        for arm in ARMS:
            beta, intercept, off_exposure, def_exposure = fits[arm]
            games = _game_margin_frame(design, beta, intercept, test_mask, train_mask)
            games["test_season"] = test_season
            games["arm"] = arm
            games["squared_error"] = (
                games["actual_margin"] - games["predicted_margin"]
            ) ** 2
            fold_games.append(games)
            fold_metrics.append(
                {
                    "test_season": test_season,
                    "arm": arm,
                    "training_start": train_seasons[0],
                    "training_end": train_seasons[-1],
                    "selected_half_life_years": selected_label,
                    **_game_metrics(games),
                }
            )
            fold_ratings.append(
                _rating_table(
                    design,
                    beta,
                    off_exposure,
                    def_exposure,
                    names=names,
                    target_season=test_season,
                    arm=arm,
                )
            )
        fold_metrics_frame = pd.DataFrame(fold_metrics)
        fold_games_frame = pd.concat(fold_games, ignore_index=True)
        fold_ratings_frame = pd.concat(fold_ratings, ignore_index=True)
        fold_coverage_frame = pd.DataFrame(
            [row for row in coverage_rows if row["target_season"] == test_season]
        )
        _atomic_parquet(fold_metrics_frame, paths["metrics"])
        _atomic_parquet(fold_games_frame, paths["games"])
        _atomic_parquet(fold_ratings_frame, paths["ratings"])
        _atomic_parquet(fold_coverage_frame, paths["coverage"])
        metric_rows.extend(fold_metrics)
        game_frames.append(fold_games_frame)
        rating_frames.append(fold_ratings_frame)
        rowset_hashes[str(test_season)] = hashlib.sha256(
            "\n".join(sorted(fold_games_frame["game_id"].astype(str).unique())).encode()
        ).hexdigest()
        print(f"AIO arms {test_season}: checkpointed", flush=True)

    metrics = pd.DataFrame(metric_rows)
    games = pd.concat(game_frames, ignore_index=True)
    ratings = pd.concat(rating_frames, ignore_index=True)
    development_summary = (
        metrics.loc[metrics["test_season"].isin(development)]
        .groupby("arm", as_index=False)
        .agg(
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
            folds=("test_season", "nunique"),
        )
        .sort_values(["mean_margin_rmse", "arm"], kind="stable")
        .reset_index(drop=True)
    )
    selected_arm = str(development_summary.iloc[0]["arm"])
    bootstrap = _paired_bootstrap(
        games,
        selected_arm=selected_arm,
        baseline_arm="five_year_zero_prior",
        seasons=development,
        draws=int(contract["bootstrap"]["draws"]),
        seed=int(contract["bootstrap"]["seed"]),
    )
    selected_is_aio = selected_arm.endswith("spm_prior_aio")
    final_2026 = ratings.loc[
        ratings["target_season"].eq(2026) & ratings["arm"].eq(selected_arm)
    ].copy()
    predictive_aio_2026 = ratings.loc[
        ratings["target_season"].eq(2026)
        & ratings["arm"].eq("selected_decay_spm_prior_aio")
    ].copy()
    identity_error = np.abs(ratings["net"] - ratings["offense"] - ratings["defense"])
    if float(identity_error.max()) > 1e-10:
        raise AssertionError("Predictive AIO rating components must sum to net.")

    _atomic_parquet(decay_metrics, output / "time_decay_fold_metrics.parquet")
    _atomic_parquet(decay_summary, output / "time_decay_summary.parquet")
    _atomic_parquet(metrics, output / "fold_metrics.parquet")
    _atomic_parquet(development_summary, output / "development_summary.parquet")
    _atomic_parquet(games, output / "game_predictions.parquet")
    _atomic_parquet(ratings, output / "ratings.parquet")
    _atomic_parquet(pd.DataFrame(coverage_rows), output / "prior_coverage.parquet")
    _atomic_parquet(final_2026, output / "selected_2026_ratings.parquet")
    _atomic_parquet(predictive_aio_2026, output / "predictive_aio_2026_ratings.parquet")
    run = {
        "run_id": run_id,
        "experiment_id": contract["experiment_id"],
        "estimand_id": contract["estimand_id"],
        "status": "research_model_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "selected_half_life_years": selected_label,
        "selected_arm": selected_arm,
        "selected_arm_is_predictive_aio": selected_is_aio,
        "predictive_aio_arm": "selected_decay_spm_prior_aio",
        "development_seasons": list(development),
        "reused_diagnostic_seasons": list(diagnostics),
        "source_hashes": source_hashes,
        "bootstrap": bootstrap,
        "quality": {
            "possession_rows": int(len(frame)),
            "games": int(frame["gameid"].nunique()),
            "players": int(n_players),
            "maximum_loaded_season": int(max(design.seasons)),
            "maximum_component_identity_error": float(identity_error.max()),
            "game_rowset_hashes": rowset_hashes,
            "selected_2026_rows": int(len(final_2026)),
            "predictive_aio_2026_rows": int(len(predictive_aio_2026)),
        },
        "paths": {
            "time_decay_fold_metrics": "time_decay_fold_metrics.parquet",
            "time_decay_summary": "time_decay_summary.parquet",
            "fold_metrics": "fold_metrics.parquet",
            "development_summary": "development_summary.parquet",
            "game_predictions": "game_predictions.parquet",
            "ratings": "ratings.parquet",
            "prior_coverage": "prior_coverage.parquet",
            "selected_2026_ratings": "selected_2026_ratings.parquet",
            "predictive_aio_2026_ratings": "predictive_aio_2026_ratings.parquet",
        },
        "forbidden_interpretation": (
            "This uses observed held-out lineups and reused 2025-26 diagnostics. It is "
            "not an untouched forecast confirmation, roster forecast, or public model."
        ),
    }
    write_json_atomic(run, output / "run.json")
    return run
