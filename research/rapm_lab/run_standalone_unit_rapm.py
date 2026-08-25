"""Compare standalone k-player unit RAPM with ordinary one-player RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import (
    game_margin_metrics,
    offense_defense_lineups,
)
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_unified_terminal_possessions,
)
from nba_impact.models.standalone_unit_rapm import (
    fit_unit_rapm,
    predict_unit_rapm,
    unit_slot_coverage,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "rapm/data/possession_cache"
POSSESSIONS = REPO_ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = REPO_ROOT / "data/lake/silver/possession_lineup_segments.parquet"
OUTPUT_ROOT = REPO_ROOT / "research/rapm_lab/outputs/standalone_unit_rapm"
ORDERS = (2, 3, 4, 5)
PENALTIES = (100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
MINIMUM_EXPOSURE = {2: 500, 3: 250, 4: 100, 5: 50}
HOME_PENALTY = 300.0
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260825


def _paired_game_bootstrap(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    paired = reference.merge(
        candidate,
        on="gameid",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if len(paired) != len(reference) or len(paired) != len(candidate):
        raise ValueError("Baseline and candidate must score identical games.")
    actual = paired["actual_margin_reference"].to_numpy(dtype=float)
    if not np.array_equal(actual, paired["actual_margin_candidate"].to_numpy(dtype=float)):
        raise ValueError("Baseline and candidate actual margins differ.")
    reference_error = actual - paired["predicted_margin_reference"].to_numpy(dtype=float)
    candidate_error = actual - paired["predicted_margin_candidate"].to_numpy(dtype=float)
    observed = float(
        np.sqrt(np.mean(candidate_error**2)) - np.sqrt(np.mean(reference_error**2))
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(draws, len(paired)))
    reference_rmse = np.sqrt(np.mean(reference_error[indices] ** 2, axis=1))
    candidate_rmse = np.sqrt(np.mean(candidate_error[indices] ** 2, axis=1))
    differences = candidate_rmse - reference_rmse
    return {
        "games": int(len(paired)),
        "observed_rmse_delta": observed,
        "rmse_delta_95_low": float(np.quantile(differences, 0.025)),
        "rmse_delta_95_high": float(np.quantile(differences, 0.975)),
        "probability_candidate_better": float(np.mean(differences < 0)),
        "draws": int(draws),
        "seed": int(seed),
    }


def _fit_reference(
    frame: pd.DataFrame,
    design,
    *,
    train_seasons: tuple[int, ...],
    test_season: int,
) -> tuple[np.ndarray, dict, pd.DataFrame]:
    train_mask = np.isin(design.seasons, train_seasons)
    test_mask = design.seasons == test_season
    config = RapmConfig(
        seasons=(*train_seasons, test_season),
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_home=HOME_PENALTY,
        data_scope="standalone_unit_rapm_reference",
    )
    beta, intercept = fit_coefficients(design, config, row_mask=train_mask)
    prediction = intercept + np.asarray(design.X[test_mask] @ beta).ravel()
    metrics, games = game_margin_metrics(
        frame.loc[test_mask].reset_index(drop=True), prediction
    )
    return prediction, metrics, games


def _stage(
    frame: pd.DataFrame,
    design,
    offense: np.ndarray,
    defense: np.ndarray,
    *,
    train_seasons: tuple[int, ...],
    test_season: int,
    selected_penalties: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, object], dict[int, dict]]:
    train_mask = np.isin(design.seasons, train_seasons)
    test_mask = design.seasons == test_season
    _, reference_metrics, reference_games = _fit_reference(
        frame,
        design,
        train_seasons=train_seasons,
        test_season=test_season,
    )
    stage = "selection" if selected_penalties is None else "diagnostic"
    metric_rows = [
        {
            "stage": stage,
            "train_seasons": "|".join(str(value) for value in train_seasons),
            "test_season": test_season,
            "order": 1,
            "penalty": np.nan,
            "minimum_exposure": np.nan,
            "units": len(design.players),
            "test_unit_slot_coverage": 1.0,
            "fit_iterations": np.nan,
            "fit_stop_code": np.nan,
            **reference_metrics,
        }
    ]
    game_frames = [reference_games.assign(order=1, penalty=np.nan)]
    fits: dict[int, object] = {}
    bootstraps: dict[int, dict] = {}
    for order in ORDERS:
        penalties = PENALTIES if selected_penalties is None else (selected_penalties[order],)
        for penalty in penalties:
            print(
                f"stage={stage} test={test_season} order={order} lambda={penalty:g}",
                flush=True,
            )
            fit = fit_unit_rapm(
                offense[train_mask],
                defense[train_mask],
                design.home_offense[train_mask],
                design.y[train_mask],
                order=order,
                unit_penalty=penalty,
                home_penalty=HOME_PENALTY,
                minimum_exposure=MINIMUM_EXPOSURE[order],
                maximum_iterations=500,
            )
            prediction = predict_unit_rapm(
                fit,
                offense[test_mask],
                defense[test_mask],
                design.home_offense[test_mask],
            )
            metrics, games = game_margin_metrics(
                frame.loc[test_mask].reset_index(drop=True), prediction
            )
            metric_rows.append(
                {
                    "stage": stage,
                    "train_seasons": "|".join(str(value) for value in train_seasons),
                    "test_season": test_season,
                    "order": order,
                    "penalty": penalty,
                    "minimum_exposure": MINIMUM_EXPOSURE[order],
                    "units": len(fit.combinations),
                    "test_unit_slot_coverage": unit_slot_coverage(
                        fit, offense[test_mask], defense[test_mask]
                    ),
                    "fit_iterations": fit.fit_iterations,
                    "fit_stop_code": fit.fit_stop_code,
                    **metrics,
                }
            )
            game_frames.append(games.assign(order=order, penalty=penalty))
            fits[order] = fit
            if selected_penalties is not None:
                bootstraps[order] = _paired_game_bootstrap(reference_games, games)
    return pd.DataFrame(metric_rows), pd.concat(game_frames, ignore_index=True), fits, bootstraps


def _coefficient_table(fits: dict[int, object]) -> pd.DataFrame:
    rows = []
    for order, fit in fits.items():
        n_units = len(fit.combinations)
        for index, players in enumerate(fit.combinations):
            offense = 100.0 * fit.coefficients[index]
            defense = -100.0 * fit.coefficients[n_units + index]
            rows.append(
                {
                    "order": order,
                    "players": "|".join(str(int(value)) for value in players),
                    "offense_per_100": offense,
                    "defense_per_100": defense,
                    "net_per_100": offense + defense,
                }
            )
    return pd.DataFrame(rows)


def run(output_root: Path = OUTPUT_ROOT) -> dict:
    seasons = tuple(range(2020, 2027))
    frame = load_unified_terminal_possessions(
        CACHE,
        POSSESSIONS,
        SEGMENTS,
        seasons,
        transition_season=2024,
        game_types=("regular",),
    )
    if frame["season"].max() >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    design = build_design(frame, include_home=True)
    offense, defense = offense_defense_lineups(frame)
    selection, selection_games, _, _ = _stage(
        frame,
        design,
        offense,
        defense,
        train_seasons=(2020, 2021, 2022, 2023, 2024),
        test_season=2025,
    )
    selected_penalties: dict[int, float] = {}
    for order in ORDERS:
        candidates = selection.loc[selection["order"].eq(order)]
        selected_penalties[order] = float(
            candidates.sort_values(["margin_rmse", "penalty"], kind="stable").iloc[0]["penalty"]
        )
    diagnostic, diagnostic_games, diagnostic_fits, bootstraps = _stage(
        frame,
        design,
        offense,
        defense,
        train_seasons=(2021, 2022, 2023, 2024, 2025),
        test_season=2026,
        selected_penalties=selected_penalties,
    )
    metrics = pd.concat([selection, diagnostic], ignore_index=True)
    games = pd.concat(
        [selection_games.assign(stage="selection"), diagnostic_games.assign(stage="diagnostic")],
        ignore_index=True,
    )
    source_hashes = {
        f"legacy_{season}": sha256_file(CACHE / f"matchups_{season}.parquet")
        for season in range(2020, 2024)
    }
    source_hashes.update(
        {"current_possessions": sha256_file(POSSESSIONS), "current_segments": sha256_file(SEGMENTS)}
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "sources": source_hashes,
                "penalties": PENALTIES,
                "minimum_exposure": MINIMUM_EXPOSURE,
                "selection_train": [2020, 2021, 2022, 2023, 2024],
                "diagnostic_train": [2021, 2022, 2023, 2024, 2025],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"standalone_unit_rapm_v1_{identity}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output / "metrics.parquet", index=False)
    games.to_parquet(output / "game_predictions.parquet", index=False)
    _coefficient_table(diagnostic_fits).to_parquet(
        output / "diagnostic_unit_ratings.parquet", index=False
    )
    reference_selection = selection.loc[selection["order"].eq(1)].iloc[0]
    reference_diagnostic = diagnostic.loc[diagnostic["order"].eq(1)].iloc[0]
    comparisons = []
    for order in ORDERS:
        selected = selection.loc[
            selection["order"].eq(order)
            & selection["penalty"].eq(selected_penalties[order])
        ].iloc[0]
        checked = diagnostic.loc[diagnostic["order"].eq(order)].iloc[0]
        comparisons.append(
            {
                "order": order,
                "selected_penalty": selected_penalties[order],
                "selection_rmse_delta_vs_one_player_rapm": float(
                    selected["margin_rmse"] - reference_selection["margin_rmse"]
                ),
                "diagnostic_rmse_delta_vs_one_player_rapm": float(
                    checked["margin_rmse"] - reference_diagnostic["margin_rmse"]
                ),
                "diagnostic_correlation_delta_vs_one_player_rapm": float(
                    checked["margin_correlation"] - reference_diagnostic["margin_correlation"]
                ),
                "diagnostic_test_unit_slot_coverage": float(
                    checked["test_unit_slot_coverage"]
                ),
                "paired_game_bootstrap": bootstraps[order],
            }
        )
    manifest = {
        "run_id": run_id,
        "status": "diagnostic_only",
        "model_family": "standalone_k_player_unit_rapm",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "retrospective_k_player_unit_points_per_100",
        "definition": (
            "Each challenger contains only k-player offensive units, k-player "
            "defensive units, and home advantage. It has no individual player "
            "columns and is not fitted to player-RAPM residuals."
        ),
        "config": {
            "loaded_seasons": list(seasons),
            "selection": {"train": [2020, 2021, 2022, 2023, 2024], "test": 2025},
            "diagnostic": {"train": [2021, 2022, 2023, 2024, 2025], "test": 2026},
            "reference_penalties": {"offense": 3000, "defense": 3000, "home": 300},
            "candidate_unit_penalties": list(PENALTIES),
            "home_penalty": HOME_PENALTY,
            "minimum_training_exposure": MINIMUM_EXPOSURE,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "source_hashes": source_hashes,
        "comparisons": comparisons,
        "paths": {
            "metrics": str((output / "metrics.parquet").relative_to(REPO_ROOT)),
            "game_predictions": str((output / "game_predictions.parquet").relative_to(REPO_ROOT)),
            "diagnostic_unit_ratings": str(
                (output / "diagnostic_unit_ratings.parquet").relative_to(REPO_ROOT)
            ),
        },
        "forbidden_interpretation": (
            "Individual player value, causal chemistry, or a deployable forecast."
        ),
        "caveats": [
            "The units overlap mechanically within each possession.",
            "Higher-order units are sparse and tied to lineup context.",
            "2025 selects shrinkage; 2026 is a reused diagnostic, not untouched confirmation.",
            "Season 2027 is not loaded.",
        ],
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
