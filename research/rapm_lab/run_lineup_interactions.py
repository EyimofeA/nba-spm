"""Select and diagnose 2- through 5-player residual RAPM interactions."""

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
    fit_interaction_layer,
    game_margin_metrics,
    offense_defense_lineups,
    predict_interaction_layer,
)
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_unified_terminal_possessions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "rapm/data/possession_cache"
POSSESSIONS = REPO_ROOT / "data/lake/silver/possessions.parquet"
SEGMENTS = REPO_ROOT / "data/lake/silver/possession_lineup_segments.parquet"
OUTPUT_ROOT = REPO_ROOT / "research/rapm_lab/outputs/lineup_interactions"
ORDERS = (2, 3, 4, 5)
PENALTIES = (300.0, 1000.0, 3000.0, 10000.0)
MINIMUM_EXPOSURE = {2: 500, 3: 250, 4: 100, 5: 50}


def _stage(
    frame: pd.DataFrame,
    design,
    offense: np.ndarray,
    defense: np.ndarray,
    *,
    train_end: int,
    test_season: int,
    selected_penalties: dict[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, object]]:
    train_mask = design.seasons <= train_end
    test_mask = design.seasons == test_season
    config = RapmConfig(
        seasons=tuple(range(2022, train_end + 1)),
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_home=300.0,
        data_scope="research_lineup_interactions",
    )
    beta, intercept = fit_coefficients(design, config, row_mask=train_mask)
    baseline = intercept + np.asarray(design.X @ beta).ravel()
    residual = design.y[train_mask] - baseline[train_mask]
    baseline_metrics, baseline_games = game_margin_metrics(
        frame.loc[test_mask].reset_index(drop=True), baseline[test_mask]
    )
    metric_rows = [
        {
            "stage": "selection" if selected_penalties is None else "diagnostic",
            "train_end": train_end,
            "test_season": test_season,
            "order": 1,
            "penalty": np.nan,
            "minimum_exposure": np.nan,
            "combinations": len(design.players),
            "fit_iterations": np.nan,
            "fit_stop_code": np.nan,
            **baseline_metrics,
        }
    ]
    game_frames = [baseline_games.assign(order=1, penalty=np.nan)]
    fits: dict[int, object] = {}
    for order in ORDERS:
        penalties = PENALTIES if selected_penalties is None else (selected_penalties[order],)
        for penalty in penalties:
            print(
                f"stage={test_season} order={order} lambda={penalty:g}",
                flush=True,
            )
            fit = fit_interaction_layer(
                offense[train_mask],
                defense[train_mask],
                residual,
                order=order,
                penalty=penalty,
                minimum_exposure=MINIMUM_EXPOSURE[order],
                maximum_iterations=300,
            )
            interaction = predict_interaction_layer(
                fit, offense[test_mask], defense[test_mask]
            )
            metrics, games = game_margin_metrics(
                frame.loc[test_mask].reset_index(drop=True),
                baseline[test_mask] + interaction,
            )
            metric_rows.append(
                {
                    "stage": "selection" if selected_penalties is None else "diagnostic",
                    "train_end": train_end,
                    "test_season": test_season,
                    "order": order,
                    "penalty": penalty,
                    "minimum_exposure": MINIMUM_EXPOSURE[order],
                    "combinations": len(fit.combinations),
                    "fit_iterations": fit.fit_iterations,
                    "fit_stop_code": fit.fit_stop_code,
                    **metrics,
                }
            )
            game_frames.append(games.assign(order=order, penalty=penalty))
            fits[order] = fit
    return pd.DataFrame(metric_rows), pd.concat(game_frames, ignore_index=True), fits


def _coefficient_table(fits: dict[int, object], stage: str) -> pd.DataFrame:
    rows = []
    for order, fit in fits.items():
        n = len(fit.combinations)
        for index, players in enumerate(fit.combinations):
            rows.append(
                {
                    "stage": stage,
                    "order": order,
                    "players": "|".join(str(int(value)) for value in players),
                    "offense_residual_per_100": 100.0 * fit.coefficients[index],
                    "defense_residual_per_100": -100.0 * fit.coefficients[n + index],
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["net_residual_per_100"] = (
            result["offense_residual_per_100"] + result["defense_residual_per_100"]
        )
    return result


def run(output_root: Path = OUTPUT_ROOT) -> dict:
    frame = load_unified_terminal_possessions(
        CACHE,
        POSSESSIONS,
        SEGMENTS,
        tuple(range(2022, 2027)),
        transition_season=2024,
        game_types=("regular",),
    )
    if frame["season"].max() >= 2027:
        raise ValueError("Season 2027 must remain untouched.")
    design = build_design(frame, include_home=True)
    offense, defense = offense_defense_lineups(frame)
    selection, selection_games, _ = _stage(
        frame,
        design,
        offense,
        defense,
        train_end=2024,
        test_season=2025,
    )
    selected_penalties: dict[int, float] = {}
    for order in ORDERS:
        candidates = selection.loc[selection["order"].eq(order)]
        selected_penalties[order] = float(
            candidates.sort_values(["margin_rmse", "penalty"], kind="stable").iloc[0]["penalty"]
        )
    diagnostic, diagnostic_games, diagnostic_fits = _stage(
        frame,
        design,
        offense,
        defense,
        train_end=2025,
        test_season=2026,
        selected_penalties=selected_penalties,
    )
    metrics = pd.concat([selection, diagnostic], ignore_index=True)
    games = pd.concat(
        [selection_games.assign(stage="selection"), diagnostic_games.assign(stage="diagnostic")],
        ignore_index=True,
    )
    identity = hashlib.sha256(
        json.dumps(
            {
                "possession_source": sha256_file(POSSESSIONS),
                "segment_source": sha256_file(SEGMENTS),
                "penalties": PENALTIES,
                "minimum_exposure": MINIMUM_EXPOSURE,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:10]
    run_id = f"lineup_interactions_v1_{identity}"
    output = output_root / run_id
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(output / "metrics.parquet", index=False)
    games.to_parquet(output / "game_predictions.parquet", index=False)
    coefficients = _coefficient_table(diagnostic_fits, "diagnostic")
    coefficients.to_parquet(output / "diagnostic_coefficients.parquet", index=False)
    baseline_selection = selection.loc[selection["order"].eq(1)].iloc[0]
    baseline_diagnostic = diagnostic.loc[diagnostic["order"].eq(1)].iloc[0]
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
                "selection_rmse_delta_vs_player_rapm": float(
                    selected["margin_rmse"] - baseline_selection["margin_rmse"]
                ),
                "diagnostic_rmse_delta_vs_player_rapm": float(
                    checked["margin_rmse"] - baseline_diagnostic["margin_rmse"]
                ),
                "selection_correlation_delta_vs_player_rapm": float(
                    selected["margin_correlation"] - baseline_selection["margin_correlation"]
                ),
                "diagnostic_correlation_delta_vs_player_rapm": float(
                    checked["margin_correlation"] - baseline_diagnostic["margin_correlation"]
                ),
            }
        )
    manifest = {
        "run_id": run_id,
        "status": "diagnostic_only",
        "model_family": "player_rapm_plus_zero_centered_unit_residual_ridge",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimand": "incremental_lineup_unit_points_per_possession_beyond_player_rapm",
        "config": {
            "seasons": [2022, 2023, 2024, 2025, 2026],
            "selection": {"train": [2022, 2023, 2024], "test": 2025},
            "diagnostic": {"train": [2022, 2023, 2024, 2025], "test": 2026},
            "player_rapm_penalties": {"offense": 3000, "defense": 3000, "home": 300},
            "interaction_orders": list(ORDERS),
            "candidate_penalties": list(PENALTIES),
            "minimum_exposure": MINIMUM_EXPOSURE,
            "selected_penalties": selected_penalties,
        },
        "comparisons": comparisons,
        "paths": {
            "metrics": str((output / "metrics.parquet").relative_to(REPO_ROOT)),
            "game_predictions": str((output / "game_predictions.parquet").relative_to(REPO_ROOT)),
            "diagnostic_coefficients": str(
                (output / "diagnostic_coefficients.parquet").relative_to(REPO_ROOT)
            ),
        },
        "forbidden_interpretation": "Causal chemistry, portable combination talent, or production player value.",
        "caveats": [
            "The interaction layer fits residuals after player RAPM rather than jointly refitting all player and unit columns.",
            "Higher-order units are sparse and strongly lineup-context dependent.",
            "2025 selects one penalty per order; 2026 is a reused diagnostic, not untouched confirmation.",
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
