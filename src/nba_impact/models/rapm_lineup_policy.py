"""Identical-row RAPM comparison for within-possession lineup assignment."""
from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.possessions import (
    AWAY_LINEUP_COLUMNS,
    HOME_LINEUP_COLUMNS,
)
from nba_impact.models.rapm import (
    RapmConfig,
    RapmDesign,
    _game_margin_frame,
    _game_margin_metrics,
    build_design,
    fit_coefficients,
    load_current_possessions,
    ratings_table,
)


def fractional_segment_weights(segments: pd.DataFrame) -> pd.Series:
    """Allocate a possession across lineup segments without using its outcome."""
    ordered = segments.sort_values(["possession_id", "segment_number"], kind="stable").copy()
    next_start = ordered.groupby("possession_id", sort=False)["start_seconds_elapsed"].shift(-1)
    interval_end = next_start.fillna(ordered["possession_end_seconds_elapsed"])
    duration = (interval_end - ordered["start_seconds_elapsed"]).clip(lower=0.0)
    duration_total = duration.groupby(ordered["possession_id"]).transform("sum")
    action_count = ordered["action_count"].astype(float)
    action_total = action_count.groupby(ordered["possession_id"]).transform("sum")
    if action_count.le(0).any() or action_total.le(0).any():
        raise ValueError("Every lineup segment must contain at least one action.")
    weights = np.where(duration_total.gt(0), duration / duration_total, action_count / action_total)
    result = pd.Series(weights, index=ordered.index, name="segment_weight")
    sums = result.groupby(ordered["possession_id"]).sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-10):
        raise ValueError("Fractional lineup weights do not sum to one per possession.")
    return result.reindex(segments.index)


def build_fractional_current_design(
    possessions_path: str | Path,
    segments_path: str | Path,
    *,
    game_types: tuple[str, ...] = ("regular",),
) -> RapmDesign:
    possessions = pd.read_parquet(possessions_path)
    possessions = possessions.loc[possessions["season_type"].isin(game_types)].sort_values(
        ["game_id", "possession_number"], kind="stable"
    ).reset_index(drop=True)
    possessions["possession_row"] = np.arange(len(possessions), dtype=np.int64)
    segments = pd.read_parquet(segments_path)
    segments = segments.loc[segments["possession_id"].isin(possessions["possession_id"])].copy()
    context = possessions[
        ["possession_id", "possession_row", "end_seconds_elapsed"]
    ].rename(columns={"end_seconds_elapsed": "possession_end_seconds_elapsed"})
    segments = segments.merge(context, on="possession_id", validate="many_to_one")
    segments["segment_weight"] = fractional_segment_weights(segments)

    home = segments[list(HOME_LINEUP_COLUMNS)].to_numpy(dtype=np.int64, copy=False)
    away = segments[list(AWAY_LINEUP_COLUMNS)].to_numpy(dtype=np.int64, copy=False)
    home_offense_segment = possessions.loc[
        segments["possession_row"].to_numpy(), "offense_is_home"
    ].to_numpy(dtype=bool)
    offense_players = np.where(home_offense_segment[:, None], home, away)
    defense_players = np.where(home_offense_segment[:, None], away, home)
    players = np.unique(np.concatenate([offense_players.ravel(), defense_players.ravel()]))
    players = np.asarray(sorted(int(player) for player in players), dtype=np.int64)
    offense_columns = np.searchsorted(players, offense_players)
    defense_columns = np.searchsorted(players, defense_players) + len(players)
    segment_rows = segments["possession_row"].to_numpy(dtype=np.int64)
    sparse_rows = np.repeat(segment_rows, 5)
    values = np.repeat(segments["segment_weight"].to_numpy(dtype=float), 5)
    home_offense = possessions["offense_is_home"].to_numpy(dtype=bool)
    matrix = csr_matrix(
        (
            np.concatenate([values, values, np.where(home_offense, 1.0, -1.0)]),
            (
                np.concatenate([sparse_rows, sparse_rows, np.arange(len(possessions))]),
                np.concatenate(
                    [
                        offense_columns.ravel(),
                        defense_columns.ravel(),
                        np.full(len(possessions), 2 * len(players), dtype=np.int64),
                    ]
                ),
            ),
        ),
        shape=(len(possessions), 2 * len(players) + 1),
    )
    if not np.allclose(np.asarray(matrix[:, : len(players)].sum(axis=1)).ravel(), 5.0):
        raise ValueError("Fractional offensive exposure does not sum to five players.")
    if not np.allclose(np.asarray(matrix[:, len(players) : 2 * len(players)].sum(axis=1)).ravel(), 5.0):
        raise ValueError("Fractional defensive exposure does not sum to five players.")
    return RapmDesign(
        X=matrix,
        y=possessions["points"].to_numpy(dtype=float),
        players=players,
        game_ids=possessions["game_id"].astype(str).to_numpy(),
        seasons=possessions["season_end"].to_numpy(dtype=np.int32),
        home_offense=home_offense,
        off_possessions=np.asarray(matrix[:, : len(players)].sum(axis=0)).ravel(),
        def_possessions=np.asarray(matrix[:, len(players) : 2 * len(players)].sum(axis=0)).ravel(),
    )


def _paired_policy_bootstrap(
    predictions: pd.DataFrame,
    candidate: str,
    baseline: str,
    *,
    repetitions: int,
    seed: int,
) -> dict:
    baseline_rows = predictions.loc[predictions["policy"].eq(baseline)]
    challenger = predictions.loc[predictions["policy"].eq(candidate)]
    paired = challenger.merge(
        baseline_rows,
        on="game_id",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if not np.allclose(paired["actual_margin_candidate"], paired["actual_margin_baseline"]):
        raise ValueError("Lineup policies are not scored on identical game outcomes.")
    actual = paired["actual_margin_candidate"].to_numpy(dtype=float)
    delta = (actual - paired["predicted_margin_candidate"].to_numpy()) ** 2 - (
        actual - paired["predicted_margin_baseline"].to_numpy()
    ) ** 2
    rng = np.random.default_rng(seed)
    draws = np.array(
        [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(repetitions)]
    )
    return {
        "candidate": candidate,
        "baseline": baseline,
        "games": int(len(delta)),
        "mean_game_squared_error_delta_candidate_minus_baseline": float(delta.mean()),
        "probability_candidate_better": float((draws < 0).mean()),
        "delta_ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
    }


def _rating_correlations(ratings: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for candidate in ("terminal", "fractional"):
        paired = ratings[candidate].merge(
            ratings["start"], on="player_id", suffixes=("_candidate", "_start"),
            validate="one_to_one",
        )
        rows.append(
            {
                "candidate": candidate,
                "players": int(len(paired)),
                **{
                    f"{metric}_correlation": float(
                        paired[[f"{metric}_candidate", f"{metric}_start"]].corr().iloc[0, 1]
                    )
                    for metric in ("offense_per_100", "defense_per_100", "net_per_100")
                },
            }
        )
    return rows


def run_rapm_lineup_policy_comparison(
    possessions_path: str | Path,
    segments_path: str | Path,
    *,
    artifact_root: str | Path,
    config: RapmConfig | None = None,
    bootstrap_repetitions: int = 5000,
    seed: int = 7,
) -> dict:
    config = config or RapmConfig(seasons=(2025, 2026))
    if tuple(config.seasons) != (2025, 2026):
        raise ValueError("Current lineup-policy comparison is frozen to seasons 2025 and 2026.")
    frames = {
        policy: load_current_possessions(
            possessions_path, segments_path, lineup_policy=policy, game_types=config.game_types
        )
        for policy in ("start", "terminal")
    }
    designs = {
        "start": build_design(frames["start"], include_home=config.include_home),
        "terminal": build_design(frames["terminal"], include_home=config.include_home),
        "fractional": build_fractional_current_design(
            possessions_path, segments_path, game_types=config.game_types
        ),
    }
    baseline_design = designs["start"]
    for policy, design in designs.items():
        identical = (
            np.array_equal(design.game_ids, baseline_design.game_ids)
            and np.array_equal(design.seasons, baseline_design.seasons)
            and np.array_equal(design.y, baseline_design.y)
        )
        if not identical:
            raise ValueError(
                f"Lineup policy {policy} does not contain identical ordered possession outcomes."
            )

    policy_metrics = {}
    game_predictions = []
    ratings = {}
    for policy, design in designs.items():
        train = design.seasons == 2025
        test = design.seasons == 2026
        beta, intercept = fit_coefficients(design, config, train)
        policy_metrics[policy] = _game_margin_metrics(design, beta, intercept, test, train)
        games = _game_margin_frame(design, beta, intercept, test, train)
        games["policy"] = policy
        game_predictions.append(games)
        ratings[policy] = ratings_table(design, beta)
    predictions = pd.concat(game_predictions, ignore_index=True)
    paired = {
        "terminal_vs_start": _paired_policy_bootstrap(
            predictions, "terminal", "start", repetitions=bootstrap_repetitions, seed=seed + 1
        ),
        "fractional_vs_start": _paired_policy_bootstrap(
            predictions, "fractional", "start", repetitions=bootstrap_repetitions, seed=seed + 2
        ),
        "fractional_vs_terminal": _paired_policy_bootstrap(
            predictions, "fractional", "terminal", repetitions=bootstrap_repetitions, seed=seed + 3
        ),
    }

    run_id = f"rapm_lineup_policy_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "rapm_lineup_policy" / run_id
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "game_predictions.parquet", index=False)
    for policy, table in ratings.items():
        table.to_parquet(output / f"ratings_{policy}.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "zero_prior_rapm_lineup_policy_comparison",
        "estimand": "lineup_conditioned_descriptive_points_per_100",
        "status": "research_diagnostic_single_outer_fold",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "train_season": 2025,
            "test_season": 2026,
            "fractional_policy": "elapsed-time share; action-count fallback for zero-clock-span possessions",
            "bootstrap_repetitions": bootstrap_repetitions,
            "seed": seed,
            "source_hashes": {
                "possessions": sha256_file(possessions_path),
                "segments": sha256_file(segments_path),
                "source_code": sha256_file(Path(__file__)),
                "rapm_source_code": sha256_file(Path(__file__).with_name("rapm.py")),
            },
        },
        "metrics": {
            "identical_possessions": int(len(designs["start"].y)),
            "train_possessions": int((designs["start"].seasons == 2025).sum()),
            "test_possessions": int((designs["start"].seasons == 2026).sum()),
            "policies": policy_metrics,
            "paired_game_bootstrap": paired,
            "rating_correlations_vs_start": _rating_correlations(ratings),
        },
        "caveats": [
            "Observed test-season lineups make this retrodiction, not a deployable forecast.",
            "Only one chronological outer fold is available.",
            "Fractional exposure is an attribution policy, not observed sub-second player tracking.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
