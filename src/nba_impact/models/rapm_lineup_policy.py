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

DEFAULT_OUTER_FOLDS = ((2024, 2025), (2025, 2026))


def fractional_segment_weights(segments: pd.DataFrame) -> pd.Series:
    """Allocate a possession across lineup segments without using its outcome."""
    ordered = segments.sort_values(
        ["possession_id", "segment_number"], kind="stable"
    ).copy()
    next_start = ordered.groupby("possession_id", sort=False)[
        "start_seconds_elapsed"
    ].shift(-1)
    interval_end = next_start.fillna(ordered["possession_end_seconds_elapsed"])
    duration = (interval_end - ordered["start_seconds_elapsed"]).clip(lower=0.0)
    duration_total = duration.groupby(ordered["possession_id"]).transform("sum")
    action_count = ordered["action_count"].astype(float)
    action_total = action_count.groupby(ordered["possession_id"]).transform("sum")
    if action_count.le(0).any() or action_total.le(0).any():
        raise ValueError("Every lineup segment must contain at least one action.")
    weights = np.where(
        duration_total.gt(0), duration / duration_total, action_count / action_total
    )
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
    seasons: tuple[int, ...] | None = None,
    include_home: bool = True,
) -> RapmDesign:
    possessions = pd.read_parquet(possessions_path)
    possessions = (
        possessions.loc[possessions["season_type"].isin(game_types)]
        .sort_values(["game_id", "possession_number"], kind="stable")
        .reset_index(drop=True)
    )
    if seasons is not None:
        possessions = possessions.loc[
            possessions["season_end"].isin(seasons)
        ].reset_index(drop=True)
    possessions["possession_row"] = np.arange(len(possessions), dtype=np.int64)
    segments = pd.read_parquet(segments_path)
    segments = segments.loc[
        segments["possession_id"].isin(possessions["possession_id"])
    ].copy()
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
    players = np.unique(
        np.concatenate([offense_players.ravel(), defense_players.ravel()])
    )
    players = np.asarray(sorted(int(player) for player in players), dtype=np.int64)
    offense_columns = np.searchsorted(players, offense_players)
    defense_columns = np.searchsorted(players, defense_players) + len(players)
    segment_rows = segments["possession_row"].to_numpy(dtype=np.int64)
    sparse_rows = np.repeat(segment_rows, 5)
    values = np.repeat(segments["segment_weight"].to_numpy(dtype=float), 5)
    home_offense = possessions["offense_is_home"].to_numpy(dtype=bool)
    row_parts = [sparse_rows, sparse_rows]
    column_parts = [offense_columns.ravel(), defense_columns.ravel()]
    value_parts = [values, values]
    if include_home:
        row_parts.append(np.arange(len(possessions)))
        column_parts.append(np.full(len(possessions), 2 * len(players), dtype=np.int64))
        value_parts.append(np.where(home_offense, 1.0, -1.0))
    matrix = csr_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(len(possessions), 2 * len(players) + int(include_home)),
    )
    if not np.allclose(np.asarray(matrix[:, : len(players)].sum(axis=1)).ravel(), 5.0):
        raise ValueError("Fractional offensive exposure does not sum to five players.")
    if not np.allclose(
        np.asarray(matrix[:, len(players) : 2 * len(players)].sum(axis=1)).ravel(), 5.0
    ):
        raise ValueError("Fractional defensive exposure does not sum to five players.")
    return RapmDesign(
        X=matrix,
        y=possessions["points"].to_numpy(dtype=float),
        players=players,
        game_ids=possessions["game_id"].astype(str).to_numpy(),
        seasons=possessions["season_end"].to_numpy(dtype=np.int32),
        home_offense=home_offense,
        off_possessions=np.asarray(matrix[:, : len(players)].sum(axis=0)).ravel(),
        def_possessions=np.asarray(
            matrix[:, len(players) : 2 * len(players)].sum(axis=0)
        ).ravel(),
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
    join_columns = ["game_id"]
    if "outer_fold" in predictions.columns:
        join_columns.insert(0, "outer_fold")
    paired = challenger.merge(
        baseline_rows,
        on=join_columns,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if not np.allclose(
        paired["actual_margin_candidate"], paired["actual_margin_baseline"]
    ):
        raise ValueError("Lineup policies are not scored on identical game outcomes.")
    actual = paired["actual_margin_candidate"].to_numpy(dtype=float)
    delta = (actual - paired["predicted_margin_candidate"].to_numpy()) ** 2 - (
        actual - paired["predicted_margin_baseline"].to_numpy()
    ) ** 2
    rng = np.random.default_rng(seed)
    draws = np.array(
        [
            rng.choice(delta, size=len(delta), replace=True).mean()
            for _ in range(repetitions)
        ]
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
            ratings["start"],
            on="player_id",
            suffixes=("_candidate", "_start"),
            validate="one_to_one",
        )
        rows.append(
            {
                "candidate": candidate,
                "players": int(len(paired)),
                **{
                    f"{metric}_correlation": float(
                        paired[[f"{metric}_candidate", f"{metric}_start"]]
                        .corr()
                        .iloc[0, 1]
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
    folds: tuple[tuple[int, int], ...] = DEFAULT_OUTER_FOLDS,
    bootstrap_repetitions: int = 5000,
    seed: int = 7,
) -> dict:
    required_seasons = tuple(sorted({season for fold in folds for season in fold}))
    config = config or RapmConfig(seasons=required_seasons)
    if not folds:
        raise ValueError("At least one chronological outer fold is required.")
    if any(train_season >= test_season for train_season, test_season in folds):
        raise ValueError("Each outer fold must train before it tests.")
    if not set(required_seasons).issubset(config.seasons):
        raise ValueError("RAPM config seasons must contain every outer-fold season.")
    frames = {
        policy: load_current_possessions(
            possessions_path,
            segments_path,
            lineup_policy=policy,
            game_types=config.game_types,
        )
        for policy in ("start", "terminal")
    }
    frames = {
        policy: frame.loc[frame["season"].isin(config.seasons)]
        .sort_values(["gameid", "num"], kind="stable")
        .reset_index(drop=True)
        for policy, frame in frames.items()
    }
    designs = {
        "start": build_design(frames["start"], include_home=config.include_home),
        "terminal": build_design(frames["terminal"], include_home=config.include_home),
        "fractional": build_fractional_current_design(
            possessions_path,
            segments_path,
            game_types=config.game_types,
            seasons=config.seasons,
            include_home=config.include_home,
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

    fold_metrics = []
    game_predictions = []
    rating_outputs: dict[tuple[int, str], pd.DataFrame] = {}
    comparisons = (
        ("terminal_vs_start", "terminal", "start"),
        ("fractional_vs_start", "fractional", "start"),
        ("fractional_vs_terminal", "fractional", "terminal"),
    )
    for fold_number, (train_season, test_season) in enumerate(folds, start=1):
        outer_fold = f"{train_season}_to_{test_season}"
        policy_metrics = {}
        fold_predictions = []
        fold_ratings = {}
        for policy, design in designs.items():
            train = design.seasons == train_season
            test = design.seasons == test_season
            if not train.any() or not test.any():
                raise ValueError(
                    f"Outer fold {outer_fold} has no train or test possessions."
                )
            beta, intercept = fit_coefficients(design, config, train)
            policy_metrics[policy] = _game_margin_metrics(
                design, beta, intercept, test, train
            )
            games = _game_margin_frame(design, beta, intercept, test, train)
            games["policy"] = policy
            games["outer_fold"] = outer_fold
            games["train_season"] = train_season
            games["test_season"] = test_season
            fold_predictions.append(games)
            fold_ratings[policy] = ratings_table(design, beta)
            rating_outputs[(fold_number, policy)] = fold_ratings[policy]
        fold_prediction_frame = pd.concat(fold_predictions, ignore_index=True)
        game_predictions.append(fold_prediction_frame)
        paired = {
            name: _paired_policy_bootstrap(
                fold_prediction_frame,
                candidate,
                baseline,
                repetitions=bootstrap_repetitions,
                seed=seed + fold_number * 10 + comparison_number,
            )
            for comparison_number, (name, candidate, baseline) in enumerate(
                comparisons, start=1
            )
        }
        fold_metrics.append(
            {
                "outer_fold": outer_fold,
                "train_season": train_season,
                "test_season": test_season,
                "train_possessions": int(
                    (baseline_design.seasons == train_season).sum()
                ),
                "test_possessions": int((baseline_design.seasons == test_season).sum()),
                "policies": policy_metrics,
                "paired_game_bootstrap": paired,
                "rating_correlations_vs_start": _rating_correlations(fold_ratings),
            }
        )
    predictions = pd.concat(game_predictions, ignore_index=True)
    pooled_paired = {
        name: _paired_policy_bootstrap(
            predictions,
            candidate,
            baseline,
            repetitions=bootstrap_repetitions,
            seed=seed + 100 + comparison_number,
        )
        for comparison_number, (name, candidate, baseline) in enumerate(
            comparisons, start=1
        )
    }

    run_id = f"rapm_lineup_policy_v2_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "rapm_lineup_policy" / run_id
    output.mkdir(parents=True, exist_ok=False)
    predictions.to_parquet(output / "game_predictions.parquet", index=False)
    for (fold_number, policy), table in rating_outputs.items():
        table.to_parquet(
            output / f"ratings_fold_{fold_number}_{policy}.parquet", index=False
        )
    run = {
        "run_id": run_id,
        "model_family": "zero_prior_rapm_lineup_policy_comparison",
        "estimand": "lineup_conditioned_descriptive_points_per_100",
        "status": "research_diagnostic_two_outer_folds",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "outer_folds": [
                {"train_season": train_season, "test_season": test_season}
                for train_season, test_season in folds
            ],
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
            "folds": fold_metrics,
            "pooled_paired_game_bootstrap": pooled_paired,
        },
        "caveats": [
            "Observed test-season lineups make this retrodiction, not a deployable forecast.",
            "The comparison uses two one-season-train chronological outer folds.",
            "Fractional exposure is an attribution policy, not observed sub-second player tracking.",
        ],
        "artifact_path": str(output.resolve()),
    }
    write_json_atomic(run, output / "run.json")
    return run
