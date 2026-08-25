"""Test possession-context adjustments on rolling five-year RAPM ratings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, hstack
from scipy.sparse.linalg import cg, spsolve

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import build_design, load_unified_terminal_possessions


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "research" / "experiments" / "rolling_5y_context_adjustments_v1.json"
)
DEFAULT_CACHE = REPO_ROOT / "rapm" / "data" / "possession_cache"
DEFAULT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
DEFAULT_SEGMENTS = REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "research" / "rapm_lab" / "outputs" / "context_adjustments"
GT_BASE = {1: 25.0, 2: 20.0, 3: 17.0, 4: 12.0}


@dataclass(frozen=True)
class FitResult:
    beta: np.ndarray
    intercept: float
    n_players: int
    home_column: int | None
    context_columns: tuple[int, ...]
    training_rows: int
    dropped_rows: int


@dataclass(frozen=True)
class PreparedFold:
    X: csr_matrix
    y: np.ndarray
    evaluation: pd.DataFrame
    train_rows: int
    n_players: int
    column_names: tuple[str, ...]
    is_garbage: np.ndarray


@dataclass(frozen=True)
class TrainingStats:
    xtx: csr_matrix
    rhs: np.ndarray
    intercept: float
    training_rows: int
    dropped_rows: int


def annotate_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add pre-possession score and common clock-state fields without lookahead."""
    required = {"gameid", "period", "num", "pts", "home_poss"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Context input is missing columns: {missing}")
    result = frame.loc[:, sorted(required)].copy()
    result["_context_source_order"] = np.arange(len(result), dtype=np.int64)
    result = result.sort_values(["gameid", "period", "num"], kind="stable")
    home = result["home_poss"].astype(bool).to_numpy()
    points = pd.to_numeric(result["pts"], errors="raise").to_numpy(dtype=np.float64)
    result["home_points_on_row"] = np.where(home, points, 0.0)
    result["away_points_on_row"] = np.where(home, 0.0, points)
    grouped = result.groupby("gameid", sort=False)
    result["home_points_before"] = (
        grouped["home_points_on_row"].cumsum() - result["home_points_on_row"]
    )
    result["away_points_before"] = (
        grouped["away_points_on_row"].cumsum() - result["away_points_on_row"]
    )
    result["home_margin_before"] = (
        result["home_points_before"] - result["away_points_before"]
    )
    result["offense_margin_before"] = np.where(
        home, result["home_margin_before"], -result["home_margin_before"]
    )
    period_group = result.groupby(["gameid", "period"], sort=False)
    period_rank = period_group.cumcount().to_numpy(dtype=np.float64) + 1.0
    period_count = period_group["num"].transform("size").to_numpy(dtype=np.float64)
    result["quarter_progress"] = period_rank / period_count
    period = pd.to_numeric(result["period"], errors="raise").to_numpy(dtype=np.int64)
    period_bucket = np.minimum(period, 4)
    base = np.asarray([GT_BASE[int(value)] for value in period_bucket])
    threshold = base + 8.0 * (1.0 - result["quarter_progress"].to_numpy())
    result["is_garbage"] = np.abs(result["home_margin_before"].to_numpy()) >= threshold
    result["clock_fatigue"] = np.clip(
        0.8 * (period_bucket - 1.0) / 3.0
        + 0.2 * result["quarter_progress"].to_numpy(),
        0.0,
        1.0,
    )
    return (
        result.sort_values("_context_source_order", kind="stable")
        .drop(
            columns=[
                "home_points_on_row",
                "away_points_on_row",
                "_context_source_order",
            ]
        )
        .reset_index(drop=True)
    )


def context_matrix(
    annotated: pd.DataFrame,
    *,
    train_rows: int,
    quarter_rubberband: bool,
    clock_fatigue: bool,
) -> tuple[csr_matrix, tuple[str, ...]]:
    """Build context columns standardized from training rows only."""
    columns: list[np.ndarray] = []
    names: list[str] = []
    periods = np.minimum(
        pd.to_numeric(annotated["period"], errors="raise").to_numpy(dtype=np.int64), 4
    )
    margin = annotated["offense_margin_before"].to_numpy(dtype=np.float64)
    if quarter_rubberband:
        for quarter in range(1, 5):
            active = periods == quarter
            train_active = active[:train_rows]
            active_values = margin[:train_rows][train_active]
            scale = float(np.std(active_values)) if len(active_values) else 1.0
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0
            columns.append(np.where(active, margin / scale, 0.0))
            names.append(f"rubberband_q{quarter if quarter < 4 else '4plus'}")
    if clock_fatigue:
        values = annotated["clock_fatigue"].to_numpy(dtype=np.float64)
        mean = float(values[:train_rows].mean())
        scale = float(values[:train_rows].std())
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        columns.append((values - mean) / scale)
        names.append("clock_fatigue")
    if not columns:
        return csr_matrix((len(annotated), 0), dtype=np.float64), tuple()
    return csr_matrix(np.column_stack(columns)), tuple(names)


def _season_adjust_train(train: pd.DataFrame) -> pd.DataFrame:
    season_means = train.groupby("season")["pts"].mean()
    overall = float(train["pts"].mean())
    adjusted = train.copy()
    adjusted["pts"] = adjusted["pts"] - adjusted["season"].map(season_means) + overall
    return adjusted


def prepare_fold(train: pd.DataFrame, evaluation: pd.DataFrame) -> PreparedFold:
    """Build one superset design so every variant uses identical columns and rows."""
    train_rows = len(train)
    raw = pd.concat([train, evaluation], ignore_index=True)
    context_frame = annotate_context(raw)
    model_frame = raw.copy()
    model_frame["pts"] = pd.to_numeric(model_frame["pts"], errors="raise").astype(float)
    model_frame.loc[: train_rows - 1, "pts"] = (
        _season_adjust_train(train)["pts"].to_numpy()
    )
    base = build_design(model_frame, include_home=True)
    context, context_names = context_matrix(
        context_frame,
        train_rows=train_rows,
        quarter_rubberband=True,
        clock_fatigue=True,
    )
    names = (
        *(f"off_{player}" for player in base.players),
        *(f"def_{player}" for player in base.players),
        "home",
        *context_names,
    )
    return PreparedFold(
        X=hstack([base.X, context], format="csr"),
        y=base.y,
        evaluation=evaluation.reset_index(drop=True),
        train_rows=train_rows,
        n_players=len(base.players),
        column_names=tuple(names),
        is_garbage=context_frame["is_garbage"].to_numpy(dtype=bool),
    )


def training_stats(prepared: PreparedFold, *, garbage_filter: bool) -> TrainingStats:
    mask = np.arange(prepared.X.shape[0]) < prepared.train_rows
    if garbage_filter:
        mask[: prepared.train_rows] &= ~prepared.is_garbage[: prepared.train_rows]
    X_train = prepared.X[mask]
    y_train = prepared.y[mask]
    intercept = float(y_train.mean())
    return TrainingStats(
        xtx=(X_train.T @ X_train).tocsr(),
        rhs=np.asarray(X_train.T @ (y_train - intercept)).ravel(),
        intercept=intercept,
        training_rows=int(mask.sum()),
        dropped_rows=int(prepared.train_rows - mask.sum()),
    )


def fit_variant(
    prepared: PreparedFold,
    stats: TrainingStats,
    spec: dict,
    penalties: dict,
) -> tuple[FitResult, pd.DataFrame]:
    n_players = prepared.n_players
    selected_columns = list(range(2 * n_players))
    if spec["home"]:
        selected_columns.append(prepared.column_names.index("home"))
    if spec["quarter_rubberband"]:
        selected_columns.extend(
            index
            for index, name in enumerate(prepared.column_names)
            if name.startswith("rubberband_")
        )
    if spec["clock_fatigue"]:
        selected_columns.append(prepared.column_names.index("clock_fatigue"))
    selected = np.asarray(selected_columns, dtype=np.int64)
    context_count = 4 * int(bool(spec["quarter_rubberband"])) + int(
        bool(spec["clock_fatigue"])
    )
    penalty_values = np.concatenate(
        [
            np.full(n_players, float(penalties["lambda_off"])),
            np.full(n_players, float(penalties["lambda_def"])),
            np.asarray(
                [float(penalties["lambda_home"])] if spec["home"] else [],
                dtype=np.float64,
            ),
            np.full(context_count, float(penalties["lambda_context"])),
        ]
    )
    lhs = stats.xtx[selected][:, selected].tocsr() + diags(
        penalty_values, format="csr"
    )
    rhs = stats.rhs[selected]
    try:
        beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        beta = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(beta, dtype=np.float64)
    diagonal = stats.xtx.diagonal()
    off_counts = diagonal[:n_players]
    def_counts = diagonal[n_players : 2 * n_players]
    off_mean = float(np.average(beta[:n_players], weights=off_counts))
    def_mean = float(
        np.average(beta[n_players : 2 * n_players], weights=def_counts)
    )
    beta[:n_players] -= off_mean
    beta[n_players : 2 * n_players] -= def_mean
    intercept = stats.intercept + 5.0 * (off_mean + def_mean)
    home_column = 2 * n_players if spec["home"] else None
    context_start = 2 * n_players + int(bool(spec["home"]))
    fit = FitResult(
        beta=beta,
        intercept=intercept,
        n_players=n_players,
        home_column=home_column,
        context_columns=tuple(range(context_start, len(selected_columns))),
        training_rows=stats.training_rows,
        dropped_rows=stats.dropped_rows,
    )
    evaluation_design = prepared.X[prepared.train_rows :, selected].tocsr()
    return fit, _game_predictions(
        evaluation_design,
        prepared.evaluation,
        fit,
    )


def _game_predictions(
    evaluation_design: csr_matrix,
    evaluation: pd.DataFrame,
    fit: FitResult,
) -> pd.DataFrame:
    player_home_beta = fit.beta.copy()
    if fit.context_columns:
        player_home_beta[list(fit.context_columns)] = 0.0
    primary_rows = np.asarray(evaluation_design @ player_home_beta).ravel() + fit.intercept
    conditional_rows = np.asarray(evaluation_design @ fit.beta).ravel() + fit.intercept
    signs = np.where(evaluation["home_poss"].to_numpy(dtype=bool), 1.0, -1.0)
    scored = pd.DataFrame(
        {
            "game_id": evaluation["gameid"].astype(str).to_numpy(),
            "actual": signs * evaluation["pts"].to_numpy(dtype=np.float64),
            "primary": signs * primary_rows,
            "conditional": signs * conditional_rows,
        }
    )
    return scored.groupby("game_id", sort=True, as_index=False).sum(numeric_only=True)


def metrics(predictions: pd.DataFrame, prediction_column: str) -> dict[str, float | int]:
    actual = predictions["actual"].to_numpy(dtype=np.float64)
    predicted = predictions[prediction_column].to_numpy(dtype=np.float64)
    error = actual - predicted
    return {
        "games": int(len(predictions)),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
    }


def select_variant(summary: pd.DataFrame, tolerance: float = 0.0005) -> pd.Series:
    best = float(summary["selection_primary_correlation"].max())
    eligible = summary.loc[summary["selection_primary_correlation"] >= best - tolerance]
    return eligible.sort_values(
        ["selection_primary_rmse", "variant"], kind="stable"
    ).iloc[0]


def paired_bootstrap(
    predictions: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    seasons = [frame.reset_index(drop=True) for _, frame in predictions.groupby("season")]
    improvements = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        baseline_losses: list[np.ndarray] = []
        candidate_losses: list[np.ndarray] = []
        for frame in seasons:
            idx = rng.integers(0, len(frame), len(frame))
            actual = frame["actual"].to_numpy()[idx]
            baseline = frame["baseline"].to_numpy()[idx]
            candidate = frame["candidate"].to_numpy()[idx]
            baseline_losses.append((actual - baseline) ** 2)
            candidate_losses.append((actual - candidate) ** 2)
        improvements[draw] = np.mean(np.concatenate(baseline_losses)) - np.mean(
            np.concatenate(candidate_losses)
        )
    return {
        "draws": draws,
        "seed": seed,
        "mean_mse_improvement": float(improvements.mean()),
        "lower_95": float(np.quantile(improvements, 0.025)),
        "upper_95": float(np.quantile(improvements, 0.975)),
        "probability_mse_improvement": float(np.mean(improvements > 0.0)),
    }


def run_experiment(
    contract_path: Path,
    *,
    evaluation_seasons: tuple[int, ...] | None = None,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Context experiment requires a frozen research contract.")
    allowed = tuple(contract["selection_seasons"] + contract["reused_diagnostic_seasons"])
    seasons = evaluation_seasons or allowed
    if sorted(set(seasons) - set(allowed)):
        raise ValueError(f"Evaluation seasons must be within {allowed}.")
    if int(contract["untouched_confirmation_season"]) in seasons:
        raise ValueError("Season 2027 must remain untouched.")
    identity_payload = {
        "contract_hash": sha256_file(contract_path),
        "script_hash": sha256_file(Path(__file__)),
        "evaluation_seasons": list(seasons),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    mode = "full" if seasons == allowed else "pilot"
    output = output_root / f"context_adjustments_v1_{mode}_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "run.json"
    if completed.exists():
        return json.loads(completed.read_text())

    folds: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    checkpoints = output / "fold_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    started = time.perf_counter()
    for season in seasons:
        fold_started = time.perf_counter()
        fold_checkpoint = checkpoints / f"fold_{season}.parquet"
        prediction_checkpoint = checkpoints / f"predictions_{season}.parquet"
        if fold_checkpoint.exists() and prediction_checkpoint.exists():
            folds.extend(pd.read_parquet(fold_checkpoint).to_dict("records"))
            prediction_frames.append(pd.read_parquet(prediction_checkpoint))
            print(f"context fold {season}: checkpointed", flush=True)
            continue
        train_seasons = tuple(range(season - int(contract["window_seasons"]), season))
        train = load_unified_terminal_possessions(
            DEFAULT_CACHE,
            DEFAULT_POSSESSIONS,
            DEFAULT_SEGMENTS,
            train_seasons,
            game_types=("regular",),
        )
        evaluation = load_unified_terminal_possessions(
            DEFAULT_CACHE,
            DEFAULT_POSSESSIONS,
            DEFAULT_SEGMENTS,
            (season,),
            game_types=("regular",),
        )
        prepared = prepare_fold(train, evaluation)
        stats_by_filter = {
            False: training_stats(prepared, garbage_filter=False),
            True: training_stats(prepared, garbage_filter=True),
        }
        game_set: set[str] | None = None
        for variant, spec in contract["variants"].items():
            fit, predictions = fit_variant(
                prepared,
                stats_by_filter[bool(spec["garbage_filter"])],
                spec,
                contract["player_penalties"],
            )
            observed = set(predictions["game_id"].astype(str))
            if game_set is None:
                game_set = observed
            elif observed != game_set:
                raise ValueError("Variants did not score identical held-out games.")
            primary = metrics(predictions, "primary")
            conditional = metrics(predictions, "conditional")
            folds.append(
                {
                    "evaluation_season": int(season),
                    "train_start": int(train_seasons[0]),
                    "train_end": int(train_seasons[-1]),
                    "variant": variant,
                    "training_rows": fit.training_rows,
                    "dropped_training_rows": fit.dropped_rows,
                    **{f"primary_{key}": value for key, value in primary.items()},
                    **{f"conditional_{key}": value for key, value in conditional.items()},
                }
            )
            predictions = predictions.rename(
                columns={"primary": "predicted_margin"}
            )[["game_id", "actual", "predicted_margin"]]
            predictions.insert(0, "variant", variant)
            predictions.insert(0, "season", int(season))
            prediction_frames.append(predictions)
        pd.DataFrame(
            [row for row in folds if row["evaluation_season"] == int(season)]
        ).to_parquet(fold_checkpoint, index=False)
        pd.concat(
            prediction_frames[-len(contract["variants"]) :], ignore_index=True
        ).to_parquet(prediction_checkpoint, index=False)
        print(
            f"context fold {season}: {len(train):,} train rows, {len(evaluation):,} test rows, "
            f"{time.perf_counter() - fold_started:.1f}s",
            flush=True,
        )

    fold_frame = pd.DataFrame(folds)
    prediction_frame = pd.concat(prediction_frames, ignore_index=True)
    fold_frame.to_parquet(output / "fold_results.parquet", index=False)
    prediction_frame.to_parquet(output / "game_predictions.parquet", index=False)
    summary_rows: list[dict] = []
    selection = [season for season in contract["selection_seasons"] if season in seasons]
    diagnostics = [
        season for season in contract["reused_diagnostic_seasons"] if season in seasons
    ]
    for variant, frame in fold_frame.groupby("variant", sort=False):
        row: dict[str, str | float] = {"variant": variant}
        for scope, scoped_seasons in (("selection", selection), ("diagnostic", diagnostics)):
            scoped = frame.loc[frame["evaluation_season"].isin(scoped_seasons)]
            for kind in ("primary", "conditional"):
                for metric_name in ("correlation", "mae", "rmse"):
                    row[f"{scope}_{kind}_{metric_name}"] = (
                        float(scoped[f"{kind}_{metric_name}"].mean())
                        if len(scoped)
                        else float("nan")
                    )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(output / "variant_summary.parquet", index=False)

    selected = select_variant(summary) if set(selection) == set(contract["selection_seasons"]) else None
    bootstrap = None
    gate_passed = None
    if selected is not None and set(diagnostics) == set(contract["reused_diagnostic_seasons"]):
        candidate_name = str(selected["variant"])
        baseline_name = str(contract["evaluation"]["baseline"])
        diagnostic_predictions = prediction_frame.loc[
            prediction_frame["season"].isin(diagnostics)
        ]
        baseline = diagnostic_predictions.loc[
            diagnostic_predictions["variant"].eq(baseline_name)
        ].rename(columns={"predicted_margin": "baseline"})
        candidate = diagnostic_predictions.loc[
            diagnostic_predictions["variant"].eq(candidate_name)
        ].rename(columns={"predicted_margin": "candidate"})
        paired = baseline.merge(
            candidate[["season", "game_id", "actual", "candidate"]],
            on=["season", "game_id", "actual"],
            validate="one_to_one",
        )
        bootstrap = paired_bootstrap(
            paired,
            draws=int(contract["evaluation"]["paired_bootstrap_draws"]),
            seed=int(contract["evaluation"]["paired_bootstrap_seed"]),
        )
        baseline_summary = summary.loc[summary["variant"].eq(baseline_name)].iloc[0]
        candidate_summary = summary.loc[summary["variant"].eq(candidate_name)].iloc[0]
        gate_passed = bool(
            candidate_name != baseline_name
            and candidate_summary["diagnostic_primary_correlation"]
            >= baseline_summary["diagnostic_primary_correlation"]
            and candidate_summary["diagnostic_primary_rmse"]
            <= baseline_summary["diagnostic_primary_rmse"]
            and bootstrap["probability_mse_improvement"]
            >= float(contract["evaluation"]["required_probability_mse_improvement"])
        )

    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "research_challenger" if gate_passed else "research_null"
            if gate_passed is not None
            else "research_pilot_complete"
        ),
        "contract": str(contract_path.relative_to(REPO_ROOT)),
        "contract_hash": identity_payload["contract_hash"],
        "script_hash": identity_payload["script_hash"],
        "evaluation_seasons": list(seasons),
        "folds": int(len(seasons)),
        "variants": list(contract["variants"]),
        "selected_variant": str(selected["variant"]) if selected is not None else None,
        "gate_passed": gate_passed,
        "paired_diagnostic_bootstrap": bootstrap,
        "elapsed_seconds": float(time.perf_counter() - started),
        "season_2027_loaded": False,
        "artifacts": {
            "fold_results": "fold_results.parquet",
            "variant_summary": "variant_summary.parquet",
            "game_predictions": "game_predictions.parquet",
        },
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, completed)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--evaluation-seasons", type=int, nargs="*")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_experiment(
        args.contract.resolve(),
        evaluation_seasons=(
            tuple(args.evaluation_seasons) if args.evaluation_seasons else None
        ),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
