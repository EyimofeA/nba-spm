"""Test partially pooled team-specific home effects in rolling RAPM."""

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
from scipy.sparse import bmat, csr_matrix, diags, hstack
from scipy.sparse.linalg import cg, spsolve

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import build_design, load_unified_terminal_possessions


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "research" / "experiments" / "rolling_5y_team_home_v1.json"
)
DEFAULT_CACHE = REPO_ROOT / "rapm" / "data" / "possession_cache"
DEFAULT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
DEFAULT_SEGMENTS = REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"
DEFAULT_OLD_SCORES = (
    REPO_ROOT
    / "research"
    / "rapm_lab"
    / "data"
    / "team_home_scores_2014_2016"
    / "official_game_scores.parquet"
)
DEFAULT_CURRENT_SCORES = (
    REPO_ROOT
    / "data"
    / "lake"
    / "bronze"
    / "official_game_scores"
    / "official_game_scores.parquet"
)
DEFAULT_TEAM_DIM = REPO_ROOT / "data" / "lake" / "silver" / "team_dim.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "research" / "rapm_lab" / "outputs" / "team_home"


@dataclass(frozen=True)
class PreparedFold:
    X: csr_matrix
    y: np.ndarray
    evaluation: pd.DataFrame
    train_rows: int
    n_players: int
    teams: np.ndarray
    team_weights: np.ndarray
    xtx: csr_matrix
    rhs: np.ndarray
    intercept: float


@dataclass(frozen=True)
class FitResult:
    beta: np.ndarray
    intercept: float
    n_players: int
    teams: np.ndarray
    team_home_lambda: float | None


def load_home_team_map(
    old_scores_path: Path = DEFAULT_OLD_SCORES,
    current_scores_path: Path = DEFAULT_CURRENT_SCORES,
    team_dim_path: Path = DEFAULT_TEAM_DIM,
) -> pd.DataFrame:
    """Load one verified regular-season game-to-home-team map for 2014-26."""
    frames = []
    for path in (old_scores_path, current_scores_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing official game-score source: {path}")
        frame = pd.read_parquet(
            path,
            columns=["project_season", "season_type", "game_id", "home_team_id"],
        )
        frames.append(frame.loc[frame["season_type"].eq("regular")])
    games = pd.concat(frames, ignore_index=True)
    games["game_id"] = games["game_id"].astype(str).str.zfill(10)
    games["project_season"] = pd.to_numeric(
        games["project_season"], errors="raise"
    ).astype(int)
    games["home_team_id"] = pd.to_numeric(
        games["home_team_id"], errors="raise"
    ).astype("int64")
    games = games.loc[games["project_season"].between(2014, 2026)].copy()
    if games.duplicated("game_id").any():
        raise ValueError("Official home-team map has duplicate game IDs.")
    if sorted(games["project_season"].unique()) != list(range(2014, 2027)):
        raise ValueError("Official home-team map does not cover every 2014-26 season.")
    if games["home_team_id"].nunique() != 30:
        raise ValueError("Official home-team map must contain exactly 30 teams.")
    teams = pd.read_parquet(
        team_dim_path, columns=["team_id", "canonical_tricode"]
    ).rename(columns={"team_id": "home_team_id"})
    teams["home_team_id"] = pd.to_numeric(
        teams["home_team_id"], errors="raise"
    ).astype("int64")
    games = games.merge(teams, on="home_team_id", how="left", validate="many_to_one")
    if games["canonical_tricode"].isna().any():
        raise ValueError("Home-team map contains an unresolved team ID.")
    return games.rename(columns={"game_id": "gameid"})[
        ["gameid", "project_season", "home_team_id", "canonical_tricode"]
    ]


def attach_home_team(frame: pd.DataFrame, game_map: pd.DataFrame) -> pd.DataFrame:
    result = frame.merge(game_map, on="gameid", how="left", validate="many_to_one")
    if result["home_team_id"].isna().any():
        missing = result.loc[result["home_team_id"].isna(), "gameid"].nunique()
        raise ValueError(f"Home-team map is missing {missing} possession games.")
    mismatch = result["season"].astype(int).ne(result["project_season"].astype(int))
    if mismatch.any():
        raise ValueError("Home-team season disagrees with possession season.")
    result["home_team_id"] = result["home_team_id"].astype("int64")
    return result


def _season_adjust_train(train: pd.DataFrame) -> pd.DataFrame:
    season_means = train.groupby("season")["pts"].mean()
    overall = float(train["pts"].mean())
    adjusted = train.copy()
    adjusted["pts"] = adjusted["pts"] - adjusted["season"].map(season_means) + overall
    return adjusted


def prepare_fold(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    game_map: pd.DataFrame,
) -> PreparedFold:
    train_rows = len(train)
    raw = attach_home_team(
        pd.concat([train, evaluation], ignore_index=True), game_map
    )
    model_frame = raw.copy()
    model_frame["pts"] = pd.to_numeric(model_frame["pts"], errors="raise").astype(float)
    model_frame.loc[: train_rows - 1, "pts"] = _season_adjust_train(train)[
        "pts"
    ].to_numpy()
    base = build_design(model_frame, include_home=True)
    teams = np.asarray(sorted(raw["home_team_id"].unique()), dtype=np.int64)
    team_index = np.searchsorted(teams, raw["home_team_id"].to_numpy(dtype=np.int64))
    signs = np.where(raw["home_poss"].to_numpy(dtype=bool), 1.0, -1.0)
    team_design = csr_matrix(
        (
            signs,
            (np.arange(len(raw), dtype=np.int64), team_index),
        ),
        shape=(len(raw), len(teams)),
    )
    X = hstack([base.X, team_design], format="csr")
    X_train = X[:train_rows]
    y_train = base.y[:train_rows]
    intercept = float(y_train.mean())
    return PreparedFold(
        X=X,
        y=base.y,
        evaluation=raw.iloc[train_rows:].reset_index(drop=True),
        train_rows=train_rows,
        n_players=len(base.players),
        teams=teams,
        team_weights=np.asarray(
            (team_design[:train_rows].T @ team_design[:train_rows]).diagonal()
        ).ravel(),
        xtx=(X_train.T @ X_train).tocsr(),
        rhs=np.asarray(X_train.T @ (y_train - intercept)).ravel(),
        intercept=intercept,
    )


def fit_model(
    prepared: PreparedFold,
    *,
    lambda_off: float,
    lambda_def: float,
    lambda_global_home: float,
    team_home_lambda: float | None,
) -> FitResult:
    n_players = prepared.n_players
    base_columns = 2 * n_players + 1
    selected = np.arange(
        base_columns + (len(prepared.teams) if team_home_lambda is not None else 0),
        dtype=np.int64,
    )
    penalty = np.concatenate(
        [
            np.full(n_players, lambda_off),
            np.full(n_players, lambda_def),
            np.asarray([lambda_global_home]),
            (
                np.full(len(prepared.teams), team_home_lambda)
                if team_home_lambda is not None
                else np.asarray([], dtype=np.float64)
            ),
        ]
    )
    lhs = prepared.xtx[selected][:, selected].tocsr() + diags(penalty, format="csr")
    rhs = prepared.rhs[selected]
    if team_home_lambda is None:
        try:
            beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
        except TypeError:
            beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
        if info != 0:
            beta = spsolve(lhs.tocsc(), rhs)
    else:
        # The global signed-home column is exactly the sum of the raw team
        # columns. Enforce a weighted-zero team deviation inside the solve;
        # post-fit centering would preserve predictions but would not preserve
        # the intended independent global/team penalties.
        constraint = np.zeros(len(selected), dtype=np.float64)
        constraint[base_columns:] = prepared.team_weights / prepared.team_weights.sum()
        c = csr_matrix(constraint[:, None])
        kkt = bmat([[lhs, c], [c.T, None]], format="csc")
        solution = spsolve(kkt, np.concatenate([rhs, [0.0]]))
        beta = solution[:-1]
    beta = np.asarray(beta, dtype=np.float64)
    diagonal = prepared.xtx.diagonal()
    off_mean = float(np.average(beta[:n_players], weights=diagonal[:n_players]))
    def_mean = float(
        np.average(
            beta[n_players : 2 * n_players],
            weights=diagonal[n_players : 2 * n_players],
        )
    )
    beta[:n_players] -= off_mean
    beta[n_players : 2 * n_players] -= def_mean
    intercept = prepared.intercept + 5.0 * (off_mean + def_mean)
    if team_home_lambda is not None:
        team_start = base_columns
        constrained_mean = float(
            np.average(beta[team_start:], weights=prepared.team_weights)
        )
        if abs(constrained_mean) > 1e-8:
            raise ValueError("Team-home deviations failed the weighted-zero constraint.")
    return FitResult(
        beta=beta,
        intercept=intercept,
        n_players=n_players,
        teams=prepared.teams,
        team_home_lambda=team_home_lambda,
    )


def game_predictions(prepared: PreparedFold, fit: FitResult) -> pd.DataFrame:
    columns = len(fit.beta)
    rows = np.asarray(prepared.X[prepared.train_rows :, :columns] @ fit.beta).ravel()
    rows += fit.intercept
    signs = np.where(
        prepared.evaluation["home_poss"].to_numpy(dtype=bool), 1.0, -1.0
    )
    scored = pd.DataFrame(
        {
            "game_id": prepared.evaluation["gameid"].astype(str),
            "actual": signs * prepared.evaluation["pts"].to_numpy(dtype=np.float64),
            "predicted": signs * rows,
        }
    )
    return scored.groupby("game_id", sort=True, as_index=False).sum(numeric_only=True)


def score(predictions: pd.DataFrame) -> dict[str, float | int]:
    actual = predictions["actual"].to_numpy(dtype=np.float64)
    predicted = predictions["predicted"].to_numpy(dtype=np.float64)
    error = actual - predicted
    variance = float(np.var(predicted))
    slope = (
        float(np.cov(actual, predicted, ddof=0)[0, 1] / variance)
        if variance > 0
        else float("nan")
    )
    return {
        "games": int(len(predictions)),
        "correlation": float(np.corrcoef(actual, predicted)[0, 1]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(error**2))),
        "calibration_intercept": float(actual.mean() - slope * predicted.mean()),
        "calibration_slope": slope,
    }


def team_effects(
    fit: FitResult,
    team_labels: dict[int, str],
    *,
    evaluation_season: int,
    variant: str,
) -> pd.DataFrame:
    global_home = float(fit.beta[2 * fit.n_players])
    deviations = (
        fit.beta[2 * fit.n_players + 1 :]
        if fit.team_home_lambda is not None
        else np.zeros(len(fit.teams), dtype=np.float64)
    )
    return pd.DataFrame(
        {
            "evaluation_season": evaluation_season,
            "variant": variant,
            "team_home_lambda": fit.team_home_lambda,
            "team_id": fit.teams,
            "team": [team_labels[int(team)] for team in fit.teams],
            "global_home_coefficient": global_home,
            "team_home_deviation": deviations,
            "home_net_advantage_per_100": 200.0 * (global_home + deviations),
        }
    )


def config_id(team_home_lambda: float | None) -> str:
    return "baseline_global" if team_home_lambda is None else f"team_lambda_{team_home_lambda:g}"


def select_candidate(summary: pd.DataFrame, *, tolerance: float) -> pd.Series:
    candidates = summary.loc[summary["team_home_lambda"].notna()].copy()
    best = float(candidates["mean_rmse"].min())
    eligible = candidates.loc[candidates["mean_rmse"] <= best + tolerance]
    return eligible.sort_values("team_home_lambda", ascending=False, kind="stable").iloc[0]


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
        baseline_losses = []
        candidate_losses = []
        for frame in seasons:
            indices = rng.integers(0, len(frame), len(frame))
            actual = frame["actual"].to_numpy()[indices]
            baseline = frame["baseline"].to_numpy()[indices]
            candidate = frame["candidate"].to_numpy()[indices]
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


def _fold(
    season: int,
    lambdas: tuple[float | None, ...],
    contract: dict,
    game_map: pd.DataFrame,
    team_labels: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    started = time.perf_counter()
    window = int(contract["window_seasons"])
    train_seasons = tuple(range(season - window, season))
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
    prepared = prepare_fold(train, evaluation, game_map)
    rows = []
    predictions = []
    effects = []
    model = contract["model"]
    game_set = None
    for team_lambda in lambdas:
        variant = config_id(team_lambda)
        fit = fit_model(
            prepared,
            lambda_off=float(model["lambda_off"]),
            lambda_def=float(model["lambda_def"]),
            lambda_global_home=float(model["lambda_global_home"]),
            team_home_lambda=team_lambda,
        )
        predicted = game_predictions(prepared, fit)
        observed = set(predicted["game_id"])
        if game_set is None:
            game_set = observed
        elif observed != game_set:
            raise ValueError("Team-home variants did not score identical games.")
        rows.append(
            {
                "evaluation_season": season,
                "train_start": train_seasons[0],
                "train_end": train_seasons[-1],
                "variant": variant,
                "team_home_lambda": team_lambda,
                **score(predicted),
            }
        )
        predicted.insert(0, "variant", variant)
        predicted.insert(0, "season", season)
        predictions.append(predicted)
        effects.append(
            team_effects(
                fit,
                team_labels,
                evaluation_season=season,
                variant=variant,
            )
        )
    return (
        pd.DataFrame(rows),
        pd.concat(predictions, ignore_index=True),
        pd.concat(effects, ignore_index=True),
        time.perf_counter() - started,
    )


def _summary(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby(["variant", "team_home_lambda"], dropna=False, as_index=False)
        .agg(
            seasons=("evaluation_season", "nunique"),
            mean_correlation=("correlation", "mean"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_calibration_intercept=("calibration_intercept", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
    )


def run_experiment(
    contract_path: Path,
    *,
    pilot_season: int | None = None,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Team-home experiment requires a frozen contract.")
    if pilot_season == int(contract["untouched_confirmation_season"]):
        raise ValueError("Season 2027 must remain untouched.")
    identity_payload = {
        "contract_hash": sha256_file(contract_path),
        "script_hash": sha256_file(Path(__file__)),
        "old_scores_hash": sha256_file(DEFAULT_OLD_SCORES),
        "current_scores_hash": sha256_file(DEFAULT_CURRENT_SCORES),
        "team_dim_hash": sha256_file(DEFAULT_TEAM_DIM),
        "pilot_season": pilot_season,
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    mode = f"pilot_{pilot_season}" if pilot_season else "full"
    output = output_root / f"team_home_v1_{mode}_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "run.json"
    if completed.exists():
        return json.loads(completed.read_text())
    game_map = load_home_team_map()
    team_labels = (
        game_map[["home_team_id", "canonical_tricode"]]
        .drop_duplicates("home_team_id")
        .set_index("home_team_id")["canonical_tricode"]
        .to_dict()
    )
    model = contract["model"]
    candidate_lambdas = tuple(float(value) for value in model["team_home_lambdas"])
    all_lambdas: tuple[float | None, ...] = (None, *candidate_lambdas)
    fold_frames = []
    prediction_frames = []
    effect_frames = []
    started = time.perf_counter()

    if pilot_season is not None:
        folds, predictions, effects, elapsed = _fold(
            pilot_season, all_lambdas, contract, game_map, team_labels
        )
        fold_frames.append(folds)
        prediction_frames.append(predictions)
        effect_frames.append(effects)
        print(f"team-home pilot {pilot_season}: {elapsed:.1f}s", flush=True)
        selected = None
        diagnostic_summary = None
        bootstrap = None
        gate = None
    else:
        for season in contract["selection_seasons"]:
            folds, predictions, effects, elapsed = _fold(
                int(season), all_lambdas, contract, game_map, team_labels
            )
            fold_frames.append(folds)
            prediction_frames.append(predictions)
            effect_frames.append(effects)
            print(f"team-home selection {season}: {elapsed:.1f}s", flush=True)
        selection_folds = pd.concat(fold_frames, ignore_index=True)
        selection_summary = _summary(selection_folds)
        selected = select_candidate(
            selection_summary,
            tolerance=float(contract["selection"]["rmse_tolerance"]),
        )
        selected_lambda = float(selected["team_home_lambda"])
        for season in contract["reused_diagnostic_seasons"]:
            folds, predictions, effects, elapsed = _fold(
                int(season), (None, selected_lambda), contract, game_map, team_labels
            )
            fold_frames.append(folds)
            prediction_frames.append(predictions)
            effect_frames.append(effects)
            print(f"team-home diagnostic {season}: {elapsed:.1f}s", flush=True)
        all_folds = pd.concat(fold_frames, ignore_index=True)
        diagnostic_folds = all_folds.loc[
            all_folds["evaluation_season"].isin(contract["reused_diagnostic_seasons"])
        ]
        diagnostic_summary = _summary(diagnostic_folds)
        all_predictions = pd.concat(prediction_frames, ignore_index=True)
        diagnostic_predictions = all_predictions.loc[
            all_predictions["season"].isin(contract["reused_diagnostic_seasons"])
        ]
        baseline = diagnostic_predictions.loc[
            diagnostic_predictions["variant"].eq("baseline_global")
        ].rename(columns={"predicted": "baseline"})
        candidate_name = config_id(selected_lambda)
        candidate = diagnostic_predictions.loc[
            diagnostic_predictions["variant"].eq(candidate_name)
        ].rename(columns={"predicted": "candidate"})
        paired = baseline.merge(
            candidate[["season", "game_id", "actual", "candidate"]],
            on=["season", "game_id", "actual"],
            validate="one_to_one",
        )
        bootstrap = paired_bootstrap(
            paired,
            draws=int(contract["promotion_gate"]["paired_bootstrap_draws"]),
            seed=int(contract["promotion_gate"]["paired_bootstrap_seed"]),
        )
        baseline_selection = selection_folds.loc[
            selection_folds["variant"].eq("baseline_global")
        ].set_index("evaluation_season")
        candidate_selection = selection_folds.loc[
            selection_folds["variant"].eq(candidate_name)
        ].set_index("evaluation_season")
        baseline_diagnostic = diagnostic_folds.loc[
            diagnostic_folds["variant"].eq("baseline_global")
        ].set_index("evaluation_season")
        candidate_diagnostic = diagnostic_folds.loc[
            diagnostic_folds["variant"].eq(candidate_name)
        ].set_index("evaluation_season")
        selection_win_fraction = float(
            (candidate_selection["rmse"] < baseline_selection["rmse"]).mean()
        )
        diagnostic_win_fraction = float(
            (candidate_diagnostic["rmse"] < baseline_diagnostic["rmse"]).mean()
        )
        base_diag = diagnostic_summary.loc[
            diagnostic_summary["variant"].eq("baseline_global")
        ].iloc[0]
        cand_diag = diagnostic_summary.loc[
            diagnostic_summary["variant"].eq(candidate_name)
        ].iloc[0]
        relative_rmse = float(
            (base_diag["mean_rmse"] - cand_diag["mean_rmse"])
            / base_diag["mean_rmse"]
        )
        gate_contract = contract["promotion_gate"]
        gate = {
            "relative_diagnostic_rmse_improvement": relative_rmse,
            "selection_fold_win_fraction": selection_win_fraction,
            "diagnostic_fold_win_fraction": diagnostic_win_fraction,
            "diagnostic_mae_change": float(cand_diag["mean_mae"] - base_diag["mean_mae"]),
            "diagnostic_correlation_change": float(
                cand_diag["mean_correlation"] - base_diag["mean_correlation"]
            ),
        }
        gate["passed"] = bool(
            relative_rmse
            >= float(gate_contract["minimum_relative_diagnostic_rmse_improvement"])
            and gate["diagnostic_mae_change"]
            <= float(gate_contract["maximum_diagnostic_mae_increase"])
            and gate["diagnostic_correlation_change"]
            >= -float(gate_contract["maximum_diagnostic_correlation_decline"])
            and selection_win_fraction
            >= float(gate_contract["minimum_selection_fold_win_fraction"])
            and diagnostic_win_fraction
            >= float(gate_contract["minimum_diagnostic_fold_win_fraction"])
            and bootstrap["probability_mse_improvement"]
            >= float(gate_contract["required_probability_mse_improvement"])
        )

    all_folds = pd.concat(fold_frames, ignore_index=True)
    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    all_effects = pd.concat(effect_frames, ignore_index=True)
    all_folds.to_parquet(output / "fold_results.parquet", index=False)
    all_predictions.to_parquet(output / "game_predictions.parquet", index=False)
    all_effects.to_parquet(output / "team_effects.parquet", index=False)
    _summary(
        all_folds.loc[
            all_folds["evaluation_season"].isin(contract["selection_seasons"])
        ]
    ).to_parquet(output / "selection_summary.parquet", index=False)
    if diagnostic_summary is not None:
        diagnostic_summary.to_parquet(output / "diagnostic_summary.parquet", index=False)
    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": (
            "research_pilot_complete"
            if pilot_season is not None
            else "research_challenger"
            if gate and gate["passed"]
            else "research_null"
        ),
        "contract": str(contract_path.relative_to(REPO_ROOT)),
        "hashes": identity_payload,
        "pilot_season": pilot_season,
        "selected_team_home_lambda": (
            float(selected["team_home_lambda"]) if selected is not None else None
        ),
        "gate": gate,
        "paired_diagnostic_bootstrap": bootstrap,
        "elapsed_seconds": float(time.perf_counter() - started),
        "season_2027_loaded": False,
        "artifacts": {
            "fold_results": "fold_results.parquet",
            "selection_summary": "selection_summary.parquet",
            "diagnostic_summary": (
                "diagnostic_summary.parquet" if diagnostic_summary is not None else None
            ),
            "game_predictions": "game_predictions.parquet",
            "team_effects": "team_effects.parquet",
        },
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, completed)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--pilot-season", type=int)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = run_experiment(
        args.contract.resolve(),
        pilot_season=args.pilot_season,
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
