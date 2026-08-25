"""Compare clock and possession-progress rubber-band adjusted RAPM."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    load_current_possessions,
    load_current_player_names,
    ratings_table,
)
from nba_impact.models.rubberband_adjustment import (
    RubberbandFit,
    RubberbandSpec,
    annotate_score_context,
    coefficient_table,
    fit_rubberband,
    paired_game_bootstrap,
    score_adjustment,
)
from nba_impact.models.rubberband_progress import (
    annotate_possession_progress,
    possession_coefficient_table,
    slope_only_adjustment,
    use_possession_progress,
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = (
    REPO_ROOT / "research" / "experiments" / "rubberband_progress_rapm_v2.json"
)
DEFAULT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
DEFAULT_SEGMENTS = (
    REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"
)
DEFAULT_PLAYER_GAMES = REPO_ROOT / "data" / "lake" / "silver" / "player_games.parquet"
DEFAULT_NAMES = REPO_ROOT / "rapm" / "data" / "all_names.csv"
CLOCK_OUTPUT = REPO_ROOT / "research" / "rapm_lab" / "outputs" / "rubberband"
DEFAULT_OUTPUT = (
    REPO_ROOT / "research" / "rapm_lab" / "outputs" / "rubberband_progress_rapm"
)


def load_clocked_terminal_possessions(
    possessions_path: Path,
    segments_path: Path,
    seasons: tuple[int, ...],
) -> pd.DataFrame:
    """Load canonical terminal-lineup RAPM rows with exact possession clock."""
    model = load_current_possessions(
        possessions_path,
        segments_path,
        lineup_policy="terminal",
        game_types=("regular",),
    )
    model = model.loc[model["season"].isin(seasons)].copy()
    clock = pd.read_parquet(
        possessions_path,
        columns=[
            "game_id",
            "possession_number",
            "period",
            "start_seconds_elapsed",
            "season_end",
            "season_type",
        ],
    )
    clock = clock.loc[
        clock["season_end"].isin(seasons) & clock["season_type"].eq("regular")
    ].rename(columns={"game_id": "gameid", "possession_number": "num"})
    clock["gameid"] = clock["gameid"].astype(str)
    if clock.duplicated(["gameid", "period", "num"]).any():
        raise ValueError("Clock source has duplicate possession keys.")
    result = model.merge(
        clock[["gameid", "period", "num", "start_seconds_elapsed"]],
        on=["gameid", "period", "num"],
        how="left",
        validate="one_to_one",
    )
    if result["start_seconds_elapsed"].isna().any():
        raise ValueError("Every modeled possession must resolve to an actual start time.")
    return annotate_score_context(result)


def _latest_clock_run() -> tuple[dict, Path]:
    paths = sorted(
        CLOCK_OUTPUT.glob("rubberband_adjustment_v1_*/run.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not paths:
        raise FileNotFoundError("Run the actual-clock rubber-band experiment first.")
    return json.loads(paths[-1].read_text()), paths[-1].parent


def _bucket_intercept_fit(frame: pd.DataFrame, spec: RubberbandSpec) -> RubberbandFit:
    clean = frame.loc[frame["regulation"].astype(bool)]
    six_minute = clean["six_minute_bucket"].to_numpy(dtype=int)
    if spec.time_buckets == 1:
        bucket = np.zeros(len(clean), dtype=int)
    elif spec.time_buckets == 4:
        bucket = six_minute // 2
    else:
        bucket = six_minute
    intercepts = (
        clean.assign(_bucket=bucket)
        .groupby("_bucket")["lineup_residual_points"]
        .mean()
        .reindex(range(spec.time_buckets))
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    return RubberbandFit(
        spec=spec,
        coefficients=np.concatenate([intercepts, np.zeros(spec.time_buckets)]),
        covariance=None,
        rows=len(clean),
        games=clean["gameid"].nunique(),
    )


def _paired_rmse_bootstrap(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    paired = baseline.merge(
        candidate,
        on="gameid",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError("RAPM variants must score identical games.")
    actual = paired["actual_margin_baseline"].to_numpy(dtype=float)
    other_actual = paired["actual_margin_candidate"].to_numpy(dtype=float)
    if not np.array_equal(actual, other_actual):
        raise ValueError("RAPM variants must use identical actual margins.")
    base_error = actual - paired["predicted_margin_baseline"].to_numpy(dtype=float)
    candidate_error = (
        actual - paired["predicted_margin_candidate"].to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(paired), size=(draws, len(paired)))
    changes = np.sqrt(np.mean(candidate_error[indices] ** 2, axis=1)) - np.sqrt(
        np.mean(base_error[indices] ** 2, axis=1)
    )
    return {
        "draws": int(draws),
        "seed": int(seed),
        "observed_rmse_delta": float(
            math.sqrt(np.mean(candidate_error**2)) - math.sqrt(np.mean(base_error**2))
        ),
        "lower_95": float(np.quantile(changes, 0.025)),
        "upper_95": float(np.quantile(changes, 0.975)),
        "probability_improvement": float(np.mean(changes < 0)),
    }


def _fit_player_model(design, target: np.ndarray, mask: np.ndarray | None):
    adjusted_design = replace(design, y=np.asarray(target, dtype=float))
    config = RapmConfig(
        seasons=(2024, 2025, 2026),
        lambda_off=3000.0,
        lambda_def=3000.0,
        lambda_home=300.0,
        data_scope="rubberband_adjusted_rapm_research",
    )
    beta, intercept = fit_coefficients(adjusted_design, config, row_mask=mask)
    return adjusted_design, beta, intercept


def _rating_variant(
    design,
    target: np.ndarray,
    names: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    adjusted_design, beta, _ = _fit_player_model(design, target, None)
    ratings = ratings_table(adjusted_design, beta, names=names)
    return ratings.rename(
        columns={
            "offense_per_100": f"{prefix}_offense",
            "defense_per_100": f"{prefix}_defense",
            "net_per_100": f"{prefix}_net",
        }
    ).drop(columns="uncertainty_status")


def _row_set_hash(frame: pd.DataFrame) -> str:
    keys = frame[["gameid", "period", "num"]].astype(str)
    values = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def run_experiment(
    contract_path: Path = DEFAULT_CONTRACT,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Rubber-band RAPM requires a frozen research contract.")
    if int(contract["untouched_confirmation_season"]) in contract["seasons"]:
        raise ValueError("Season 2027 must remain untouched.")
    clock_run, clock_path = _latest_clock_run()
    identity_payload = {
        "contract": sha256_file(contract_path),
        "runner": sha256_file(Path(__file__)),
        "progress_model": sha256_file(
            REPO_ROOT / "src" / "nba_impact" / "models" / "rubberband_progress.py"
        ),
        "clock_run": clock_run["run_id"],
        "possessions": sha256_file(DEFAULT_POSSESSIONS),
        "segments": sha256_file(DEFAULT_SEGMENTS),
        "player_games": sha256_file(DEFAULT_PLAYER_GAMES),
        "names": sha256_file(DEFAULT_NAMES),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = output_root / f"{contract['experiment_id']}_{identity}"
    run_path = output / "run.json"
    if run_path.exists():
        return json.loads(run_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    seasons = tuple(int(value) for value in contract["seasons"])
    source = load_clocked_terminal_possessions(
        DEFAULT_POSSESSIONS, DEFAULT_SEGMENTS, seasons
    )
    source = annotate_possession_progress(
        source,
        bucket_size=int(
            contract["possession_progress_spec"][
                "combined_regulation_possessions_per_bucket"
            ]
        ),
        buckets=int(contract["possession_progress_spec"]["time_buckets"]),
    )
    residuals = pd.concat(
        [
            pd.read_parquet(clock_path / f"lineup_residuals_{season}.parquet")
            for season in seasons
        ],
        ignore_index=True,
    )
    residuals = annotate_possession_progress(
        residuals,
        bucket_size=int(
            contract["possession_progress_spec"][
                "combined_regulation_possessions_per_bucket"
            ]
        ),
        buckets=int(contract["possession_progress_spec"]["time_buckets"]),
    )
    source_keys = source[["gameid", "period", "num", "pts"]].reset_index(drop=True)
    residual_keys = residuals[["gameid", "period", "num", "pts"]].reset_index(drop=True)
    if not source_keys.equals(residual_keys):
        raise ValueError("Clock residual checkpoints do not match current RAPM rows.")

    train_curve = residuals["season"].isin(contract["curve_training_seasons"])
    diagnostic_curve = residuals["season"].eq(contract["reused_diagnostic_season"])
    clock_spec = RubberbandSpec(
        "actual_clock",
        int(contract["clock_spec"]["time_buckets"]),
        float(contract["clock_spec"]["margin_clip"]),
    )
    progress_spec = RubberbandSpec(
        "possession_progress",
        int(contract["possession_progress_spec"]["time_buckets"]),
        float(contract["possession_progress_spec"]["margin_clip"]),
    )
    clock_fit = fit_rubberband(
        residuals.loc[train_curve], clock_spec, cluster_covariance=True
    )
    progress_residuals = use_possession_progress(residuals)
    progress_fit = fit_rubberband(
        progress_residuals.loc[train_curve],
        progress_spec,
        cluster_covariance=True,
    )
    clock_context_metrics, clock_context_games = score_adjustment(
        residuals.loc[diagnostic_curve],
        _bucket_intercept_fit(residuals.loc[train_curve], clock_spec),
        clock_fit,
    )
    progress_context_metrics, progress_context_games = score_adjustment(
        progress_residuals.loc[diagnostic_curve],
        _bucket_intercept_fit(progress_residuals.loc[train_curve], progress_spec),
        progress_fit,
    )
    context_bootstrap = {
        "clock": paired_game_bootstrap(
            clock_context_games,
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=int(contract["paired_game_bootstrap_seed"]),
        ),
        "possession_progress": paired_game_bootstrap(
            progress_context_games,
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=int(contract["paired_game_bootstrap_seed"]) + 1,
        ),
    }

    progress_source = use_possession_progress(source)
    clock_adjustment = slope_only_adjustment(clock_fit, source)
    progress_adjustment = slope_only_adjustment(progress_fit, progress_source)
    design = build_design(source, include_home=True)
    train_mask = design.seasons < int(contract["reused_diagnostic_season"])
    test_mask = design.seasons == int(contract["reused_diagnostic_season"])
    variants = {
        "normal": np.zeros(len(source), dtype=float),
        "clock": clock_adjustment,
        "possession_progress": progress_adjustment,
    }
    evaluation_rows: list[dict] = []
    evaluation_games: dict[str, pd.DataFrame] = {}
    for name, adjustment in variants.items():
        _, beta, intercept = _fit_player_model(
            design, design.y - adjustment, train_mask
        )
        neutral_prediction = intercept + np.asarray(
            design.X[test_mask] @ beta
        ).ravel()
        neutral_metrics, neutral_games = game_margin_metrics(
            source.loc[test_mask].reset_index(drop=True), neutral_prediction
        )
        evaluation_rows.append(
            {"variant": name, "prediction_mode": "neutral_player_only", **neutral_metrics}
        )
        evaluation_games[f"{name}_neutral"] = neutral_games
        if name != "normal":
            conditional_metrics, conditional_games = game_margin_metrics(
                source.loc[test_mask].reset_index(drop=True),
                neutral_prediction + adjustment[test_mask],
            )
            evaluation_rows.append(
                {
                    "variant": name,
                    "prediction_mode": "conditional_score_path",
                    **conditional_metrics,
                }
            )
            evaluation_games[f"{name}_conditional"] = conditional_games
    evaluation = pd.DataFrame(evaluation_rows)
    rapm_bootstrap = {
        name: _paired_rmse_bootstrap(
            evaluation_games["normal_neutral"],
            evaluation_games[f"{name}_neutral"],
            draws=int(contract["paired_game_bootstrap_draws"]),
            seed=int(contract["paired_game_bootstrap_seed"]) + offset,
        )
        for offset, name in enumerate(("clock", "possession_progress"), start=10)
    }

    names = load_current_player_names(DEFAULT_NAMES, DEFAULT_PLAYER_GAMES)
    ratings = _rating_variant(design, design.y, names, "normal")
    for prefix, adjustment in (
        ("clock", clock_adjustment),
        ("possession", progress_adjustment),
    ):
        candidate = _rating_variant(design, design.y - adjustment, names, prefix)
        keep = [
            "player_id",
            f"{prefix}_offense",
            f"{prefix}_defense",
            f"{prefix}_net",
        ]
        ratings = ratings.merge(candidate[keep], on="player_id", validate="one_to_one")
    ratings["clock_net_change"] = ratings["clock_net"] - ratings["normal_net"]
    ratings["possession_net_change"] = (
        ratings["possession_net"] - ratings["normal_net"]
    )
    for prefix in ("normal", "clock", "possession"):
        ratings[f"{prefix}_net_rank"] = (
            ratings[f"{prefix}_net"].rank(method="min", ascending=False).astype(int)
        )
    ratings = ratings.sort_values("clock_net", ascending=False, kind="stable")
    ratings.to_parquet(output / "ratings.parquet", index=False)

    clock_coefficients = coefficient_table(clock_fit).assign(basis="actual_clock")
    progress_coefficients = possession_coefficient_table(
        progress_fit,
        bucket_size=int(
            contract["possession_progress_spec"][
                "combined_regulation_possessions_per_bucket"
            ]
        ),
    ).assign(basis="possession_progress")
    coefficients = pd.concat(
        [clock_coefficients, progress_coefficients], ignore_index=True
    )
    coefficients.to_parquet(output / "coefficients.parquet", index=False)
    evaluation.to_parquet(output / "rapm_evaluation.parquet", index=False)
    for name, games in evaluation_games.items():
        games.assign(variant=name).to_parquet(
            output / f"game_predictions_{name}.parquet", index=False
        )

    qualified = ratings.loc[
        ratings[["off_possessions", "def_possessions"]].min(axis=1).ge(5000)
    ]
    maximum_identity_error = max(
        float(
            np.abs(
                ratings[f"{prefix}_offense"]
                + ratings[f"{prefix}_defense"]
                - ratings[f"{prefix}_net"]
            ).max()
        )
        for prefix in ("normal", "clock", "possession")
    )
    run = {
        "run_id": output.name,
        "status": "research_diagnostic",
        "model_family": "clock_and_possession_progress_rubberband_adjusted_rapm",
        "estimand_id": contract["estimand_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {**contract, **identity_payload},
        "quality": {
            "row_set_hash": _row_set_hash(source),
            "possessions": int(len(source)),
            "games": int(source["gameid"].nunique()),
            "missing_actual_clock": int(source["start_seconds_elapsed"].isna().sum()),
            "missing_possession_progress": int(
                source.loc[source["regulation"], "regulation_possession_number"]
                .isna()
                .sum()
            ),
            "identical_checkpoint_rows": True,
            "maximum_rating_identity_error": maximum_identity_error,
            "qualified_leaderboard_players": int(len(qualified)),
        },
        "context_effect": {
            "clock": clock_context_metrics,
            "possession_progress": progress_context_metrics,
            "bootstrap": context_bootstrap,
            "slope_correlation": float(
                np.corrcoef(
                    clock_fit.coefficients[clock_spec.time_buckets :],
                    progress_fit.coefficients[progress_spec.time_buckets :],
                )[0, 1]
            ),
        },
        "rapm_evaluation": evaluation.to_dict("records"),
        "rapm_bootstrap_vs_normal": rapm_bootstrap,
        "conditional_rapm_bootstrap_vs_normal": {
            name: _paired_rmse_bootstrap(
                evaluation_games["normal_neutral"],
                evaluation_games[f"{name}_conditional"],
                draws=int(contract["paired_game_bootstrap_draws"]),
                seed=int(contract["paired_game_bootstrap_seed"]) + offset,
            )
            for offset, name in enumerate(
                ("clock", "possession_progress"), start=20
            )
        },
        "rating_effect": {
            "clock_net_correlation_with_normal": float(
                ratings[["normal_net", "clock_net"]].corr().iloc[0, 1]
            ),
            "possession_net_correlation_with_normal": float(
                ratings[["normal_net", "possession_net"]].corr().iloc[0, 1]
            ),
            "clock_mean_absolute_net_change": float(
                ratings["clock_net_change"].abs().mean()
            ),
            "possession_mean_absolute_net_change": float(
                ratings["possession_net_change"].abs().mean()
            ),
        },
        "leaderboards": {
            "clock_net": qualified.nlargest(15, "clock_net")[
                ["player_id", "player_name", "clock_net", "clock_net_change"]
            ].to_dict("records"),
            "possession_net": qualified.nlargest(15, "possession_net")[
                [
                    "player_id",
                    "player_name",
                    "possession_net",
                    "possession_net_change",
                ]
            ].to_dict("records"),
            "largest_clock_changes": qualified.reindex(
                qualified["clock_net_change"].abs().sort_values(ascending=False).index
            )[
                ["player_id", "player_name", "normal_net", "clock_net", "clock_net_change"]
            ].head(15).to_dict("records"),
        },
        "artifacts": {
            "ratings": "ratings.parquet",
            "coefficients": "coefficients.parquet",
            "rapm_evaluation": "rapm_evaluation.parquet",
        },
        "forbidden_interpretation": (
            "These are descriptive score-context normalizations on observed lineups. "
            "They are not causal effort estimates, forecasts, or promoted public ratings."
        ),
        "caveats": [
            "Exact clock is available on canonical 2024-26 rows but not on the legacy RAPM cache.",
            "Historical event possessions and legacy RAPM possessions have different boundaries, so clock is not blindly joined into 2022-23 legacy rows.",
            "The possession-progress proxy uses fixed 25-possession bins and no final-game-length information.",
            "Score margin remains endogenous even after out-of-fold lineup residualization.",
            "The 2026 comparison is reused diagnostic evidence. Season 2027 is untouched.",
        ],
    }
    write_json_atomic(run, run_path)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.contract, args.output), indent=2))


if __name__ == "__main__":
    main()
