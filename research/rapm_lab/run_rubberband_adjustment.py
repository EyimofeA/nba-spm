"""Estimate the rubber-band scoring curve from actual clock and score state."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm import load_current_possessions
from nba_impact.models.rubberband_adjustment import (
    RubberbandFit,
    RubberbandSpec,
    annotate_score_context,
    coefficient_table,
    cross_fitted_lineup_residuals,
    fit_rubberband,
    paired_game_bootstrap,
    score_adjustment,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "research" / "experiments" / "rubberband_adjustment_v1.json"
DEFAULT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
DEFAULT_SEGMENTS = (
    REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"
)
DEFAULT_OUTPUT = REPO_ROOT / "research" / "rapm_lab" / "outputs" / "rubberband"


def load_clocked_terminal_possessions(
    possessions_path: Path,
    segments_path: Path,
    seasons: tuple[int, ...],
) -> pd.DataFrame:
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


def time_only_fit(frame: pd.DataFrame, spec: RubberbandSpec) -> RubberbandFit:
    clean = frame.loc[frame["regulation"].astype(bool)]
    if spec.time_buckets == 1:
        bucket = np.zeros(len(clean), dtype=int)
    elif spec.time_buckets == 4:
        bucket = clean["six_minute_bucket"].to_numpy(dtype=int) // 2
    else:
        bucket = clean["six_minute_bucket"].to_numpy(dtype=int)
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


def run_experiment(
    contract_path: Path = DEFAULT_CONTRACT,
    possessions_path: Path = DEFAULT_POSSESSIONS,
    segments_path: Path = DEFAULT_SEGMENTS,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict:
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != "frozen_research_contract":
        raise ValueError("Rubber-band estimation requires a frozen contract.")
    seasons = (
        int(contract["development_season"]),
        int(contract["selection_season"]),
        int(contract["reused_diagnostic_season"]),
    )
    if int(contract["untouched_confirmation_season"]) in seasons:
        raise ValueError("Season 2027 must remain untouched.")
    identity_payload = {
        "contract_hash": sha256_file(contract_path),
        "script_hash": sha256_file(Path(__file__)),
        "model_hash": sha256_file(
            REPO_ROOT / "src" / "nba_impact" / "models" / "rubberband_adjustment.py"
        ),
        "possessions_hash": sha256_file(possessions_path),
        "segments_hash": sha256_file(segments_path),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = output_root / f"rubberband_adjustment_v1_{identity}"
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "run.json"
    if completed.exists():
        return json.loads(completed.read_text())

    started = time.perf_counter()
    source = load_clocked_terminal_possessions(possessions_path, segments_path, seasons)
    residual_parts: list[pd.DataFrame] = []
    for season in seasons:
        checkpoint = output / f"lineup_residuals_{season}.parquet"
        if checkpoint.exists():
            residual_parts.append(pd.read_parquet(checkpoint))
            continue
        season_frame = source.loc[source["season"].eq(season)].reset_index(drop=True)
        residual = cross_fitted_lineup_residuals(
            season_frame,
            folds=int(contract["game_crossfit_folds"]),
            lambda_off=float(contract["player_penalties"]["lambda_off"]),
            lambda_def=float(contract["player_penalties"]["lambda_def"]),
            lambda_home=float(contract["player_penalties"]["lambda_home"]),
        )
        residual.to_parquet(checkpoint, index=False)
        residual_parts.append(residual)
        print(
            f"rubber-band residuals {season}: {len(residual):,} possessions, "
            f"{residual['gameid'].nunique():,} games",
            flush=True,
        )
    residuals = pd.concat(residual_parts, ignore_index=True)
    development = residuals.loc[residuals["season"].eq(seasons[0])].reset_index(drop=True)
    selection = residuals.loc[residuals["season"].eq(seasons[1])].reset_index(drop=True)
    diagnostic = residuals.loc[residuals["season"].eq(seasons[2])].reset_index(drop=True)

    selection_rows: list[dict] = []
    selection_game_losses: dict[str, pd.DataFrame] = {}
    specs: list[RubberbandSpec] = []
    for raw in contract["candidate_specs"]:
        spec = RubberbandSpec(
            name=str(raw["name"]),
            time_buckets=int(raw["time_buckets"]),
            margin_clip=(
                None if raw["margin_clip"] is None else float(raw["margin_clip"])
            ),
        )
        specs.append(spec)
        baseline_fit = time_only_fit(development, spec)
        candidate_fit = fit_rubberband(development, spec)
        metrics, game_losses = score_adjustment(selection, baseline_fit, candidate_fit)
        selection_game_losses[spec.name] = game_losses
        selection_rows.append({"stage": "selection", "spec": spec.name, **metrics})
    selection_frame = pd.DataFrame(selection_rows)
    ranked_selection = selection_frame.sort_values(
        ["mean_squared_error_improvement", "spec"],
        ascending=[False, True],
        kind="stable",
    )
    winner_name = str(ranked_selection.iloc[0]["spec"])
    runner_up_name = str(ranked_selection.iloc[1]["spec"])
    winner = next(spec for spec in specs if spec.name == winner_name)
    runner_up_losses = selection_game_losses[runner_up_name][
        ["gameid", "rows", "candidate_squared_error"]
    ].rename(columns={"candidate_squared_error": "baseline_squared_error"})
    winner_losses = selection_game_losses[winner_name]
    winner_vs_runner_up_games = runner_up_losses.merge(
        winner_losses[["gameid", "candidate_squared_error"]],
        on="gameid",
        validate="one_to_one",
    )
    winner_vs_runner_up = paired_game_bootstrap(
        winner_vs_runner_up_games,
        draws=int(contract["paired_game_bootstrap_draws"]),
        seed=int(contract["paired_game_bootstrap_seed"]) + 1,
    )

    training = pd.concat([development, selection], ignore_index=True)
    final_baseline = time_only_fit(training, winner)
    final_fit = fit_rubberband(training, winner, cluster_covariance=True)
    diagnostic_metrics, diagnostic_games = score_adjustment(
        diagnostic, final_baseline, final_fit
    )
    bootstrap = paired_game_bootstrap(
        diagnostic_games,
        draws=int(contract["paired_game_bootstrap_draws"]),
        seed=int(contract["paired_game_bootstrap_seed"]),
    )
    all_metrics = pd.concat(
        [
            selection_frame,
            pd.DataFrame(
                [{"stage": "diagnostic", "spec": winner.name, **diagnostic_metrics}]
            ),
        ],
        ignore_index=True,
    )

    coefficients = coefficient_table(final_fit)
    coefficients.insert(0, "fit_seasons", f"{seasons[0]}-{seasons[1]}")
    stability_parts: list[pd.DataFrame] = []
    for season in seasons:
        fit = fit_rubberband(
            residuals.loc[residuals["season"].eq(season)].reset_index(drop=True),
            winner,
        )
        table = coefficient_table(fit)
        table.insert(0, "season", season)
        stability_parts.append(table)
    stability = pd.concat(stability_parts, ignore_index=True)

    clip = winner.margin_clip
    curve_rows: list[dict] = []
    slopes = final_fit.coefficients[winner.time_buckets :]
    for bucket, slope in enumerate(slopes):
        for margin in range(-30, 31):
            effective = float(margin if clip is None else np.clip(margin, -clip, clip))
            curve_rows.append(
                {
                    "time_bucket": bucket,
                    "minutes_elapsed_start": bucket * 48 / winner.time_buckets,
                    "minutes_elapsed_end": (bucket + 1) * 48 / winner.time_buckets,
                    "offense_margin_before": margin,
                    "adjustment_points_per_100": 100.0 * slope * effective,
                }
            )
    curve = pd.DataFrame(curve_rows)

    all_metrics.to_parquet(output / "metrics.parquet", index=False)
    coefficients.to_parquet(output / "coefficients.parquet", index=False)
    stability.to_parquet(output / "season_stability.parquet", index=False)
    curve.to_parquet(output / "curve.parquet", index=False)
    diagnostic_games.to_parquet(output / "diagnostic_game_losses.parquet", index=False)
    slopes_by_season = stability.pivot(
        index="time_bucket",
        columns="season",
        values="slope_points_per_100_per_margin_point",
    )
    run = {
        "run_id": output.name,
        "model_family": "cross_fitted_actual_clock_rubberband_adjustment",
        "estimand_id": contract["estimand_id"],
        "status": "research_estimate",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": {
            "development_season": seasons[0],
            "selection_season": seasons[1],
            "reused_diagnostic_season": seasons[2],
            "season_2027_loaded": False,
            "game_crossfit_folds": int(contract["game_crossfit_folds"]),
            "selected_spec": winner.name,
            "selected_time_buckets": winner.time_buckets,
            "selected_margin_clip": winner.margin_clip,
            **identity_payload,
        },
        "quality": {
            "possessions": int(len(residuals)),
            "regulation_possessions": int(residuals["regulation"].sum()),
            "games": int(residuals["gameid"].nunique()),
            "expected_regular_season_games": 3 * 1230,
            "game_coverage_rate": float(residuals["gameid"].nunique() / (3 * 1230)),
            "season_games": {
                str(season): int(
                    residuals.loc[residuals["season"].eq(season), "gameid"].nunique()
                )
                for season in seasons
            },
            "missing_actual_clock": 0,
            "unique_game_fold_assignments": int(
                residuals[["gameid", "game_fold"]].drop_duplicates()["gameid"].nunique()
            ),
            "maximum_absolute_pre_possession_margin": float(
                residuals["offense_margin_before"].abs().max()
            ),
        },
        "selection": selection_frame.to_dict("records"),
        "selection_winner_vs_runner_up": {
            "winner": winner_name,
            "runner_up": runner_up_name,
            **winner_vs_runner_up,
        },
        "diagnostic": diagnostic_metrics,
        "paired_diagnostic_bootstrap": bootstrap,
        "coefficient_stability": {
            "minimum_pairwise_season_correlation": float(
                slopes_by_season.corr().to_numpy()[np.triu_indices(3, 1)].min()
            )
        },
        "artifacts": {
            "metrics": "metrics.parquet",
            "coefficients": "coefficients.parquet",
            "season_stability": "season_stability.parquet",
            "curve": "curve.parquet",
            "diagnostic_game_losses": "diagnostic_game_losses.parquet",
        },
        "forbidden_interpretation": (
            "This is a descriptive pre-possession score-context association. It is not "
            "causal effort, a garbage-time label, player impact, or a promoted RAPM correction."
        ),
        "caveats": [
            "The lineup baseline is cross-fitted by whole game within each season.",
            "Actual game clock replaces the prior possession-order proxy.",
            "Score margin remains endogenous to team strength and strategy despite lineup controls.",
            "The adjustment is zero in overtime in this first contract.",
            "Nine regular-season games remain outside the canonical lineup-quality panel; no quality gate was weakened.",
            "2026 is reused diagnostic evidence and Season 2027 is untouched.",
        ],
    }
    write_json_atomic(run, completed)
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--possessions", type=Path, default=DEFAULT_POSSESSIONS)
    parser.add_argument("--segments", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run = run_experiment(args.contract, args.possessions, args.segments, args.output)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
