#!/usr/bin/env python3
"""Compare BoxPIPM and a public PIPM reference before and after RAPM."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from run_aio_prior_bakeoff import _game_metrics
from run_aio_prior_canonical_followup import _annual_from_frame, _center, _remap_annual, _solve
from run_pipm_breaker import _load_attached_pipm


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "pipm_four_way_comparison_v1"
RATING_SEASONS = (2021, 2022, 2023)
BOX_SOURCE_CANDIDATE = "box_courtsignal_ridge"
MINUTES_THRESHOLD = 250.0
REPLACEMENT_OFFENSE = -1.0
REPLACEMENT_DEFENSE = -1.0
MODEL_ORDER = (
    "box_prior",
    "pipm_reference",
    "box_prior_plus_rapm",
    "pipm_reference_plus_rapm",
)
PRIMARY_PAIRS = {
    frozenset(("box_prior", "pipm_reference")),
    frozenset(("box_prior", "box_prior_plus_rapm")),
    frozenset(("pipm_reference", "pipm_reference_plus_rapm")),
    frozenset(("box_prior_plus_rapm", "pipm_reference_plus_rapm")),
}


def _box_priors(path: Path) -> pd.DataFrame:
    source = pd.read_parquet(path)
    output = source.loc[
        source["Window_End"].isin(RATING_SEASONS)
        & source["candidate"].eq(BOX_SOURCE_CANDIDATE),
        [
            "PLAYER_ID",
            "Window_End",
            "prior_offense_per_100",
            "prior_defense_per_100",
            "prior_net_per_100",
        ],
    ].copy()
    if output.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("BoxPIPM has duplicate player-window rows.")
    if set(output["Window_End"].unique()) != set(RATING_SEASONS):
        raise ValueError("BoxPIPM does not cover every comparison season.")
    output["candidate"] = "box_prior"
    return output


def _pipm_reference(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_parquet(path)
    source = source.loc[source["rating_season"].isin(RATING_SEASONS)].copy()
    if source.duplicated(["PLAYER_ID", "rating_season"]).any():
        raise ValueError("PIPM reference has duplicate player-season rows.")
    source["below_minutes_threshold"] = source["minutes"].lt(MINUTES_THRESHOLD)
    source.loc[
        source["below_minutes_threshold"],
        ["pipm_offense", "pipm_defense", "pipm_net"],
    ] = [
        REPLACEMENT_OFFENSE,
        REPLACEMENT_DEFENSE,
        REPLACEMENT_OFFENSE + REPLACEMENT_DEFENSE,
    ]
    output = source.rename(
        columns={
            "rating_season": "Window_End",
            "pipm_offense": "prior_offense_per_100",
            "pipm_defense": "prior_defense_per_100",
            "pipm_net": "prior_net_per_100",
        }
    )[
        [
            "PLAYER_ID",
            "Window_End",
            "prior_offense_per_100",
            "prior_defense_per_100",
            "prior_net_per_100",
        ]
    ].copy()
    output["candidate"] = "pipm_reference"
    coverage = (
        source.groupby("rating_season", as_index=False)
        .agg(
            source_players=("PLAYER_ID", "size"),
            replaced_players=("below_minutes_threshold", "sum"),
            source_minutes=("minutes", "sum"),
        )
        .rename(columns={"rating_season": "Window_End"})
    )
    return output, coverage


def _audit_attached_pipm(path: Path, player_sheet_root: Path) -> pd.DataFrame:
    matched, identity_coverage = _load_attached_pipm(path, player_sheet_root)
    latest = matched.loc[matched["Season"].eq(matched["Season"].max())]
    return identity_coverage.assign(
        latest_season=int(latest["Season"].max()),
        latest_season_players=int(latest["PLAYER_ID"].nunique()),
        latest_season_max_games=float(latest["GP"].max()),
        latest_season_median_games=float(latest["GP"].median()),
        complete_latest_season=False,
        scored_in_comparison=False,
        exclusion_reason="The 2020-21 rows stop about 20 games into the season.",
    )


def _agreement(box: pd.DataFrame, pipm: pd.DataFrame) -> pd.DataFrame:
    matched = box.merge(
        pipm,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        suffixes=("_box", "_pipm"),
        validate="one_to_one",
    )
    rows = []
    for season, season_frame in matched.groupby("Window_End"):
        for component in ("offense", "defense", "net"):
            box_values = season_frame[f"prior_{component}_per_100_box"]
            pipm_values = season_frame[f"prior_{component}_per_100_pipm"]
            rows.append(
                {
                    "rating_season": int(season),
                    "component": component,
                    "matched_players": len(season_frame),
                    "pearson": float(box_values.corr(pipm_values)),
                    "spearman": float(box_values.corr(pipm_values, method="spearman")),
                    "rmse": float(np.sqrt(np.mean((box_values - pipm_values) ** 2))),
                    "mean_box_minus_pipm": float((box_values - pipm_values).mean()),
                }
            )
    return pd.DataFrame(rows)


def _score_models(
    priors: pd.DataFrame,
    possession_root: Path,
    matrix_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rating_rows = []
    game_rows = []
    coverage_rows = []
    for season in RATING_SEASONS:
        matrix_dir = matrix_root / f"5y_end_{season}"
        possessions = pd.read_parquet(possession_root / f"matchups_{season}.parquet")
        direct = _annual_from_frame(possessions, season)
        annual = _remap_annual(direct, np.load(matrix_dir / "player_ids.npy"))
        zero_beta, zero_intercept = _solve(
            annual,
            np.zeros(2 * len(annual.players) + 1),
            scale=0.0,
        )

        for prior_name in ("box_prior", "pipm_reference"):
            prior = priors.loc[
                priors["candidate"].eq(prior_name)
                & priors["Window_End"].eq(season)
            ]
            center, coverage = _center(prior, annual)
            standalone = center.copy()
            standalone[-1] = zero_beta[-1]
            posterior, posterior_intercept = _solve(annual, center, scale=1.0)
            candidates = (
                (prior_name, standalone, zero_intercept, "standalone_prior"),
                (
                    f"{prior_name}_plus_rapm",
                    posterior,
                    posterior_intercept,
                    "rapm_posterior",
                ),
            )
            for candidate, beta, intercept, stage in candidates:
                prediction = stored_evaluation_predictions(matrix_dir, beta, intercept)
                prediction["candidate"] = candidate
                prediction["rating_season"] = season
                prediction["test_season"] = season + 1
                prediction["squared_error"] = (
                    prediction["actual_margin"] - prediction["predicted_margin"]
                ) ** 2
                game_rows.append(prediction)

                n = len(annual.players)
                rating = pd.DataFrame(
                    {
                        "PLAYER_ID": annual.players,
                        "offense": 100.0 * beta[:n],
                        "defense": -100.0 * beta[n : 2 * n],
                        "Poss_Off": annual.off_possessions,
                        "Poss_Def": annual.def_possessions,
                    }
                )
                rating["net"] = rating["offense"] + rating["defense"]
                rating["candidate"] = candidate
                rating["stage"] = stage
                rating["rating_season"] = season
                rating_rows.append(rating)
            coverage_rows.append(
                {"candidate": prior_name, "rating_season": season, **coverage}
            )

    games = pd.concat(game_rows, ignore_index=True)
    for season, frame in games.groupby("test_season"):
        counts = frame.groupby("candidate")["game_id"].nunique()
        if len(counts) != len(MODEL_ORDER) or counts.nunique() != 1:
            raise ValueError(f"Models do not score identical {season} game counts.")
        hashes = frame.groupby("candidate").apply(
            lambda candidate_frame: hashlib.sha256(
                "|".join(
                    sorted(
                        candidate_frame["game_id"].astype(str)
                        + ":"
                        + candidate_frame["actual_margin"].astype(str)
                    )
                ).encode()
            ).hexdigest(),
            include_groups=False,
        )
        if hashes.nunique() != 1:
            raise ValueError(f"Models do not score identical {season} outcomes.")
    return pd.concat(rating_rows, ignore_index=True), games, pd.DataFrame(coverage_rows)


def paired_game_bootstrap(
    games: pd.DataFrame,
    *,
    draws: int = 5_000,
    seed: int = 20260827,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample games inside each season and average season MSE equally."""
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive.")
    season_errors = []
    for _, frame in games.groupby("test_season", sort=True):
        wide = frame.pivot(
            index="game_id",
            columns="candidate",
            values=["actual_margin", "predicted_margin"],
        )
        actual = wide["actual_margin"]
        if actual.isna().any().any() or not actual.nunique(axis=1).eq(1).all():
            raise ValueError("Candidates must score the same complete outcomes.")
        predictions = wide["predicted_margin"].reindex(columns=MODEL_ORDER)
        if predictions.isna().any().any():
            raise ValueError("Candidates must score every game.")
        actual_values = actual.iloc[:, 0].to_numpy(dtype=float)
        season_errors.append(
            (actual_values[:, None] - predictions.to_numpy(dtype=float)) ** 2
        )

    point_mse = np.mean([values.mean(axis=0) for values in season_errors], axis=0)
    season_mse = np.asarray([values.mean(axis=0) for values in season_errors])
    rng = np.random.default_rng(seed)
    draw_mse = np.empty((draws, len(MODEL_ORDER)), dtype=float)
    for draw in range(draws):
        draw_mse[draw] = np.mean(
            [
                values[rng.integers(0, len(values), size=len(values))].mean(axis=0)
                for values in season_errors
            ],
            axis=0,
        )

    ranks = np.argsort(np.argsort(draw_mse, axis=1), axis=1) + 1
    model_rows = []
    for index, candidate in enumerate(MODEL_ORDER):
        mse_low, mse_high = np.quantile(draw_mse[:, index], [0.025, 0.975])
        model_rows.append(
            {
                "candidate": candidate,
                "folds": len(season_errors),
                "equal_season_mse": float(point_mse[index]),
                "equal_season_rmse": float(np.sqrt(point_mse[index])),
                "bootstrap_mse_95_low": float(mse_low),
                "bootstrap_mse_95_high": float(mse_high),
                "bootstrap_rmse_95_low": float(np.sqrt(mse_low)),
                "bootstrap_rmse_95_high": float(np.sqrt(mse_high)),
                "probability_best": float(np.mean(ranks[:, index] == 1)),
                "mean_bootstrap_rank": float(ranks[:, index].mean()),
                "bootstrap_draws": draws,
            }
        )

    pair_rows = []
    for left, right in itertools.combinations(MODEL_ORDER, 2):
        left_index = MODEL_ORDER.index(left)
        right_index = MODEL_ORDER.index(right)
        delta_draws = draw_mse[:, left_index] - draw_mse[:, right_index]
        low, high = np.quantile(delta_draws, [0.025, 0.975])
        pair_rows.append(
            {
                "candidate": left,
                "reference": right,
                "primary_comparison": frozenset((left, right)) in PRIMARY_PAIRS,
                "mean_mse_delta": float(point_mse[left_index] - point_mse[right_index]),
                "bootstrap_95_low": float(low),
                "bootstrap_95_high": float(high),
                "probability_candidate_lower_mse": float(np.mean(delta_draws < 0)),
                "candidate_fold_wins": int(
                    np.sum(season_mse[:, left_index] < season_mse[:, right_index])
                ),
                "reference_fold_wins": int(
                    np.sum(season_mse[:, right_index] < season_mse[:, left_index])
                ),
                "bootstrap_draws": draws,
            }
        )
    return (
        pd.DataFrame(model_rows).sort_values("equal_season_mse", kind="stable"),
        pd.DataFrame(pair_rows).sort_values(
            ["primary_comparison", "mean_mse_delta"],
            ascending=[False, True],
            kind="stable",
        ),
    )


def _game_metrics_frames(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (test_season, candidate), frame in games.groupby(
        ["test_season", "candidate"]
    ):
        rows.append(
            {
                "test_season": int(test_season),
                "rating_season": int(test_season) - 1,
                "candidate": candidate,
                **_game_metrics(frame),
            }
        )
    folds = pd.DataFrame(rows)
    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values("mean_margin_rmse", kind="stable")
    )
    return folds, summary


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attached-pipm",
        type=Path,
        default=Path("/Users/eadebayo/Downloads/PIPM Player Finder through 2021 - Database.csv"),
    )
    parser.add_argument(
        "--pipm-reference",
        type=Path,
        default=ROOT
        / "artifacts/research/pipm_reference_comparison/pipm_reference_comparison_v1_49a3c2c973/reference.parquet",
    )
    parser.add_argument(
        "--pipm-reference-run",
        type=Path,
        default=ROOT
        / "artifacts/research/pipm_reference_comparison/pipm_reference_comparison_v1_49a3c2c973/run.json",
    )
    parser.add_argument(
        "--box-priors",
        type=Path,
        default=ROOT
        / "artifacts/research/pipm_breaker/pipm_breaker_v1_d154ebea55/prior_ratings.parquet",
    )
    parser.add_argument(
        "--player-sheet-root",
        type=Path,
        default=ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals",
    )
    parser.add_argument(
        "--possession-root",
        type=Path,
        default=ROOT / "research/rapm_lab/external/external/poss_data/derived",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT
        / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices",
    )
    parser.add_argument("--draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    matrix_paths = {
        f"matrix_manifest_{season}": args.matrix_root / f"5y_end_{season}/manifest.json"
        for season in RATING_SEASONS
    }
    possession_paths = {
        f"possessions_{season}": args.possession_root / f"matchups_{season}.parquet"
        for season in RATING_SEASONS
    }
    inputs = {
        "attached_pipm": args.attached_pipm,
        "pipm_reference": args.pipm_reference,
        "pipm_reference_run": args.pipm_reference_run,
        "box_priors": args.box_priors,
        "runner": Path(__file__),
        **matrix_paths,
        **possession_paths,
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing comparison inputs: {missing}")

    box = _box_priors(args.box_priors)
    pipm, source_coverage = _pipm_reference(args.pipm_reference)
    attached_audit = _audit_attached_pipm(args.attached_pipm, args.player_sheet_root)
    priors = pd.concat([box, pipm], ignore_index=True)
    agreement = _agreement(box, pipm)
    ratings, games, prior_coverage = _score_models(
        priors,
        args.possession_root,
        args.matrix_root,
    )
    fold_metrics, summary = _game_metrics_frames(games)
    bootstrap_models, bootstrap_pairs = paired_game_bootstrap(
        games,
        draws=args.draws,
        seed=args.seed,
    )

    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_seasons": list(RATING_SEASONS),
        "test_seasons": [season + 1 for season in RATING_SEASONS],
        "models": list(MODEL_ORDER),
        "box_source_candidate": BOX_SOURCE_CANDIDATE,
        "low_minute_rule": {
            "threshold": MINUTES_THRESHOLD,
            "replacement_offense": REPLACEMENT_OFFENSE,
            "replacement_defense": REPLACEMENT_DEFENSE,
        },
        "rapm_penalties": {"offense": 3000.0, "defense": 3000.0, "home": 300.0},
        "rapm_center_scale": 1.0,
        "standalone_nuisance_terms": "shared zero-prior intercept and home coefficient",
        "bootstrap": {
            "unit": "whole game within test season",
            "season_aggregation": "equal-season mean MSE",
            "draws": args.draws,
            "seed": args.seed,
            "interval": "2.5th and 97.5th percentiles of paired MSE differences",
        },
        "inputs": {
            name: {"path": _relative(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = (
        args.artifact_root
        / "research"
        / "pipm_four_way_comparison"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "source_priors.parquet": priors,
        "ratings.parquet": ratings,
        "game_predictions.parquet": games,
        "fold_metrics.parquet": fold_metrics,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": bootstrap_models,
        "paired_bootstrap.parquet": bootstrap_pairs,
        "prior_agreement.parquet": agreement,
        "source_coverage.parquet": source_coverage,
        "attached_source_audit.parquet": attached_audit,
        "prior_coverage.parquet": prior_coverage,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)

    run = {
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "reused_historical_diagnostic",
        "estimand": "next-season game margin from prior-season player ratings",
        "source_scope": (
            "The scored PIPM source is a third-party regular-season table reconstruction. "
            "The attached original PIPM file has a partial 2020-21 season and is not scored."
        ),
        "fold_definition": (
            "Build ratings through season t, apply the optional same-season RAPM update, "
            "then score the fixed t+1 games."
        ),
        "config": config,
        "quality": {
            "models": len(MODEL_ORDER),
            "folds": len(RATING_SEASONS),
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
            "attached_pipm_excluded": True,
            "season_2027_loaded": False,
        },
        "files": {},
        "forbidden_interpretation": (
            "The PIPM reference is not an original release. These reused folds do not establish "
            "a universal winner or support public promotion before the 2027 confirmation."
        ),
    }
    for name, frame in outputs.items():
        run["files"][name] = {"sha256": sha256_file(output / name), "rows": len(frame)}
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print("\nPrimary paired comparisons")
    print(bootstrap_pairs.loc[bootstrap_pairs["primary_comparison"]].to_string(index=False))


if __name__ == "__main__":
    main()
