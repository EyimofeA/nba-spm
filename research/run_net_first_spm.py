#!/usr/bin/env python3
"""Compare direct-net five-year SPM priors with component-first baselines.

Both challengers learn CourtSignal five-year RAPM net directly and learn
offense separately. Defense is the exact residual ``net - offense``. The only
difference between the challengers is the offense feature bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.predictive_spm import _predictive_metrics
from nba_impact.models.rapm import load_legacy_possessions
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_model_comparison import _fit_model
from run_aio_prior_bakeoff import _game_metrics, _paired_bootstrap, _prior_frame
from run_aio_prior_canonical_followup import (
    _annual_from_frame,
    _center,
    _remap_annual,
    _solve,
)
from run_ryan_target_spm import _panel


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "net_first_spm_v1"
RATING_SEASONS = (2021, 2022, 2023)
CHALLENGERS = (
    "net_first_full_for_both",
    "net_first_full_net_offense_only",
)
BASELINES = (
    "courtsignal_target_weighted",
    "selected_five_year_spm",
    "box_pipm_style_prior",
    "ryan_target_weighted_rescaled",
)


def feature_sets(selected: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    full = tuple(dict.fromkeys((*selected["offense"], *selected["defense"])))
    return {
        "full": full,
        "offense_only": selected["offense"],
    }


def fit_net_first_priors(
    panel: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    banks = feature_sets(selected)
    rows = []
    for season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train["Window_End"].nunique() < 3 or test.empty:
            raise ValueError(f"Net-first fold {season} lacks chronological history.")
        train["sample_weight"] = train["courtsignal_exposure"]

        net_model = _fit_model(
            _frozen_model("offense"), train, banks["full"], "courtsignal_net"
        )
        predicted_net = net_model.predict(test.loc[:, banks["full"]])
        offense_banks = {
            "net_first_full_for_both": banks["full"],
            "net_first_full_net_offense_only": banks["offense_only"],
        }
        for candidate, offense_features in offense_banks.items():
            offense_model = _fit_model(
                _frozen_model("offense"),
                train,
                offense_features,
                "courtsignal_offense",
            )
            offense = offense_model.predict(test.loc[:, offense_features])
            prior = test[["PLAYER_ID", "Window_End"]].copy()
            prior["offense"] = offense
            prior["net"] = predicted_net
            prior["defense"] = prior["net"] - prior["offense"]
            framed = _prior_frame(prior, candidate)
            framed["net_feature_bank"] = "full"
            framed["offense_feature_bank"] = (
                "full" if candidate == "net_first_full_for_both" else "offense_only"
            )
            framed["training_window_start"] = int(train["Window_End"].min())
            framed["training_window_end"] = int(train["Window_End"].max())
            rows.append(framed)
    output = pd.concat(rows, ignore_index=True)
    identity_error = (
        output["prior_offense_per_100"]
        + output["prior_defense_per_100"]
        - output["prior_net_per_100"]
    ).abs()
    if identity_error.max() > 1e-12:
        raise AssertionError("Net-first prior does not satisfy offense + defense = net.")
    return output


def fit_aio(
    priors: pd.DataFrame,
    *,
    legacy_cache: Path,
    matrix_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = []
    games = []
    coverage = []
    for season in RATING_SEASONS:
        direct = _annual_from_frame(
            load_legacy_possessions(legacy_cache, (season,), game_types=("regular",)),
            season,
        )
        matrix_dir = matrix_root / f"5y_end_{season}"
        annual = _remap_annual(direct, np.load(matrix_dir / "player_ids.npy"))
        for candidate in CHALLENGERS:
            prior = priors.loc[
                priors["candidate"].eq(candidate)
                & priors["Window_End"].eq(season)
            ]
            center, report = _center(prior, annual)
            beta, intercept = _solve(annual, center, scale=1.0)
            game = stored_evaluation_predictions(matrix_dir, beta, intercept)
            game["candidate"] = candidate
            game["rating_season"] = season
            game["test_season"] = season + 1
            game["squared_error"] = (
                game["actual_margin"] - game["predicted_margin"]
            ) ** 2
            games.append(game)
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
            rating["rating_season"] = season
            ratings.append(rating)
            coverage.append({"candidate": candidate, "rating_season": season, **report})
        print(f"Net-first AIO fold {season}->{season + 1}: complete", flush=True)
    return (
        pd.concat(ratings, ignore_index=True),
        pd.concat(games, ignore_index=True),
        pd.DataFrame(coverage),
    )


def common_spm_metrics(
    panel: pd.DataFrame,
    challenger_priors: pd.DataFrame,
    ryan_priors: pd.DataFrame,
    bakeoff_priors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    priors = pd.concat(
        [
            challenger_priors,
            ryan_priors.loc[
                ryan_priors["candidate"].isin(
                    ["courtsignal_target_weighted", "ryan_target_weighted_rescaled"]
                )
            ],
            bakeoff_priors.loc[
                bakeoff_priors["candidate"].isin(
                    ["selected_five_year_spm", "box_pipm_style_prior"]
                )
            ],
        ],
        ignore_index=True,
    )
    priors = priors.loc[priors["Window_End"].isin(RATING_SEASONS)].copy()
    candidates = (*BASELINES, *CHALLENGERS)
    rows = []
    scored = []
    for season in RATING_SEASONS:
        target = panel.loc[
            panel["Window_End"].eq(season),
            [
                "PLAYER_ID",
                "Window_End",
                "courtsignal_offense",
                "courtsignal_defense",
                "courtsignal_net",
                "courtsignal_exposure",
            ],
        ].copy()
        candidate_frames = {
            candidate: priors.loc[
                priors["Window_End"].eq(season) & priors["candidate"].eq(candidate),
                [
                    "PLAYER_ID",
                    "Window_End",
                    "prior_offense_per_100",
                    "prior_defense_per_100",
                    "prior_net_per_100",
                ],
            ]
            for candidate in candidates
        }
        common = set(target["PLAYER_ID"].astype(int))
        for frame in candidate_frames.values():
            common &= set(frame["PLAYER_ID"].astype(int))
        if not common:
            raise ValueError(f"No common SPM rows for rating season {season}.")
        target = target.loc[target["PLAYER_ID"].isin(common)].copy()
        for candidate, frame in candidate_frames.items():
            merged = target.merge(
                frame.loc[frame["PLAYER_ID"].isin(common)],
                on=["PLAYER_ID", "Window_End"],
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(common):
                raise ValueError(f"SPM row mismatch for {candidate}, season {season}.")
            merged["candidate"] = candidate
            scored.append(merged)
            for component in ("offense", "defense", "net"):
                rows.append(
                    {
                        "candidate": candidate,
                        "rating_season": season,
                        "component": component,
                        "players": len(merged),
                        **_predictive_metrics(
                            merged[f"courtsignal_{component}"].to_numpy(dtype=float),
                            merged[f"prior_{component}_per_100"].to_numpy(dtype=float),
                            merged["courtsignal_exposure"].to_numpy(dtype=float),
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.concat(scored, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-cache", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=Path,
        default=(
            ROOT
            / "artifacts/research/spm_target_horizon_full"
            / "spm_target_horizon_full_v1_f0777db1d4/features_5y.parquet"
        ),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=(
            ROOT
            / "artifacts/models/five_year_target_spm"
            / "five_year_target_spm_v1_65550acb79/five_year_targets.parquet"
        ),
    )
    parser.add_argument(
        "--ryan-ratings",
        type=Path,
        default=(
            ROOT
            / "research/rapm_lab/data/external/user_downloads"
            / "ryan_davis_multi_rapm.csv"
        ),
    )
    parser.add_argument(
        "--ryan-run",
        type=Path,
        default=(
            ROOT
            / "artifacts/research/ryan_target_spm"
            / "ryan_target_spm_v1_31eccca595"
        ),
    )
    parser.add_argument(
        "--bakeoff-run",
        type=Path,
        default=(
            ROOT
            / "artifacts/research/aio_prior_bakeoff"
            / "aio_prior_bakeoff_v1_0a3591a402"
        ),
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=(
            ROOT
            / "research/rapm_lab/outputs/rolling_5y_2014_2026"
            / "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
        ),
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    if 2027 in RATING_SEASONS or 2027 in tuple(season + 1 for season in RATING_SEASONS):
        raise ValueError("Season 2027 is reserved and must not be loaded.")
    panel, selected = _panel(args.features, args.targets, args.ryan_ratings)
    challenger_priors = fit_net_first_priors(panel, selected)
    ratings, challenger_games, coverage = fit_aio(
        challenger_priors,
        legacy_cache=args.legacy_cache,
        matrix_root=args.matrix_root,
    )

    ryan_priors = pd.read_parquet(args.ryan_run / "priors.parquet")
    bakeoff_priors = pd.read_parquet(args.bakeoff_run / "priors.parquet")
    spm_metrics, spm_rows = common_spm_metrics(
        panel, challenger_priors, ryan_priors, bakeoff_priors
    )
    baseline_games = pd.concat(
        [
            pd.read_parquet(args.ryan_run / "game_predictions.parquet").loc[
                lambda frame: frame["candidate"].isin(
                    ["courtsignal_target_weighted", "ryan_target_weighted_rescaled"]
                )
            ],
            pd.read_parquet(args.bakeoff_run / "game_predictions.parquet").loc[
                lambda frame: frame["candidate"].isin(
                    ["selected_five_year_spm", "box_pipm_style_prior"]
                )
            ],
        ],
        ignore_index=True,
    )
    games = pd.concat([baseline_games, challenger_games], ignore_index=True)
    game_counts = games.groupby(["test_season", "candidate"])["game_id"].nunique().unstack()
    if game_counts.isna().any().any() or not game_counts.nunique(axis=1).eq(1).all():
        raise ValueError("Candidates do not score identical game counts.")
    actual_hashes = (
        games.assign(
            actual_key=lambda frame: (
                frame["game_id"].astype(str)
                + ":"
                + frame["actual_margin"].astype(str)
            )
        )
        .groupby(["test_season", "candidate"])["actual_key"]
        .apply(lambda values: hashlib.sha256("|".join(sorted(values)).encode()).hexdigest())
        .unstack()
    )
    if not actual_hashes.nunique(axis=1).eq(1).all():
        raise ValueError("Candidates do not score identical game outcomes.")

    game_metrics = []
    for (candidate, rating_season, test_season), frame in games.groupby(
        ["candidate", "rating_season", "test_season"]
    ):
        game_metrics.append(
            {
                "candidate": candidate,
                "rating_season": int(rating_season),
                "test_season": int(test_season),
                **_game_metrics(frame),
            }
        )
    game_metrics = pd.DataFrame(game_metrics)
    game_summary = (
        game_metrics.groupby("candidate", as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        .sort_values("mean_margin_rmse", kind="stable")
    )
    spm_summary = (
        spm_metrics.groupby(["candidate", "component"], as_index=False)
        .agg(
            folds=("rating_season", "nunique"),
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_weighted_correlation=("weighted_correlation", "mean"),
        )
        .sort_values(["component", "mean_weighted_rmse"], kind="stable")
    )
    bootstrap = pd.concat(
        [
            _paired_bootstrap(games, baseline=baseline, draws=5_000, seed=20260827).assign(
                comparison_baseline=baseline
            )
            for baseline in (
                "courtsignal_target_weighted",
                "box_pipm_style_prior",
                "selected_five_year_spm",
                "ryan_target_weighted_rescaled",
            )
        ],
        ignore_index=True,
    )

    banks = feature_sets(selected)
    config = {
        "rating_seasons": list(RATING_SEASONS),
        "test_seasons": [season + 1 for season in RATING_SEASONS],
        "targets": {
            "net": "CourtSignal rolling five-year zero-prior RAPM net",
            "offense": "CourtSignal rolling five-year zero-prior RAPM offense",
            "defense": "predicted net minus predicted offense",
        },
        "learners": {
            "net": "frozen offense histogram GBM",
            "offense": "frozen offense histogram GBM",
        },
        "feature_banks": {name: list(values) for name, values in banks.items()},
        "challengers": {
            "net_first_full_for_both": {"net": "full", "offense": "full"},
            "net_first_full_net_offense_only": {
                "net": "full",
                "offense": "offense_only",
            },
        },
        "aio": {
            "lambda_off": 3000.0,
            "lambda_def": 3000.0,
            "lambda_home": 300.0,
            "center_scale": 1.0,
            "likelihood_seasons": 1,
        },
        "source_hashes": {
            "features": sha256_file(args.features),
            "targets": sha256_file(args.targets),
            "ryan_ratings": sha256_file(args.ryan_ratings),
            "ryan_run": sha256_file(args.ryan_run / "run.json"),
            "bakeoff_run": sha256_file(args.bakeoff_run / "run.json"),
            "runner": sha256_file(Path(__file__)),
        },
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = args.artifact_root / "research/net_first_spm" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "challenger_priors": challenger_priors,
        "aio_ratings": ratings,
        "aio_game_predictions": games,
        "aio_game_metrics": game_metrics,
        "aio_summary": game_summary,
        "paired_bootstrap": bootstrap,
        "spm_common_rows": spm_rows,
        "spm_metrics": spm_metrics,
        "spm_summary": spm_summary,
        "prior_coverage": coverage,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / f"{name}.parquet", index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "research_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "matched_training_rows": len(panel),
            "matched_training_windows": sorted(panel["Window_End"].unique().tolist()),
            "full_feature_count": len(banks["full"]),
            "offense_feature_count": len(banks["offense_only"]),
            "identical_game_rows": True,
            "component_identity_max_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
            "season_2027_rows": 0,
        },
        "paths": {name: f"{name}.parquet" for name in outputs},
        "caveats": [
            "Full means the union of the 151 offense and defense inputs available in the stored matched five-year panel.",
            "The stored matched panel predates later zTS and matchup-defense additions; those are not silently reconstructed here.",
            "The net-first design and all 2022-24 comparisons are reused diagnostics, not independent confirmation.",
            "The same fixed histogram GBM is used for net and offense; no learner or feature tuning follows the scored results.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(game_summary.to_string(index=False))
    print(spm_summary.loc[spm_summary["component"].eq("net")].to_string(index=False))
    print(
        bootstrap.loc[
            bootstrap["candidate"].isin(CHALLENGERS),
            [
                "candidate",
                "comparison_baseline",
                "season_wins",
                "mean_mse_delta",
                "bootstrap_95_low",
                "bootstrap_95_high",
            ],
        ].to_string(index=False)
    )
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
