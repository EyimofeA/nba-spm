#!/usr/bin/env python3
"""Test Ryan Davis five-year RAPM coefficients as SPM training labels.

Ryan's released rating tables do not contain player-window exposure weights.
The experiment therefore compares CourtSignal exposure weights with an
unweighted sensitivity run. Both use the same matched players, windows,
features, learners, AIO likelihood, and future games.
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
from rapm_lab.run_external_reproduction_benchmark import ryan_window_bounds


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "ryan_target_spm_v1"
RATING_SEASONS = (2021, 2022, 2023)
VARIANTS = {
    "courtsignal_target_weighted": {
        "target": "courtsignal",
        "weight": "courtsignal_exposure",
        "center_scale": "unit",
    },
    "ryan_target_weighted": {
        "target": "ryan",
        "weight": "courtsignal_exposure",
        "center_scale": "unit",
    },
    "ryan_target_unweighted": {
        "target": "ryan",
        "weight": "uniform",
        "center_scale": "unit",
    },
    "ryan_target_weighted_rescaled": {
        "target": "ryan_rescaled",
        "weight": "courtsignal_exposure",
        "center_scale": "unit",
    },
    "courtsignal_target_weighted_shrunken": {
        "target": "courtsignal",
        "weight": "courtsignal_exposure",
        "center_scale": "ryan_to_courtsignal_sd",
    },
}


def _ryan_targets(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    bounds = source["season"].map(ryan_window_bounds)
    source["Window_Start"] = bounds.map(lambda value: value[0])
    source["Window_End"] = bounds.map(lambda value: value[1])
    source = source.loc[
        (source["Window_End"] - source["Window_Start"] + 1).eq(5)
    ].copy()
    output = source.rename(
        columns={
            "playerId": "PLAYER_ID",
            "RAPM__Off": "ryan_offense",
            "RAPM__Def": "ryan_defense",
            "RAPM": "ryan_net",
        }
    )[
        [
            "PLAYER_ID",
            "Window_Start",
            "Window_End",
            "ryan_offense",
            "ryan_defense",
            "ryan_net",
            "playerName",
        ]
    ]
    output["PLAYER_ID"] = pd.to_numeric(output["PLAYER_ID"], errors="raise").astype(int)
    if output.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Ryan five-year RAPM keys are not unique.")
    return output


def _panel(
    features_path: Path,
    targets_path: Path,
    ryan_path: Path,
) -> tuple[pd.DataFrame, dict[str, tuple[str, ...]]]:
    features = pd.read_parquet(features_path)
    own = pd.read_parquet(targets_path).rename(
        columns={
            "target_offense": "courtsignal_offense",
            "target_defense": "courtsignal_defense",
            "target_net": "courtsignal_net",
        }
    )
    ryan = _ryan_targets(ryan_path)
    panel = features.merge(
        own[
            [
                "PLAYER_ID",
                "Window_End",
                "courtsignal_offense",
                "courtsignal_defense",
                "courtsignal_net",
                "Poss_Off",
                "Poss_Def",
            ]
        ],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    ).merge(
        ryan,
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel = panel.loc[panel["Window_End"].between(2018, 2023)].copy()
    panel["courtsignal_exposure"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    panel["uniform"] = 1.0
    manifest = json.loads((features_path.parent / "run.json").read_text())
    selected = {
        side: tuple(manifest["features"][side]) for side in ("offense", "defense")
    }
    missing = {
        side: sorted(set(columns) - set(panel.columns))
        for side, columns in selected.items()
    }
    if any(missing.values()):
        raise ValueError(f"Matched feature panel is missing columns: {missing}")
    return panel, selected


def _fit_priors(
    panel: pd.DataFrame,
    selected: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    target_metrics = []
    for season in RATING_SEASONS:
        train_base = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        if train_base["Window_End"].nunique() < 3 or test.empty:
            raise ValueError(f"Ryan target SPM fold {season} lacks history.")
        weight = train_base["courtsignal_exposure"].to_numpy(dtype=float).copy()
        weight /= weight.sum()
        scale_ratios = {}
        for side in ("offense", "defense"):
            court = train_base[f"courtsignal_{side}"].to_numpy(dtype=float)
            ryan = train_base[f"ryan_{side}"].to_numpy(dtype=float)
            court_sd = float(np.sqrt(np.sum(weight * (court - np.sum(weight * court)) ** 2)))
            ryan_sd = float(np.sqrt(np.sum(weight * (ryan - np.sum(weight * ryan)) ** 2)))
            if ryan_sd <= 0:
                raise ValueError(f"Ryan {side} target has zero weighted spread.")
            scale_ratios[side] = court_sd / ryan_sd
            train_base[f"ryan_rescaled_{side}"] = train_base[f"ryan_{side}"] * scale_ratios[side]
            test[f"ryan_rescaled_{side}"] = test[f"ryan_{side}"] * scale_ratios[side]
        shrink_scale = float(np.mean([1.0 / value for value in scale_ratios.values()]))

        for candidate, specification in VARIANTS.items():
            target_source = specification["target"]
            weight_column = specification["weight"]
            train = train_base.copy()
            train["sample_weight"] = train[weight_column]
            prior = test[["PLAYER_ID", "Window_End"]].copy()
            for side in ("offense", "defense"):
                target = f"{target_source}_{side}"
                model = _fit_model(
                    _frozen_model(side), train, selected[side], target
                )
                prediction = model.predict(test.loc[:, selected[side]])
                prior[side] = prediction
                error = test[target].to_numpy(dtype=float) - prediction
                target_metrics.append(
                    {
                        "candidate": candidate,
                        "rating_season": season,
                        "target_source": target_source,
                        "component": side,
                        "players": len(test),
                        "target_rmse": float(np.sqrt(np.mean(error**2))),
                        "target_correlation": float(
                            np.corrcoef(test[target].to_numpy(dtype=float), prediction)[0, 1]
                        ),
                    }
                )
            prior["net"] = prior["offense"] + prior["defense"]
            framed = _prior_frame(prior, candidate)
            framed["center_scale"] = (
                shrink_scale
                if specification["center_scale"] == "ryan_to_courtsignal_sd"
                else 1.0
            )
            framed["offense_target_rescale"] = scale_ratios["offense"]
            framed["defense_target_rescale"] = scale_ratios["defense"]
            rows.append(framed)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(target_metrics)


def _downstream_aio(
    priors: pd.DataFrame,
    legacy_cache: Path,
    matrix_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = []
    games = []
    metrics = []
    coverage = []
    for season in RATING_SEASONS:
        direct = _annual_from_frame(
            load_legacy_possessions(legacy_cache, (season,), game_types=("regular",)),
            season,
        )
        matrix_dir = matrix_root / f"5y_end_{season}"
        matrix_players = np.load(matrix_dir / "player_ids.npy")
        annual = _remap_annual(direct, matrix_players)
        for candidate in VARIANTS:
            prior = priors.loc[
                priors["candidate"].eq(candidate)
                & priors["Window_End"].eq(season)
            ]
            center, report = _center(prior, annual)
            scales = prior["center_scale"].unique()
            if len(scales) != 1:
                raise ValueError(f"Candidate {candidate} has inconsistent center scales.")
            beta, intercept = _solve(annual, center, scale=float(scales[0]))
            game = stored_evaluation_predictions(matrix_dir, beta, intercept)
            game["candidate"] = candidate
            game["rating_season"] = season
            game["test_season"] = season + 1
            game["squared_error"] = (
                game["actual_margin"] - game["predicted_margin"]
            ) ** 2
            games.append(game)
            metrics.append(
                {
                    "candidate": candidate,
                    "rating_season": season,
                    "test_season": season + 1,
                    **_game_metrics(game),
                }
            )
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
            coverage.append(
                {"candidate": candidate, "rating_season": season, **report}
            )
        print(f"Ryan-target AIO fold {season}->{season + 1}: complete", flush=True)
    return (
        pd.concat(ratings, ignore_index=True),
        pd.concat(games, ignore_index=True),
        pd.DataFrame(metrics),
        pd.DataFrame(coverage),
    )


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

    panel, selected = _panel(args.features, args.targets, args.ryan_ratings)
    priors, target_metrics = _fit_priors(panel, selected)
    ratings, games, game_metrics, coverage = _downstream_aio(
        priors, args.legacy_cache, args.matrix_root
    )
    summary = (
        game_metrics.groupby("candidate", as_index=False)
        .agg(
            folds=("test_season", "nunique"),
            mean_margin_rmse=("margin_rmse", "mean"),
            mean_margin_correlation=("margin_correlation", "mean"),
        )
        .sort_values("mean_margin_rmse", kind="stable")
    )
    bootstrap = _paired_bootstrap(
        games,
        baseline="courtsignal_target_weighted",
        draws=5_000,
        seed=20260827,
    )
    agreement = []
    for side in ("offense", "defense", "net"):
        agreement.append(
            {
                "component": side,
                "players": len(panel),
                "pearson": float(
                    panel[[f"courtsignal_{side}", f"ryan_{side}"]].corr().iloc[0, 1]
                ),
                "spearman": float(
                    panel[[f"courtsignal_{side}", f"ryan_{side}"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                ),
                "courtsignal_sd": float(panel[f"courtsignal_{side}"].std(ddof=0)),
                "ryan_sd": float(panel[f"ryan_{side}"].std(ddof=0)),
            }
        )
    agreement_frame = pd.DataFrame(agreement)

    config = {
        "rating_seasons": list(RATING_SEASONS),
        "test_seasons": [season + 1 for season in RATING_SEASONS],
        "features": {side: list(values) for side, values in selected.items()},
        "variants": VARIANTS,
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
            "runner": sha256_file(Path(__file__)),
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = (
        args.artifact_root
        / "research/ryan_target_spm"
        / f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "priors": priors,
        "ratings": ratings,
        "game_predictions": games,
        "game_metrics": game_metrics,
        "target_metrics": target_metrics,
        "summary": summary,
        "paired_bootstrap": bootstrap,
        "prior_coverage": coverage,
        "target_agreement": agreement_frame,
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
            "matched_panel_rows": len(panel),
            "matched_windows": sorted(panel["Window_End"].unique().tolist()),
            "identical_game_rows": not games.pivot(
                index=["test_season", "game_id"],
                columns="candidate",
                values="actual_margin",
            ).isna().any().any(),
            "component_identity_max_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"])
                .abs()
                .max()
            ),
            "season_2027_rows": 0,
        },
        "paths": {name: f"{name}.parquet" for name in outputs},
        "caveats": [
            "Ryan's historical rating files contain coefficients but no player-window exposure weights.",
            "The weighted Ryan arm uses CourtSignal exposure weights only to isolate the target change.",
            "The unweighted Ryan arm is the sensitivity test; neither arm is described as using Ryan player weights.",
            "The rescaled Ryan arm matches each side's target spread using training windows only.",
            "The shrunken CourtSignal arm keeps CourtSignal labels but applies the training-only Ryan-to-CourtSignal spread ratio to the AIO center.",
            "The matched feature set is the 126-offense and 50-defense base five-year SPM architecture, before later zTS and tracking additions.",
            "The future-game folds are reused diagnostics, not untouched confirmation.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(agreement_frame.to_string(index=False))
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
