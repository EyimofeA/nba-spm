#!/usr/bin/env python3
"""Compare frozen annual rich SPM with five-year Box15 on future games."""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions

try:
    import run_full_spm_history_ablation as base
    from run_aio_prior_calibration_precision import (
        _predictions,
        _solve as solve_precision,
        select_configuration,
    )
    from run_aio_prior_canonical_followup import _center, _remap_annual, _solve
except ModuleNotFoundError:  # Imported as research.run_* by tests.
    from research import run_full_spm_history_ablation as base
    from research.run_aio_prior_calibration_precision import (
        _predictions,
        _solve as solve_precision,
        select_configuration,
    )
    from research.run_aio_prior_canonical_followup import _center, _remap_annual, _solve


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "annual_rich_forward_aio_v1"
CONTRACT = ROOT / "research/experiments/annual_rich_forward_aio_v1.yml"
ANNUAL_RUN = ROOT / (
    "artifacts/research/annual_spm_learner_screen/"
    "annual_spm_learner_screen_v1_74808a8ae2"
)
BOX_RUN = ROOT / (
    "artifacts/research/compact_spm_comparison/"
    "compact_spm_comparison_v1_2a0f8a6f31"
)
MATRIX_ROOT = ROOT / (
    "research/rapm_lab/outputs/rolling_5y_2014_2026/"
    "rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices"
)
POSSESSION_CACHE = ROOT / "rapm/data/possession_cache"
RATING_SEASONS = tuple(range(2021, 2026))
SCORED_RATING_SEASONS = tuple(range(2022, 2026))
PENALTIES = (1500.0, 3000.0, 4500.0, 6000.0)
PRIOR_NAMES = ("annual_rich_spm", "five_year_box15")
MODEL_ORDER = (
    "annual_rich_spm",
    "five_year_box15",
    "annual_rich_spm_aio",
    "five_year_box15_aio",
)


def _annual_rich_priors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Recover the frozen selected rich learner without refitting it."""
    arms = []
    for side, learner in (("offense", "elastic_net"), ("defense", "ridge")):
        development = predictions.loc[
            predictions["Season"].eq(2021)
            & predictions["phase"].eq("learner_selection")
            & predictions["arm"].eq("audited_all")
            & predictions["side"].eq(side)
            & predictions["learner"].eq(learner),
            ["PLAYER_ID", "Season", "prediction"],
        ]
        diagnostic = predictions.loc[
            predictions["Season"].isin(RATING_SEASONS[1:])
            & predictions["phase"].eq("diagnostic")
            & predictions["arm"].eq("audited_all")
            & predictions["side"].eq(side)
            & predictions["learner"].eq(learner),
            ["PLAYER_ID", "Season", "prediction"],
        ]
        arm = pd.concat([development, diagnostic], ignore_index=True)
        if arm.duplicated(["PLAYER_ID", "Season"]).any():
            raise ValueError(f"Frozen {side} predictions have duplicate keys.")
        arms.append(arm.rename(columns={"prediction": f"prior_{side}_per_100"}))
    prior = arms[0].merge(
        arms[1], on=["PLAYER_ID", "Season"], how="inner", validate="one_to_one"
    ).rename(columns={"Season": "Window_End"})
    prior["prior_net_per_100"] = (
        prior["prior_offense_per_100"] + prior["prior_defense_per_100"]
    )
    prior["candidate"] = "annual_rich_spm"
    missing = sorted(set(RATING_SEASONS) - set(prior["Window_End"].unique()))
    if missing:
        raise ValueError(f"Frozen annual rich prior misses seasons {missing}.")
    return prior


def _load_priors() -> pd.DataFrame:
    rich = _annual_rich_priors(pd.read_parquet(ANNUAL_RUN / "predictions.parquet"))
    box = pd.read_parquet(BOX_RUN / "priors.parquet")
    box = box.loc[
        box["candidate"].eq("box_pipm") & box["Window_End"].isin(RATING_SEASONS)
    ].copy()
    box["candidate"] = "five_year_box15"
    priors = pd.concat([rich, box], ignore_index=True)
    expected = pd.MultiIndex.from_product(
        [PRIOR_NAMES, RATING_SEASONS], names=["candidate", "Window_End"]
    )
    observed = pd.MultiIndex.from_frame(
        priors[["candidate", "Window_End"]].drop_duplicates()
    )
    if not expected.equals(observed.sort_values()):
        raise ValueError("Prior candidates do not cover every required rating season.")
    return priors


def _standalone_predictions(matrix_dir: Path, annual, center: np.ndarray) -> pd.DataFrame:
    zero = np.zeros_like(center)
    zero_beta, zero_intercept = _solve(annual, zero, scale=0.0)
    beta = center.copy()
    beta[-1] = zero_beta[-1]
    frame = stored_evaluation_predictions(matrix_dir, beta, zero_intercept)
    frame["squared_error"] = (
        frame["actual_margin"] - frame["predicted_margin"]
    ) ** 2
    return frame


def _rating_frame(candidate: str, season: int, annual, beta: np.ndarray) -> pd.DataFrame:
    n = len(annual.players)
    frame = pd.DataFrame(
        {
            "PLAYER_ID": annual.players,
            "offense": 100.0 * beta[:n],
            "defense": -100.0 * beta[n : 2 * n],
            "Poss_Off": annual.off_possessions,
            "Poss_Def": annual.def_possessions,
        }
    )
    frame["net"] = frame["offense"] + frame["defense"]
    frame["candidate"] = candidate
    frame["rating_season"] = season
    return frame


def _assert_identical_games(games: pd.DataFrame) -> None:
    for season, frame in games.groupby("test_season"):
        counts = frame.groupby("candidate")["game_id"].nunique()
        if set(counts.index) != set(MODEL_ORDER) or counts.nunique() != 1:
            raise ValueError(f"Candidates do not score identical {season} games.")
        actual = frame.pivot(index="game_id", columns="candidate", values="actual_margin")
        if actual.isna().any().any() or not actual.nunique(axis=1).eq(1).all():
            raise ValueError(f"Candidates do not share complete {season} outcomes.")


def _complementarity_diagnostics(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prior in PRIOR_NAMES:
        posterior = f"{prior}_aio"
        for season in sorted(games["test_season"].unique()):
            left = games.loc[
                games["candidate"].eq(prior) & games["test_season"].eq(season)
            ].sort_values("game_id")
            right = games.loc[
                games["candidate"].eq(posterior)
                & games["test_season"].eq(season)
            ].sort_values("game_id")
            if not left["game_id"].reset_index(drop=True).equals(
                right["game_id"].reset_index(drop=True)
            ):
                raise ValueError("Prior and posterior games are not aligned.")
            actual = left["actual_margin"].to_numpy(dtype=float)
            prior_prediction = left["predicted_margin"].to_numpy(dtype=float)
            posterior_prediction = right["predicted_margin"].to_numpy(dtype=float)
            update = posterior_prediction - prior_prediction
            prior_residual = actual - prior_prediction
            rows.append(
                {
                    "prior": prior,
                    "test_season": int(season),
                    "prior_prediction_sd": float(np.std(prior_prediction, ddof=1)),
                    "rapm_update_sd": float(np.std(update, ddof=1)),
                    "prior_update_correlation": float(
                        np.corrcoef(prior_prediction, update)[0, 1]
                    ),
                    "update_prior_residual_correlation": float(
                        np.corrcoef(update, prior_residual)[0, 1]
                    ),
                    "aio_mse_improvement": float(
                        np.mean(prior_residual**2 - (actual - posterior_prediction) ** 2)
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("Experiment contract does not match the runner.")
    if (
        tuple(contract["information_cutoff"]["scored_rating_seasons"])
        != SCORED_RATING_SEASONS
    ):
        raise ValueError("Scored rating seasons changed.")

    priors = _load_priors()
    annual, reconstruction = base._annual_bundles(POSSESSION_CACHE, MATRIX_ROOT)
    centers = {}
    bundles = {}
    coverage_rows = []
    grid_rows = []
    for season in RATING_SEASONS:
        matrix_dir = MATRIX_ROOT / f"5y_end_{season}"
        players = np.load(matrix_dir / "player_ids.npy")
        bundle = _remap_annual(annual[season], players)
        bundles[season] = bundle
        for candidate in PRIOR_NAMES:
            prior = priors.loc[
                priors["candidate"].eq(candidate) & priors["Window_End"].eq(season)
            ]
            center, coverage = _center(prior, bundle)
            centers[(season, candidate)] = center
            coverage_rows.append(
                {"candidate": candidate, "rating_season": season, **coverage}
            )
            for offense_penalty, defense_penalty in itertools.product(
                PENALTIES, repeat=2
            ):
                grid_rows.append(
                    _predictions(
                        matrix_dir,
                        bundle,
                        center,
                        offense_penalty=offense_penalty,
                        defense_penalty=defense_penalty,
                    ).assign(
                        candidate=candidate,
                        rating_season=season,
                        test_season=season + 1,
                        offense_penalty=offense_penalty,
                        defense_penalty=defense_penalty,
                    )
                )
    grid = pd.concat(grid_rows, ignore_index=True)

    game_rows = []
    rating_rows = []
    selections = []
    for season in SCORED_RATING_SEASONS:
        matrix_dir = MATRIX_ROOT / f"5y_end_{season}"
        bundle = bundles[season]
        for candidate in PRIOR_NAMES:
            center = centers[(season, candidate)]
            standalone = _standalone_predictions(matrix_dir, bundle, center).assign(
                candidate=candidate,
                rating_season=season,
                test_season=season + 1,
            )
            game_rows.append(standalone)
            rating_rows.append(_rating_frame(candidate, season, bundle, center))

            history = grid.loc[
                grid["candidate"].eq(candidate) & grid["rating_season"].lt(season)
            ]
            choice = select_configuration(
                history, ["offense_penalty", "defense_penalty"]
            )
            posterior = _predictions(matrix_dir, bundle, center, **choice).assign(
                candidate=f"{candidate}_aio",
                rating_season=season,
                test_season=season + 1,
            )
            game_rows.append(posterior)
            beta, _ = solve_precision(bundle, center, **choice)
            rating_rows.append(_rating_frame(f"{candidate}_aio", season, bundle, beta))
            selections.append(
                {
                    "candidate": candidate,
                    "rating_season": season,
                    "selection_rating_seasons": ",".join(
                        map(str, sorted(history["rating_season"].unique()))
                    ),
                    **choice,
                }
            )

    games = pd.concat(game_rows, ignore_index=True).sort_values(
        ["candidate", "test_season", "game_id"]
    )
    ratings = pd.concat(rating_rows, ignore_index=True)
    _assert_identical_games(games)
    base.MODEL_ORDER = MODEL_ORDER
    base.PRIMARY_PAIRS = {
        frozenset(("annual_rich_spm", "five_year_box15")),
        frozenset(("annual_rich_spm_aio", "five_year_box15_aio")),
    }
    folds, summary = base._game_metrics_frames(games)
    intervals, pairs = base.paired_game_bootstrap(
        games,
        draws=int(contract["evaluation"]["uncertainty"]["draws"]),
        seed=int(contract["evaluation"]["uncertainty"]["seed"]),
    )
    complementarity = _complementarity_diagnostics(games)

    source_paths = {
        "contract": CONTRACT,
        "runner": Path(__file__),
        "annual_manifest": ANNUAL_RUN / "run.json",
        "annual_predictions": ANNUAL_RUN / "predictions.parquet",
        "box_manifest": BOX_RUN / "run.json",
        "box_priors": BOX_RUN / "priors.parquet",
        **{
            f"matrix_manifest_{season}": MATRIX_ROOT
            / f"5y_end_{season}/manifest.json"
            for season in RATING_SEASONS
        },
        **{
            f"annual_possessions_{season}": POSSESSION_CACHE
            / f"matchups_{season}.parquet"
            for season in range(2020, 2024)
        },
    }
    config = {
        "experiment_id": EXPERIMENT_ID,
        "rating_seasons": list(RATING_SEASONS),
        "scored_rating_seasons": list(SCORED_RATING_SEASONS),
        "model_order": list(MODEL_ORDER),
        "penalty_grid": list(PENALTIES),
        "sources": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:10]
    output = ROOT / "artifacts/research/annual_rich_forward_aio" / (
        f"{EXPERIMENT_ID}_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    outputs = {
        "priors.parquet": priors,
        "ratings.parquet": ratings,
        "precision_selections.parquet": pd.DataFrame(selections),
        "game_predictions.parquet": games,
        "fold_metrics.parquet": folds,
        "summary.parquet": summary,
        "bootstrap_model_intervals.parquet": intervals,
        "paired_bootstrap.parquet": pairs,
        "complementarity_diagnostics.parquet": complementarity,
        "prior_coverage.parquet": pd.DataFrame(coverage_rows),
        "matrix_reconstruction.parquet": reconstruction,
    }
    for name, frame in outputs.items():
        frame.to_parquet(output / name, index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": contract,
        "config": config,
        "quality": {
            "identical_games_within_fold": True,
            "maximum_component_identity_error": float(
                (ratings["offense"] + ratings["defense"] - ratings["net"]).abs().max()
            ),
            "maximum_loaded_season": 2026,
        },
        "files": {},
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    for name, frame in outputs.items():
        run["files"][name] = {
            "path": name,
            "rows": len(frame),
            "sha256": sha256_file(output / name),
        }
    write_json_atomic(run, output / "run.json")
    print(output)
    print(summary.to_string(index=False))
    print("\nPrimary paired comparisons")
    print(pairs.loc[pairs["primary_comparison"]].to_string(index=False))


if __name__ == "__main__":
    main()
