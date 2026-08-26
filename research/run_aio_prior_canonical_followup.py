#!/usr/bin/env python3
"""Run the BoxPIPM prior bake-off on canonical 2025 and 2026 games."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, load_npz
from scipy.sparse.linalg import cg, spsolve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import (
    BOX_PIPM_STYLE_FEATURES,
    _fit as _fit_box,
    _select_alpha,
)
from nba_impact.models.rapm import build_design, load_legacy_possessions
from nba_impact.models.rapm_sufficient_statistics import stored_evaluation_predictions
from nba_impact.paths import LEGACY_PLAYER_SHEETS, LEGACY_POSSESSION_CACHE
from run_aio_prior_bakeoff import (
    ALPHA_GRID,
    _annual_onoff,
    _game_metrics,
    _paired_bootstrap,
    _prior_frame,
    _rolling_onoff,
    _team_context,
)
from run_spm_cheating_ladder import _extend_features


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "aio_prior_canonical_followup_v1"
RATING_SEASONS = (2024, 2025)


def _existing_spm_priors(path: Path) -> pd.DataFrame:
    """Load the two frozen SPM arms for this follow-up's rating seasons.

    The development bake-off helper is deliberately scoped to 2021--23.  Do
    not reuse it here: doing so silently produces empty 2024--25 prior frames.
    """
    source = pd.read_parquet(path)
    source = source.loc[
        source["Window_End"].isin(RATING_SEASONS)
        & source["variant"].isin(["baseline", "selected_combined"])
    ].copy()
    source["candidate"] = source["variant"].map(
        {"baseline": "five_year_spm", "selected_combined": "selected_five_year_spm"}
    )
    output = source[[
        "PLAYER_ID",
        "Window_End",
        "prior_offense_per_100",
        "prior_defense_per_100",
        "prior_net_per_100",
        "candidate",
    ]]
    missing = sorted(set(RATING_SEASONS) - set(output["Window_End"].unique()))
    if missing:
        raise ValueError(f"Frozen SPM prior artifact is missing seasons: {missing}")
    return output


@dataclass(frozen=True)
class AnnualSufficientStatistics:
    season: int
    players: np.ndarray
    xtx: csr_matrix
    xty_centered: np.ndarray
    off_possessions: np.ndarray
    def_possessions: np.ndarray
    base_intercept: float
    rows: int


def _column_map(source_players: np.ndarray, target_players: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(target_players, source_players)
    if (
        (positions >= len(target_players)).any()
        or not np.array_equal(target_players[positions], source_players)
    ):
        raise ValueError("Target player universe does not contain every source player.")
    n_target = len(target_players)
    return np.concatenate([positions, n_target + positions, np.asarray([2 * n_target])])


def _remap_sparse(matrix: csr_matrix, mapping: np.ndarray, size: int) -> csr_matrix:
    coo = matrix.tocoo()
    return csr_matrix((coo.data, (mapping[coo.row], mapping[coo.col])), shape=(size, size))


def _remap_vector(values: np.ndarray, mapping: np.ndarray, size: int) -> np.ndarray:
    output = np.zeros(size, dtype=float)
    output[mapping] = values
    return output


def _remap_annual(
    bundle: AnnualSufficientStatistics, target_players: np.ndarray
) -> AnnualSufficientStatistics:
    active = (bundle.off_possessions + bundle.def_possessions) > 1e-8
    active_indices = np.flatnonzero(active)
    source_columns = np.concatenate(
        [
            active_indices,
            len(bundle.players) + active_indices,
            np.asarray([2 * len(bundle.players)]),
        ]
    )
    source_players = bundle.players[active]
    mapping = _column_map(source_players, target_players)
    size = 2 * len(target_players) + 1
    player_positions = mapping[: len(source_players)]
    off = np.zeros(len(target_players), dtype=float)
    deff = np.zeros(len(target_players), dtype=float)
    off[player_positions] = bundle.off_possessions[active]
    deff[player_positions] = bundle.def_possessions[active]
    compact_xtx = bundle.xtx[source_columns][:, source_columns].tocsr()
    compact_xty = bundle.xty_centered[source_columns]
    return AnnualSufficientStatistics(
        season=bundle.season,
        players=target_players,
        xtx=_remap_sparse(compact_xtx, mapping, size),
        xty_centered=_remap_vector(compact_xty, mapping, size),
        off_possessions=off,
        def_possessions=deff,
        base_intercept=bundle.base_intercept,
        rows=bundle.rows,
    )


def _annual_from_frame(frame: pd.DataFrame, season: int) -> AnnualSufficientStatistics:
    design = build_design(frame, include_home=True)
    intercept = float(design.y.mean())
    centered = design.y - intercept
    n = len(design.players)
    return AnnualSufficientStatistics(
        season=season,
        players=design.players,
        xtx=(design.X.T @ design.X).tocsr(),
        xty_centered=np.asarray(design.X.T @ centered).ravel(),
        off_possessions=np.asarray(design.X[:, :n].sum(axis=0)).ravel(),
        def_possessions=np.asarray(design.X[:, n : 2 * n].sum(axis=0)).ravel(),
        base_intercept=intercept,
        rows=len(frame),
    )


def _recover_annual(
    matrix_dir: Path,
    season: int,
    prior_annual: dict[int, AnnualSufficientStatistics],
) -> tuple[AnnualSufficientStatistics, dict]:
    manifest = json.loads((matrix_dir / "manifest.json").read_text())
    players = np.load(matrix_dir / "player_ids.npy")
    rolling_xtx = load_npz(matrix_dir / "train_xtx.npz").tocsr()
    rolling_xty = np.load(matrix_dir / "train_xty_centered.npy")
    rolling_off = np.load(matrix_dir / "train_off_possessions.npy")
    rolling_def = np.load(matrix_dir / "train_def_possessions.npy")
    size = len(rolling_xty)
    previous = tuple(range(season - 4, season))
    mapped = [_remap_annual(prior_annual[value], players) for value in previous]
    prior_xtx = sum((value.xtx for value in mapped), start=csr_matrix((size, size)))
    prior_xty = sum((value.xty_centered for value in mapped), start=np.zeros(size))
    prior_off = sum((value.off_possessions for value in mapped), start=np.zeros(len(players)))
    prior_def = sum((value.def_possessions for value in mapped), start=np.zeros(len(players)))
    prior_rows = sum(value.rows for value in mapped)
    annual = AnnualSufficientStatistics(
        season=season,
        players=players,
        xtx=(rolling_xtx - prior_xtx).tocsr(),
        xty_centered=rolling_xty - prior_xty,
        off_possessions=rolling_off - prior_off,
        def_possessions=rolling_def - prior_def,
        base_intercept=float(
            manifest["train"]["season_environment"]["season_points_per_possession"][str(season)]
        ),
        rows=int(manifest["train"]["possession_rows"]) - prior_rows,
    )
    reconstructed_xtx = annual.xtx + prior_xtx
    matrix_error = reconstructed_xtx - rolling_xtx
    quality = {
        "season": season,
        "rows": annual.rows,
        "players": len(players),
        "xtx_max_error": float(np.max(np.abs(matrix_error.data))) if matrix_error.nnz else 0.0,
        "xty_max_error": float(np.max(np.abs(annual.xty_centered + prior_xty - rolling_xty))),
        "off_possession_max_error": float(np.max(np.abs(annual.off_possessions + prior_off - rolling_off))),
        "def_possession_max_error": float(np.max(np.abs(annual.def_possessions + prior_def - rolling_def))),
        "minimum_off_possessions": float(annual.off_possessions.min()),
        "minimum_def_possessions": float(annual.def_possessions.min()),
    }
    if max(
        quality["xtx_max_error"],
        quality["xty_max_error"],
        quality["off_possession_max_error"],
        quality["def_possession_max_error"],
    ) > 1e-8:
        raise ValueError(f"Annual matrix reconstruction failed for {season}: {quality}")
    if annual.rows <= 0 or quality["minimum_off_possessions"] < -1e-8 or quality["minimum_def_possessions"] < -1e-8:
        raise ValueError(f"Recovered annual exposure is invalid for {season}: {quality}")
    return annual, quality


def _box_and_pipm_like_priors(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    onoff: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = features.merge(
        targets, on=["PLAYER_ID", "Window_End"], how="inner", validate="one_to_one"
    ).merge(
        onoff, on=["PLAYER_ID", "Window_End"], how="left", validate="one_to_one"
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    box_rows: list[pd.DataFrame] = []
    pipm_rows: list[pd.DataFrame] = []
    for season in RATING_SEASONS:
        train = panel.loc[panel["Window_End"].lt(season)].copy()
        test = panel.loc[panel["Window_End"].eq(season)].copy()
        box = test[["PLAYER_ID", "Window_End"]].copy()
        pipm = test[["PLAYER_ID", "Window_End"]].copy()
        for side in ("offense", "defense"):
            target = f"target_{side}"
            alpha = _select_alpha(
                train.rename(columns={"Window_End": "Season"}),
                BOX_PIPM_STYLE_FEATURES,
                target,
                ALPHA_GRID,
            )
            box_model = _fit_box(train, BOX_PIPM_STYLE_FEATURES, target, alpha)
            train[f"box_{side}"] = box_model.predict(train.loc[:, BOX_PIPM_STYLE_FEATURES])
            box[side] = box_model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
            context = f"raw_onoff_{side}_5y"
            residual_model = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=100.0)),
                ]
            )
            residual_model.fit(
                train[[context]],
                train[target] - train[f"box_{side}"],
                ridge__sample_weight=train["sample_weight"],
            )
            pipm[side] = box[side] + residual_model.predict(test[[context]])
        for frame in (box, pipm):
            frame["net"] = frame["offense"] + frame["defense"]
        box_rows.append(box)
        pipm_rows.append(pipm)
    return (
        _prior_frame(pd.concat(box_rows, ignore_index=True), "box_pipm_style_prior"),
        _prior_frame(pd.concat(pipm_rows, ignore_index=True), "pipm_like_raw_onoff_prior"),
    )


def _center(prior: pd.DataFrame, annual: AnnualSufficientStatistics) -> tuple[np.ndarray, dict]:
    indexed = prior.set_index("PLAYER_ID")
    offense = pd.to_numeric(indexed["prior_offense_per_100"], errors="raise").reindex(annual.players)
    defense = pd.to_numeric(indexed["prior_defense_per_100"], errors="raise").reindex(annual.players)
    present = offense.notna() & defense.notna()
    off = offense.fillna(0.0).to_numpy(dtype=float) / 100.0
    deff = -defense.fillna(0.0).to_numpy(dtype=float) / 100.0
    off -= np.average(off, weights=annual.off_possessions)
    deff -= np.average(deff, weights=annual.def_possessions)
    return np.concatenate([off, deff, np.zeros(1)]), {
        "players": len(annual.players),
        "players_with_prior": int(present.sum()),
        "off_possession_coverage": float(np.average(present, weights=annual.off_possessions)),
        "def_possession_coverage": float(np.average(present, weights=annual.def_possessions)),
    }


def _solve(
    annual: AnnualSufficientStatistics, center: np.ndarray, *, scale: float
) -> tuple[np.ndarray, float]:
    n = len(annual.players)
    penalty = np.concatenate([
        np.full(n, 3000.0), np.full(n, 3000.0), np.asarray([300.0])
    ])
    lhs = annual.xtx + diags(penalty, format="csr")
    rhs = annual.xty_centered + scale * penalty * center
    try:
        raw, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        raw, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        raw = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(raw)
    off_mean = float(np.average(beta[:n], weights=annual.off_possessions))
    def_mean = float(np.average(beta[n : 2 * n], weights=annual.def_possessions))
    beta[:n] -= off_mean
    beta[n : 2 * n] -= def_mean
    intercept = annual.base_intercept + 5.0 * (off_mean + def_mean)
    return beta, intercept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--legacy-cache",
        type=Path,
        default=Path(os.environ.get("NBA_IMPACT_LEGACY_POSSESSION_CACHE", LEGACY_POSSESSION_CACHE)),
    )
    parser.add_argument(
        "--legacy-player-sheets",
        type=Path,
        default=Path(os.environ.get("NBA_IMPACT_LEGACY_PLAYER_SHEETS", LEGACY_PLAYER_SHEETS)),
    )
    parser.add_argument(
        "--player-sheet-root",
        type=Path,
        default=ROOT / "data/lake/bronze/gabriel_player_sheets/revision=54b57cf/year_totals",
    )
    parser.add_argument(
        "--schedule-root",
        type=Path,
        default=ROOT / "data/lake/bronze/official_game_schedule_1997_2026",
    )
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "research/rapm_lab/outputs/rolling_5y_2014_2026/rolling_5y_rapm_2014_2026_a7754bfb77/lambda_matrices",
    )
    parser.add_argument(
        "--spm-priors",
        type=Path,
        default=ROOT / "artifacts/models/five_year_spm_feature_research/five_year_spm_feature_research_v1_93c148510e/spm_predictions.parquet",
    )
    parser.add_argument(
        "--reference-features",
        type=Path,
        default=ROOT / "artifacts/research/spm_target_horizon_full/spm_target_horizon_full_v1_f0777db1d4/features_5y.parquet",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "artifacts/models/five_year_target_spm/five_year_target_spm_v1_65550acb79/five_year_targets.parquet",
    )
    parser.add_argument(
        "--reference-aio-metrics",
        type=Path,
        default=ROOT / "artifacts/models/five_year_spm_feature_research/five_year_spm_feature_research_v1_93c148510e/aio_metrics.parquet",
    )
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()

    annual: dict[int, AnnualSufficientStatistics] = {}
    for season in range(2020, 2024):
        frame = load_legacy_possessions(args.legacy_cache, (season,), game_types=("regular",))
        annual[season] = _annual_from_frame(frame, season)
    reconstruction = []
    for season in RATING_SEASONS:
        bundle, quality = _recover_annual(
            args.matrix_root / f"5y_end_{season}", season, annual
        )
        annual[season] = bundle
        reconstruction.append(quality)

    features = _extend_features(args.reference_features, args.legacy_player_sheets, 2025)
    targets = pd.read_parquet(args.targets)
    team = _team_context(args.schedule_root, range(2014, 2026))
    annual_onoff = _annual_onoff(args.player_sheet_root, team, range(2014, 2026))
    rolling_onoff = _rolling_onoff(annual_onoff, tuple(range(2018, 2026)))
    box, pipm = _box_and_pipm_like_priors(features, targets, rolling_onoff)
    priors = pd.concat([_existing_spm_priors(args.spm_priors), box, pipm], ignore_index=True)
    priors = priors.loc[priors["Window_End"].isin(RATING_SEASONS)].copy()

    candidates = (
        "zero_prior_rapm",
        "five_year_spm",
        "selected_five_year_spm",
        "box_pipm_style_prior",
        "pipm_like_raw_onoff_prior",
    )
    metrics = []
    games = []
    ratings = []
    coverage = []
    for season in RATING_SEASONS:
        bundle = annual[season]
        matrix_dir = args.matrix_root / f"5y_end_{season}"
        for candidate in candidates:
            if candidate == "zero_prior_rapm":
                center = np.zeros(2 * len(bundle.players) + 1)
                scale = 0.0
                report = {
                    "players": len(bundle.players),
                    "players_with_prior": 0,
                    "off_possession_coverage": 0.0,
                    "def_possession_coverage": 0.0,
                }
            else:
                prior = priors.loc[
                    priors["candidate"].eq(candidate) & priors["Window_End"].eq(season)
                ]
                center, report = _center(prior, bundle)
                scale = 1.0
            beta, intercept = _solve(bundle, center, scale=scale)
            game = stored_evaluation_predictions(matrix_dir, beta, intercept)
            metric = _game_metrics(game)
            metrics.append({"candidate": candidate, "rating_season": season, "test_season": season + 1, **metric})
            game["candidate"] = candidate
            game["rating_season"] = season
            game["test_season"] = season + 1
            game["squared_error"] = (game["actual_margin"] - game["predicted_margin"]) ** 2
            games.append(game)
            n = len(bundle.players)
            rating = pd.DataFrame({
                "PLAYER_ID": bundle.players,
                "offense": 100.0 * beta[:n],
                "defense": -100.0 * beta[n : 2 * n],
                "Poss_Off": bundle.off_possessions,
                "Poss_Def": bundle.def_possessions,
            })
            rating["net"] = rating["offense"] + rating["defense"]
            rating["candidate"] = candidate
            rating["rating_season"] = season
            ratings.append(rating)
            coverage.append({"candidate": candidate, "rating_season": season, **report})
        print(f"canonical AIO prior fold {season}->{season + 1}: complete", flush=True)

    metric_frame = pd.DataFrame(metrics)
    game_frame = pd.concat(games, ignore_index=True)
    rating_frame = pd.concat(ratings, ignore_index=True)
    coverage_frame = pd.DataFrame(coverage)
    if game_frame.duplicated(["candidate", "test_season", "game_id"]).any():
        raise ValueError("Game prediction keys are duplicated.")
    if game_frame.groupby(["test_season", "game_id"])["actual_margin"].nunique().max() != 1:
        raise ValueError("Candidates do not share identical actual margins.")
    bootstrap = _paired_bootstrap(game_frame, baseline="selected_five_year_spm")
    summary = metric_frame.groupby("candidate", as_index=False).agg(
        folds=("test_season", "nunique"),
        mean_margin_rmse=("margin_rmse", "mean"),
        mean_margin_correlation=("margin_correlation", "mean"),
        minimum_games=("games", "min"),
    ).sort_values("mean_margin_rmse", kind="stable")

    reference = pd.read_parquet(args.reference_aio_metrics)
    reference = reference.loc[
        reference["variant"].eq("selected_combined")
        & reference["test_season"].isin([2025, 2026])
    ]
    reproduced = metric_frame.loc[metric_frame["candidate"].eq("selected_five_year_spm")]
    check = reference.merge(reproduced, on="test_season", suffixes=("_reference", "_reproduced"))
    reference_rmse_error = float(
        (check["margin_rmse_reference"] - check["margin_rmse_reproduced"]).abs().max()
    )
    reference_correlation_error = float(
        (check["margin_correlation_reference"] - check["margin_correlation_reproduced"]).abs().max()
    )
    if max(reference_rmse_error, reference_correlation_error) > 1e-8:
        raise ValueError(
            "Recovered one-season matrices do not reproduce the frozen AIO: "
            f"rmse_error={reference_rmse_error}, "
            f"correlation_error={reference_correlation_error}, "
            f"rows={check.to_dict(orient='records')}"
        )

    source_paths = {
        "contract": ROOT / "research/experiments/aio_prior_canonical_followup_v1.yml",
        "spm_priors": args.spm_priors,
        "reference_features": args.reference_features,
        "targets": args.targets,
        "reference_aio_metrics": args.reference_aio_metrics,
    }
    config = {
        "rating_seasons": list(RATING_SEASONS),
        "test_seasons": [2025, 2026],
        "lambda_off": 3000.0,
        "lambda_def": 3000.0,
        "lambda_home": 300.0,
        "center_scale": 1.0,
        "source_hashes": {
            **{name: sha256_file(path) for name, path in source_paths.items()},
            "runner": sha256_file(Path(__file__)),
            "rolling_matrices": {
                str(season): {
                    name: sha256_file(args.matrix_root / f"5y_end_{season}" / name)
                    for name in (
                        "manifest.json", "train_xtx.npz", "train_xty_centered.npy",
                        "train_off_possessions.npy", "train_def_possessions.npy",
                        "player_ids.npy", "evaluation_game_design.npz",
                        "evaluation_game_ids.npy", "evaluation_actual_margin.npy",
                        "evaluation_intercept_multiplier.npy",
                    )
                }
                for season in RATING_SEASONS
            },
        },
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    output = args.artifact_root / "research" / "aio_prior_canonical_followup" / f"{EXPERIMENT_ID}_{identity}"
    output.mkdir(parents=True, exist_ok=False)
    priors.to_parquet(output / "priors.parquet", index=False)
    rating_frame.to_parquet(output / "posterior_ratings.parquet", index=False)
    metric_frame.to_parquet(output / "game_metrics.parquet", index=False)
    game_frame.to_parquet(output / "game_predictions.parquet", index=False)
    coverage_frame.to_parquet(output / "prior_coverage.parquet", index=False)
    pd.DataFrame(reconstruction).to_parquet(output / "matrix_reconstruction.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    bootstrap.to_parquet(output / "paired_bootstrap.parquet", index=False)
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "matrix_reconstruction": reconstruction,
            "frozen_aio_rmse_reproduction_max_error": reference_rmse_error,
            "frozen_aio_correlation_reproduction_max_error": reference_correlation_error,
            "component_identity_max_error": float(
                (rating_frame["offense"] + rating_frame["defense"] - rating_frame["net"]).abs().max()
            ),
            "identical_game_rows": True,
            "season_2027_rows": 0,
        },
        "paths": {
            "priors": "priors.parquet",
            "posterior_ratings": "posterior_ratings.parquet",
            "game_metrics": "game_metrics.parquet",
            "game_predictions": "game_predictions.parquet",
            "prior_coverage": "prior_coverage.parquet",
            "matrix_reconstruction": "matrix_reconstruction.parquet",
            "summary": "summary.parquet",
            "paired_bootstrap": "paired_bootstrap.parquet",
        },
        "caveats": [
            "Seasons 2025 and 2026 are reused diagnostics, not untouched confirmation.",
            "BoxPIPM-style is the reproducible box-only portion, not full historical PIPM.",
            "The raw-on-off PIPM-like arm double counts lineup outcome evidence and is not eligible for selection.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    print(summary.to_string(index=False), flush=True)
    print(bootstrap.to_string(index=False), flush=True)
    print(json.dumps(run, indent=2), flush=True)


if __name__ == "__main__":
    main()
