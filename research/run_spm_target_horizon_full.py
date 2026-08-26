"""Full matched-window 1y/3y/5y/6y/expanding SPM and AIO comparison."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.data.statistical_features import _aggregate_window, _load_source
from nba_impact.data.statistical_features_v2 import _engineer_window
from nba_impact.models.prior_informed_rapm import build_prior_center
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    _game_margin_metrics,
    build_design,
    fit_coefficient_center_path,
    fit_coefficients,
    load_unified_terminal_possessions,
)
from nba_impact.models.single_season_spm import _selected_single_season_features


ROOT = Path(__file__).resolve().parents[1]
PILOT_RUNNER = ROOT / "research/run_spm_target_horizon_pilot.py"
PILOT_SPEC = importlib.util.spec_from_file_location(
    "run_spm_target_horizon_pilot_helpers", PILOT_RUNNER
)
if PILOT_SPEC is None or PILOT_SPEC.loader is None:
    raise ImportError(f"Cannot load target-horizon helpers from {PILOT_RUNNER}.")
PILOT_MODULE = importlib.util.module_from_spec(PILOT_SPEC)
PILOT_SPEC.loader.exec_module(PILOT_MODULE)
_fit_spm_fold = PILOT_MODULE._fit_spm_fold
_rating_table = PILOT_MODULE._rating_table
_write_parquet_atomic = PILOT_MODULE._write_parquet_atomic
DEFAULT_CONTRACT = ROOT / "research/experiments/spm_target_horizon_full_v1.yml"
REFERENCE_RUN = ROOT / "artifacts/models/single_season_spm/single_season_spm_v1_bff6060df6"
OUTPUT_ROOT = ROOT / "artifacts/research/spm_target_horizon_full"


def _load_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text())
    expected = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": "spm_target_horizon_full_v1",
        "status": "preregistered_development",
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(f"{field} must be {value!r}.")
    test_seasons = tuple(int(value) for value in contract["test_seasons"])
    window_ends = tuple(int(value) for value in contract["window_ends"])
    if tuple(sorted(set(test_seasons))) != test_seasons:
        raise ValueError("test_seasons must be unique and increasing.")
    if tuple(sorted(set(window_ends))) != window_ends:
        raise ValueError("window_ends must be unique and increasing.")
    if max(test_seasons) > 2024 or max(window_ends) > 2023:
        raise ValueError("Development must stop before Season 2025.")
    if set(contract["horizons"]) != {"1y", "3y", "5y", "6y", "expanding"}:
        raise ValueError("The full contract requires all five horizons.")
    return contract


def _window_seasons(spec: dict, end: int) -> tuple[int, ...]:
    if spec["kind"] == "rolling":
        length = int(spec["seasons"])
        return tuple(range(end - length + 1, end + 1))
    if spec["kind"] == "expanding":
        start = int(spec["start_season"])
        if end < start:
            raise ValueError("Expanding window ends before its start.")
        return tuple(range(start, end + 1))
    raise ValueError(f"Unknown horizon kind {spec['kind']!r}.")


def _target_ratings(design, beta: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    n = len(design.players)
    train = design.X[mask]
    off_count = np.asarray(train[:, :n].sum(axis=0)).ravel()
    def_count = np.asarray(train[:, n : 2 * n].sum(axis=0)).ravel()
    output = pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "target_offense": 100.0 * beta[:n],
            "target_defense": -100.0 * beta[n : 2 * n],
            "Poss_Off": off_count,
            "Poss_Def": def_count,
        }
    )
    output["target_net"] = output["target_offense"] + output["target_defense"]
    return output.loc[np.minimum(output["Poss_Off"], output["Poss_Def"]).gt(0)].copy()


def _load_feature_sources(contract: dict) -> tuple[dict[int, pd.DataFrame], dict[str, str]]:
    specs = contract["horizons"]
    ends = tuple(int(value) for value in contract["window_ends"])
    required = {
        season
        for spec in specs.values()
        for end in ends
        for season in _window_seasons(spec, end)
    }
    required.update(
        season for end in ends for season in range(end - 2, end + 1)
    )
    source_dir = ROOT / contract["features"]["source_dir"]
    loaded: dict[int, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for season in sorted(required):
        path = source_dir / f"{season}.csv"
        loaded[season] = _load_source(path, season)[0]
        hashes[str(season)] = sha256_file(path)
    return loaded, hashes


def _build_feature_panel(
    loaded: dict[int, pd.DataFrame], spec: dict, window_ends: tuple[int, ...]
) -> pd.DataFrame:
    outputs = []
    for end in window_ends:
        seasons = _window_seasons(spec, end)
        frames = [loaded[season] for season in seasons]
        temporal = [
            _aggregate_window([loaded[season]], season)
            for season in range(end - 2, end + 1)
        ]
        base = _aggregate_window(frames, end)
        outputs.append(_engineer_window(base, frames, temporal))
    output = pd.concat(outputs, ignore_index=True)
    if output.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("Matched-window feature keys are not unique.")
    return output


def _source_hashes(contract_path: Path, contract: dict) -> dict[str, object]:
    cache = ROOT / "rapm/data/possession_cache"
    possession_seasons = {
        season
        for spec in contract["horizons"].values()
        for end in contract["window_ends"]
        for season in _window_seasons(spec, int(end))
        if season < 2024
    }
    possession_seasons.update(
        int(season) for season in contract["test_seasons"] if int(season) < 2024
    )
    return {
        "contract": sha256_file(contract_path),
        "runner": sha256_file(Path(__file__)),
        "pilot_helpers": sha256_file(
            PILOT_RUNNER
        ),
        "reference_spm": sha256_file(REFERENCE_RUN / "run.json"),
        "legacy_possessions": {
            str(season): sha256_file(cache / f"matchups_{season}.parquet")
            for season in sorted(possession_seasons)
        },
        "current_possessions": sha256_file(
            ROOT / "data/lake/silver/possessions.parquet"
        ),
        "current_lineups": sha256_file(
            ROOT / "data/lake/silver/possession_lineup_segments.parquet"
        ),
    }


def _identity_inputs(
    contract_path: Path, contract: dict
) -> tuple[dict[int, pd.DataFrame], dict[str, object]]:
    """Load and hash every resume-sensitive input before choosing a run path."""
    source_hashes = _source_hashes(contract_path, contract)
    loaded_features, feature_source_hashes = _load_feature_sources(contract)
    source_hashes["statistical_sources"] = feature_source_hashes
    return loaded_features, source_hashes


def run(contract_path: Path = DEFAULT_CONTRACT) -> dict:
    contract = _load_contract(contract_path)
    loaded_features, source_hashes = _identity_inputs(contract_path, contract)
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"spm_target_horizon_full_v1_{identity}"
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())
    checkpoint = output / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)

    window_ends = tuple(int(value) for value in contract["window_ends"])
    test_seasons = tuple(int(value) for value in contract["test_seasons"])
    all_possession_seasons = sorted(
        {
            season
            for spec in contract["horizons"].values()
            for end in window_ends
            for season in _window_seasons(spec, end)
        }
        | set(test_seasons)
    )
    frame = load_unified_terminal_possessions(
        ROOT / "rapm/data/possession_cache",
        ROOT / "data/lake/silver/possessions.parquet",
        ROOT / "data/lake/silver/possession_lineup_segments.parquet",
        tuple(all_possession_seasons),
        transition_season=2024,
        game_types=("regular",),
    )
    original_design = build_design(frame, include_home=True)
    season_mean = frame.groupby("season")["pts"].mean()
    adjusted = frame["pts"].to_numpy(dtype=float) - frame["season"].map(
        season_mean
    ).to_numpy(dtype=float)
    target_design = replace(original_design, y=adjusted)
    rapm_config = RapmConfig(
        seasons=tuple(all_possession_seasons),
        lambda_off=float(contract["rapm"]["lambda_off"]),
        lambda_def=float(contract["rapm"]["lambda_def"]),
        lambda_home=float(contract["rapm"]["lambda_home"]),
        data_scope="spm_target_horizon_full",
    )

    target_frames = []
    for horizon, spec in contract["horizons"].items():
        for end in window_ends:
            path = checkpoint / f"target_{horizon}_{end}.parquet"
            if path.exists():
                target_frames.append(pd.read_parquet(path))
                continue
            seasons = _window_seasons(spec, end)
            mask = np.isin(target_design.seasons, seasons)
            beta, _ = fit_coefficients(target_design, rapm_config, row_mask=mask)
            target = _target_ratings(target_design, beta, mask)
            target["horizon"] = horizon
            target["Window_End"] = end
            target["Window_Start"] = seasons[0]
            _write_parquet_atomic(target, path)
            target_frames.append(target)
            print(f"target {horizon} {seasons[0]}-{end}: checkpointed", flush=True)
    targets = pd.concat(target_frames, ignore_index=True)
    _write_parquet_atomic(targets, output / "targets.parquet")

    feature_panels: dict[str, pd.DataFrame] = {}
    for horizon, spec in contract["horizons"].items():
        path = output / f"features_{horizon}.parquet"
        if path.exists():
            feature_panels[horizon] = pd.read_parquet(path)
            continue
        feature_panels[horizon] = _build_feature_panel(
            loaded_features, spec, window_ends
        )
        _write_parquet_atomic(feature_panels[horizon], path)
        print(f"features {horizon}: checkpointed", flush=True)

    selected = _selected_single_season_features(REFERENCE_RUN)
    common_features = {
        side: tuple(
            feature
            for feature in selected[side]
            if all(feature in panel for panel in feature_panels.values())
        )
        for side in ("offense", "defense")
    }
    panels = {}
    for horizon, features in feature_panels.items():
        panel_targets = targets.loc[targets["horizon"].eq(horizon)].drop(
            columns="horizon"
        )
        panels[horizon] = features.merge(
            panel_targets,
            on=["PLAYER_ID", "Window_End"],
            how="inner",
            validate="one_to_one",
        )
        panels[horizon]["sample_weight"] = np.sqrt(
            np.minimum(
                panels[horizon]["Poss_Off"], panels[horizon]["Poss_Def"]
            ).clip(lower=1)
        )

    metric_rows: list[dict] = []
    prior_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    game_frames: list[pd.DataFrame] = []
    rowset_hashes: dict[str, str] = {}
    for horizon, spec in contract["horizons"].items():
        for test_season in test_seasons:
            stem = f"{horizon}_{test_season}"
            paths = {
                name: checkpoint / f"{stem}_{name}.parquet"
                for name in ("metrics", "priors", "ratings", "games")
            }
            if all(path.exists() for path in paths.values()):
                metric_rows.extend(pd.read_parquet(paths["metrics"]).to_dict("records"))
                prior_frames.append(pd.read_parquet(paths["priors"]))
                rating_frames.append(pd.read_parquet(paths["ratings"]))
                games = pd.read_parquet(paths["games"])
                game_frames.append(games)
                rowset_hashes[stem] = hashlib.sha256(
                    "\n".join(sorted(games["game_id"].astype(str).unique())).encode()
                ).hexdigest()
                print(f"{stem}: resumed", flush=True)
                continue

            prior_end = test_season - 1
            priors, label_metrics = _fit_spm_fold(
                panels[horizon],
                common_features,
                prior_window_end=prior_end,
                minimum_training_window_ends=int(
                    contract["minimum_training_window_ends"]
                ),
            )
            priors["horizon"] = horizon
            priors["test_season"] = test_season
            fold_metric_rows = [
                {
                    "horizon": horizon,
                    "test_season": test_season,
                    "evaluation": "label_reconstruction",
                    "candidate": "spm",
                    **row,
                }
                for row in label_metrics
            ]
            train_seasons = _window_seasons(spec, prior_end)
            train_mask = np.isin(original_design.seasons, train_seasons)
            test_mask = original_design.seasons == test_season
            center, coverage = build_prior_center(
                original_design,
                priors,
                prior_window_end=prior_end,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            fits = fit_coefficient_center_path(
                original_design,
                rapm_config,
                center,
                center_scales=tuple(
                    float(value) for value in contract["rapm"]["center_scales"]
                ),
                row_mask=train_mask,
            )
            fold_ratings = []
            fold_games = []
            for scale, (beta, intercept) in fits.items():
                candidate = "zero_prior" if scale == 0.0 else "spm_centered_aio"
                game_metrics = _game_margin_metrics(
                    original_design, beta, intercept, test_mask, train_mask
                )
                fold_metric_rows.append(
                    {
                        "horizon": horizon,
                        "test_season": test_season,
                        "evaluation": "future_game_margin",
                        "candidate": candidate,
                        "component": "net",
                        "training_start": train_seasons[0],
                        "training_end": train_seasons[-1],
                        "training_window_ends": np.nan,
                        "rows": game_metrics["games"],
                        "weighted_rmse": game_metrics["margin_rmse"],
                        "weighted_correlation": game_metrics["margin_correlation"],
                        "dispersion_ratio": game_metrics["predicted_margin_sd"]
                        / game_metrics["actual_margin_sd"],
                        "calibration_slope": game_metrics["calibration_slope"],
                        "calibration_intercept": game_metrics[
                            "calibration_intercept"
                        ],
                        "prior_lineup_slot_coverage": coverage[
                            "test_lineup_slot_coverage"
                        ],
                    }
                )
                ratings = _rating_table(original_design, beta, train_mask)
                ratings["candidate"] = candidate
                ratings["horizon"] = horizon
                ratings["test_season"] = test_season
                fold_ratings.append(ratings)
                games = _game_margin_frame(
                    original_design, beta, intercept, test_mask, train_mask
                )
                games["candidate"] = candidate
                games["horizon"] = horizon
                games["test_season"] = test_season
                fold_games.append(games)

            fold_metrics = pd.DataFrame(fold_metric_rows)
            fold_ratings_frame = pd.concat(fold_ratings, ignore_index=True)
            fold_games_frame = pd.concat(fold_games, ignore_index=True)
            _write_parquet_atomic(fold_metrics, paths["metrics"])
            _write_parquet_atomic(priors, paths["priors"])
            _write_parquet_atomic(fold_ratings_frame, paths["ratings"])
            _write_parquet_atomic(fold_games_frame, paths["games"])
            metric_rows.extend(fold_metric_rows)
            prior_frames.append(priors)
            rating_frames.append(fold_ratings_frame)
            game_frames.append(fold_games_frame)
            rowset_hashes[stem] = hashlib.sha256(
                "\n".join(
                    sorted(fold_games_frame["game_id"].astype(str).unique())
                ).encode()
            ).hexdigest()
            print(f"{stem}: checkpointed", flush=True)

    for test_season in test_seasons:
        hashes = {
            rowset_hashes[f"{horizon}_{test_season}"]
            for horizon in contract["horizons"]
        }
        if len(hashes) != 1:
            raise ValueError(f"Horizon game rows differ for Season {test_season}.")

    metrics = pd.DataFrame(metric_rows).drop_duplicates(
        ["horizon", "test_season", "evaluation", "candidate", "component"]
    )
    priors = pd.concat(prior_frames, ignore_index=True)
    ratings = pd.concat(rating_frames, ignore_index=True)
    games = pd.concat(game_frames, ignore_index=True)
    summary = (
        metrics.groupby(
            ["evaluation", "horizon", "candidate", "component"], as_index=False
        )
        .agg(
            folds=("test_season", "nunique"),
            mean_rmse=("weighted_rmse", "mean"),
            mean_correlation=("weighted_correlation", "mean"),
            mean_dispersion=("dispersion_ratio", "mean"),
        )
        .sort_values(["evaluation", "component", "mean_rmse", "horizon"])
    )
    _write_parquet_atomic(metrics, output / "fold_metrics.parquet")
    _write_parquet_atomic(summary, output / "summary.parquet")
    _write_parquet_atomic(priors, output / "priors.parquet")
    _write_parquet_atomic(ratings, output / "ratings.parquet")
    _write_parquet_atomic(games, output / "games.parquet")
    manifest = {
        "run_id": output.name,
        "experiment_id": contract["experiment_id"],
        "estimand_id": contract["estimand_id"],
        "status": "research_development_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": contract,
        "features": {
            side: list(values) for side, values in common_features.items()
        },
        "source_hashes": source_hashes,
        "quality": {
            "possession_rows": len(frame),
            "games": int(frame["gameid"].nunique()),
            "target_rows": len(targets),
            "offense_features": len(common_features["offense"]),
            "defense_features": len(common_features["defense"]),
            "identical_future_game_rows": True,
            "game_rowset_hashes": rowset_hashes,
            "maximum_loaded_season": int(frame["season"].max()),
            "season_2027_loaded": False,
        },
        "summary": summary.to_dict("records"),
        "paths": {
            "targets": "targets.parquet",
            "fold_metrics": "fold_metrics.parquet",
            "summary": "summary.parquet",
            "priors": "priors.parquet",
            "ratings": "ratings.parquet",
            "games": "games.parquet",
            "checkpoints": "checkpoints",
        },
        "forbidden_interpretation": "Production promotion or confirmation on Seasons 2025-2027.",
    }
    write_json_atomic(manifest, output / "run.json")
    return manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
