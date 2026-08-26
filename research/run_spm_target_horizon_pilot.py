"""Frozen one- versus five-year matched-window SPM/AIO pilot."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.prior_informed_rapm import build_prior_center
from nba_impact.models.rapm import (
    RapmConfig,
    _game_margin_frame,
    _game_margin_metrics,
    build_design,
    fit_coefficient_center_path,
    load_unified_terminal_possessions,
)
from nba_impact.models.single_season_spm import _selected_single_season_features
from nba_impact.models.statistical_feature_ablation import _frozen_model
from nba_impact.models.statistical_model_comparison import _fit_model
from nba_impact.models.predictive_spm import _predictive_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "research/experiments/spm_target_horizon_pilot_v1.yml"
FEATURE_PATHS = {
    "statistical_features_v2_0e1350d95a": ROOT
    / "artifacts/features/statistical_impact/statistical_features_v2_0e1350d95a/features.parquet",
    "statistical_features_v2_c6351c128f": ROOT
    / "artifacts/research/spm_target_horizon_pilot/features/statistical_impact/statistical_features_v2_c6351c128f/features.parquet",
}
TARGET_PATH = ROOT / (
    "research/rapm_lab/outputs/rapm_target_horizon_bakeoff/"
    "rapm_target_horizon_bakeoff_v1_7c70e278cb/ratings.parquet"
)
REFERENCE_RUN = ROOT / "artifacts/models/single_season_spm/single_season_spm_v1_bff6060df6"
OUTPUT_ROOT = ROOT / "artifacts/research/spm_target_horizon_pilot/runs"


def _artifact_run_id(path: Path) -> str:
    manifest = path.parent / "run.json" if path.is_file() else path / "run.json"
    return json.loads(manifest.read_text())["run_id"]


def _load_contract(path: Path) -> dict:
    contract = yaml.safe_load(path.read_text())
    required = {
        "schema_version": "experiment_preregistration_v1",
        "experiment_id": "spm_target_horizon_pilot_v1",
        "status": "preregistered_pilot",
    }
    for field, expected in required.items():
        if contract.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}.")
    test_seasons = tuple(int(value) for value in contract["test_seasons"])
    if tuple(sorted(set(test_seasons))) != test_seasons:
        raise ValueError("test_seasons must be unique and increasing.")
    if any(season >= 2025 for season in test_seasons):
        raise ValueError("Pilot test seasons must stop at 2024.")
    if 2027 in test_seasons:
        raise ValueError("Season 2027 is forbidden.")
    if set(contract["horizons"]) != {"1y", "5y"}:
        raise ValueError("Pilot horizons must be exactly 1y and 5y.")
    return contract


def _validate_inputs(contract: dict) -> None:
    for horizon in contract["horizons"].values():
        run_id = horizon["features"]
        if run_id not in FEATURE_PATHS:
            raise ValueError(f"Unknown pinned feature run {run_id}.")
        if _artifact_run_id(FEATURE_PATHS[run_id]) != run_id:
            raise ValueError(f"Feature artifact does not match pin {run_id}.")
    if _artifact_run_id(TARGET_PATH) != contract["targets"]:
        raise ValueError("Target artifact does not match the contract pin.")
    if _artifact_run_id(REFERENCE_RUN) != contract["reference_spm"]:
        raise ValueError("Reference SPM artifact does not match the contract pin.")


def _common_features(contract: dict) -> dict[str, tuple[str, ...]]:
    selected = _selected_single_season_features(REFERENCE_RUN)
    available = {
        horizon: set(pd.read_parquet(FEATURE_PATHS[spec["features"]]).columns)
        for horizon, spec in contract["horizons"].items()
    }
    return {
        side: tuple(
            feature
            for feature in selected[side]
            if all(feature in available[horizon] for horizon in available)
        )
        for side in ("offense", "defense")
    }


def _panel(horizon: str, feature_path: Path, features: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    requested = tuple(dict.fromkeys((*features["offense"], *features["defense"])))
    feature_frame = pd.read_parquet(
        feature_path, columns=["PLAYER_ID", "Window_End", *requested]
    )
    targets = pd.read_parquet(TARGET_PATH)
    targets = targets.loc[targets["horizon"].eq(horizon)].rename(
        columns={
            "window_end": "Window_End",
            "offense": "target_offense",
            "defense": "target_defense",
            "net": "target_net",
        }
    )
    target_columns = [
        "PLAYER_ID",
        "Window_End",
        "target_offense",
        "target_defense",
        "target_net",
        "Poss_Off",
        "Poss_Def",
    ]
    panel = feature_frame.merge(
        targets[target_columns],
        on=["PLAYER_ID", "Window_End"],
        how="inner",
        validate="one_to_one",
    )
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    if panel.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError(f"{horizon} panel keys are not unique.")
    return panel


def _fit_spm_fold(
    panel: pd.DataFrame,
    features: dict[str, tuple[str, ...]],
    *,
    prior_window_end: int,
    minimum_training_window_ends: int,
) -> tuple[pd.DataFrame, list[dict]]:
    train = panel.loc[panel["Window_End"].lt(prior_window_end)]
    evaluate = panel.loc[panel["Window_End"].eq(prior_window_end)]
    training_ends = tuple(sorted(int(value) for value in train["Window_End"].unique()))
    if len(training_ends) < minimum_training_window_ends:
        raise ValueError(
            f"Window {prior_window_end} has only {len(training_ends)} training ends."
        )
    if evaluate.empty:
        raise ValueError(f"No label rows for window {prior_window_end}.")
    output = evaluate[
        [
            "PLAYER_ID",
            "Window_End",
            "target_offense",
            "target_defense",
            "target_net",
            "Poss_Off",
            "Poss_Def",
            "sample_weight",
        ]
    ].copy()
    metrics: list[dict] = []
    for side in ("offense", "defense"):
        model = _fit_model(
            _frozen_model(side), train, features[side], f"target_{side}"
        )
        output[f"prior_{side}_per_100"] = model.predict(
            evaluate.loc[:, features[side]]
        )
        values = _predictive_metrics(
            evaluate[f"target_{side}"].to_numpy(dtype=float),
            output[f"prior_{side}_per_100"].to_numpy(dtype=float),
            evaluate["sample_weight"].to_numpy(dtype=float),
        )
        metrics.append(
            {
                "component": side,
                "training_start": training_ends[0],
                "training_end": training_ends[-1],
                "training_window_ends": len(training_ends),
                "rows": len(evaluate),
                **values,
            }
        )
    output["prior_net_per_100"] = (
        output["prior_offense_per_100"] + output["prior_defense_per_100"]
    )
    values = _predictive_metrics(
        output["target_net"].to_numpy(dtype=float),
        output["prior_net_per_100"].to_numpy(dtype=float),
        output["sample_weight"].to_numpy(dtype=float),
    )
    metrics.append(
        {
            "component": "net",
            "training_start": training_ends[0],
            "training_end": training_ends[-1],
            "training_window_ends": len(training_ends),
            "rows": len(evaluate),
            **values,
        }
    )
    return output, metrics


def _rating_table(design, beta: np.ndarray, train_mask: np.ndarray) -> pd.DataFrame:
    n_players = len(design.players)
    train = design.X[train_mask]
    off_count = np.asarray(train[:, :n_players].sum(axis=0)).ravel()
    def_count = np.asarray(train[:, n_players : 2 * n_players].sum(axis=0)).ravel()
    output = pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "offense": 100.0 * beta[:n_players],
            "defense": -100.0 * beta[n_players : 2 * n_players],
            "Poss_Off": off_count,
            "Poss_Def": def_count,
        }
    )
    output["net"] = output["offense"] + output["defense"]
    return output


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def run(contract_path: Path = DEFAULT_CONTRACT) -> dict:
    contract = _load_contract(contract_path)
    _validate_inputs(contract)
    features = _common_features(contract)
    source_hashes = {
        "contract": sha256_file(contract_path),
        "runner": sha256_file(Path(__file__)),
        "targets": sha256_file(TARGET_PATH),
        "reference_spm": sha256_file(REFERENCE_RUN / "run.json"),
        **{
            f"features_{horizon}": sha256_file(FEATURE_PATHS[spec["features"]])
            for horizon, spec in contract["horizons"].items()
        },
    }
    identity = hashlib.sha256(
        json.dumps(source_hashes, sort_keys=True).encode()
    ).hexdigest()[:10]
    output = OUTPUT_ROOT / f"spm_target_horizon_pilot_v1_{identity}"
    if (output / "run.json").exists():
        return json.loads((output / "run.json").read_text())
    checkpoint = output / "checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict] = []
    prior_frames: list[pd.DataFrame] = []
    rating_frames: list[pd.DataFrame] = []
    game_frames: list[pd.DataFrame] = []
    rowset_hashes: dict[str, str] = {}
    for horizon, spec in contract["horizons"].items():
        horizon_seasons = int(spec["seasons"])
        panel = _panel(horizon, FEATURE_PATHS[spec["features"]], features)
        for test_season in contract["test_seasons"]:
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

            prior_end = int(test_season) - 1
            priors, label_metrics = _fit_spm_fold(
                panel,
                features,
                prior_window_end=prior_end,
                minimum_training_window_ends=int(
                    contract["minimum_training_window_ends"]
                ),
            )
            priors["horizon"] = horizon
            priors["test_season"] = int(test_season)
            for row in label_metrics:
                metric_rows.append(
                    {
                        "horizon": horizon,
                        "test_season": int(test_season),
                        "evaluation": "label_reconstruction",
                        "candidate": "spm",
                        **row,
                    }
                )

            train_seasons = tuple(
                range(prior_end - horizon_seasons + 1, prior_end + 1)
            )
            frame = load_unified_terminal_possessions(
                ROOT / "rapm/data/possession_cache",
                ROOT / "data/lake/silver/possessions.parquet",
                ROOT / "data/lake/silver/possession_lineup_segments.parquet",
                (*train_seasons, int(test_season)),
                transition_season=2024,
                game_types=("regular",),
            )
            design = build_design(frame, include_home=True)
            train_mask = np.isin(design.seasons, train_seasons)
            test_mask = design.seasons == int(test_season)
            config = RapmConfig(
                seasons=(*train_seasons, int(test_season)),
                lambda_off=float(contract["rapm"]["lambda_off"]),
                lambda_def=float(contract["rapm"]["lambda_def"]),
                lambda_home=float(contract["rapm"]["lambda_home"]),
                data_scope="spm_target_horizon_pilot",
            )
            center, coverage = build_prior_center(
                design,
                priors,
                prior_window_end=prior_end,
                train_mask=train_mask,
                test_mask=test_mask,
            )
            fits = fit_coefficient_center_path(
                design,
                config,
                center,
                center_scales=tuple(float(v) for v in contract["rapm"]["center_scales"]),
                row_mask=train_mask,
            )
            fold_ratings = []
            fold_games = []
            for scale, (beta, intercept) in fits.items():
                candidate = "zero_prior" if scale == 0.0 else "spm_centered_aio"
                metrics = _game_margin_metrics(
                    design, beta, intercept, test_mask, train_mask
                )
                metric_rows.append(
                    {
                        "horizon": horizon,
                        "test_season": int(test_season),
                        "evaluation": "future_game_margin",
                        "candidate": candidate,
                        "component": "net",
                        "training_start": train_seasons[0],
                        "training_end": train_seasons[-1],
                        "training_window_ends": np.nan,
                        "rows": metrics["games"],
                        "weighted_rmse": metrics["margin_rmse"],
                        "weighted_correlation": metrics["margin_correlation"],
                        "dispersion_ratio": metrics["predicted_margin_sd"]
                        / metrics["actual_margin_sd"],
                        "calibration_slope": metrics["calibration_slope"],
                        "calibration_intercept": metrics["calibration_intercept"],
                        "prior_lineup_slot_coverage": coverage[
                            "test_lineup_slot_coverage"
                        ],
                    }
                )
                ratings = _rating_table(design, beta, train_mask)
                ratings["candidate"] = candidate
                ratings["horizon"] = horizon
                ratings["test_season"] = int(test_season)
                fold_ratings.append(ratings)
                games = _game_margin_frame(
                    design, beta, intercept, test_mask, train_mask
                )
                games["candidate"] = candidate
                games["horizon"] = horizon
                games["test_season"] = int(test_season)
                fold_games.append(games)
            fold_metrics = pd.DataFrame(
                [
                    row
                    for row in metric_rows
                    if row["horizon"] == horizon
                    and row["test_season"] == int(test_season)
                ]
            )
            fold_ratings_frame = pd.concat(fold_ratings, ignore_index=True)
            fold_games_frame = pd.concat(fold_games, ignore_index=True)
            _write_parquet_atomic(fold_metrics, paths["metrics"])
            _write_parquet_atomic(priors, paths["priors"])
            _write_parquet_atomic(fold_ratings_frame, paths["ratings"])
            _write_parquet_atomic(fold_games_frame, paths["games"])
            prior_frames.append(priors)
            rating_frames.append(fold_ratings_frame)
            game_frames.append(fold_games_frame)
            rowset_hashes[stem] = hashlib.sha256(
                "\n".join(
                    sorted(fold_games_frame["game_id"].astype(str).unique())
                ).encode()
            ).hexdigest()
            print(f"{stem}: checkpointed", flush=True)

    metrics = pd.DataFrame(metric_rows).drop_duplicates(
        ["horizon", "test_season", "evaluation", "candidate", "component"]
    )
    priors = pd.concat(prior_frames, ignore_index=True)
    ratings = pd.concat(rating_frames, ignore_index=True)
    games = pd.concat(game_frames, ignore_index=True)
    for test_season in contract["test_seasons"]:
        hashes = {
            rowset_hashes[f"{horizon}_{test_season}"]
            for horizon in contract["horizons"]
        }
        if len(hashes) != 1:
            raise ValueError(f"Horizon game rows differ for Season {test_season}.")
    summary = (
        metrics.groupby(["evaluation", "horizon", "candidate", "component"], as_index=False)
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
    run_manifest = {
        "run_id": output.name,
        "experiment_id": contract["experiment_id"],
        "estimand_id": contract["estimand_id"],
        "status": "research_pilot_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": contract,
        "features": {side: list(values) for side, values in features.items()},
        "source_hashes": source_hashes,
        "quality": {
            "offense_features": len(features["offense"]),
            "defense_features": len(features["defense"]),
            "identical_future_game_rows": True,
            "game_rowset_hashes": rowset_hashes,
            "season_2027_loaded": False,
        },
        "summary": summary.to_dict("records"),
        "paths": {
            "fold_metrics": "fold_metrics.parquet",
            "summary": "summary.parquet",
            "priors": "priors.parquet",
            "ratings": "ratings.parquet",
            "games": "games.parquet",
            "checkpoints": "checkpoints",
        },
        "forbidden_interpretation": "Production promotion or a full target-horizon decision.",
    }
    write_json_atomic(run_manifest, output / "run.json")
    return run_manifest


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
