#!/usr/bin/env python3
"""Screen compact PULSE learners and era representations downstream."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES

try:
    from run_annual_spm_learner_screen import MODEL_GRIDS, _fit, _select_spec
    from run_target_window_spm_aio import _evaluate_priors, _game_metrics, _panel
except ModuleNotFoundError:
    from research.run_annual_spm_learner_screen import MODEL_GRIDS, _fit, _select_spec
    from research.run_target_window_spm_aio import _evaluate_priors, _game_metrics, _panel


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "pulse_model_screen_v1"
CONTRACT = ROOT / "research/experiments/pulse_model_screen_v1.yml"
TARGET_RUN = ROOT / (
    "artifacts/research/target_window_spm_aio/"
    "target_window_spm_aio_v1_be05a18f9b"
)
FEATURES = ROOT / (
    "artifacts/research/historical_box15_extension/"
    "historical_box15_extension_v1_08ff4c34ff/annual_box15_features.parquet"
)
OUTPUT_ROOT = ROOT / "artifacts/research/pulse_model_screen"
_ARTIFACT_FILES = (
    "priors.parquet",
    "selections.parquet",
    "game_predictions.parquet",
    "ratings.parquet",
    "prior_coverage.parquet",
    "fold_metrics.parquet",
    "summary.parquet",
    "paired_bootstrap.parquet",
)


def _load_contract() -> dict:
    contract = yaml.safe_load(CONTRACT.read_text())
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("PULSE screen contract changed.")
    rating = tuple(int(value) for value in contract["rating_seasons"])
    tests = tuple(int(value) for value in contract["test_seasons"])
    if rating != tuple(range(2014, 2026)) or tests != tuple(range(2015, 2027)):
        raise ValueError("PULSE screen seasons changed.")
    if tests != tuple(season + 1 for season in rating):
        raise ValueError("Every rating season must predict its next season.")
    return contract


def _weighted_location_scale(values: pd.Series, weights: pd.Series) -> tuple[float, float]:
    finite = values.notna() & weights.gt(0)
    if not finite.any():
        return 0.0, 1.0
    x = values.loc[finite].to_numpy(dtype=float)
    w = weights.loc[finite].to_numpy(dtype=float)
    location = float(np.average(x, weights=w))
    variance = float(np.average((x - location) ** 2, weights=w))
    return location, max(float(np.sqrt(variance)), 1e-12)


def _represent(panel: pd.DataFrame, representation: str) -> pd.DataFrame:
    """Transform inputs using only the contemporaneous season cross-section."""
    if representation == "raw":
        return panel.copy()
    if representation not in {"season_relative", "season_standardized"}:
        raise ValueError(f"Unknown representation {representation}.")
    transformed = panel.copy()
    weight = np.sqrt(
        np.minimum(transformed["OffPoss"], transformed["DefPoss"]).clip(lower=1)
    )
    for _, indices in transformed.groupby("Window_End", sort=True).groups.items():
        season_weight = weight.loc[indices]
        for feature in BOX_PIPM_STYLE_FEATURES:
            location, scale = _weighted_location_scale(
                transformed.loc[indices, feature], season_weight
            )
            transformed.loc[indices, feature] = (
                transformed.loc[indices, feature] - location
            )
            if representation == "season_standardized":
                transformed.loc[indices, feature] /= scale
    return transformed


def _fit_priors(panel: pd.DataFrame, contract: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    selections: list[dict] = []
    for representation in contract["features"]["representations"]:
        represented = _represent(panel, representation).rename(
            columns={"Window_End": "Season"}
        )
        for family in contract["learners"]["families"]:
            candidate = f"box15_{family}_{representation}"
            for season in contract["rating_seasons"]:
                train = represented.loc[represented["Season"].lt(season)].copy()
                test = represented.loc[represented["Season"].eq(season)].copy()
                if train.empty or test.empty:
                    continue
                prior = test[["PLAYER_ID", "Season"]].rename(
                    columns={"Season": "Window_End"}
                )
                for side in ("offense", "defense"):
                    target = f"target_{side}"
                    spec, _ = _select_spec(
                        train,
                        BOX_PIPM_STYLE_FEATURES,
                        family,
                        target,
                        1.01,
                    )
                    model = _fit(spec, train, BOX_PIPM_STYLE_FEATURES, target)
                    prediction = model.predict(test.loc[:, BOX_PIPM_STYLE_FEATURES])
                    prior[f"prior_{side}_per_100"] = prediction
                    selections.append(
                        {
                            "candidate": candidate,
                            "rating_season": int(season),
                            "side": side,
                            "selected_params": json.dumps(spec.params, sort_keys=True),
                            "training_start": int(train["Season"].min()),
                            "training_end": int(train["Season"].max()),
                            "training_rows": len(train),
                            "prediction_rows": len(test),
                        }
                    )
                prior["prior_net_per_100"] = (
                    prior["prior_offense_per_100"]
                    + prior["prior_defense_per_100"]
                )
                prior["candidate"] = candidate
                rows.append(prior)
            print(f"PULSE prior {candidate}: complete", flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(selections)


def _paired_bootstrap(
    games: pd.DataFrame,
    baseline: str,
    *,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_games = games.loc[games["candidate"].eq(baseline)]
    rows = []
    for candidate in sorted(games["candidate"].unique()):
        if candidate == baseline or not candidate.endswith("_aio"):
            continue
        challenger = games.loc[games["candidate"].eq(candidate)]
        seasons = sorted(
            set(baseline_games["test_season"]) & set(challenger["test_season"])
        )
        differences = []
        for season in seasons:
            left = baseline_games.loc[
                baseline_games["test_season"].eq(season)
            ].set_index("game_id").sort_index()
            right = challenger.loc[
                challenger["test_season"].eq(season)
            ].set_index("game_id").sort_index()
            if not left.index.equals(right.index):
                raise ValueError(f"{baseline} and {candidate} score different games.")
            differences.append(
                right["squared_error"].to_numpy(dtype=float)
                - left["squared_error"].to_numpy(dtype=float)
            )
        samples = np.empty(draws, dtype=float)
        for draw in range(draws):
            samples[draw] = float(
                np.mean(
                    [
                        values[rng.integers(0, len(values), len(values))].mean()
                        for values in differences
                    ]
                )
            )
        low, high = np.quantile(samples, (0.025, 0.975))
        rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "folds": len(seasons),
                "candidate_minus_baseline_mse": float(
                    np.mean([values.mean() for values in differences])
                ),
                "lower_95": float(low),
                "upper_95": float(high),
                "probability_candidate_better": float(np.mean(samples < 0)),
            }
        )
    return pd.DataFrame(rows)


def _phase_summary(folds: pd.DataFrame, contract: dict) -> pd.DataFrame:
    rows = []
    phases = {
        "complete": tuple(contract["test_seasons"]),
        "selection": tuple(contract["selection_test_seasons"]),
        "diagnostic": tuple(contract["diagnostic_test_seasons"]),
    }
    for phase, seasons in phases.items():
        selected = folds.loc[folds["test_season"].isin(seasons)]
        summary = selected.groupby("candidate", as_index=False).agg(
            folds=("test_season", "nunique"),
            equal_season_mse=("mse", "mean"),
            mean_correlation=("correlation", "mean"),
            mean_calibration_slope=("calibration_slope", "mean"),
        )
        summary["equal_season_rmse"] = np.sqrt(summary["equal_season_mse"])
        summary["phase"] = phase
        rows.append(summary)
    return pd.concat(rows, ignore_index=True).sort_values(
        ["phase", "equal_season_mse", "candidate"]
    )


def _finalize_run(
    output: Path,
    contract: dict,
    hashes: dict[str, str],
    *,
    panel_rows: int,
    started: float,
) -> dict:
    priors = pd.read_parquet(output / "priors.parquet")
    ratings = pd.read_parquet(output / "ratings.parquet")
    games = pd.read_parquet(output / "game_predictions.parquet")
    summary = pd.read_parquet(output / "summary.parquet")
    selection = summary.loc[
        summary["phase"].eq("selection")
        & summary["candidate"].str.endswith("_aio")
    ].sort_values("equal_season_mse")
    run = {
        "run_id": output.name,
        "experiment_id": EXPERIMENT_ID,
        "status": "reused_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_on_2017_2021": str(selection.iloc[0]["candidate"]),
        "baseline": "box15_ridge_raw_aio",
        "source_hashes": hashes,
        "quality": {
            "panel_rows": panel_rows,
            "prior_rows": len(priors),
            "rating_rows": len(ratings),
            "game_rows": len(games),
            "duplicate_prior_keys": int(
                priors.duplicated(["PLAYER_ID", "Window_End", "candidate"]).sum()
            ),
            "nonfinite_game_predictions": int(
                (~np.isfinite(games["predicted_margin"])).sum()
            ),
            "runtime_seconds": time.perf_counter() - started,
        },
        "files": {
            name: {"sha256": sha256_file(output / name)} for name in _ARTIFACT_FILES
        },
        "forbidden_interpretation": contract["forbidden_interpretation"],
    }
    write_json_atomic(run, output / "run.json")
    return run


def main() -> None:
    started = time.perf_counter()
    contract = _load_contract()
    sources = {
        "contract": CONTRACT,
        "runner": Path(__file__),
        "features": FEATURES,
        "targets": TARGET_RUN / "targets.parquet",
    }
    hashes = {name: sha256_file(path) for name, path in sources.items()}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:10]
    output = OUTPUT_ROOT / f"{EXPERIMENT_ID}_{identity}"
    if (output / "run.json").exists():
        print(output)
        return
    output.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(FEATURES)
    targets = pd.read_parquet(TARGET_RUN / "targets.parquet")
    targets = targets.loc[
        targets["horizon"].eq(contract["target"]["horizon_seasons"])
        & targets["target_variant"].eq(contract["target"]["variant"])
    ]
    panel = _panel(features, targets)
    if panel.duplicated(["PLAYER_ID", "Window_End"]).any():
        raise ValueError("PULSE screen panel has duplicate player-season keys.")

    if all((output / name).exists() for name in _ARTIFACT_FILES):
        _finalize_run(
            output,
            contract,
            hashes,
            panel_rows=len(panel),
            started=started,
        )
        print(output)
        return

    priors, selections = _fit_priors(panel, contract)
    evaluation_contract = {
        "rating_seasons": contract["rating_seasons"],
        "aio": contract["aio"],
    }
    games, ratings, coverage = _evaluate_priors(priors, evaluation_contract)
    folds, _ = _game_metrics(games)
    summary = _phase_summary(folds, contract)
    baseline = "box15_ridge_raw_aio"
    bootstrap = _paired_bootstrap(
        games,
        baseline,
        draws=int(contract["evaluation"]["bootstrap_draws"]),
        seed=int(contract["evaluation"]["bootstrap_seed"]),
    )

    priors.to_parquet(output / "priors.parquet", index=False)
    selections.to_parquet(output / "selections.parquet", index=False)
    games.to_parquet(output / "game_predictions.parquet", index=False)
    ratings.to_parquet(output / "ratings.parquet", index=False)
    coverage.to_parquet(output / "prior_coverage.parquet", index=False)
    folds.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    bootstrap.to_parquet(output / "paired_bootstrap.parquet", index=False)

    _finalize_run(
        output,
        contract,
        hashes,
        panel_rows=len(panel),
        started=started,
    )
    print(output)


if __name__ == "__main__":
    main()
