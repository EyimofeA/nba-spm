"""Matched annual RAPM benchmark for internal and external player ratings."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_impact import _metrics


INTERNAL_CANDIDATES = ("spm", "box_pipm_style")
EXTERNAL_CANDIDATES = ("bpm", "xrapm")
COMPONENTS = ("offense", "defense", "net")


def build_annual_rating_benchmark(
    external_benchmark_path: str | Path,
    box_pipm_oof_path: str | Path,
    *,
    artifact_root: str | Path,
    minimum_possessions_per_side: float = 1000.0,
) -> dict:
    """Compare ratings on an identical high-exposure annual RAPM population.

    SPM and the BoxPIPM-style baseline already predict in RAPM units. BPM and
    xRAPM receive a component-specific affine calibration fitted on all other
    seasons. That makes their RMSE comparable without leaking the test season.
    Correlation remains unaffected by that calibration.
    """
    if minimum_possessions_per_side <= 0:
        raise ValueError("Minimum possession exposure must be positive.")
    benchmark = pd.read_parquet(external_benchmark_path)
    pipm = pd.read_parquet(box_pipm_oof_path)
    pipm_columns = [
        "PLAYER_ID",
        "Season",
        *[f"box_pipm_style_{component}" for component in COMPONENTS],
    ]
    missing = sorted(set(pipm_columns) - set(pipm.columns))
    if missing:
        raise ValueError(f"BoxPIPM-style predictions are missing {missing}.")
    if pipm.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("BoxPIPM-style predictions have duplicate player-season keys.")
    panel = benchmark.merge(
        pipm.loc[:, pipm_columns],
        on=["PLAYER_ID", "Season"],
        how="inner",
        validate="one_to_one",
    )
    required = [
        "PLAYER_ID",
        "Season",
        "Poss_Off",
        "Poss_Def",
        "sample_weight",
        *[
            f"{candidate}_{component}"
            for candidate in (*INTERNAL_CANDIDATES, *EXTERNAL_CANDIDATES)
            for component in COMPONENTS
        ],
        *[f"target_{component}" for component in COMPONENTS],
    ]
    absent = sorted(set(required) - set(panel.columns))
    if absent:
        raise ValueError(f"Annual-rating benchmark is missing {absent}.")
    eligible = panel.loc[
        panel["Poss_Off"].ge(minimum_possessions_per_side)
        & panel["Poss_Def"].ge(minimum_possessions_per_side)
    ].copy()
    eligible = eligible.dropna(subset=required)
    if eligible.empty:
        raise ValueError("Annual-rating benchmark has no matched eligible rows.")
    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    for season in sorted(int(value) for value in eligible["Season"].unique()):
        train = eligible.loc[eligible["Season"].ne(season)]
        test = eligible.loc[eligible["Season"].eq(season)].copy()
        if train.empty or test.empty:
            raise ValueError(
                f"Annual-rating benchmark fold {season} has an empty partition."
            )
        for component in COMPONENTS:
            target = f"target_{component}"
            for candidate in (*INTERNAL_CANDIDATES, *EXTERNAL_CANDIDATES):
                source = f"{candidate}_{component}"
                if candidate in EXTERNAL_CANDIDATES:
                    calibrator = LinearRegression().fit(
                        train[[source]],
                        train[target],
                        sample_weight=train["sample_weight"],
                    )
                    prediction = calibrator.predict(test[[source]])
                    calibration = "affine_leave_one_season_out"
                else:
                    prediction = test[source].to_numpy(dtype=float)
                    calibration = "native_rapm_units"
                test[f"prediction_{candidate}_{component}"] = prediction
                metric_rows.append(
                    {
                        "test_season": season,
                        "component": component,
                        "candidate": candidate,
                        "calibration": calibration,
                        "rows": len(test),
                        **_metrics(
                            test[target].to_numpy(dtype=float),
                            prediction,
                            test["sample_weight"].to_numpy(dtype=float),
                        ),
                    }
                )
        prediction_rows.append(test)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby(["component", "candidate", "calibration"], as_index=False)
        .agg(
            mean_weighted_rmse=("weighted_rmse", "mean"),
            mean_correlation=("correlation", "mean"),
            folds=("test_season", "nunique"),
            matched_rows=("rows", "mean"),
        )
        .sort_values(["component", "mean_weighted_rmse", "candidate"])
    )
    config = {
        "minimum_possessions_per_side": minimum_possessions_per_side,
        "internal_candidates": list(INTERNAL_CANDIDATES),
        "external_candidates": list(EXTERNAL_CANDIDATES),
        "components": list(COMPONENTS),
        "external_benchmark_sha256": sha256_file(external_benchmark_path),
        "box_pipm_oof_sha256": sha256_file(box_pipm_oof_path),
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[
        :10
    ]
    output = (
        Path(artifact_root)
        / "models"
        / "annual_rating_benchmark"
        / f"annual_rating_benchmark_v1_{identity}"
    )
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    predictions.to_parquet(output / "matched_predictions.parquet", index=False)
    run = {
        "run_id": output.name,
        "model_family": "annual_rapm_matched_rating_benchmark",
        "status": "research_evaluation",
        "estimand": "annual zero-prior RAPM on identical high-exposure player-season rows",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "quality": {
            "matched_rows": len(eligible),
            "seasons": sorted(int(value) for value in eligible["Season"].unique()),
            "duplicate_prediction_keys": int(
                predictions.duplicated(["PLAYER_ID", "Season"]).sum()
            ),
        },
        "metrics": summary.to_dict(orient="records"),
        "artifact_path": str(output.resolve()),
        "caveats": [
            "This is a retrodictive annual benchmark, not next-season forecast evidence.",
            "xRAPM includes adjusted-plus-minus information and is not an independent box-only baseline.",
            "BPM and xRAPM RMSE use affine calibration fitted outside each scored season.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
