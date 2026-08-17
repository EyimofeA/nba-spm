"""Forward and reverse aging-balanced diagnostics for annual player ratings."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from nba_impact.data.manifest import sha256_file, write_json_atomic
from nba_impact.models.statistical_impact import _metrics


COMPONENTS = ("offense", "defense", "net")
AGE_KNOTS = (22.0, 25.0, 28.0, 31.0, 34.0)


def _age_basis(age: pd.Series) -> np.ndarray:
    values = pd.to_numeric(age, errors="coerce").to_numpy(dtype=float)
    centered = values - 27.0
    columns = [np.ones(len(values)), centered]
    columns.extend(np.maximum(values - knot, 0.0) for knot in AGE_KNOTS)
    return np.column_stack(columns)


def _load_age_panel(source_dir: str | Path, seasons: tuple[int, ...]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        path = Path(source_dir) / f"{season}.csv"
        frame = pd.read_csv(path, usecols=["PLAYER_ID", "AGE"])
        frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="coerce")
        frame["AGE"] = pd.to_numeric(frame["AGE"], errors="coerce")
        frame = frame.dropna(subset=["PLAYER_ID", "AGE"])
        conflicting = frame.groupby("PLAYER_ID")["AGE"].nunique().gt(1)
        if conflicting.any():
            raise ValueError(f"Age source {path} has conflicting duplicate player ages.")
        frame = frame.drop_duplicates("PLAYER_ID")
        frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
        frame["Season"] = season
        rows.append(frame[["PLAYER_ID", "Season", "AGE"]])
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Age panel keys are not unique.")
    return result


def build_transition_pairs(
    predictions: pd.DataFrame,
    ages: pd.DataFrame,
    *,
    direction: str,
) -> pd.DataFrame:
    """Join an origin rating with the adjacent annual RAPM target."""
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be 'forward' or 'reverse'.")
    required = {
        "PLAYER_ID", "Season", "sample_weight",
        *(f"target_{side}" for side in COMPONENTS),
        *(f"spm_{side}" for side in COMPONENTS),
    }
    if missing := sorted(required - set(predictions.columns)):
        raise ValueError(f"Prediction panel is missing {missing}.")
    if predictions.duplicated(["PLAYER_ID", "Season"]).any():
        raise ValueError("Prediction keys are not unique.")
    origin = predictions[list(required)].copy()
    origin = origin.merge(ages, on=["PLAYER_ID", "Season"], validate="one_to_one")
    step = 1 if direction == "forward" else -1
    adjacent = predictions[
        ["PLAYER_ID", "Season", "sample_weight", *(f"target_{side}" for side in COMPONENTS)]
    ].copy()
    adjacent["Season"] = adjacent["Season"] - step
    adjacent = adjacent.rename(
        columns={
            "sample_weight": "adjacent_sample_weight",
            **{f"target_{side}": f"adjacent_target_{side}" for side in COMPONENTS},
        }
    )
    pair = origin.merge(adjacent, on=["PLAYER_ID", "Season"], validate="one_to_one")
    pair["evaluation_weight"] = np.minimum(
        pair["sample_weight"], pair["adjacent_sample_weight"]
    )
    pair["direction"] = direction
    return pair


def evaluate_aging_balanced_pairs(
    pairs: pd.DataFrame,
    *,
    minimum_training_origins: int = 3,
    aging_ridge_alpha: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit earlier-only age curves and score raw and adjusted adjacent targets."""
    seasons = sorted(pairs["Season"].unique().astype(int).tolist())
    metrics = []
    predictions = []
    for test_season in seasons:
        train = pairs.loc[pairs["Season"].lt(test_season)].copy()
        test = pairs.loc[pairs["Season"].eq(test_season)].copy()
        if train["Season"].nunique() < minimum_training_origins or test.empty:
            continue
        fold = test[["PLAYER_ID", "Season", "AGE", "direction", "evaluation_weight"]].copy()
        for side in COMPONENTS:
            change = train[f"adjacent_target_{side}"] - train[f"target_{side}"]
            model = Ridge(alpha=aging_ridge_alpha, fit_intercept=False)
            model.fit(
                _age_basis(train["AGE"]),
                change,
                sample_weight=train["evaluation_weight"],
            )
            expected_change = model.predict(_age_basis(test["AGE"]))
            raw_target = test[f"adjacent_target_{side}"].to_numpy()
            adjusted_target = raw_target - expected_change
            rating = test[f"spm_{side}"].to_numpy()
            weights = test["evaluation_weight"].to_numpy()
            for variant, target in (("raw", raw_target), ("aging_adjusted", adjusted_target)):
                metrics.append(
                    {
                        "direction": test["direction"].iloc[0],
                        "origin_season": test_season,
                        "component": side,
                        "variant": variant,
                        "train_origin_seasons": int(train["Season"].nunique()),
                        "train_rows": int(len(train)),
                        "test_rows": int(len(test)),
                        **_metrics(target, rating, weights),
                    }
                )
            fold[f"spm_{side}"] = rating
            fold[f"raw_target_{side}"] = raw_target
            fold[f"expected_age_change_{side}"] = expected_change
            fold[f"aging_adjusted_target_{side}"] = adjusted_target
        predictions.append(fold)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def run_aging_balanced_validation(
    predictions_path: str | Path,
    age_source_dir: str | Path,
    *,
    artifact_root: str | Path,
    minimum_training_origins: int = 3,
    aging_ridge_alpha: float = 10.0,
) -> dict:
    predictions = pd.read_parquet(predictions_path)
    seasons = tuple(sorted(predictions["Season"].unique().astype(int).tolist()))
    ages = _load_age_panel(age_source_dir, seasons)
    metric_frames = []
    prediction_frames = []
    coverage = []
    for direction in ("forward", "reverse"):
        pairs = build_transition_pairs(predictions, ages, direction=direction)
        coverage.append(
            {
                "direction": direction, "rows": int(len(pairs)),
                "players": int(pairs["PLAYER_ID"].nunique()),
                "origin_seasons": sorted(pairs["Season"].unique().astype(int).tolist()),
            }
        )
        metrics, scored = evaluate_aging_balanced_pairs(
            pairs,
            minimum_training_origins=minimum_training_origins,
            aging_ridge_alpha=aging_ridge_alpha,
        )
        metric_frames.append(metrics)
        prediction_frames.append(scored)
    metrics = pd.concat(metric_frames, ignore_index=True)
    scored = pd.concat(prediction_frames, ignore_index=True)
    summary = metrics.groupby(["direction", "component", "variant"], as_index=False).agg(
        folds=("origin_season", "nunique"),
        rows=("test_rows", "sum"),
        mean_weighted_rmse=("weighted_rmse", "mean"),
        mean_correlation=("correlation", "mean"),
    )
    run_id = f"aging_balanced_validation_v1_{uuid.uuid4().hex[:10]}"
    output = Path(artifact_root) / "models" / "aging_balanced_validation" / run_id
    output.mkdir(parents=True, exist_ok=False)
    metrics.to_parquet(output / "fold_metrics.parquet", index=False)
    summary.to_parquet(output / "summary.parquet", index=False)
    scored.to_parquet(output / "predictions.parquet", index=False)
    run = {
        "run_id": run_id,
        "model_family": "aging_balanced_rating_validation_v1",
        "status": "diagnostic_only",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "minimum_training_origins": minimum_training_origins,
            "aging_ridge_alpha": aging_ridge_alpha,
            "age_knots": list(AGE_KNOTS),
            "source_hashes": {
                "predictions": sha256_file(predictions_path),
                "age_seasons": {
                    str(season): sha256_file(Path(age_source_dir) / f"{season}.csv")
                    for season in seasons
                },
                "source_code": sha256_file(Path(__file__)),
            },
        },
        "coverage": coverage,
        "metrics": {"summary": summary.to_dict(orient="records")},
        "fold_metrics_path": str(output / "fold_metrics.parquet"),
        "summary_path": str(output / "summary.parquet"),
        "predictions_path": str(output / "predictions.parquet"),
        "artifact_path": str(output),
        "caveats": [
            "Forward raw scores are deployment-like diagnostics; reverse scores are symmetry diagnostics only.",
            "Age curves are fitted only on earlier origin seasons and never enter the retrospective SPM.",
            "This run evaluates an already-inspected model and cannot supply promotion evidence.",
            "The spline is a small diagnostic adjustment, not a final NBA aging model.",
        ],
    }
    write_json_atomic(run, output / "run.json")
    return run
