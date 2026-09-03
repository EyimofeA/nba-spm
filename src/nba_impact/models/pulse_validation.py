"""Fail closed when descriptive PULSE ratings enter a validation export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_impact.models.canonical_pulse import game_metrics

_CANDIDATES = {"prior", "pulse", "rapm"}


def _validate_frame_keys(
    frame: pd.DataFrame,
    unique_keys: list[str],
    expected_seasons: set[int],
) -> None:
    if frame.empty or frame[unique_keys].isna().any().any() or frame.duplicated(unique_keys).any():
        raise ValueError("PULSE validation has empty, missing, or duplicate keys.")
    if set(frame["candidate"]) != _CANDIDATES:
        raise ValueError("PULSE validation must use canonical candidates, not renamed older models.")
    if not frame["outcome_season"].eq(frame["rating_season"] + 1).all():
        raise ValueError("PULSE must score the next season, not the fitted season.")
    if not frame["outcome_season"].between(2015, 2026).all():
        raise ValueError("PULSE validation is restricted to the released 2015–2026 outcomes.")
    if not frame["rating_season"].isin(expected_seasons).all():
        raise ValueError("PULSE rating seasons must be declared integer years.")


def _validate_priors(priors: pd.DataFrame, folds: pd.DataFrame) -> None:
    invalid = (
        priors.empty
        or priors[["PLAYER_ID", "rating_season", "Window_End"]].isna().any().any()
        or priors.duplicated(["PLAYER_ID", "rating_season"]).any()
        or not priors["Window_End"].eq(priors["rating_season"]).all()
        or set(priors["rating_season"]) != set(folds["rating_season"])
    )
    if invalid:
        raise ValueError("PULSE validation priors do not match the rated seasons.")


def _validate_fold_summaries(games: pd.DataFrame, folds: pd.DataFrame, keys: list[str]) -> None:
    measured = games.groupby(keys).apply(
        lambda group: pd.Series(game_metrics(group)), include_groups=False
    ).add_prefix("checked_")
    compared = folds.merge(measured, on=keys, how="outer", validate="one_to_one", indicator=True)
    metrics_match = all(
        np.allclose(
            compared[column],
            compared[f"checked_{column}"],
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )
        for column in ("mse", "correlation", "calibration_slope")
    )
    if (
        not compared["_merge"].eq("both").all()
        or not compared["games"].eq(compared["checked_games"]).all()
        or not metrics_match
    ):
        raise ValueError("PULSE fold summaries disagree with their game predictions.")


def load_pulse_validation(run_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load audited fold predictions, never full-history descriptive ratings.

    This checks artifact consistency and declared fit cutoffs. It cannot prove
    that historical model-selection decisions did not reuse evaluation years.
    """
    manifest = json.loads((run_path / "run.json").read_text())
    interpretation = manifest.get("config", {}).get("interpretation", {})
    if interpretation.get("validation_rows") != "past_only":
        raise ValueError("PULSE validation requires an explicit past-only contract.")
    for name in ("validation_games", "validation_folds", "validation_priors"):
        if manifest.get("artifacts", {}).get(name) != f"{name}.parquet":
            raise ValueError(f"PULSE validation cannot substitute descriptive data for {name}.")
    games = pd.read_parquet(run_path / "validation_games.parquet")
    folds = pd.read_parquet(run_path / "validation_folds.parquet")
    priors = pd.read_parquet(run_path / "validation_priors.parquet")
    first, last = manifest["validation"]["rating_seasons"]
    expected = set(range(first, last + 1))
    if not expected or set(games["rating_season"]) != expected or set(folds["rating_season"]) != expected:
        raise ValueError("PULSE validation is missing declared rating seasons.")
    keys = ["candidate", "rating_season", "outcome_season"]
    for frame, unique_keys in ((games, keys + ["game_id"]), (folds, keys)):
        _validate_frame_keys(frame, unique_keys, expected)
    if not (
        folds["training_start"].le(folds["training_end"])
        & folds["training_end"].lt(folds["rating_season"])
    ).all():
        raise ValueError("PULSE prior training overlaps the rating or outcome season.")
    _validate_priors(priors, folds)
    if not np.isfinite(games[["actual_margin", "predicted_margin"]].to_numpy(float)).all():
        raise ValueError("PULSE game margins must be finite.")
    shared = games.pivot(
        index=["rating_season", "outcome_season", "game_id"],
        columns="candidate", values="actual_margin",
    )
    if shared.isna().any().any() or not shared.eq(shared["pulse"], axis=0).all().all():
        raise ValueError("PULSE candidates must score identical games and actual margins.")
    _validate_fold_summaries(games, folds, keys)
    return games, folds
