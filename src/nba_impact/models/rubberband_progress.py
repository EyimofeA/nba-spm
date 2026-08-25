"""Possession-progress helpers for rubber-band score adjustment research."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_impact.models.rubberband_adjustment import (
    RubberbandFit,
    coefficient_table,
    rubberband_design,
)


def annotate_possession_progress(
    frame: pd.DataFrame,
    *,
    bucket_size: int = 25,
    buckets: int = 8,
) -> pd.DataFrame:
    """Add pre-possession regulation progress without using final game length."""
    required = {"gameid", "period", "num"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Possession-progress input is missing columns: {missing}")
    if bucket_size < 1 or buckets < 1:
        raise ValueError("Possession bucket size and count must be positive.")
    result = frame.copy()
    result["_source_order"] = np.arange(len(result), dtype=np.int64)
    result = result.sort_values(["gameid", "period", "num"], kind="stable")
    regulation = result["period"].astype(int).le(4)
    game_order = result.groupby("gameid", sort=False).cumcount().to_numpy() + 1
    result["regulation_possession_number"] = np.where(
        regulation,
        game_order,
        np.nan,
    )
    progress_bucket = np.minimum((game_order - 1) // bucket_size, buckets - 1)
    result["possession_progress_bucket"] = np.where(
        regulation,
        progress_bucket,
        0,
    ).astype(int)
    return (
        result.sort_values("_source_order", kind="stable")
        .drop(columns="_source_order")
        .reset_index(drop=True)
    )


def use_possession_progress(frame: pd.DataFrame) -> pd.DataFrame:
    """Adapt possession-progress buckets to the shared rubber-band fitter."""
    if "possession_progress_bucket" not in frame:
        raise ValueError("Possession progress must be annotated before fitting.")
    result = frame.copy()
    result["six_minute_bucket"] = result["possession_progress_bucket"].astype(int)
    return result


def slope_only_adjustment(fit: RubberbandFit, frame: pd.DataFrame) -> np.ndarray:
    """Return the signed-margin term, excluding time-bucket intercepts."""
    prediction = np.zeros(len(frame), dtype=float)
    regulation = frame["regulation"].astype(bool).to_numpy()
    if regulation.any():
        coefficients = fit.coefficients.copy()
        coefficients[: fit.spec.time_buckets] = 0.0
        prediction[regulation] = (
            rubberband_design(frame.loc[regulation], fit.spec) @ coefficients
        )
    return prediction


def possession_coefficient_table(
    fit: RubberbandFit,
    *,
    bucket_size: int,
) -> pd.DataFrame:
    """Label a shared rubber-band fit in combined regulation possessions."""
    table = coefficient_table(fit).drop(
        columns=["minutes_elapsed_start", "minutes_elapsed_end"]
    )
    table["possessions_elapsed_start"] = table["time_bucket"] * bucket_size
    table["possessions_elapsed_end"] = (table["time_bucket"] + 1) * bucket_size
    table.loc[
        table["time_bucket"].eq(fit.spec.time_buckets - 1),
        "possessions_elapsed_end",
    ] = np.nan
    return table
