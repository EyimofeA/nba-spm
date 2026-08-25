"""Factor and multinomial possession-outcome research models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import cg, spsolve
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS
from nba_impact.models.lineup_interactions import game_margin_metrics
from nba_impact.models.rapm import (
    RapmConfig,
    build_design,
    fit_coefficients,
    ratings_table,
    _penalty,
)


MULTINOMIAL_CLASSES = np.array([0, 1, 2, 3], dtype=np.int64)


@dataclass(frozen=True)
class FactorLedger:
    possessions: pd.DataFrame
    shots: pd.DataFrame
    ts_opportunities: pd.DataFrame
    rebound_opportunities: pd.DataFrame
    quality: dict[str, int | float]


def canonical_terminal_frame(
    possessions: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    seasons: tuple[int, ...],
) -> pd.DataFrame:
    """Retain canonical IDs and terminal lineups in the RAPM row contract."""
    base = possessions.loc[
        possessions["season_type"].eq("regular")
        & possessions["season_end"].isin(seasons)
    ].copy()
    selected = (
        segments.loc[segments["possession_id"].isin(base["possession_id"])]
        .sort_values(["possession_id", "segment_number"], kind="stable")
        .groupby("possession_id", as_index=False, sort=False)
        .tail(1)
    )
    selected = selected[
        [
            "possession_id",
            *(f"home_player_{index}" for index in range(1, 6)),
            *(f"away_player_{index}" for index in range(1, 6)),
        ]
    ]
    frame = base.merge(selected, on="possession_id", validate="one_to_one")
    rename = {
        **{f"away_player_{index}": f"a{index}" for index in range(1, 6)},
        **{f"home_player_{index}": f"h{index}" for index in range(1, 6)},
    }
    frame = frame.rename(columns=rename)
    frame["home_poss"] = frame["offense_is_home"].astype(int)
    frame["pts"] = frame["points"].astype(float)
    frame["season"] = frame["season_end"].astype(int)
    frame["gameid"] = frame["game_id"].astype(str)
    frame["num"] = frame["possession_number"].astype(int)
    for column in (*AWAY_PLAYER_COLUMNS, *HOME_PLAYER_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype("int64")
    if frame.duplicated("possession_id").any():
        raise ValueError("Canonical terminal possession IDs must be unique.")
    return frame


def assign_events_to_possessions(
    possessions: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Map an event to the canonical possession interval containing its action."""
    outputs: list[pd.DataFrame] = []
    event_columns = list(events.columns)
    for game_id, game_events in events.groupby("game_id", sort=False):
        game_possessions = possessions.loc[possessions["game_id"].eq(game_id)].sort_values(
            "start_action_number", kind="stable"
        )
        if game_possessions.empty:
            continue
        ordered = game_events.sort_values("actionNumber", kind="stable").copy()
        starts = game_possessions["start_action_number"].to_numpy(dtype=np.int64)
        positions = np.searchsorted(
            starts, ordered["actionNumber"].to_numpy(dtype=np.int64), side="right"
        ) - 1
        clipped = np.clip(positions, 0, len(game_possessions) - 1)
        ends = game_possessions["end_action_number"].to_numpy(dtype=np.int64)
        valid = (positions >= 0) & (
            ordered["actionNumber"].to_numpy(dtype=np.int64) <= ends[clipped]
        )
        if not valid.any():
            continue
        mapped = ordered.loc[valid, event_columns].copy()
        mapped_positions = clipped[valid]
        mapped["possession_id"] = game_possessions["possession_id"].to_numpy()[mapped_positions]
        mapped["possession_offense_team_id"] = game_possessions[
            "offense_team_id"
        ].to_numpy()[mapped_positions]
        outputs.append(mapped)
    if not outputs:
        return pd.DataFrame(columns=[*event_columns, "possession_id", "possession_offense_team_id"])
    return pd.concat(outputs, ignore_index=True)


def _rebound_opportunities(
    events: pd.DataFrame,
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    """Match a field-goal miss to the first subsequent rebound in the game."""
    possession_by_event = mapped.set_index(["game_id", "actionNumber"])["possession_id"]
    rows: list[dict[str, object]] = []
    for game_id, game in events.groupby("game_id", sort=False):
        pending: tuple[int, str, int] | None = None
        for event in game.sort_values("actionNumber", kind="stable").itertuples(index=False):
            action = str(event.actionType)
            event_key = (str(game_id), int(event.actionNumber))
            if action == "Missed Shot" and event_key in possession_by_event.index:
                team = int(event.teamId) if pd.notna(event.teamId) else 0
                pending = (team, str(possession_by_event.loc[event_key]), int(event.actionNumber))
            elif action == "Made Shot":
                pending = None
            elif action == "Rebound" and pending is not None:
                rebound_team = int(event.teamId) if pd.notna(event.teamId) else 0
                if rebound_team:
                    shot_team, possession_id, miss_action = pending
                    rows.append(
                        {
                            "possession_id": possession_id,
                            "miss_action_number": miss_action,
                            "offensive_rebound": float(rebound_team == shot_team),
                        }
                    )
                pending = None
            elif action in {"Turnover", "period"}:
                pending = None
    return pd.DataFrame(
        rows,
        columns=["possession_id", "miss_action_number", "offensive_rebound"],
    )


def build_factor_ledger(
    possessions: pd.DataFrame,
    segments: pd.DataFrame,
    events: pd.DataFrame,
    *,
    seasons: tuple[int, ...] = (2024, 2025, 2026),
) -> FactorLedger:
    """Build eFG-shot, turnover-possession, and OREB-opportunity targets."""
    frame = canonical_terminal_frame(possessions, segments, seasons=seasons)
    event_subset = events.loc[
        events["season_type"].eq("regular")
        & events["season_end"].isin(seasons)
        & events["actionType"].isin(
            ["Made Shot", "Missed Shot", "Free Throw", "Turnover", "Rebound", "period"]
        )
    ].copy()
    mapped = assign_events_to_possessions(frame, event_subset)
    mapped = mapped.merge(
        frame[["possession_id", "offense_team_id"]],
        on="possession_id",
        validate="many_to_one",
    )
    offense_event = pd.to_numeric(mapped["teamId"], errors="coerce").eq(
        mapped["offense_team_id"]
    )
    field_goal = mapped["actionType"].isin(["Made Shot", "Missed Shot"])
    source_shot_value = pd.to_numeric(mapped["shotValue"], errors="coerce")
    inferred_shot_value = np.where(
        mapped["description"].astype("string").str.contains("3PT", case=False, na=False),
        3.0,
        2.0,
    )
    mapped["factor_shot_value"] = source_shot_value.where(
        source_shot_value.isin([2, 3]), inferred_shot_value
    )
    inferred_shot_values = int(
        (field_goal & offense_event & ~source_shot_value.isin([2, 3])).sum()
    )
    shots = mapped.loc[field_goal & offense_event].copy()
    shots["efg_value"] = np.where(
        shots["actionType"].eq("Made Shot"),
        pd.to_numeric(shots["factor_shot_value"], errors="raise") / 2.0,
        0.0,
    )
    shots = shots.merge(frame, on="possession_id", suffixes=("_event", ""), validate="many_to_one")
    free_throw = mapped["actionType"].eq("Free Throw")
    ts_events = mapped.loc[(field_goal | free_throw) & offense_event].copy()
    points_added = pd.to_numeric(
        ts_events["points_added"]
        if "points_added" in ts_events
        else pd.Series(np.nan, index=ts_events.index),
        errors="coerce",
    )
    inferred_ft_make = ~ts_events["description"].astype("string").str.contains(
        "MISS", case=False, na=False
    )
    ts_events["ts_points"] = np.where(
        ts_events["actionType"].eq("Made Shot"),
        pd.to_numeric(ts_events["factor_shot_value"], errors="coerce").fillna(0.0),
        np.where(
            ts_events["actionType"].eq("Free Throw"),
            points_added.where(points_added.notna(), inferred_ft_make.astype(float)),
            0.0,
        ),
    )
    ts_events["ts_attempt_weight"] = np.where(free_throw.loc[ts_events.index], 0.44, 1.0)
    ts_opportunities = (
        ts_events.groupby("possession_id", as_index=False)
        .agg(
            ts_points=("ts_points", "sum"),
            ts_attempt_weight=("ts_attempt_weight", "sum"),
        )
    )
    ts_opportunities["ts_value"] = ts_opportunities["ts_points"] / (
        2.0 * ts_opportunities["ts_attempt_weight"]
    )
    ts_opportunities = ts_opportunities.merge(
        frame, on="possession_id", validate="one_to_one"
    )
    turnover_ids = set(
        mapped.loc[mapped["actionType"].eq("Turnover") & offense_event, "possession_id"]
    )
    possession_targets = frame.copy()
    possession_targets["turnover"] = possession_targets["possession_id"].isin(turnover_ids).astype(float)
    opportunities = _rebound_opportunities(event_subset, mapped)
    opportunities = opportunities.merge(
        frame, on="possession_id", validate="many_to_one"
    )
    quality = {
        "possessions": int(len(frame)),
        "events_considered": int(len(event_subset)),
        "events_mapped": int(len(mapped)),
        "event_mapping_rate": float(len(mapped) / len(event_subset)) if len(event_subset) else float("nan"),
        "shot_attempts": int(len(shots)),
        "ts_possessions": int(len(ts_opportunities)),
        "ts_attempt_weight": float(ts_opportunities["ts_attempt_weight"].sum()),
        "inferred_shot_values": inferred_shot_values,
        "turnover_possessions": int(possession_targets["turnover"].sum()),
        "rebound_opportunities": int(len(opportunities)),
        "offensive_rebounds": int(opportunities["offensive_rebound"].sum()) if len(opportunities) else 0,
    }
    return FactorLedger(
        possession_targets,
        shots,
        ts_opportunities,
        opportunities,
        quality,
    )


def fit_weighted_factor_ratings(
    frame: pd.DataFrame,
    target: str,
    weight: str,
    *,
    factor: str,
    config: RapmConfig,
) -> pd.DataFrame:
    """Fit a denominator-weighted factor RAPM, used for true shooting."""
    model_frame = frame.copy()
    model_frame["pts"] = pd.to_numeric(model_frame[target], errors="raise").astype(float)
    weights = pd.to_numeric(model_frame[weight], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Factor weights must be finite and positive.")
    design = build_design(model_frame, include_home=config.include_home)
    root_weight = np.sqrt(weights)
    weighted_x = diags(root_weight) @ design.X
    intercept = float(np.average(design.y, weights=weights))
    penalty = diags(_penalty(config, len(design.players)), format="csr")
    lhs = (weighted_x.T @ weighted_x).tocsr() + penalty
    rhs = np.asarray(weighted_x.T @ ((design.y - intercept) * root_weight)).ravel()
    try:
        beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        beta = spsolve(lhs.tocsc(), rhs)
    beta = np.asarray(beta, dtype=float)
    n_players = len(design.players)
    off_counts = np.asarray(
        design.X[:, :n_players].T @ weights
    ).ravel()
    def_counts = np.asarray(
        design.X[:, n_players : 2 * n_players].T @ weights
    ).ravel()
    off_mean = float(np.average(beta[:n_players], weights=off_counts))
    def_mean = float(np.average(beta[n_players : 2 * n_players], weights=def_counts))
    beta[:n_players] -= off_mean
    beta[n_players : 2 * n_players] -= def_mean
    ratings = ratings_table(design, beta)
    ratings["off_possessions"] = off_counts
    ratings["def_possessions"] = def_counts
    return ratings.rename(
        columns={
            "offense_per_100": f"{factor}_offense",
            "defense_per_100": f"{factor}_defense",
            "net_per_100": f"{factor}_net",
            "off_possessions": f"{factor}_off_exposure",
            "def_possessions": f"{factor}_def_exposure",
        }
    )


def fit_factor_ratings(
    frame: pd.DataFrame,
    target: str,
    *,
    factor: str,
    higher_is_good_for_offense: bool,
    config: RapmConfig,
) -> pd.DataFrame:
    """Fit one factor with the normal player lineup ridge design."""
    model_frame = frame.copy()
    model_frame["pts"] = pd.to_numeric(model_frame[target], errors="raise").astype(float)
    design = build_design(model_frame, include_home=config.include_home)
    beta, _ = fit_coefficients(design, config)
    ratings = ratings_table(design, beta)
    if not higher_is_good_for_offense:
        ratings[["offense_per_100", "defense_per_100", "net_per_100"]] *= -1.0
    return ratings.rename(
        columns={
            "offense_per_100": f"{factor}_offense",
            "defense_per_100": f"{factor}_defense",
            "net_per_100": f"{factor}_net",
            "off_possessions": f"{factor}_off_exposure",
            "def_possessions": f"{factor}_def_exposure",
        }
    )


def points_class(points: pd.Series) -> np.ndarray:
    values = pd.to_numeric(points, errors="raise").to_numpy(dtype=float)
    if not np.allclose(values, np.rint(values), atol=1e-12) or (values < 0).any():
        raise ValueError("Multinomial outcomes require nonnegative integer points.")
    return np.minimum(np.rint(values).astype(np.int64), 3)


def fit_multinomial_lineup(
    X,
    y: np.ndarray,
    *,
    alpha: float,
    seed: int = 7,
) -> SGDClassifier:
    """Fit a sparse softmax model with one global L2 penalty."""
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        max_iter=1000,
        tol=1e-5,
        shuffle=True,
        random_state=seed,
        average=True,
    )
    model.fit(X, y)
    if not np.array_equal(model.classes_, MULTINOMIAL_CLASSES):
        raise ValueError("All four point-outcome classes must be present.")
    return model


def multinomial_metrics(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    three_plus_value: float,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    actual_class = points_class(frame["pts"])
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(frame), 4):
        raise ValueError("Expected one probability column for each outcome class.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Multinomial probabilities must sum to one.")
    one_hot = np.eye(4)[actual_class]
    class_values = np.array([0.0, 1.0, 2.0, float(three_plus_value)])
    expected_points = probabilities @ class_values
    actual_points = frame["pts"].to_numpy(dtype=float)
    point_error = actual_points - expected_points
    margin, games = game_margin_metrics(frame, expected_points)
    metrics = {
        "possessions": int(len(frame)),
        "log_loss": float(log_loss(actual_class, probabilities, labels=MULTINOMIAL_CLASSES)),
        "multiclass_brier": float(np.mean(np.sum((one_hot - probabilities) ** 2, axis=1))),
        "expected_points_rmse": float(math.sqrt(np.mean(point_error**2))),
        "expected_points_mae": float(np.mean(np.abs(point_error))),
        **margin,
    }
    return metrics, games
