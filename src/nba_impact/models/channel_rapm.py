"""Efficient multi-target and teammate-only RAPM research helpers.

The teammate model is not ordinary RAPM with the event owner deleted.  Each
player receives one focal row per opportunity.  The target removes that focal
player's own event, while separate nuisance blocks control the other four
teammates and the five opponents.  This makes the published focal coefficient
an association with *teammate* outcomes rather than the player's own box event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import factorized

from nba_impact.data.contracts import AWAY_PLAYER_COLUMNS, HOME_PLAYER_COLUMNS
from nba_impact.models.rapm import RapmConfig, build_design


@dataclass(frozen=True)
class TeammateChannelFit:
    players: np.ndarray
    coefficients: np.ndarray
    exposures: np.ndarray
    intercepts: np.ndarray
    targets: tuple[str, ...]
    opportunities: int


def lineup_sides(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return offensive and defensive terminal lineups for each row."""
    away = frame.loc[:, AWAY_PLAYER_COLUMNS].to_numpy(dtype=np.int64, copy=False)
    home = frame.loc[:, HOME_PLAYER_COLUMNS].to_numpy(dtype=np.int64, copy=False)
    home_offense = frame["home_poss"].to_numpy(dtype=bool, copy=False)
    return (
        np.where(home_offense[:, None], home, away),
        np.where(home_offense[:, None], away, home),
    )


def own_contribution_matrix(
    frame: pd.DataFrame,
    focal_lineup: np.ndarray,
    contributions: pd.DataFrame,
    *,
    targets: tuple[str, ...],
) -> np.ndarray:
    """Align player-event contributions to each opportunity's five focal players."""
    required = {"possession_id", "player_id", *targets}
    if missing := sorted(required - set(contributions.columns)):
        raise ValueError(f"Contribution table is missing columns: {missing}")
    if contributions.duplicated(["possession_id", "player_id"]).any():
        raise ValueError("Contributions must be unique by possession and player.")
    keys = pd.MultiIndex.from_arrays(
        [
            np.repeat(frame["possession_id"].astype(str).to_numpy(), 5),
            focal_lineup.ravel(),
        ],
        names=["possession_id", "player_id"],
    )
    indexed = contributions.copy()
    indexed["possession_id"] = indexed["possession_id"].astype(str)
    indexed["player_id"] = pd.to_numeric(indexed["player_id"], errors="raise").astype(int)
    aligned = indexed.set_index(["possession_id", "player_id"])[list(targets)].reindex(keys)
    return aligned.fillna(0.0).to_numpy(dtype=float).reshape(len(frame), 5, len(targets))


def _teammate_batch_design(
    focal: np.ndarray,
    opponents: np.ndarray,
    home_focal: np.ndarray,
    player_to_column: dict[int, int],
) -> csr_matrix:
    """Build focal, teammate-nuisance, and opponent-nuisance blocks."""
    n_opportunities = len(focal)
    n_players = len(player_to_column)
    expanded_rows = n_opportunities * 5
    row_index = np.arange(expanded_rows, dtype=np.int64)
    focal_slot = np.tile(np.arange(5, dtype=np.int64), n_opportunities)
    focal_players = focal.ravel()
    focal_columns = np.fromiter(
        (player_to_column[int(player)] for player in focal_players),
        dtype=np.int64,
        count=expanded_rows,
    )

    rows = [row_index]
    columns = [focal_columns]
    values = [np.ones(expanded_rows, dtype=float)]
    for teammate_slot in range(5):
        keep = focal_slot != teammate_slot
        teammate_players = np.repeat(focal[:, teammate_slot], 5)[keep]
        rows.append(row_index[keep])
        columns.append(
            n_players
            + np.fromiter(
                (player_to_column[int(player)] for player in teammate_players),
                dtype=np.int64,
                count=int(keep.sum()),
            )
        )
        values.append(np.ones(int(keep.sum()), dtype=float))
    for opponent_slot in range(5):
        opponent_players = np.repeat(opponents[:, opponent_slot], 5)
        rows.append(row_index)
        columns.append(
            2 * n_players
            + np.fromiter(
                (player_to_column[int(player)] for player in opponent_players),
                dtype=np.int64,
                count=expanded_rows,
            )
        )
        values.append(np.ones(expanded_rows, dtype=float))
    rows.append(row_index)
    columns.append(np.full(expanded_rows, 3 * n_players, dtype=np.int64))
    values.append(np.repeat(np.where(home_focal, 1.0, -1.0), 5))
    return coo_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
        shape=(expanded_rows, 3 * n_players + 1),
    ).tocsr()


def fit_teammate_channels(
    frame: pd.DataFrame,
    *,
    focal_side: str,
    team_targets: pd.DataFrame,
    contributions: pd.DataFrame,
    targets: tuple[str, ...],
    focal_penalty: float = 3000.0,
    nuisance_penalty: float = 12000.0,
    opponent_penalty: float = 15000.0,
    home_penalty: float = 300.0,
    batch_size: int = 50_000,
) -> TeammateChannelFit:
    """Fit lineup-adjusted effects on teammates' outcomes.

    ``team_targets`` contains one team total per opportunity.  ``contributions``
    contains each actor's part of that total.  The focal target is therefore
    ``team total - focal contribution``.
    """
    if focal_side not in {"offense", "defense"}:
        raise ValueError("focal_side must be offense or defense")
    if not targets:
        raise ValueError("At least one target is required.")
    target_frame = frame[["possession_id"]].merge(
        team_targets[["possession_id", *targets]],
        on="possession_id",
        how="left",
        validate="one_to_one",
    )
    target_values = target_frame[list(targets)].fillna(0.0).to_numpy(dtype=float)
    offense, defense = lineup_sides(frame)
    focal, opponents = (offense, defense) if focal_side == "offense" else (defense, offense)
    players = np.unique(np.concatenate([focal.ravel(), opponents.ravel()]))
    player_to_column = {int(player): index for index, player in enumerate(players)}
    own = own_contribution_matrix(frame, focal, contributions, targets=targets)
    home_offense = frame["home_poss"].to_numpy(dtype=bool)
    home_focal = home_offense if focal_side == "offense" else ~home_offense

    n_columns = 3 * len(players) + 1
    xtx = csr_matrix((n_columns, n_columns), dtype=float)
    xty = np.zeros((n_columns, len(targets)), dtype=float)
    x_sums = np.zeros(n_columns, dtype=float)
    target_sums = np.zeros(len(targets), dtype=float)
    exposures = np.zeros(len(players), dtype=float)
    rows_total = len(frame) * 5
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        matrix = _teammate_batch_design(
            focal[start:stop], opponents[start:stop], home_focal[start:stop], player_to_column
        )
        outcome = np.repeat(target_values[start:stop, None, :], 5, axis=1)
        outcome -= own[start:stop]
        outcome = outcome.reshape((stop - start) * 5, len(targets))
        xtx += matrix.T @ matrix
        xty += np.asarray(matrix.T @ outcome)
        x_sums += np.asarray(matrix.sum(axis=0)).ravel()
        exposures += np.asarray(matrix[:, : len(players)].sum(axis=0)).ravel()
        target_sums += outcome.sum(axis=0)
    intercepts = target_sums / rows_total
    centered_xty = xty - x_sums[:, None] * intercepts[None, :]
    penalties = np.concatenate(
        [
            np.full(len(players), focal_penalty),
            np.full(len(players), nuisance_penalty),
            np.full(len(players), opponent_penalty),
            np.asarray([home_penalty]),
        ]
    )
    solve = factorized(
        (xtx.astype(float) + diags(penalties, format="csr", dtype=float)).tocsc()
    )
    coefficients = np.column_stack(
        [np.asarray(solve(centered_xty[:, index])).ravel() for index in range(len(targets))]
    )
    focal_coefficients = coefficients[: len(players)].copy()
    for index in range(len(targets)):
        focal_coefficients[:, index] -= np.average(
            focal_coefficients[:, index], weights=exposures
        )
    return TeammateChannelFit(
        players=players,
        coefficients=100.0 * focal_coefficients,
        exposures=exposures,
        intercepts=intercepts,
        targets=targets,
        opportunities=len(frame),
    )


def teammate_ratings_table(
    fit: TeammateChannelFit,
    *,
    names: dict[int, str] | pd.DataFrame | None = None,
    invert: tuple[str, ...] = (),
) -> pd.DataFrame:
    if isinstance(names, pd.DataFrame):
        name_lookup = dict(
            zip(
                pd.to_numeric(names["PLAYER_ID"], errors="coerce").astype("Int64"),
                names["PLAYER_NAME"].astype(str),
            )
        )
    else:
        name_lookup = names or {}
    output = pd.DataFrame(
        {
            "player_id": fit.players,
            "player_name": [
                name_lookup.get(int(player), str(int(player))) for player in fit.players
            ],
            "opportunities": fit.exposures,
        }
    )
    for index, target in enumerate(fit.targets):
        sign = -1.0 if target in invert else 1.0
        output[target] = sign * fit.coefficients[:, index]
    return output


def fit_possession_channels(
    frame: pd.DataFrame,
    *,
    targets: tuple[str, ...],
    config: RapmConfig,
    names: dict[int, str] | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit several possession targets against one shared RAPM design."""
    if missing := sorted(set(targets) - set(frame.columns)):
        raise ValueError(f"Channel frame is missing targets: {missing}")
    design = build_design(frame.assign(pts=0.0), include_home=config.include_home)
    target = frame[list(targets)].to_numpy(dtype=float)
    intercepts = target.mean(axis=0)
    penalty = np.concatenate(
        [
            np.full(len(design.players), config.lambda_off),
            np.full(len(design.players), config.lambda_def),
            np.asarray([config.lambda_home]) if config.include_home else np.empty(0),
        ]
    )
    lhs = (design.X.T @ design.X).astype(float).tocsr() + diags(
        penalty, format="csr", dtype=float
    )
    rhs = np.asarray(design.X.T @ (target - intercepts))
    solve = factorized(lhs.tocsc())
    coefficients = np.column_stack(
        [np.asarray(solve(rhs[:, index])).ravel() for index in range(len(targets))]
    )
    n_players = len(design.players)
    off_exposure = np.asarray(design.X[:, :n_players].sum(axis=0)).ravel()
    def_exposure = np.asarray(design.X[:, n_players : 2 * n_players].sum(axis=0)).ravel()
    if isinstance(names, pd.DataFrame):
        name_lookup = dict(
            zip(
                pd.to_numeric(names["PLAYER_ID"], errors="coerce").astype("Int64"),
                names["PLAYER_NAME"].astype(str),
            )
        )
    else:
        name_lookup = names or {}
    output = pd.DataFrame(
        {
            "player_id": design.players,
            "player_name": [
                name_lookup.get(int(player), str(int(player))) for player in design.players
            ],
            "off_possessions": off_exposure,
            "def_possessions": def_exposure,
        }
    )
    for index, target_name in enumerate(targets):
        beta = coefficients[:, index].copy()
        off_mean = np.average(beta[:n_players], weights=off_exposure)
        def_mean = np.average(beta[n_players : 2 * n_players], weights=def_exposure)
        beta[:n_players] -= off_mean
        beta[n_players : 2 * n_players] -= def_mean
        output[f"{target_name}_offense"] = 100.0 * beta[:n_players]
        output[f"{target_name}_defense"] = -100.0 * beta[n_players : 2 * n_players]
        output[f"{target_name}_net"] = (
            output[f"{target_name}_offense"] + output[f"{target_name}_defense"]
        )
    return output
