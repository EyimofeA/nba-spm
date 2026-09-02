"""Weighted RAPM for canonical score-conserving lineup stints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg, spsolve

from nba_impact.models.rapm import RapmConfig, _penalty


@dataclass
class StintRapmDesign:
    X: csr_matrix
    points: np.ndarray
    possessions: np.ndarray
    players: np.ndarray
    game_ids: np.ndarray
    seasons: np.ndarray
    home_offense: np.ndarray
    off_possessions: np.ndarray
    def_possessions: np.ndarray


def load_canonical_stints(
    root: str | Path,
    seasons: tuple[int, ...],
) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = Path(root) / f"season={season}" / "regular.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    if not frames:
        raise ValueError("At least one season is required.")
    return pd.concat(frames, ignore_index=True)


def build_stint_design(frame: pd.DataFrame, *, include_home: bool = True) -> StintRapmDesign:
    player_columns = [
        f"{side}_player_{number}"
        for side in ("home", "away")
        for number in range(1, 6)
    ]
    required = {
        "season", "game_id", "home_possessions", "away_possessions",
        "home_points", "away_points", *player_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical stint frame is missing columns: {missing}")

    home = frame[[f"home_player_{number}" for number in range(1, 6)]].to_numpy(
        dtype=np.int64, copy=False
    )
    away = frame[[f"away_player_{number}" for number in range(1, 6)]].to_numpy(
        dtype=np.int64, copy=False
    )
    players = np.unique(np.concatenate([home.ravel(), away.ravel()]))
    players = np.asarray(sorted(int(player) for player in players), dtype=np.int64)
    n_stints = len(frame)
    n_rows = 2 * n_stints
    n_players = len(players)

    offense = np.vstack([home, away])
    defense = np.vstack([away, home])
    offense_indices = np.searchsorted(players, offense)
    defense_indices = np.searchsorted(players, defense)
    row_base = np.repeat(np.arange(n_rows, dtype=np.int64), 5)
    row_parts = [row_base, row_base]
    column_parts = [offense_indices.ravel(), n_players + defense_indices.ravel()]
    value_parts = [
        np.ones(n_rows * 5, dtype=np.float64),
        np.ones(n_rows * 5, dtype=np.float64),
    ]
    home_offense = np.concatenate(
        [np.ones(n_stints, dtype=bool), np.zeros(n_stints, dtype=bool)]
    )
    if include_home:
        row_parts.append(np.arange(n_rows, dtype=np.int64))
        column_parts.append(np.full(n_rows, 2 * n_players, dtype=np.int64))
        value_parts.append(np.where(home_offense, 1.0, -1.0))
    X = csr_matrix(
        (
            np.concatenate(value_parts),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=(n_rows, 2 * n_players + int(include_home)),
        dtype=np.float64,
    )
    possessions = np.concatenate(
        [frame["home_possessions"].to_numpy(float), frame["away_possessions"].to_numpy(float)]
    )
    points = np.concatenate(
        [frame["home_points"].to_numpy(float), frame["away_points"].to_numpy(float)]
    )
    if (possessions < 0).any() or (points < 0).any():
        raise ValueError("Canonical stints cannot contain negative possessions or points.")
    if (possessions == 0).all():
        raise ValueError("Canonical stints contain no possessions.")
    weighted_x = X.multiply(possessions[:, None]).tocsr()
    return StintRapmDesign(
        X=X,
        points=points,
        possessions=possessions,
        players=players,
        game_ids=np.concatenate([frame["game_id"].astype(str), frame["game_id"].astype(str)]),
        seasons=np.concatenate([frame["season"].to_numpy(int), frame["season"].to_numpy(int)]),
        home_offense=home_offense,
        off_possessions=np.asarray(weighted_x[:, :n_players].sum(axis=0)).ravel(),
        def_possessions=np.asarray(weighted_x[:, n_players : 2 * n_players].sum(axis=0)).ravel(),
    )


def fit_stint_center_path(
    design: StintRapmDesign,
    config: RapmConfig,
    center: np.ndarray,
    *,
    center_scales: tuple[float, ...],
    row_mask: np.ndarray | None = None,
) -> dict[float, tuple[np.ndarray, float]]:
    center = np.asarray(center, dtype=float)
    if center.shape != (design.X.shape[1],):
        raise ValueError("RAPM center must match the design column count.")
    if not np.isfinite(center).all():
        raise ValueError("RAPM center must contain only finite values.")
    if not center_scales or any(not 0 <= scale <= 1 for scale in center_scales):
        raise ValueError("center_scales must contain values between zero and one.")
    mask = np.ones(len(design.points), dtype=bool) if row_mask is None else row_mask
    X = design.X[mask]
    points = design.points[mask]
    possessions = design.possessions[mask]
    total_possessions = float(possessions.sum())
    intercept = float(points.sum() / total_possessions)
    weighted_x = X.multiply(possessions[:, None]).tocsr()
    penalty = _penalty(config, len(design.players))
    lhs = (X.T @ weighted_x).tocsr() + diags(penalty, format="csr")
    base_rhs = np.asarray(X.T @ (points - possessions * intercept)).ravel()
    center_rhs = penalty * center
    results = {}
    for scale in center_scales:
        rhs = base_rhs + scale * center_rhs
        try:
            beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
        except TypeError:
            beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
        if info != 0:
            beta = spsolve(lhs.tocsc(), rhs)
        beta = np.asarray(beta)
        n = len(design.players)
        off_weights = np.asarray(weighted_x[:, :n].sum(axis=0)).ravel()
        def_weights = np.asarray(weighted_x[:, n : 2 * n].sum(axis=0)).ravel()
        off_mean = float(np.average(beta[:n], weights=off_weights))
        def_mean = float(np.average(beta[n : 2 * n], weights=def_weights))
        beta[:n] -= off_mean
        beta[n : 2 * n] -= def_mean
        results[scale] = (beta, intercept + 5 * (off_mean + def_mean))
    return results


def stint_ratings(
    design: StintRapmDesign,
    beta: np.ndarray,
    *,
    row_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    mask = np.ones(len(design.points), dtype=bool) if row_mask is None else row_mask
    weighted_x = design.X[mask].multiply(design.possessions[mask, None]).tocsr()
    n = len(design.players)
    offense = 100 * beta[:n]
    defense = -100 * beta[n : 2 * n]
    result = pd.DataFrame(
        {
            "PLAYER_ID": design.players,
            "offense": offense,
            "defense": defense,
            "Poss_Off": np.asarray(weighted_x[:, :n].sum(axis=0)).ravel(),
            "Poss_Def": np.asarray(weighted_x[:, n : 2 * n].sum(axis=0)).ravel(),
        }
    )
    result["net"] = result["offense"] + result["defense"]
    return result
