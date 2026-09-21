"""Shared PULSE-search protocol: non-leaky Box15 priors, stint fits, H7 arms.

Research-only. Does not replace canonical PULSE artifacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import bmat, csr_matrix, diags
from scipy.sparse.linalg import cg, spsolve
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nba_impact.data.possessions import LINEUP_COLUMNS
from nba_impact.models.box_pipm_style import BOX_PIPM_STYLE_FEATURES
from nba_impact.models.rapm import RapmConfig, _penalty
from nba_impact.models.stint_rapm import StintRapmDesign


def fit_box15_prior(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    training_before: int,
    offense_alpha: float = 300.0,
    defense_alpha: float = 1000.0,
    learner: str = "ridge",
) -> tuple[pd.DataFrame, dict]:
    """Past-only Box15 prior. Labels with Window_End >= training_before are excluded."""
    labels = targets.loc[targets["Window_End"].lt(training_before)].copy()
    if labels.empty:
        raise ValueError(f"No Box15 labels before {training_before}")
    panel = features.merge(labels, on=["PLAYER_ID", "Window_End"], validate="one_to_one")
    panel["sample_weight"] = np.sqrt(
        np.minimum(panel["Poss_Off"], panel["Poss_Def"]).clip(lower=1)
    )
    if panel["Window_End"].ge(training_before).any():
        raise ValueError("Box15 training panel leaked the rating season.")
    prediction_rows = features.loc[features["Window_End"].eq(training_before)].copy()
    if prediction_rows.empty:
        raise ValueError(f"No Box15 features for {training_before}")
    models = {
        "offense": _fit_learner(
            panel, "target_offense", offense_alpha, learner, panel["sample_weight"].to_numpy(float)
        ),
        "defense": _fit_learner(
            panel, "target_defense", defense_alpha, learner, panel["sample_weight"].to_numpy(float)
        ),
    }
    priors = prediction_rows[["PLAYER_ID", "Window_End"]].copy()
    x = prediction_rows.loc[:, BOX_PIPM_STYLE_FEATURES]
    for side, model in models.items():
        priors[f"prior_{side}_per_100"] = model.predict(x)
    priors["prior_net_per_100"] = priors["prior_offense_per_100"] + priors["prior_defense_per_100"]
    return priors, {
        "training_rows": int(len(panel)),
        "training_start": int(panel["Window_End"].min()),
        "training_end": int(panel["Window_End"].max()),
        "prediction_rows": int(len(priors)),
        "learner": learner,
        "training_before": int(training_before),
    }


def _fit_learner(panel: pd.DataFrame, target: str, alpha: float, learner: str, weights: np.ndarray):
    x = panel.loc[:, BOX_PIPM_STYLE_FEATURES]
    y = panel[target].to_numpy(float)
    if learner == "ridge":
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(x, y, ridge__sample_weight=weights)
        return model
    if learner == "elasticnet":
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
                ("model", ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=8000)),
            ]
        )
        model.fit(x, y, model__sample_weight=weights)
        return model
    imputer = SimpleImputer(strategy="median")
    filled = imputer.fit_transform(x)
    if learner == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=250, max_depth=6, min_samples_leaf=8, random_state=20260921, n_jobs=1
        )
        model.fit(filled, y, sample_weight=weights)
        return _ArrayPredictor(model, imputer)
    if learner == "hgb":
        model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=20260921
        )
        model.fit(filled, y, sample_weight=weights)
        return _ArrayPredictor(model, imputer)
    raise ValueError(learner)


class _ArrayPredictor:
    def __init__(self, model, imputer):
        self.model = model
        self.imputer = imputer

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        filled = self.imputer.transform(frame.loc[:, BOX_PIPM_STYLE_FEATURES])
        return self.model.predict(filled)


def hustle_rates(hustle: pd.DataFrame, season: int, players: np.ndarray, possessions: np.ndarray) -> pd.DataFrame:
    frame = hustle.loc[pd.to_numeric(hustle["year"], errors="coerce").eq(season)].copy()
    frame["PLAYER_ID"] = pd.to_numeric(frame["PLAYER_ID"], errors="coerce")
    frame = frame.dropna(subset=["PLAYER_ID"])
    frame["PLAYER_ID"] = frame["PLAYER_ID"].astype(int)
    lookup = pd.Series(possessions, index=players)
    frame["Poss_Def"] = frame["PLAYER_ID"].map(lookup)
    frame = frame.loc[frame["Poss_Def"].ge(200)].copy()
    rates = pd.DataFrame({"PLAYER_ID": players.astype(int)})
    columns = {
        "gabriel_contested_2pt_p100": "CONTESTED_SHOTS_2PT",
        "gabriel_contested_3pt_p100": "CONTESTED_SHOTS_3PT",
        "gabriel_deflections_p100": "DEFLECTIONS",
        "gabriel_charges_p100": "CHARGES_DRAWN",
        "gabriel_def_boxouts_p100": "DEF_BOXOUTS",
        "gabriel_loose_def_p100": "DEF_LOOSE_BALLS_RECOVERED",
    }
    for name, source in columns.items():
        if source not in frame.columns:
            rates[name] = 0.0
            continue
        values = 100.0 * pd.to_numeric(frame[source], errors="coerce") / frame["Poss_Def"]
        mapped = pd.Series(values.to_numpy(float), index=frame["PLAYER_ID"]).reindex(rates["PLAYER_ID"])
        rates[name] = mapped.fillna(0.0).to_numpy(float)
    return rates


def add_z_increment(
    center: np.ndarray,
    design: StintRapmDesign,
    features: pd.DataFrame,
    columns: tuple[str, ...],
    scale: float,
    *,
    side: str = "defense",
) -> np.ndarray:
    n = len(design.players)
    lookup = features.set_index("PLAYER_ID")
    extra = np.zeros(n)
    for column in columns:
        extra += lookup[column].reindex(design.players).fillna(0.0).to_numpy(float)
    extra = extra / max(len(columns), 1)
    weights = design.def_possessions if side == "defense" else design.off_possessions
    std = float(np.sqrt(np.average((extra - np.average(extra, weights=weights)) ** 2, weights=weights)))
    if std > 1e-12:
        extra = extra / std
    extra -= np.average(extra, weights=weights)
    out = center.copy()
    if side == "defense":
        out[n : 2 * n] += -scale * extra / 100.0
    else:
        out[:n] += scale * extra / 100.0
    return out


def segments_to_stints_mode(
    possessions: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    mode: str = "first",
) -> pd.DataFrame:
    from research.run_pulse_search_h1_official_final import _attach_possession_meta

    extra = tuple(
        column
        for column in ("points", "points_excl_tech", "technical_points")
        if column in possessions.columns
    )
    frame = _attach_possession_meta(possessions, segments, extra=extra)
    first = frame.groupby("possession_id", as_index=False)["segment_number"].min().rename(
        columns={"segment_number": "first_segment"}
    )
    last = frame.groupby("possession_id", as_index=False)["segment_number"].max().rename(
        columns={"segment_number": "last_segment"}
    )
    frame = frame.merge(first, on="possession_id", how="left").merge(last, on="possession_id", how="left")
    nseg = frame.groupby("possession_id")["segment_number"].transform("nunique").clip(lower=1)
    excl = frame["points_excl_tech"] if "points_excl_tech" in frame else frame["points"]
    tech = frame["technical_points"] if "technical_points" in frame else 0
    if mode == "first":
        poss_weight = frame["segment_number"].eq(frame["first_segment"]).astype(float)
        point_weight = 1.0
    elif mode == "terminal":
        poss_weight = frame["segment_number"].eq(frame["last_segment"]).astype(float)
        point_weight = frame["segment_number"].eq(frame["last_segment"]).astype(float)
    elif mode == "fractional":
        poss_weight = 1.0 / nseg
        point_weight = 1.0 / nseg
    else:
        raise ValueError(mode)
    frame["home_possessions"] = np.where(frame["offense_is_home"], poss_weight, 0.0)
    frame["away_possessions"] = np.where(~frame["offense_is_home"], poss_weight, 0.0)
    frame["home_points"] = np.where(frame["offense_is_home"], excl * point_weight, 0.0)
    frame["away_points"] = np.where(~frame["offense_is_home"], excl * point_weight, 0.0)
    frame["home_points_excl"] = frame["home_points"]
    frame["away_points_excl"] = frame["away_points"]
    frame["home_technical_points_excluded"] = np.where(frame["offense_is_home"], tech * point_weight, 0.0)
    frame["away_technical_points_excluded"] = np.where(~frame["offense_is_home"], tech * point_weight, 0.0)
    grouped = frame.groupby(["game_id", "season_end", *LINEUP_COLUMNS], as_index=False, sort=False).agg(
        home_possessions=("home_possessions", "sum"),
        away_possessions=("away_possessions", "sum"),
        home_points=("home_points", "sum"),
        away_points=("away_points", "sum"),
        home_points_excl=("home_points_excl", "sum"),
        away_points_excl=("away_points_excl", "sum"),
        home_technical_points_excluded=("home_technical_points_excluded", "sum"),
        away_technical_points_excluded=("away_technical_points_excluded", "sum"),
    )
    grouped["season"] = grouped["season_end"].astype(int)
    return grouped


def _prepare_system(design: StintRapmDesign, config: RapmConfig, row_mask: np.ndarray | None = None):
    mask = np.ones(len(design.points), dtype=bool) if row_mask is None else row_mask
    x = design.X[mask]
    points = design.points[mask]
    possessions = design.possessions[mask]
    total = float(possessions.sum())
    intercept = float(points.sum() / total)
    weighted_x = x.multiply(possessions[:, None]).tocsr()
    penalty = _penalty(config, len(design.players))
    lhs = (x.T @ weighted_x).tocsr() + diags(penalty, format="csr")
    base_rhs = np.asarray(x.T @ (points - possessions * intercept)).ravel()
    return lhs, base_rhs, intercept, penalty, weighted_x, x


def _center_beta(beta: np.ndarray, weighted_x, intercept: float, n: int) -> tuple[np.ndarray, float]:
    off_weights = np.asarray(weighted_x[:, :n].sum(axis=0)).ravel()
    def_weights = np.asarray(weighted_x[:, n : 2 * n].sum(axis=0)).ravel()
    off_mean = float(np.average(beta[:n], weights=off_weights))
    def_mean = float(np.average(beta[n : 2 * n], weights=def_weights))
    beta = np.asarray(beta, dtype=float).copy()
    beta[:n] -= off_mean
    beta[n : 2 * n] -= def_mean
    return beta, intercept + 5 * (off_mean + def_mean)


def _solve(lhs, rhs) -> np.ndarray:
    try:
        beta, info = cg(lhs, rhs, rtol=1e-8, maxiter=10_000)
    except TypeError:
        beta, info = cg(lhs, rhs, tol=1e-8, maxiter=10_000)
    if info != 0:
        beta = spsolve(lhs.tocsc(), rhs)
    return np.asarray(beta, dtype=float)


def fit_many_centers(
    design: StintRapmDesign,
    config: RapmConfig,
    centers: dict[str, np.ndarray],
    *,
    scale: float = 1.0,
    row_mask: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, float]]:
    lhs, base_rhs, intercept, penalty, weighted_x, _x = _prepare_system(design, config, row_mask)
    n = len(design.players)
    results = {}
    for name, center in centers.items():
        center = np.asarray(center, dtype=float)
        rhs = base_rhs if np.allclose(center, 0) else base_rhs + scale * penalty * center
        beta = _solve(lhs, rhs)
        results[name] = _center_beta(beta, weighted_x, intercept, n)
    return results


def fit_constrained(
    design: StintRapmDesign,
    config: RapmConfig,
    center: np.ndarray,
    *,
    scale: float = 1.0,
    row_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    lhs, base_rhs, intercept, penalty, weighted_x, _x = _prepare_system(design, config, row_mask)
    n = len(design.players)
    rhs = base_rhs + scale * penalty * np.asarray(center, dtype=float)
    off_w = np.asarray(weighted_x[:, :n].sum(axis=0)).ravel()
    def_w = np.asarray(weighted_x[:, n : 2 * n].sum(axis=0)).ravel()
    constraint = np.zeros((2, lhs.shape[0]))
    constraint[0, :n] = off_w / max(float(off_w.sum()), 1.0)
    constraint[1, n : 2 * n] = def_w / max(float(def_w.sum()), 1.0)
    c = csr_matrix(constraint)
    kkt = bmat([[lhs, c.T], [c, None]], format="csr")
    kkt_rhs = np.concatenate([rhs, np.zeros(2)])
    solution = spsolve(kkt.tocsc(), kkt_rhs)
    beta = np.asarray(solution[: lhs.shape[0]], dtype=float)
    return _center_beta(beta, weighted_x, intercept, n)


def game_split_mask(design: StintRapmDesign, *, second_half: bool = False) -> np.ndarray:
    games = pd.Index(pd.unique(design.game_ids.astype(str))).sort_values()
    cut = games[len(games) // 2]
    is_late = pd.Series(design.game_ids.astype(str)).ge(str(cut)).to_numpy()
    return is_late if second_half else ~is_late
