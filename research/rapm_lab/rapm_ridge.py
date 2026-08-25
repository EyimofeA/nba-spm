"""Ridge RAPM (production-style lambdas) with walk-forward predictiveness.

Spec
----
Two player blocks (N offense + N defense), entries +1; one home column (+/-1).
y_i = points on possession i.
beta = argmin ||X b - y||^2 + lam_off||b_off||^2 + lam_def||b_def||^2
       + lam_home*b_home^2                       (augmented LSQR)
OFF_j = 100*beta_off_j ; DEF_j = -100*beta_def_j ; NET = OFF + DEF.

Predictiveness
--------------
Train/test share one player universe (union); unseen players have beta=0.
Predicted home margin per 100 = mean over possessions of
(sign-flipped X beta), correlated with the ACTUAL home-perspective margin
(mean signed points *100) of the SAME held-out possessions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from scipy.sparse.linalg import lsqr

LAB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

CACHE = REPO_ROOT / "rapm" / "data" / "possession_cache"
CURRENT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
POSSESSION_SEGMENTS = REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"
NAMES = REPO_ROOT / "rapm" / "data" / "all_names.csv"

PLAYER_COLUMNS = [f"a{i}" for i in range(1, 6)] + [f"h{i}" for i in range(1, 6)]
LAMBDA_OFF, LAMBDA_DEF, LAMBDA_HOME = 3000.0, 3000.0, 300.0
LAMBDA_INTERCEPT = 1.0


def load_cache(labels: list[int]) -> pd.DataFrame:
    frames = []
    for year in labels:
        path = CACHE / f"matchups_{year}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path, columns=["home_poss", "pts", "gameid", *PLAYER_COLUMNS]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_canonical(start_years: tuple[int, ...]) -> pd.DataFrame:
    from nba_impact.models.rapm import load_current_possessions

    frame = load_current_possessions(
        CURRENT_POSSESSIONS, POSSESSION_SEGMENTS, lineup_policy="terminal", game_types=("regular",)
    )
    columns = ["home_poss", "pts", *PLAYER_COLUMNS]
    extra = "gameid" if "gameid" in frame.columns else "game_id"
    return frame.loc[frame["season"].isin(start_years), columns + [extra]].rename(columns={extra: "gameid"})


def load_window(seasons: list[tuple[str, int]]) -> pd.DataFrame:
    cache_labels = [v for kind, v in seasons if kind == "cache"]
    canon_starts = tuple(v for kind, v in seasons if kind == "canonical")
    frames = []
    if cache_labels:
        frames.append(load_cache(cache_labels))
    if canon_starts:
        frames.append(load_canonical(canon_starts))
    return pd.concat(frames, ignore_index=True)


def player_universe(*frames: pd.DataFrame) -> np.ndarray:
    values = np.concatenate([f[PLAYER_COLUMNS].to_numpy(dtype=np.int64).ravel() for f in frames])
    values = values[values != 0]
    return np.unique(values)


def build_design(frame: pd.DataFrame, players: np.ndarray):
    values = frame[PLAYER_COLUMNS].to_numpy(dtype=np.int64)
    home = frame["home_poss"].to_numpy(dtype=bool)
    offense = np.where(home[:, None], values[:, 5:], values[:, :5])
    defense = np.where(home[:, None], values[:, :5], values[:, 5:])
    n_players = len(players)
    index_of = {int(pid): i for i, pid in enumerate(players)}
    n_rows = len(frame)

    def block(side: np.ndarray, offset: int):
        flat = side.ravel()
        keep = flat != 0
        known_mask = np.fromiter((int(p) in index_of for p in flat[keep]), dtype=bool, count=int(keep.sum()))
        rows = np.repeat(np.arange(n_rows), 5)[keep][known_mask]
        cols = np.fromiter((index_of[int(p)] for p in flat[keep][known_mask]),
                           dtype=np.int64, count=int(known_mask.sum())) + offset
        return rows, cols, np.ones(int(known_mask.sum()), dtype=np.float64)

    off_row, off_col, off_val = block(offense, 0)
    def_row, def_col, def_val = block(defense, n_players)
    home_row = np.arange(n_rows)
    home_col = np.full(n_rows, 2 * n_players)
    home_val = np.where(home, 1.0, -1.0)
    int_row = np.arange(n_rows)
    int_col = np.full(n_rows, 2 * n_players + 1)
    int_val = np.ones(n_rows)
    X = csr_matrix(
        (
            np.concatenate([off_val, def_val, home_val, int_val]),
            (
                np.concatenate([off_row, def_row, home_row, int_row]),
                np.concatenate([off_col, def_col, home_col, int_col]),
            ),
        ),
        shape=(n_rows, 2 * n_players + 2),
    )
    y = frame["pts"].to_numpy(dtype=np.float64)

    off_pos = np.vectorize(lambda v: index_of.get(int(v), -1), otypes=[np.int64])(offense)
    def_pos = np.vectorize(lambda v: index_of.get(int(v), -1), otypes=[np.int64])(defense)
    meta = {
        "game": frame["gameid"].to_numpy(),
        "home": home,
        "offense_idx": off_pos,
        "defense_idx": def_pos,
        "off_poss": np.bincount(off_pos[off_pos >= 0], minlength=n_players).astype(float),
        "def_poss": np.bincount(def_pos[def_pos >= 0], minlength=n_players).astype(float),
    }
    return X, y, meta


def ridge_solve(X: csr_matrix, y: np.ndarray, n_players: int,
                lam_off: float = LAMBDA_OFF, lam_def: float = LAMBDA_DEF,
                lam_home: float = LAMBDA_HOME):
    pen_off = csr_matrix((np.full(n_players, np.sqrt(lam_off)),
                          (np.arange(n_players), np.arange(n_players))), shape=(n_players, X.shape[1]))
    pen_def = csr_matrix((np.full(n_players, np.sqrt(lam_def)),
                          (np.arange(n_players), n_players + np.arange(n_players))),
                         shape=(n_players, X.shape[1]))
    pen_home = csr_matrix(([np.sqrt(lam_home)], ([0], [2 * n_players])), shape=(1, X.shape[1]))
    pen_int = csr_matrix(([np.sqrt(LAMBDA_INTERCEPT)], ([0], [2 * n_players + 1])), shape=(1, X.shape[1]))
    augmented = vstack([X.tocsr(), pen_off, pen_def, pen_home, pen_int]).tocsr()
    target = np.concatenate([y, np.zeros(X.shape[1])])
    out = lsqr(augmented, target, iter_lim=60000)
    print(f"  solver: istop={out[1]} itn={out[2]}", flush=True)
    return out[0]


def game_predictions(beta: np.ndarray, meta: dict, n_players: int) -> pd.Series:
    """Predicted HOME margin per 100 possessions.

    X beta estimates raw offensive scoring on each possession (both blocks
    enter positively, home column included). Sign each possession's prediction
    by which side has the ball, then average per game *100.
    """
    safe_off = np.where(meta["offense_idx"] >= 0, meta["offense_idx"], 0)
    safe_def = np.where(meta["defense_idx"] >= 0, meta["defense_idx"], 0)
    off_mask = (meta["offense_idx"] >= 0).astype(float)
    def_mask = (meta["defense_idx"] >= 0).astype(float)
    off_pts = (beta[:n_players][safe_off] * off_mask).sum(axis=1)
    def_pts = (beta[n_players : 2 * n_players][safe_def] * def_mask).sum(axis=1)
    home_flip = np.where(meta["home"], 1.0, -1.0)
    raw = off_pts + def_pts + beta[2 * n_players] * home_flip + beta[2 * n_players + 1]
    return (pd.DataFrame({"game": meta["game"], "pred": raw * home_flip})
            .groupby("game")["pred"].mean() * 100.0)


def actual_margins(frame: pd.DataFrame) -> pd.Series:
    signed = frame["pts"] * np.where(frame["home_poss"].astype(bool), 1.0, -1.0)
    return (pd.DataFrame({"game": frame["gameid"], "signed": signed})
            .groupby("game")["signed"].mean() * 100.0)


def leaderboard(beta: np.ndarray, players: np.ndarray, meta: dict, names: pd.DataFrame,
                name_column: str, min_poss: int, label: str) -> pd.DataFrame:
    ratings = pd.DataFrame({"PLAYER_ID": players})
    ratings["OFF"] = beta[: len(players)] * 100.0
    ratings["DEF"] = -beta[len(players) : 2 * len(players)] * 100.0
    ratings["NET"] = ratings["OFF"] + ratings["DEF"]
    ratings["OFF_POSS"] = meta["off_poss"]
    ratings["DEF_POSS"] = meta["def_poss"]
    ratings["PLAYER_NAME"] = ratings["PLAYER_ID"].map(names[name_column])
    ratings = ratings.sort_values("NET", ascending=False)
    qualified = ratings.loc[ratings["OFF_POSS"].ge(min_poss)]
    show = ["PLAYER_NAME", "OFF", "DEF", "NET", "OFF_POSS"]
    print(qualified.head(10)[show].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    out = LAB_ROOT / "outputs" / f"rapm_{label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(out, index=False)
    print(f"wrote {len(ratings)} -> {out}", flush=True)
    return ratings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", nargs="*", default=["last3", "career"])
    parser.add_argument("--lam-off", type=float, default=LAMBDA_OFF)
    parser.add_argument("--lam-def", type=float, default=LAMBDA_DEF)
    args = parser.parse_args()

    names = pd.read_csv(NAMES).drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")
    name_column = "PLAYER_NAME" if "PLAYER_NAME" in names.columns else names.columns[0]

    LAST3 = [("canonical", 2024), ("canonical", 2025), ("canonical", 2026)]
    CAREER_TRAIN = ([("cache", y) for y in range(1997, 2024)]
                    + [("canonical", 2024), ("canonical", 2025), ("canonical", 2026)])

    for window in args.windows:
        min_poss = {"last3": 6000, "career": 20000}[window]

        if window == "last3":
            frames = [load_window([s]) for s in LAST3]
            print(f"\n=== LAST3 | {[len(f) for f in frames]} poss/season ===", flush=True)
            universe = player_universe(*frames)
            X_full, y_full, meta_full = build_design(pd.concat(frames, ignore_index=True), universe)
            print(f"design: {X_full.shape[0]:,} x {X_full.shape[1]}", flush=True)
            beta = ridge_solve(X_full, y_full, len(universe), args.lam_off, args.lam_def)
            leaderboard(beta, universe, meta_full, names, name_column, min_poss, "last3")

            print("\npredictiveness (FORWARD folds, no future data):")
            prior_cache = load_cache(list(range(1997, 2024)))
            for held in range(3):
                tr = pd.concat([prior_cache] + frames[:held], ignore_index=True)
                te_frame = frames[held]
                U = player_universe(tr, te_frame)
                Xtr, ytr, _ = build_design(tr, U)
                b_tr = ridge_solve(Xtr, ytr, len(U), args.lam_off, args.lam_def)
                _, _, meta_te = build_design(te_frame, U)
                pred = game_predictions(b_tr, meta_te, len(U))
                al = pd.concat([pred.rename("p"), actual_margins(te_frame).rename("a")], axis=1).dropna()
                print(f"  forward -> end={LAST3[held][1]}: corr={np.corrcoef(al.p, al.a)[0, 1]:.3f} "
                      f"mae={(al.p - al.a).abs().mean():.2f} games={len(al)}", flush=True)

        if window == "career":
            held_starts = (2024, 2025, 2026)  # end-year labels; each trained WITHOUT itself
            print(f"\n=== CAREER | train cache 1997-2022 + canonical as available ===", flush=True)

            # Full leaderboard fit: every season included exactly once.
            full_frames = [load_cache(list(range(1997, 2024))), load_canonical((2024, 2025, 2026))]
            full = pd.concat(full_frames, ignore_index=True)
            universe = player_universe(full)
            X_f, y_f, meta_f = build_design(full, universe)
            print(f"design: {X_f.shape[0]:,} x {X_f.shape[1]}", flush=True)
            beta = ridge_solve(X_f, y_f, len(universe), args.lam_off, args.lam_def)
            leaderboard(beta, universe, meta_f, names, name_column, min_poss, "career")

            print("\npredictiveness (FORWARD folds on canonical era, no future data):")
            canon_frames = {s: load_canonical((s,)) for s in (2024, 2025, 2026)}
            for held_start in held_starts:
                parts = []
                parts.append(load_cache(list(range(1997, 2024))))
                canon_frames_kept = [canon_frames[s] for s in (2024, 2025, 2026) if s < held_start]
                for frame_kept in canon_frames_kept:
                    parts.append(frame_kept)
                tr = pd.concat(parts, ignore_index=True)
                te_frame = load_canonical((held_start,))
                U = player_universe(tr, te_frame)
                Xtr, ytr, _ = build_design(tr, U)
                b_tr = ridge_solve(Xtr, ytr, len(U), args.lam_off, args.lam_def)
                _, _, meta_te = build_design(te_frame, U)
                pred = game_predictions(b_tr, meta_te, len(U))
                al = pd.concat([pred.rename("p"), actual_margins(te_frame).rename("a")], axis=1).dropna()
                print(f"  forward -> end={held_start}: corr={np.corrcoef(al.p, al.a)[0, 1]:.3f} "
                      f"mae={(al.p - al.a).abs().mean():.2f} games={len(al)}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
