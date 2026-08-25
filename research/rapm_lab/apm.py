"""Simple APM over possession data (matchups caches 1997-2024 + optional 2024-26 canonical).

OLS, no regularization: the pre-ridge baseline every variant must beat.
X_ij = +1 if player j is on offense for possession i, -1 if on defense;
plus one home indicator column (+1 home possession, -1 away). y = points on
the possession. OFF_j = 100*beta_j; DEF_j = -100*beta_(N+j); NET = OFF + DEF.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

LAB_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE = REPO_ROOT / "rapm" / "data" / "possession_cache"
NAMES = REPO_ROOT / "rapm" / "data" / "all_names.csv"
CURRENT_POSSESSIONS = REPO_ROOT / "data" / "lake" / "silver" / "possessions.parquet"
POSSESSION_SEGMENTS = REPO_ROOT / "data" / "lake" / "silver" / "possession_lineup_segments.parquet"

PLAYER_COLUMNS = [f"a{i}" for i in range(1, 6)] + [f"h{i}" for i in range(1, 6)]


def load_caches(start: int, end: int) -> pd.DataFrame:
    frames = []
    for year in range(start, end + 1):
        path = CACHE / f"matchups_{year}.parquet"
        if not path.exists():
            print(f"skip {year}: missing", flush=True)
            continue
        frame = pd.read_parquet(path, columns=["home_poss", "pts", *PLAYER_COLUMNS])
        frames.append(frame)
        print(f"loaded {year}: {len(frame):,} rows", flush=True)
    return pd.concat(frames, ignore_index=True)


def load_current(from_season: int) -> pd.DataFrame:
    from nba_impact.models.rapm import load_current_possessions

    current = load_current_possessions(
        CURRENT_POSSESSIONS,
        POSSESSION_SEGMENTS,
        lineup_policy="terminal",
        game_types=("regular",),
    )
    current = current.loc[current["season"].ge(from_season), ["home_poss", "pts", *PLAYER_COLUMNS]]
    print(f"current era appended: {len(current):,} rows (season >= {from_season})", flush=True)
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=1997)
    parser.add_argument("--end", type=int, default=2024)
    parser.add_argument("--include-current", action="store_true", help="append canonical seasons >= end+1")
    parser.add_argument("--show-sample", action="store_true")
    parser.add_argument("--min-poss", type=int, default=20000, help="exposure filter for printed tables")
    parser.add_argument("--damp", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        label = f"{args.start}_{args.end}" + ("+cur" if args.include_current else "")
        args.out = LAB_ROOT / "outputs" / f"apm_{label}.csv"

    frame = load_caches(args.start, args.end)
    if args.include_current:
        frame = pd.concat([frame, load_current(max(args.end + 1, 2025))], ignore_index=True)

    if args.show_sample:
        print("\nMODEL: y_i = points on possession i. Two player blocks, 2N+1 columns.")
        print("X[i, off_j]=+1 when j on OFFENSE; X[i, def_j]=+1 when on DEFENSE; home column +/-1. beta = argmin ||X b - y||^2 (OLS) or +lambda||b||^2 (ridge).")
        print("OFF_j = 100*beta_off_j | DEF_j = -100*beta_def_j | NET_j = OFF_j + DEF_j\n")
        print("sample possession row:")
        print(frame.iloc[0].to_dict(), "\n", flush=True)

    values = frame[PLAYER_COLUMNS].to_numpy(dtype=np.int64)
    offense = np.where(frame["home_poss"].to_numpy(dtype=bool)[:, None], values[:, 5:], values[:, :5])
    defense = np.where(frame["home_poss"].to_numpy(dtype=bool)[:, None], values[:, :5], values[:, 5:])

    players = np.unique(values.ravel())
    players = players[players != 0]
    index_of = {int(pid): i for i, pid in enumerate(players)}
    n_players = len(players)
    n_rows = len(frame)

    def block(side: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flat = side.ravel()
        keep = flat != 0
        row = np.repeat(np.arange(n_rows), 5)[keep]
        col = np.fromiter((index_of[int(p)] for p in flat[keep]), dtype=np.int64, count=int(keep.sum())) + offset
        return row, col, np.ones(int(keep.sum()), dtype=np.float64)

    off_row, off_col, off_val = block(offense, 0)
    def_row, def_col, def_val = block(defense, n_players)
    home_row = np.arange(n_rows)
    home_col = np.full(n_rows, 2 * n_players)
    home_val = np.where(frame["home_poss"].to_numpy(dtype=bool), 1.0, -1.0)

    X = csr_matrix(
        (
            np.concatenate([off_val, def_val, home_val]),
            (
                np.concatenate([off_row, def_row, home_row]),
                np.concatenate([off_col, def_col, home_col]),
            ),
        ),
        shape=(n_rows, 2 * n_players + 1),
    )
    y = frame["pts"].to_numpy(dtype=np.float64)
    print(f"design: {n_rows:,} possessions x {X.shape[1]} columns | nnz={X.nnz:,}", flush=True)

    solution = lsqr(X, y, damp=args.damp, iter_lim=60000)
    beta = solution[0]
    print(f"solved: iterations={solution[2]}, reason={solution[1]}", flush=True)

    names = pd.read_csv(NAMES).drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")
    name_column = "PLAYER_NAME" if "PLAYER_NAME" in names.columns else names.columns[0]

    ratings = pd.DataFrame({"PLAYER_ID": players})
    ratings["OFF"] = beta[:n_players] * 100.0
    ratings["DEF"] = -beta[n_players : 2 * n_players] * 100.0
    ratings["NET"] = ratings["OFF"] + ratings["DEF"]
    off_exposure = np.zeros(n_players)
    def_exposure = np.zeros(n_players)
    np.add.at(off_exposure, off_col, 1)
    np.add.at(def_exposure, def_col - n_players, 1)
    ratings["OFF_POSS"] = off_exposure
    ratings["DEF_POSS"] = def_exposure
    ratings["PLAYER_NAME"] = ratings["PLAYER_ID"].map(names[name_column])
    ratings = ratings.sort_values("NET", ascending=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(args.out, index=False)

    qualified = ratings.loc[ratings["OFF_POSS"].ge(args.min_poss)]
    show = ["PLAYER_NAME", "OFF", "DEF", "NET", "OFF_POSS"]
    print(f"\nTOP 15 NET (>= {args.min_poss:,} poss):")
    print(qualified.head(15)[show].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print("\nBOTTOM 8 NET:")
    print(qualified.tail(8)[show].to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"\nwrote {len(ratings)} players -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
