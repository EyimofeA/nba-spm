"""Joint full-span career RAPM with player-season coefficients.

This fits one model across all selected possessions rather than stitching
single-season runs together. Player value is estimated as player-season
offense/defense coefficients with:

* context columns: rubber band, home, playoff, season effects
* level ridge penalties on player-season coefficients
* adjacent-season smoothness penalties, targeted to the learned aging curve

The solver streams possessions in chunks and accumulates normal equations, so it
does not materialize the full possession design matrix at once.
"""
from __future__ import annotations
import os

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import cg, splu

from paths import (
    AGING_CURVE_CSV,
    ALL_NAMES_CSV,
    JOINT_CAREER_CONTEXT_CSV,
    JOINT_CAREER_META_JSON,
    JOINT_CAREER_PEAK_3YR_CSV,
    JOINT_CAREER_PLAYER_SEASONS_CSV,
    JOINT_CAREER_SUMMARY_CSV,
    PLAYERSHEETS_YEAR_TOTALS,
    ensure_dirs,
)


RUBBER_BAND_CLIP = 25.0
TEAM_ABBREVIATION_MAP = {"PHX": "PHO"}
PLAYOFF_OVERRIDES = {
    2020: (Date(2020, 8, 17), Date(2020, 10, 11)),
    2021: (Date(2021, 5, 22), Date(2021, 7, 20)),
}


@dataclass
class ColumnMaps:
    player_cols: dict[tuple[int, int, str], int]
    col_names: dict[int, str]
    context_cols: dict[str, int]


def ensure_mysql_driver():
    try:
        import MySQLdb  # type: ignore

        return MySQLdb
    except Exception:
        import pymysql  # type: ignore

        pymysql.install_as_MySQLdb()
        import MySQLdb  # type: ignore

        return MySQLdb


def connect_db():
    mysql = ensure_mysql_driver()
    return mysql.connect(
        host="localhost",
        user="root",
        password=os.environ.get("NBA_DB_PASSWORD", ""),
        db="nba_api",
        unix_socket="/tmp/mysql.sock",
    )


def as_date(value) -> Date | None:
    if value is None:
        return None
    if isinstance(value, Date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def playoff_window(season: int) -> tuple[Date, Date]:
    return PLAYOFF_OVERRIDES.get(season, (Date(season, 4, 12), Date(season, 6, 30)))


def is_playoff_game(season: int, game_date) -> bool:
    parsed = as_date(game_date)
    if parsed is None:
        return False
    lo, hi = playoff_window(season)
    return lo <= parsed <= hi


def season_list(start: int, end: int) -> list[int]:
    return list(range(start, end + 1))


def placeholders(seasons: list[int]) -> str:
    return ",".join(["%s"] * len(seasons))


def iter_possessions(seasons: list[int], chunk_size: int):
    query = f"""
        SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5,
               season, date, period, num, gameid
        FROM matchups
        WHERE season IN ({placeholders(seasons)})
          AND pts IS NOT NULL
        ORDER BY season, date, gameid, period, num
    """
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(query, seasons)
    try:
        while True:
            rows = cur.fetchmany(chunk_size)
            if not rows:
                break
            yield rows
    finally:
        cur.close()
        conn.close()


def load_player_seasons(seasons: list[int]) -> set[tuple[int, int]]:
    unions = []
    for col in ("a1", "a2", "a3", "a4", "a5", "h1", "h2", "h3", "h4", "h5"):
        unions.append(
            f"SELECT season, {col} AS player_id FROM matchups "
            f"WHERE season IN ({placeholders(seasons)}) AND {col} IS NOT NULL"
        )
    query = " UNION ".join(unions)
    params = seasons * 10
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(query, params)
    out = {(int(season), int(player_id)) for season, player_id in cur.fetchall()}
    cur.close()
    conn.close()
    return out


def build_column_maps(seasons: list[int], player_seasons: set[tuple[int, int]]) -> ColumnMaps:
    player_cols: dict[tuple[int, int, str], int] = {}
    col_names: dict[int, str] = {}
    for season, player_id in sorted(player_seasons):
        for side in ("off", "def"):
            col = len(col_names)
            player_cols[(season, player_id, side)] = col
            col_names[col] = f"{player_id}_{season}_{side}"

    context_cols: dict[str, int] = {}
    for name in ("rubber_band", "home_effect", "playoff_indicator"):
        col = len(col_names)
        context_cols[name] = col
        col_names[col] = name
    for season in seasons[1:]:
        name = f"season_{season}"
        col = len(col_names)
        context_cols[name] = col
        col_names[col] = name
    return ColumnMaps(player_cols=player_cols, col_names=col_names, context_cols=context_cols)


def update_score_and_rubber(game_score: dict, item: tuple) -> float:
    home_poss = bool(item[0])
    pts = float(item[1])
    gameid = item[16]
    if gameid not in game_score:
        game_score[gameid] = [0.0, 0.0]
    home_pts, away_pts = game_score[gameid]
    margin_home = home_pts - away_pts
    margin_from_offense = margin_home if home_poss else -margin_home
    rubber = float(np.clip(margin_from_offense, -RUBBER_BAND_CLIP, RUBBER_BAND_CLIP))
    if home_poss:
        game_score[gameid][0] += pts
    else:
        game_score[gameid][1] += pts
    return rubber


def first_pass(seasons: list[int], chunk_size: int) -> dict[str, float]:
    n = 0
    y_sum = 0.0
    rubber_sum = 0.0
    rubber_sq_sum = 0.0
    game_score: dict = {}
    for rows in iter_possessions(seasons, chunk_size):
        for item in rows:
            pts = float(item[1])
            rubber = update_score_and_rubber(game_score, item)
            n += 1
            y_sum += pts
            rubber_sum += rubber
            rubber_sq_sum += rubber * rubber
    if n == 0:
        raise RuntimeError("No possessions found.")
    rubber_mean = rubber_sum / n
    rubber_var = max(rubber_sq_sum / n - rubber_mean * rubber_mean, 1e-12)
    return {
        "n_possessions": float(n),
        "y_mean": y_sum / n,
        "rubber_mean": rubber_mean,
        "rubber_std": math.sqrt(rubber_var),
    }


def chunk_matrix(
    rows: list[tuple],
    maps: ColumnMaps,
    y_mean: float,
    rubber_mean: float,
    rubber_std: float,
    game_score: dict,
) -> tuple[coo_matrix, np.ndarray, np.ndarray]:
    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []
    y_centered = np.zeros(len(rows), dtype=np.float64)
    poss_counts = np.zeros(len(maps.col_names), dtype=np.float64)

    for row, item in enumerate(rows):
        home_poss = bool(item[0])
        season = int(item[12])
        game_date = item[13]
        away_players = [int(item[i]) for i in range(2, 7)]
        home_players = [int(item[i]) for i in range(7, 12)]
        off_players, def_players = (home_players, away_players) if home_poss else (away_players, home_players)

        for player_id in off_players:
            col = maps.player_cols[(season, player_id, "off")]
            row_idx.append(row)
            col_idx.append(col)
            values.append(1.0)
            poss_counts[col] += 1.0
        for player_id in def_players:
            col = maps.player_cols[(season, player_id, "def")]
            row_idx.append(row)
            col_idx.append(col)
            values.append(1.0)
            poss_counts[col] += 1.0

        rubber = (update_score_and_rubber(game_score, item) - rubber_mean) / rubber_std
        for name, value in (
            ("rubber_band", rubber),
            ("home_effect", 1.0 if home_poss else -1.0),
            ("playoff_indicator", 1.0 if is_playoff_game(season, game_date) else 0.0),
        ):
            col = maps.context_cols[name]
            row_idx.append(row)
            col_idx.append(col)
            values.append(float(value))
            poss_counts[col] += abs(float(value))

        season_name = f"season_{season}"
        if season_name in maps.context_cols:
            col = maps.context_cols[season_name]
            row_idx.append(row)
            col_idx.append(col)
            values.append(1.0)
            poss_counts[col] += 1.0

        y_centered[row] = float(item[1]) - y_mean

    X = coo_matrix((values, (row_idx, col_idx)), shape=(len(rows), len(maps.col_names)), dtype=np.float64)
    return X, y_centered, poss_counts


def load_aging_curve(path: Path) -> pd.DataFrame:
    curve = pd.read_csv(path)
    required = {"age", "dOff", "dDef"}
    missing = required - set(curve.columns)
    if missing:
        raise ValueError(f"Aging curve missing columns: {sorted(missing)}")
    return curve


def age_for_player_season(player_id: int, season: int) -> int | None:
    path = PLAYERSHEETS_YEAR_TOTALS / f"{season}.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=["PLAYER_ID", "AGE"])
    except ValueError:
        return None
    match = df[df["PLAYER_ID"].astype(int) == int(player_id)]
    if match.empty:
        return None
    return int(match["AGE"].dropna().astype(float).iloc[0])


def load_age_lookup(player_seasons: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    by_season: dict[int, set[int]] = defaultdict(set)
    for season, player_id in player_seasons:
        by_season[season].add(player_id)
    lookup: dict[tuple[int, int], int] = {}
    for season, player_ids in by_season.items():
        path = PLAYERSHEETS_YEAR_TOTALS / f"{season}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["PLAYER_ID", "AGE"]).dropna()
        df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
        df["AGE"] = df["AGE"].astype(float).round().astype(int)
        for row in df.itertuples(index=False):
            if int(row.PLAYER_ID) in player_ids:
                lookup[(season, int(row.PLAYER_ID))] = int(row.AGE)
    return lookup


def aging_delta(curve: pd.DataFrame, age: int | None, side: str) -> float:
    if age is None:
        return 0.0
    age_min = int(curve["age"].min())
    age_max = int(curve["age"].max())
    clipped = int(np.clip(age, age_min, age_max))
    col = "dOff" if side == "off" else "dDef"
    return float(curve.loc[curve["age"].astype(int) == clipped, col].iloc[0]) / 100.0


def add_smoothing_penalties(
    lhs: csc_matrix,
    rhs: np.ndarray,
    maps: ColumnMaps,
    player_seasons: set[tuple[int, int]],
    age_lookup: dict[tuple[int, int], int],
    curve: pd.DataFrame,
    lam_smooth_off: float,
    lam_smooth_def: float,
) -> tuple[csc_matrix, np.ndarray, int]:
    player_to_seasons: dict[int, list[int]] = defaultdict(list)
    for season, player_id in player_seasons:
        player_to_seasons[player_id].append(season)

    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []
    smooth_rhs: list[float] = []
    row = 0
    for player_id, seasons in player_to_seasons.items():
        sorted_seasons = sorted(set(seasons))
        for prev, nxt in zip(sorted_seasons, sorted_seasons[1:]):
            if nxt != prev + 1:
                continue
            for side, lam in (("off", lam_smooth_off), ("def", lam_smooth_def)):
                prev_col = maps.player_cols.get((prev, player_id, side))
                next_col = maps.player_cols.get((nxt, player_id, side))
                if prev_col is None or next_col is None:
                    continue
                scale = math.sqrt(lam)
                row_idx.extend([row, row])
                col_idx.extend([next_col, prev_col])
                values.extend([scale, -scale])
                age = age_lookup.get((prev, player_id))
                smooth_rhs.append(scale * aging_delta(curve, age, side))
                row += 1

    if row == 0:
        return lhs, rhs, 0
    D = coo_matrix((values, (row_idx, col_idx)), shape=(row, len(maps.col_names)), dtype=np.float64).tocsc()
    lhs = lhs + (D.T @ D).tocsc()
    rhs = rhs + np.asarray(D.T @ np.asarray(smooth_rhs, dtype=np.float64)).ravel()
    return lhs, rhs, row


def level_penalty(maps: ColumnMaps, args: argparse.Namespace) -> np.ndarray:
    penalty = np.zeros(len(maps.col_names), dtype=np.float64)
    for col, name in maps.col_names.items():
        if name.endswith("_off"):
            penalty[col] = args.lam_off
        elif name.endswith("_def"):
            penalty[col] = args.lam_def
        elif name == "rubber_band":
            penalty[col] = args.lam_rubber
        elif name == "home_effect":
            penalty[col] = args.lam_home
        elif name == "playoff_indicator":
            penalty[col] = args.lam_playoff
        elif name.startswith("season_"):
            penalty[col] = args.lam_season
    return penalty


def accumulate_normal_equations(
    seasons: list[int],
    maps: ColumnMaps,
    stats: dict[str, float],
    args: argparse.Namespace,
) -> tuple[csc_matrix, np.ndarray, np.ndarray]:
    n_cols = len(maps.col_names)
    lhs = diags(level_penalty(maps, args), format="csc")
    rhs = np.zeros(n_cols, dtype=np.float64)
    poss_counts = np.zeros(n_cols, dtype=np.float64)
    game_score: dict = {}
    processed = 0
    for rows in iter_possessions(seasons, args.chunk_size):
        X, y_centered, chunk_counts = chunk_matrix(
            rows,
            maps,
            y_mean=stats["y_mean"],
            rubber_mean=stats["rubber_mean"],
            rubber_std=stats["rubber_std"],
            game_score=game_score,
        )
        X_csr = X.tocsr()
        lhs = lhs + (X_csr.T @ X_csr).tocsc()
        rhs = rhs + np.asarray(X_csr.T @ y_centered).ravel()
        poss_counts += chunk_counts
        processed += len(rows)
        if args.progress and processed % (args.chunk_size * 5) == 0:
            print(f"  processed {processed:,} possessions")
    return lhs, rhs, poss_counts


def solve(lhs: csc_matrix, rhs: np.ndarray) -> np.ndarray:
    try:
        beta, info = cg(lhs.tocsr(), rhs, rtol=1e-7, maxiter=10000)
    except TypeError:
        beta, info = cg(lhs.tocsr(), rhs, tol=1e-7, maxiter=10000)
    if info != 0:
        print(f"CG did not converge cleanly (info={info}); falling back to sparse LU.")
        beta = splu(lhs).solve(rhs)
    return np.asarray(beta, dtype=np.float64)


def load_names() -> dict[int, str]:
    df = pd.read_csv(ALL_NAMES_CSV)
    return dict(zip(df["PLAYER_ID"].astype(int), df["PLAYER_NAME"].astype(str)))


def cumulative_age_adjustment(curve: pd.DataFrame, age: int | None, side: str, ref_age: int) -> float:
    if age is None:
        return 0.0
    cum_col = "cumOff" if side == "off" else "cumDef"
    age_min = int(curve["age"].min())
    age_max = int(curve["age"].max())
    age_i = int(np.clip(age, age_min, age_max))
    ref_i = int(np.clip(ref_age, age_min, age_max))
    by_age = curve.set_index(curve["age"].astype(int))
    return float(by_age.loc[ref_i, cum_col] - by_age.loc[age_i, cum_col])


def player_season_table(
    beta: np.ndarray,
    maps: ColumnMaps,
    player_seasons: set[tuple[int, int]],
    poss_counts: np.ndarray,
    age_lookup: dict[tuple[int, int], int],
    curve: pd.DataFrame,
    ref_age: int,
) -> pd.DataFrame:
    names = load_names()
    rows = []
    for season, player_id in sorted(player_seasons):
        off_col = maps.player_cols[(season, player_id, "off")]
        def_col = maps.player_cols[(season, player_id, "def")]
        off = float(beta[off_col]) * 100.0
        defense = float(beta[def_col]) * 100.0
        age = age_lookup.get((season, player_id))
        off_at_ref = off + cumulative_age_adjustment(curve, age, "off", ref_age)
        def_at_ref = defense + cumulative_age_adjustment(curve, age, "def", ref_age)
        rows.append({
            "Season": season,
            "Age": age,
            "PLAYER_ID": player_id,
            "Name": names.get(player_id, str(player_id)),
            "Off": off,
            "Def": defense,
            "RAPM": off - defense,
            "Off_at_ref": off_at_ref,
            "Def_at_ref": def_at_ref,
            "RAPM_at_ref": off_at_ref - def_at_ref,
            "Poss_Off": float(poss_counts[off_col]),
            "Poss_Def": float(poss_counts[def_col]),
            "Poss": float(poss_counts[off_col] + poss_counts[def_col]),
        })
    return pd.DataFrame(rows).sort_values(["Season", "RAPM"], ascending=[True, False])


def weighted_mean(group: pd.DataFrame, col: str) -> float:
    weights = group["Poss"].to_numpy(dtype=float)
    if weights.sum() <= 0:
        return float("nan")
    return float(np.average(group[col], weights=weights))


def career_summary(player_seasons: pd.DataFrame, min_poss: int, min_seasons: int) -> pd.DataFrame:
    eligible = player_seasons[player_seasons["Poss"] >= min_poss].copy()
    rows = []
    for (player_id, name), group in eligible.groupby(["PLAYER_ID", "Name"]):
        if group["Season"].nunique() < min_seasons:
            continue
        rows.append({
            "PLAYER_ID": player_id,
            "Name": name,
            "N_Seasons": int(group["Season"].nunique()),
            "First_Season": int(group["Season"].min()),
            "Last_Season": int(group["Season"].max()),
            "Total_Poss": float(group["Poss"].sum()),
            "Career_Wmean_RAPM": weighted_mean(group, "RAPM"),
            "Career_Wmean_Off": weighted_mean(group, "Off"),
            "Career_Wmean_Def": weighted_mean(group, "Def"),
            "Career_Wmean_RAPM_at_Ref": weighted_mean(group, "RAPM_at_ref"),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "PLAYER_ID", "Name", "N_Seasons", "First_Season", "Last_Season",
            "Total_Poss", "Career_Wmean_RAPM", "Career_Wmean_Off",
            "Career_Wmean_Def", "Career_Wmean_RAPM_at_Ref",
        ])
    return pd.DataFrame(rows).sort_values("Career_Wmean_RAPM", ascending=False)


def peak_3yr(player_seasons: pd.DataFrame, min_total_poss: int, age_min: int, age_max: int) -> pd.DataFrame:
    rows = []
    for (player_id, name), group in player_seasons.groupby(["PLAYER_ID", "Name"]):
        group = group.sort_values("Season")
        for start in range(int(group["Season"].min()), int(group["Season"].max()) - 1):
            window = group[(group["Season"] >= start) & (group["Season"] <= start + 2)]
            if window["Poss"].sum() < min_total_poss:
                continue
            avg_age = weighted_mean(window.dropna(subset=["Age"]), "Age") if window["Age"].notna().any() else float("nan")
            if not np.isfinite(avg_age) or avg_age < age_min or avg_age > age_max:
                continue
            rows.append({
                "PLAYER_ID": player_id,
                "Name": name,
                "Start_Season": start,
                "End_Season": start + 2,
                "Poss_Weighted_Age": avg_age,
                "Total_Poss": float(window["Poss"].sum()),
                "Peak_3yr_RAPM": weighted_mean(window, "RAPM"),
                "Peak_3yr_Off": weighted_mean(window, "Off"),
                "Peak_3yr_Def": weighted_mean(window, "Def"),
                "Peak_3yr_RAPM_at_Ref": weighted_mean(window, "RAPM_at_ref"),
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    idx = out.groupby("PLAYER_ID")["Peak_3yr_RAPM"].idxmax()
    return out.loc[idx].sort_values("Peak_3yr_RAPM", ascending=False)


def context_table(beta: np.ndarray, maps: ColumnMaps) -> pd.DataFrame:
    rows = []
    for name, col in maps.context_cols.items():
        rows.append({"context": name, "coef": float(beta[col]), "coef_per_100": float(beta[col]) * 100.0})
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joint full-span player-season career RAPM.")
    parser.add_argument("--start-season", type=int, default=1997)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--aging-curve", default=str(AGING_CURVE_CSV))
    parser.add_argument("--ref-age", type=int, default=27)
    parser.add_argument("--lam-off", type=float, default=2500.0)
    parser.add_argument("--lam-def", type=float, default=4000.0)
    parser.add_argument("--lam-rubber", type=float, default=2000.0)
    parser.add_argument("--lam-home", type=float, default=5000.0)
    parser.add_argument("--lam-playoff", type=float, default=5000.0)
    parser.add_argument("--lam-season", type=float, default=5000.0)
    parser.add_argument("--lam-smooth-off", type=float, default=3000.0)
    parser.add_argument("--lam-smooth-def", type=float, default=3000.0)
    parser.add_argument("--min-poss", type=int, default=1500)
    parser.add_argument("--min-seasons", type=int, default=5)
    parser.add_argument("--peak-3yr-min-poss", type=int, default=4500)
    parser.add_argument("--peak-age-min", type=int, default=23)
    parser.add_argument("--peak-age-max", type=int, default=34)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_dirs()
    start = time.time()
    seasons = season_list(args.start_season, args.end_season)
    print(f"Joint career RAPM seasons={args.start_season}-{args.end_season}")

    print("Loading player-season universe...")
    player_seasons = load_player_seasons(seasons)
    maps = build_column_maps(seasons, player_seasons)
    print(f"  player-seasons={len(player_seasons):,} columns={len(maps.col_names):,}")

    print("First pass: y mean and rubber-band standardization...")
    stats = first_pass(seasons, args.chunk_size)
    print(
        f"  possessions={int(stats['n_possessions']):,} y_mean={stats['y_mean']:.6f} "
        f"rubber_mean={stats['rubber_mean']:.3f} rubber_std={stats['rubber_std']:.3f}"
    )

    print("Accumulating normal equations...")
    lhs, rhs, poss_counts = accumulate_normal_equations(seasons, maps, stats, args)

    print("Adding aging-curve smoothness penalties...")
    curve = load_aging_curve(Path(args.aging_curve))
    age_lookup = load_age_lookup(player_seasons)
    lhs, rhs, n_smooth = add_smoothing_penalties(
        lhs,
        rhs,
        maps,
        player_seasons,
        age_lookup,
        curve,
        lam_smooth_off=args.lam_smooth_off,
        lam_smooth_def=args.lam_smooth_def,
    )
    print(f"  smoothness rows={n_smooth:,}")

    print("Solving...")
    beta = solve(lhs, rhs)

    print("Writing outputs...")
    player_table = player_season_table(
        beta,
        maps,
        player_seasons,
        poss_counts,
        age_lookup,
        curve,
        ref_age=args.ref_age,
    )
    summary = career_summary(player_table, min_poss=args.min_poss, min_seasons=args.min_seasons)
    peaks = peak_3yr(
        player_table,
        min_total_poss=args.peak_3yr_min_poss,
        age_min=args.peak_age_min,
        age_max=args.peak_age_max,
    )
    contexts = context_table(beta, maps)

    player_table.to_csv(JOINT_CAREER_PLAYER_SEASONS_CSV, index=False)
    summary.to_csv(JOINT_CAREER_SUMMARY_CSV, index=False)
    peaks.to_csv(JOINT_CAREER_PEAK_3YR_CSV, index=False)
    contexts.to_csv(JOINT_CAREER_CONTEXT_CSV, index=False)
    JOINT_CAREER_META_JSON.write_text(json.dumps({
        "seasons": seasons,
        "n_possessions": int(stats["n_possessions"]),
        "n_player_seasons": len(player_seasons),
        "n_columns": len(maps.col_names),
        "n_smoothness_rows": n_smooth,
        "y_mean": stats["y_mean"],
        "rubber_band": {
            "clip": RUBBER_BAND_CLIP,
            "mean": stats["rubber_mean"],
            "std": stats["rubber_std"],
        },
        "lambdas": {
            "offense": args.lam_off,
            "defense": args.lam_def,
            "rubber_band": args.lam_rubber,
            "home": args.lam_home,
            "playoff": args.lam_playoff,
            "season": args.lam_season,
            "smooth_off": args.lam_smooth_off,
            "smooth_def": args.lam_smooth_def,
        },
        "aging_curve": args.aging_curve,
        "ref_age": args.ref_age,
        "elapsed_seconds": time.time() - start,
        "outputs": {
            "player_seasons": str(JOINT_CAREER_PLAYER_SEASONS_CSV),
            "summary": str(JOINT_CAREER_SUMMARY_CSV),
            "peak_3yr": str(JOINT_CAREER_PEAK_3YR_CSV),
            "context": str(JOINT_CAREER_CONTEXT_CSV),
        },
    }, indent=2))

    print(f"Player seasons -> {JOINT_CAREER_PLAYER_SEASONS_CSV}")
    print(f"Summary -> {JOINT_CAREER_SUMMARY_CSV}")
    print(f"Peak 3yr -> {JOINT_CAREER_PEAK_3YR_CSV}")
    print(f"Context -> {JOINT_CAREER_CONTEXT_CSV}")
    print(f"Meta -> {JOINT_CAREER_META_JSON}")
    print("\nTop career weighted RAPM:")
    print(summary[["Name", "N_Seasons", "Total_Poss", "Career_Wmean_RAPM", "Career_Wmean_RAPM_at_Ref"]].head(20).to_string(index=False))
    print("\nTop raw 3-year prime peaks:")
    print(peaks[["Name", "Start_Season", "End_Season", "Poss_Weighted_Age", "Peak_3yr_RAPM", "Peak_3yr_RAPM_at_Ref"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
