#!/usr/bin/env python3
"""Historical team-level RAPM retrodiction.

Train on rolling prior seasons, aggregate player RAPM with the target season's
team minutes, and compare to actual team net rating / wins.
"""
from __future__ import annotations
import os

import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import cg, splu


SCRIPT_DIR = Path(__file__).resolve().parent
RAPM_ROOT = SCRIPT_DIR.parents[0]
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = RAPM_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paths import PLAYERSHEETS_YEAR_TOTALS  # noqa: E402


OUTPUT_DIR = SCRIPT_DIR / "outputs"
TEAM_RATINGS = PROJECT_ROOT / "zts" / "data" / "processed" / "team_ratings.csv"
WINDOW = 3
FIRST_TRAIN_SEASON = 1997
LAST_TARGET_SEASON = 2025
RUBBER_BAND_CLIP = 25.0
FIXED_LAMBDAS = {
    "offense": 2500.0,
    "defense": 4000.0,
    "rubber_band": 2000.0,
    "home": 5000.0,
    "playoff": 5000.0,
    "season": 5000.0,
}
TEAM_ABBREVIATION_MAP = {
    "PHX": "PHO",
}
PLAYOFF_OVERRIDES = {
    2020: (Date(2020, 8, 17), Date(2020, 10, 11)),
    2021: (Date(2021, 5, 22), Date(2021, 7, 20)),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    use_rubber_band: bool = False
    use_home: bool = False
    use_playoff: bool = False
    use_season_effects: bool = False


SPECS = [
    ModelSpec("simple"),
    ModelSpec(
        "complex",
        use_rubber_band=True,
        use_home=True,
        use_playoff=True,
        use_season_effects=True,
    ),
]


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


def fetch_possessions(seasons: list[int]) -> list[tuple]:
    placeholders = ",".join(["%s"] * len(seasons))
    query = f"""
        SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5,
               season, date, period, num, gameid
        FROM matchups
        WHERE season IN ({placeholders})
          AND pts IS NOT NULL
        ORDER BY season, date, gameid, period, num
    """
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(query, seasons)
    rows = list(cur.fetchall())
    cur.close()
    conn.close()
    return rows


def rubber_band_values(data: list[tuple]) -> np.ndarray:
    game_score: dict = {}
    values = np.zeros(len(data), dtype=np.float64)
    for idx, item in enumerate(data):
        home_poss = bool(item[0])
        pts = float(item[1])
        gameid = item[16]
        if gameid not in game_score:
            game_score[gameid] = [0.0, 0.0]
        home_pts, away_pts = game_score[gameid]
        margin_home = home_pts - away_pts
        margin_from_offense = margin_home if home_poss else -margin_home
        values[idx] = np.clip(margin_from_offense, -RUBBER_BAND_CLIP, RUBBER_BAND_CLIP)
        if home_poss:
            game_score[gameid][0] += pts
        else:
            game_score[gameid][1] += pts
    return values


def build_matrix(
    data: list[tuple],
    train_seasons: list[int],
    spec: ModelSpec,
) -> tuple[csr_matrix, np.ndarray, dict[int, str]]:
    all_players = sorted({int(item[i]) for item in data for i in range(2, 12)})
    player_to_col: dict[str, int] = {}
    col_to_name: dict[int, str] = {}
    for player in all_players:
        for side in ("off", "def"):
            key = f"{player}_{side}"
            col = len(col_to_name)
            player_to_col[key] = col
            col_to_name[col] = key

    rubber_band_col = None
    home_col = None
    playoff_col = None
    season_cols: dict[int, int] = {}
    if spec.use_rubber_band:
        rubber_band_col = len(col_to_name)
        col_to_name[rubber_band_col] = "rubber_band"
    if spec.use_home:
        home_col = len(col_to_name)
        col_to_name[home_col] = "home_effect"
    if spec.use_playoff:
        playoff_col = len(col_to_name)
        col_to_name[playoff_col] = "playoff_indicator"
    if spec.use_season_effects:
        for season in train_seasons[1:]:
            col = len(col_to_name)
            season_cols[season] = col
            col_to_name[col] = f"season_{season}"

    rubber_z = None
    if spec.use_rubber_band:
        rubber_raw = rubber_band_values(data)
        rubber_std = float(rubber_raw.std())
        if rubber_std <= 1e-12:
            rubber_std = 1.0
        rubber_z = (rubber_raw - float(rubber_raw.mean())) / rubber_std

    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []
    y = np.zeros(len(data), dtype=np.float64)
    for row, item in enumerate(data):
        home_poss = bool(item[0])
        season = int(item[12])
        game_date = item[13]
        away_players = [int(item[i]) for i in range(2, 7)]
        home_players = [int(item[i]) for i in range(7, 12)]
        off_players, def_players = (home_players, away_players) if home_poss else (away_players, home_players)

        for player in off_players:
            row_idx.append(row)
            col_idx.append(player_to_col[f"{player}_off"])
            values.append(1.0)
        for player in def_players:
            row_idx.append(row)
            col_idx.append(player_to_col[f"{player}_def"])
            values.append(1.0)
        if rubber_band_col is not None and rubber_z is not None:
            row_idx.append(row)
            col_idx.append(rubber_band_col)
            values.append(float(rubber_z[row]))
        if home_col is not None:
            row_idx.append(row)
            col_idx.append(home_col)
            values.append(1.0 if home_poss else -1.0)
        if playoff_col is not None and is_playoff_game(season, game_date):
            row_idx.append(row)
            col_idx.append(playoff_col)
            values.append(1.0)
        if season in season_cols:
            row_idx.append(row)
            col_idx.append(season_cols[season])
            values.append(1.0)
        y[row] = float(item[1])

    X = coo_matrix(
        (values, (row_idx, col_idx)),
        shape=(len(data), len(col_to_name)),
        dtype=np.float64,
    ).tocsr()
    return X, y, col_to_name


def lambda_vector(col_to_name: dict[int, str]) -> np.ndarray:
    penalty = np.zeros(len(col_to_name), dtype=np.float64)
    for col, key in col_to_name.items():
        if key.endswith("_off"):
            penalty[col] = FIXED_LAMBDAS["offense"]
        elif key.endswith("_def"):
            penalty[col] = FIXED_LAMBDAS["defense"]
        elif key == "rubber_band":
            penalty[col] = FIXED_LAMBDAS["rubber_band"]
        elif key == "home_effect":
            penalty[col] = FIXED_LAMBDAS["home"]
        elif key == "playoff_indicator":
            penalty[col] = FIXED_LAMBDAS["playoff"]
        elif key.startswith("season_"):
            penalty[col] = FIXED_LAMBDAS["season"]
        else:
            raise ValueError(f"Unknown column: {key}")
    return penalty


def fit_ridge(X: csr_matrix, y_centered: np.ndarray, col_to_name: dict[int, str]) -> np.ndarray:
    lhs = (X.T @ X).tocsc() + diags(lambda_vector(col_to_name), format="csc")
    rhs = X.T @ y_centered
    try:
        beta, info = cg(lhs.tocsr(), rhs, rtol=1e-6, maxiter=5000)
    except TypeError:
        beta, info = cg(lhs.tocsr(), rhs, tol=1e-6, maxiter=5000)
    if info != 0:
        beta = splu(lhs).solve(rhs)
    return beta


def player_ratings(beta: np.ndarray, col_to_name: dict[int, str]) -> pd.DataFrame:
    off: dict[int, float] = {}
    defense: dict[int, float] = {}
    for col, key in col_to_name.items():
        if key.endswith("_off"):
            off[int(key.split("_")[0])] = float(beta[col]) * 100.0
        elif key.endswith("_def"):
            defense[int(key.split("_")[0])] = float(beta[col]) * 100.0
    rows = []
    for player_id in sorted(set(off) | set(defense)):
        off_val = off.get(player_id, 0.0)
        def_val = defense.get(player_id, 0.0)
        rows.append({
            "PLAYER_ID": player_id,
            "Off": off_val,
            "Def": def_val,
            "RAPM": off_val - def_val,
        })
    return pd.DataFrame(rows)


def player_sheet_path(season: int) -> Path:
    return PLAYERSHEETS_YEAR_TOTALS / f"{season}.csv"


def load_team_minutes(season: int) -> pd.DataFrame:
    path = player_sheet_path(season)
    cols = ["PLAYER_ID", "TEAM_ABBREVIATION", "MIN", "Minutes"]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    minute_col = "MIN" if "MIN" in df.columns and df["MIN"].notna().any() else "Minutes"
    df = df.rename(columns={minute_col: "minutes", "TEAM_ABBREVIATION": "team"})
    df = df.dropna(subset=["PLAYER_ID", "team", "minutes"])
    df["team"] = df["team"].replace(TEAM_ABBREVIATION_MAP)
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(int)
    df["minutes"] = df["minutes"].astype(float)
    return df.groupby(["team", "PLAYER_ID"], as_index=False).agg(minutes=("minutes", "sum"))


def load_actual_from_team_ratings(season: int) -> pd.DataFrame | None:
    if not TEAM_RATINGS.exists():
        return None
    ratings = pd.read_csv(TEAM_RATINGS)
    ratings = ratings[ratings["Season"] == season].copy()
    if ratings.empty:
        return None
    ratings["team"] = ratings["Team"].replace(TEAM_ABBREVIATION_MAP)
    ratings["actual_net_rating"] = ratings["team_ortg"] - ratings["team_drtg"]
    ratings = ratings.rename(columns={"team_wins": "actual_wins"})
    return ratings[["team", "actual_net_rating", "actual_wins"]]


def load_actual_from_player_sheet(season: int) -> pd.DataFrame:
    path = player_sheet_path(season)
    needed = {"TEAM_ABBREVIATION", "W", "W_PCT", "MIN", "Minutes", "sp_work_NET_RATING", "NET_RATING"}
    df = pd.read_csv(path, usecols=lambda c: c in needed)
    minute_col = "MIN" if "MIN" in df.columns and df["MIN"].notna().any() else "Minutes"
    net_col = "sp_work_NET_RATING" if "sp_work_NET_RATING" in df.columns else "NET_RATING"
    df = df.rename(columns={minute_col: "minutes", "TEAM_ABBREVIATION": "team", net_col: "actual_net_rating"})
    df = df.dropna(subset=["team"])
    df["team"] = df["team"].replace(TEAM_ABBREVIATION_MAP)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce").fillna(0.0)
    df["actual_net_rating"] = pd.to_numeric(df["actual_net_rating"], errors="coerce")
    df["W"] = pd.to_numeric(df["W"], errors="coerce")

    rows = []
    for team, group in df.groupby("team"):
        valid_net = group.dropna(subset=["actual_net_rating"])
        if valid_net.empty:
            actual_net = float("nan")
        elif valid_net["minutes"].sum() > 0:
            actual_net = float(np.average(valid_net["actual_net_rating"], weights=valid_net["minutes"]))
        else:
            actual_net = float(valid_net["actual_net_rating"].mean())
        wins = float(group["W"].dropna().max()) if group["W"].notna().any() else float("nan")
        rows.append({"team": team, "actual_net_rating": actual_net, "actual_wins": wins})
    return pd.DataFrame(rows)


def load_actual_team_results(season: int) -> tuple[pd.DataFrame, str]:
    actual = load_actual_from_team_ratings(season)
    if actual is not None:
        return actual, "team_ratings"
    return load_actual_from_player_sheet(season), "player_sheet"


def aggregate_team_predictions(ratings: pd.DataFrame, minutes: pd.DataFrame, model: str, season: int) -> pd.DataFrame:
    merged = minutes.merge(ratings, on="PLAYER_ID", how="left", indicator=True)
    merged["rating_matched"] = merged["_merge"].eq("both")
    merged[["Off", "Def", "RAPM"]] = merged[["Off", "Def", "RAPM"]].fillna(0.0)
    rows = []
    for team, group in merged.groupby("team"):
        total_minutes = float(group["minutes"].sum())
        if total_minutes <= 0:
            continue
        rows.append({
            "target_season": season,
            "model": model,
            "team": team,
            "pred_net_rating": float(np.average(group["RAPM"], weights=group["minutes"])),
            "pred_off": float(np.average(group["Off"], weights=group["minutes"])),
            "pred_def": float(np.average(group["Def"], weights=group["minutes"])),
            "total_minutes": total_minutes,
            "matched_minutes": float(group.loc[group["rating_matched"], "minutes"].sum()),
            "minute_coverage": float(group.loc[group["rating_matched"], "minutes"].sum() / total_minutes),
        })
    return pd.DataFrame(rows)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    total = float(np.sum((y_true[mask] - y_true[mask].mean()) ** 2))
    if total <= 0:
        return float("nan")
    residual = float(np.sum((y_pred[mask] - y_true[mask]) ** 2))
    return 1.0 - residual / total


def corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(a.corr(b))


def metrics(group: pd.DataFrame) -> dict:
    err = group["pred_net_rating"] - group["actual_net_rating"]
    return {
        "n_teams": int(len(group)),
        "mean_minute_coverage": float(group["minute_coverage"].mean()),
        "net_rmse": float(math.sqrt(float(np.mean(err**2)))),
        "net_mae": float(np.mean(np.abs(err))),
        "net_r2": r_squared(group["actual_net_rating"].to_numpy(), group["pred_net_rating"].to_numpy()),
        "net_corr": corr(group["pred_net_rating"], group["actual_net_rating"]),
        "wins_corr": corr(group["pred_net_rating"], group["actual_wins"]),
    }


def main() -> None:
    start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_predictions = []
    season_metrics = []
    train_cache: dict[tuple[int, ...], list[tuple]] = {}

    for target_season in range(FIRST_TRAIN_SEASON + WINDOW, LAST_TARGET_SEASON + 1):
        train_seasons = list(range(target_season - WINDOW, target_season))
        if not player_sheet_path(target_season).exists():
            print(f"Skipping {target_season}: missing player sheet")
            continue
        key = tuple(train_seasons)
        if key not in train_cache:
            train_cache[key] = fetch_possessions(train_seasons)
        data = train_cache[key]
        if not data:
            print(f"Skipping {target_season}: no possessions for {train_seasons}")
            continue
        minutes = load_team_minutes(target_season)
        actual, actual_source = load_actual_team_results(target_season)

        for spec in SPECS:
            print(f"Fitting {spec.name}: train={train_seasons} target={target_season} rows={len(data)}")
            X, y, col_to_name = build_matrix(data, train_seasons, spec)
            intercept = float(y.mean())
            beta = fit_ridge(X, y - intercept, col_to_name)
            ratings = player_ratings(beta, col_to_name)
            team_pred = aggregate_team_predictions(ratings, minutes, spec.name, target_season)
            merged = team_pred.merge(actual, on="team", how="inner")
            merged["train_seasons"] = "-".join(str(s) for s in train_seasons)
            merged["actual_source"] = actual_source
            all_predictions.append(merged)
            row = {
                "target_season": target_season,
                "model": spec.name,
                "train_seasons": "-".join(str(s) for s in train_seasons),
                "actual_source": actual_source,
                **metrics(merged),
            }
            season_metrics.append(row)
            print(
                f"  {spec.name}: teams={row['n_teams']} rmse={row['net_rmse']:.3f} "
                f"r2={row['net_r2']:.3f} corr={row['net_corr']:.3f}"
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    by_season = pd.DataFrame(season_metrics)
    pooled_rows = []
    for model, group in predictions.groupby("model"):
        pooled = metrics(group)
        pooled.update({
            "model": model,
            "n_seasons": int(group["target_season"].nunique()),
            "first_target": int(group["target_season"].min()),
            "last_target": int(group["target_season"].max()),
            "avg_season_net_rmse": float(by_season.loc[by_season["model"] == model, "net_rmse"].mean()),
            "avg_season_net_r2": float(by_season.loc[by_season["model"] == model, "net_r2"].mean()),
            "avg_season_net_corr": float(by_season.loc[by_season["model"] == model, "net_corr"].mean()),
        })
        pooled_rows.append(pooled)
    summary = pd.DataFrame(pooled_rows).sort_values(["net_rmse", "net_r2"], ascending=[True, False])

    predictions_path = OUTPUT_DIR / "historical_team_retrodiction_predictions.csv"
    season_path = OUTPUT_DIR / "historical_team_retrodiction_by_season.csv"
    summary_path = OUTPUT_DIR / "historical_team_retrodiction_summary.csv"
    meta_path = OUTPUT_DIR / "historical_team_retrodiction_summary.json"
    predictions.to_csv(predictions_path, index=False)
    by_season.to_csv(season_path, index=False)
    summary.to_csv(summary_path, index=False)
    with open(meta_path, "w") as handle:
        json.dump(
            {
                "window": WINDOW,
                "first_train_season": FIRST_TRAIN_SEASON,
                "last_target_season": LAST_TARGET_SEASON,
                "include_playoffs": True,
                "season_weights": "uniform",
                "fixed_lambdas": FIXED_LAMBDAS,
                "models": [spec.__dict__ for spec in SPECS],
                "elapsed_seconds": time.time() - start,
                "outputs": {
                    "predictions": str(predictions_path),
                    "by_season": str(season_path),
                    "summary": str(summary_path),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    print(summary.to_string(index=False))
    print(f"Predictions -> {predictions_path}")
    print(f"By season -> {season_path}")
    print(f"Summary -> {summary_path}")
    print(f"Metadata -> {meta_path}")


if __name__ == "__main__":
    main()
