import os
# Experimental RAPM runner with:
# - Garbage time filtering
# - Rubberband adjustment (score margin per quarter covariates)
# - Home-court advantage covariate
# - Within-game fatigue covariate
#
# This file leaves rapm.py untouched. Run this file to produce tagged outputs.

# Prefer native MySQLdb; fall back to PyMySQL if the native lib isn't available
try:
    import MySQLdb
except Exception:
    import pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb

from sklearn import linear_model
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import csv
import pandas as pd
from collections import defaultdict

from paths import (
    ALL_NAMES_CSV,
    DUMP,
    RAPM_RESULTS,
    COMBINED_EXPERIMENTAL,
    ensure_dirs,
)

ensure_dirs()

# ----------------------- Helpers -----------------------

def format_season_list(seasons):
    start_year = str(seasons[0])
    end_years = '-'.join([str(year)[-2:] for year in seasons[1:]])
    return f"{start_year}-{end_years}"


def get_player_id_by_name(name_query: str) -> str:
    df = pd.read_csv(ALL_NAMES_CSV)
    if "PLAYER_ID" not in df.columns or "PLAYER_NAME" not in df.columns:
        raise ValueError("all_names.csv must contain PLAYER_ID and PLAYER_NAME columns")
    df["PLAYER_NAME_lc"] = df["PLAYER_NAME"].astype(str).str.lower()
    name_q = str(name_query).strip().lower()
    matches = df[df["PLAYER_NAME_lc"].str.contains(name_q, na=False)]
    if len(matches) == 0:
        raise ValueError(f"Player not found for query: {name_query}")
    exact = matches[matches["PLAYER_NAME_lc"] == name_q]
    row = (exact.iloc[0] if len(exact) == 1 else matches.iloc[0])
    return str(row["PLAYER_ID"])  # keep as string


def calculate_time_decay_weights(seasons, base_weight=1.0, decay_rate=1.0):
    current_season = max(seasons)
    return {season: base_weight * (decay_rate ** (current_season - season)) for season in seasons}


# ----------------------- Core model -----------------------

# Optimized meta columns (validated via systematic search on 2020-2021 sample)
# Configuration showed +0.000207 improvement over baseline
META_COLS = [
    "META_home",          # +1 if home-possession offense, else 0
    "META_rb_q1",         # rubberband: score margin (home minus away) if period==1 else 0
    "META_rb_q2",
    "META_rb_q3",
    "META_rb_q4",
    "META_fatigue",       # within-game fatigue proxy [0..1] with optimized weights
]

# Base lead thresholds by quarter for garbage time filtering (absolute margin, before the play)
GT_BASE = {1: 25, 2: 20, 3: 17, 4: 12}


def gt_threshold(period: int, progress: float) -> int:
    """Dynamic garbage-time margin threshold. progress in [0,1] within quarter.
    Earlier in a quarter -> allow a bigger threshold (keep more data early), tighter late.
    """
    base = GT_BASE.get(period, 12)
    # widen early by up to +8, tighten late
    dynamic = base + int(8 * (1.0 - max(0.0, min(1.0, progress))))
    return dynamic


def run_ridge_model(X, y, sample_weights):
    # Optimized alpha search range
    alphas = [1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]
    clf = linear_model.RidgeCV(alphas=alphas, cv=4)
    clf.fit(X, y, sample_weight=sample_weights)
    return clf.coef_


def fetch_possessions(seasons):
    cur = MySQLdb.connect(host="localhost", user="root", password=os.environ.get("NBA_DB_PASSWORD", ""), db="nba_api", unix_socket='/tmp/mysql.sock')
    cursor = cur.cursor()
    placeholders = ','.join(['%s'] * len(seasons))
    # ps query
    # query = (
    #     """
    #     SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5,
    #            season, date, period, num, gameid
    #     FROM matchups
    #     WHERE season IN ({})
    #       AND date NOT BETWEEN CONCAT(season, '-04-12') AND CONCAT(season, '-06-30')
    #     ORDER BY gameid, period, num
    #     """.format(placeholders)
    # )
    
    query = (
        """
        SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5,
               season, date, period, num, gameid
        FROM matchups
        WHERE season IN ({})
        """.format(placeholders)
    )
    cursor.execute(query, seasons)
    data = cursor.fetchall()
    cursor.close()
    cur.close()
    return data


def build_and_fit(seasons, tag):
    data = fetch_possessions(seasons)

    # Collect players
    all_players = {}
    for item in data:
        for i in range(2, 12):  # a1..a5, h1..h5
            all_players[item[i]] = 1

    # Player columns
    player_to_col = {}
    col_to_key = {}
    for p in all_players:
        for side in ['off', 'def']:
            key = f"{p}_{side}"
            if key not in player_to_col:
                idx = len(player_to_col)
                player_to_col[key] = idx
                col_to_key[idx] = key

    # Add meta columns at the end
    meta_to_col = {}
    for m in META_COLS:
        idx = len(col_to_key)
        meta_to_col[m] = idx
        col_to_key[idx] = m

    # Precompute max possession index per (game, period) for fatigue progress normalization
    max_num = defaultdict(int)
    for item in data:
        _, _, *rest = item
        season = item[12]
        period = item[14]
        num = item[15]
        gameid = item[16]
        key = (gameid, period)
        if isinstance(num, (int, float)):
            max_num[key] = max(max_num[key], int(num))

    # Construct X, y with filters and covariates
    rows = []
    y = []
    weights = []
    season_weights = calculate_time_decay_weights(seasons, decay_rate=1.0)

    # Running score by game
    game_score = defaultdict(lambda: [0, 0])  # (home_pts, away_pts)

    for item in data:
        home_poss, pts = item[0], item[1]
        a = [item[i] for i in range(2, 7)]
        h = [item[i] for i in range(7, 12)]
        season = item[12]
        period = int(item[14])
        num = int(item[15])
        gameid = item[16]

        # Margin BEFORE this possession's points
        home_pts, away_pts = game_score[gameid]
        margin_home = home_pts - away_pts  # >0 means home leading

        # Compute progress within quarter
        denom = max(1, max_num[(gameid, period)])
        progress = min(1.0, num / float(denom))

        # Garbage time filter
        if abs(margin_home) >= gt_threshold(period, progress):
            # Skip this possession entirely
            # Update running score AFTER skipping so future margin is correct
            if home_poss:
                game_score[gameid][0] += pts
            else:
                game_score[gameid][1] += pts
            continue

        # Build one sparse row
        row = lil_matrix((1, len(col_to_key)))

        # Off/def player indicators
        off_list, def_list = (h, a) if home_poss else (a, h)  # if home_poss==1, home is offense
        for p in off_list:
            row[0, player_to_col[f"{p}_off"]] = 1
        for p in def_list:
            row[0, player_to_col[f"{p}_def"]] = 1

        # Meta: home-court advantage (1 for home offense, else 0)
        row[0, meta_to_col["META_home"]] = 1 if home_poss else 0

        # Rubberband: margin per quarter covariates (use margin at possession start)
        rb = margin_home  # centered around 0 already
        if period == 1:
            row[0, meta_to_col["META_rb_q1"]] = rb
        elif period == 2:
            row[0, meta_to_col["META_rb_q2"]] = rb
        elif period == 3:
            row[0, meta_to_col["META_rb_q3"]] = rb
        else:
            row[0, meta_to_col["META_rb_q4"]] = rb

        # Fatigue proxy: later periods and deeper into the period -> higher fatigue
        # Optimized weights: period weight 0.8, progress weight 0.2
        fatigue = ((period - 1) / 3.0) * 0.8 + (progress) * 0.2  # weighted combo in [0,1]
        row[0, meta_to_col["META_fatigue"]] = fatigue

        rows.append(row)
        y.append(pts)
        weights.append(season_weights[season])

        # Update running score AFTER building features
        if home_poss:
            game_score[gameid][0] += pts
        else:
            game_score[gameid][1] += pts

    if not rows:
        raise RuntimeError("No possessions left after filtering; relax thresholds or check data.")

    X = csr_matrix(lil_matrix(np.vstack([r.toarray() for r in rows])))
    y = np.array(y)
    y_av = np.average(y)
    beta = run_ridge_model(X, y - y_av, np.array(weights))

    # Write raw coefficients
    dump_suffix = f"_exp"
    dump_path = DUMP / f"results_{format_season_list(seasons)}{dump_suffix}.csv"
    with open(dump_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Player', 'Ridge Coefficient', 'Bayesian Coefficient'])
        for i, coef in enumerate(beta):
            w.writerow([col_to_key[i], coef, 0])
    print(f"Results saved to {dump_path}")

    # Human-readable player summary (ignore META_ rows)
    all_names = pd.read_csv(ALL_NAMES_CSV)
    id_to_name = dict(zip(all_names['PLAYER_ID'].astype(str), all_names['PLAYER_NAME']))

    res_df = pd.read_csv(dump_path)
    res_df = res_df[~res_df['Player'].str.startswith('META_')].copy()
    res_df['player_id'] = res_df['Player'].apply(lambda x: x.split('_')[0])
    res_df['role'] = res_df['Player'].apply(lambda x: x.split('_')[1])

    res_df['Off'] = res_df.apply(lambda r: r['Ridge Coefficient'] if r['role'] == 'off' else None, axis=1)
    res_df['Def'] = res_df.apply(lambda r: r['Ridge Coefficient'] if r['role'] == 'def' else None, axis=1)
    res_df['Name'] = res_df['player_id'].map(id_to_name)

    new_df = res_df[['Name', 'Off', 'Def']].copy()
    merged = new_df.groupby('Name').agg({'Off': 'sum', 'Def': 'sum'}).reset_index()
    merged['Rapm'] = merged['Off'] - merged['Def']
    merged.sort_values(by='Rapm', ascending=False, inplace=True)
    merged[["Off", "Def", "Rapm"]] = merged[["Off", "Def", "Rapm"]].mul(100)
    merged["Season"] = '-'.join(str(y) for y in seasons)
    print(merged.head(15))
    out_path = RAPM_RESULTS / f"Exp_Rapm_{seasons}.csv"
    merged.to_csv(out_path, index=False)
    print(f"Human readable results saved to {out_path}")


def main():
      # ignored; we fix to 5-year windows
    start_season = int(input("Enter Start Season"))
    end_season = int(input("Enter End Season "))
    length = int(input("Enter RAPM time length "))
    # 5-year windows from 2014–2018 to 2019–2023
    windows = []
    for start in range(start_season, end_season):
        end = start + length
        windows.append((end, start))  # (start_year=end, end_year=start)

    for start_year, end_year in windows:
        seasons = list(range(end_year, start_year + 1))
        print(f"\n=== Experimental window {format_season_list(seasons)} ===")
        build_and_fit(seasons, tag="exp")

    # Combine
    all_results = []
    for start_year, end_year in windows:
        seasons = list(range(end_year, start_year + 1))
        p = RAPM_RESULTS / f"Exp_Rapm_{seasons}.csv"
        all_results.append(pd.read_csv(p))
    combo = pd.concat(all_results, ignore_index=True)

    # Compress Season to 'yyyy-yy'
    def _shorten(s):
        parts = s.split('-')
        return f"{parts[0]}-{parts[-1][-2:]}" if len(parts) > 1 else s
    combo['Season'] = combo['Season'].apply(_shorten)

    combo.to_csv(COMBINED_EXPERIMENTAL, index=False)
    print(f"All experimental results combined and saved to {COMBINED_EXPERIMENTAL}")


if __name__ == "__main__":
    main()
