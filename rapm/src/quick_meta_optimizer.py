import os
#!/usr/bin/env python3
"""
Quick meta-column optimization focusing on the most promising candidates.
"""

try:
    import MySQLdb
except Exception:
    import pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
import json

from paths import QUICK_OPT_RESULTS, ensure_dirs

ensure_dirs()


def fetch_possessions(seasons):
    cur = MySQLdb.connect(
        host="localhost", 
        user="root", 
        password=os.environ.get("NBA_DB_PASSWORD", ""), 
        db="nba_api", 
        unix_socket='/tmp/mysql.sock'
    )
    cursor = cur.cursor()
    placeholders = ','.join(['%s'] * len(seasons))
    
    query = f"""
    SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5,
           season, date, period, num, gameid
    FROM matchups
    WHERE season IN ({placeholders})
    ORDER BY gameid, period, num
    """
    
    cursor.execute(query, seasons)
    data = cursor.fetchall()
    cursor.close()
    cur.close()
    return data


def calculate_time_decay_weights(seasons, base_weight=1.0, decay_rate=1.0):
    current_season = max(seasons)
    return {season: base_weight * (decay_rate ** (current_season - season)) 
            for season in seasons}


def gt_threshold(period: int, progress: float) -> int:
    GT_BASE = {1: 25, 2: 20, 3: 17, 4: 12}
    base = GT_BASE.get(period, 12)
    dynamic = base + int(8 * (1.0 - max(0.0, min(1.0, progress))))
    return dynamic


def build_X_y_with_meta(data, meta_config):
    """Build X, y with specified meta configuration"""
    
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

    # Add meta columns
    meta_to_col = {}
    for m in meta_config:
        idx = len(col_to_key)
        meta_to_col[m] = idx
        col_to_key[idx] = m

    # Precompute max possession index per (game, period) for fatigue
    max_num = defaultdict(int)
    for item in data:
        period = item[14]
        num = item[15]
        gameid = item[16]
        key = (gameid, period)
        if isinstance(num, (int, float)):
            max_num[key] = max(max_num[key], int(num))

    # Build features
    rows = []
    y = []
    weights = []
    seasons = list(set(item[12] for item in data))
    season_weights = calculate_time_decay_weights(seasons, decay_rate=1.0)

    # Running score by game
    game_score = defaultdict(lambda: [0, 0])

    for item in data:
        home_poss, pts = item[0], item[1]
        a = [item[i] for i in range(2, 7)]
        h = [item[i] for i in range(7, 12)]
        season = item[12]
        period = int(item[14])
        num = int(item[15])
        gameid = item[16]

        # Margin BEFORE this possession
        home_pts, away_pts = game_score[gameid]
        margin_home = home_pts - away_pts

        # Progress within quarter
        denom = max(1, max_num[(gameid, period)])
        progress = min(1.0, num / float(denom))

        # Garbage time filter
        if abs(margin_home) >= gt_threshold(period, progress):
            if home_poss:
                game_score[gameid][0] += pts
            else:
                game_score[gameid][1] += pts
            continue

        # Build sparse row
        row = lil_matrix((1, len(col_to_key)))

        # Player indicators
        off_list, def_list = (h, a) if home_poss else (a, h)
        for p in off_list:
            row[0, player_to_col[f"{p}_off"]] = 1
        for p in def_list:
            row[0, player_to_col[f"{p}_def"]] = 1

        # Meta columns
        if "META_home" in meta_config:
            row[0, meta_to_col["META_home"]] = 1 if home_poss else 0

        if "META_rb_q1" in meta_config and period == 1:
            row[0, meta_to_col["META_rb_q1"]] = margin_home
        if "META_rb_q2" in meta_config and period == 2:
            row[0, meta_to_col["META_rb_q2"]] = margin_home
        if "META_rb_q3" in meta_config and period == 3:
            row[0, meta_to_col["META_rb_q3"]] = margin_home
        if "META_rb_q4" in meta_config and period >= 4:
            row[0, meta_to_col["META_rb_q4"]] = margin_home

        if "META_rb_all" in meta_config:
            row[0, meta_to_col["META_rb_all"]] = margin_home

        if "META_fatigue" in meta_config:
            fatigue = ((period - 1) / 3.0) * 0.6 + progress * 0.4
            row[0, meta_to_col["META_fatigue"]] = fatigue

        if "META_close_game" in meta_config:
            row[0, meta_to_col["META_close_game"]] = 1 if abs(margin_home) <= 5 else 0

        if "META_blowout" in meta_config:
            row[0, meta_to_col["META_blowout"]] = 1 if abs(margin_home) > 15 else 0

        if "META_late_game" in meta_config:
            row[0, meta_to_col["META_late_game"]] = 1 if (period >= 4 and progress > 0.75) else 0

        if "META_overtime" in meta_config:
            row[0, meta_to_col["META_overtime"]] = 1 if period > 4 else 0

        rows.append(row)
        y.append(pts)
        weights.append(season_weights[season])

        # Update running score
        if home_poss:
            game_score[gameid][0] += pts
        else:
            game_score[gameid][1] += pts

    if not rows:
        raise RuntimeError("No possessions left after filtering")

    X = csr_matrix(lil_matrix(np.vstack([r.toarray() for r in rows])))
    y = np.array(y)
    weights = np.array(weights)

    return X, y, weights


def evaluate_meta_config(data, meta_config):
    """Evaluate a meta configuration using cross-validation"""
    try:
        X, y, weights = build_X_y_with_meta(data, meta_config)
        
        y_centered = y - np.average(y)
        
        clf = linear_model.Ridge(alpha=10000)
        
        scores = cross_val_score(clf, X, y_centered, cv=3, 
                               scoring='neg_mean_squared_error',
                               fit_params={'sample_weight': weights})
        
        return scores.mean()
        
    except Exception as e:
        print(f"Error evaluating {meta_config}: {e}")
        return -np.inf


def main():
    print("Quick meta column optimization...")
    
    # Test seasons
    seasons = [2019, 2020, 2021]
    print(f"Loading data for seasons {seasons}...")
    
    data = fetch_possessions(seasons)
    print(f"Loaded {len(data)} possessions")

    # Define candidate meta columns (focus on most promising)
    candidates = {
        "home_court": ["META_home"],
        "rubberband_basic": ["META_rb_all"],
        "rubberband_quarters": ["META_rb_q1", "META_rb_q2", "META_rb_q3", "META_rb_q4"],
        "fatigue": ["META_fatigue"],
        "game_situation": ["META_close_game", "META_blowout"],
        "timing": ["META_late_game", "META_overtime"],
        "current_baseline": ["META_home", "META_rb_q1", "META_rb_q2", "META_rb_q3", "META_rb_q4", "META_fatigue"],
    }

    print("\n=== Testing meta column groups ===")
    results = {}
    
    for name, config in candidates.items():
        print(f"Testing {name}: {config}")
        score = evaluate_meta_config(data, config)
        results[name] = {"config": config, "score": score}
        print(f"  Score: {score:.6f}")

    # Test combinations of top performers
    print("\n=== Testing combinations ===")
    
    # Sort by performance
    sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)
    print("Individual group rankings:")
    for name, result in sorted_results:
        print(f"  {name}: {result['score']:.6f}")

    # Test some specific combinations
    test_combinations = [
        # Current + improvements
        ["META_home", "META_rb_q1", "META_rb_q2", "META_rb_q3", "META_rb_q4", "META_fatigue", "META_close_game"],
        ["META_home", "META_rb_all", "META_fatigue", "META_close_game"],
        ["META_home", "META_rb_all", "META_fatigue"],
        # Minimal effective set
        ["META_home", "META_rb_all"],
        ["META_home", "META_fatigue"],
        ["META_rb_all", "META_fatigue"],
        # Current baseline
        ["META_home", "META_rb_q1", "META_rb_q2", "META_rb_q3", "META_rb_q4", "META_fatigue"],
    ]

    combo_results = []
    for i, combo in enumerate(test_combinations):
        print(f"Testing combination {i+1}: {combo}")
        score = evaluate_meta_config(data, combo)
        combo_results.append({"config": combo, "score": score})
        print(f"  Score: {score:.6f}")

    # Find best combination
    all_results = list(results.values()) + combo_results
    best_result = max(all_results, key=lambda x: x["score"])

    print(f"\n=== RESULTS ===")
    print(f"Best configuration: {best_result['config']}")
    print(f"Best score: {best_result['score']:.6f}")

    # Save results
    output = {
        "seasons": seasons,
        "individual_groups": {name: result for name, result in results.items()},
        "combinations": combo_results,
        "best": best_result
    }

    with open(QUICK_OPT_RESULTS, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {QUICK_OPT_RESULTS}")
    
    # Generate code snippet
    print(f"\nOptimal META_COLS for rapm.py:")
    print("META_COLS = [")
    for col in best_result['config']:
        print(f'    "{col}",')
    print("]")


if __name__ == "__main__":
    main()