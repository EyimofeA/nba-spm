import os
#!/usr/bin/env python3
"""
Fast meta-column and parameter optimization.
Tests individual columns first, then optimizes parameters for the best ones.
"""

try:
    import MySQLdb
except Exception:
    import pymysql
    pymysql.install_as_MySQLdb()
    import MySQLdb

from sklearn import linear_model
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
import json

from paths import FAST_OPT_RESULTS, ensure_dirs

ensure_dirs()


def fetch_small_sample(seasons, limit=50000):
    """Fetch a smaller sample for faster testing"""
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
    ORDER BY RAND()
    LIMIT {limit}
    """
    
    cursor.execute(query, seasons)
    data = cursor.fetchall()
    cursor.close()
    cur.close()
    return data


def simple_ridge_score(X, y, alpha=10000):
    """Simple ridge regression score without cross-validation for speed"""
    try:
        clf = linear_model.Ridge(alpha=alpha)
        clf.fit(X, y)
        y_pred = clf.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return -mse  # negative MSE for consistency
    except:
        return -np.inf


def build_features_fast(data, meta_config, parameters=None):
    """Fast feature building with parameter optimization"""
    
    if parameters is None:
        parameters = {}
    
    # Default parameters
    gt_base = parameters.get('gt_base', {1: 25, 2: 20, 3: 17, 4: 12})
    fatigue_period_weight = parameters.get('fatigue_period_weight', 0.6)
    fatigue_progress_weight = parameters.get('fatigue_progress_weight', 0.4)
    close_game_threshold = parameters.get('close_game_threshold', 5)
    blowout_threshold = parameters.get('blowout_threshold', 15)
    
    # Collect players
    all_players = set()
    for item in data:
        for i in range(2, 12):
            all_players.add(item[i])
    
    all_players = list(all_players)
    
    # Player columns
    player_to_col = {}
    for i, p in enumerate(all_players):
        player_to_col[f"{p}_off"] = i * 2
        player_to_col[f"{p}_def"] = i * 2 + 1
    
    n_player_cols = len(all_players) * 2
    n_meta_cols = len(meta_config)
    n_total_cols = n_player_cols + n_meta_cols
    
    # Meta column mapping
    meta_to_col = {}
    for i, m in enumerate(meta_config):
        meta_to_col[m] = n_player_cols + i
    
    # Precompute max nums for fatigue
    max_num = defaultdict(int)
    for item in data:
        period = item[14]
        num = item[15]
        gameid = item[16]
        key = (gameid, period)
        if isinstance(num, (int, float)):
            max_num[key] = max(max_num[key], int(num))
    
    def gt_threshold(period, progress):
        base = gt_base.get(period, 12)
        return base + int(8 * (1.0 - max(0.0, min(1.0, progress))))
    
    # Build features
    rows = []
    y = []
    game_score = defaultdict(lambda: [0, 0])
    
    for item in data:
        home_poss, pts = item[0], item[1]
        a = [item[i] for i in range(2, 7)]
        h = [item[i] for i in range(7, 12)]
        period = int(item[14])
        num = int(item[15])
        gameid = item[16]
        
        # Score margin
        home_pts, away_pts = game_score[gameid]
        margin_home = home_pts - away_pts
        
        # Progress
        denom = max(1, max_num[(gameid, period)])
        progress = min(1.0, num / float(denom))
        
        # Garbage time filter
        if abs(margin_home) >= gt_threshold(period, progress):
            if home_poss:
                game_score[gameid][0] += pts
            else:
                game_score[gameid][1] += pts
            continue
        
        # Build row
        row = np.zeros(n_total_cols)
        
        # Players
        off_list, def_list = (h, a) if home_poss else (a, h)
        for p in off_list:
            if f"{p}_off" in player_to_col:
                row[player_to_col[f"{p}_off"]] = 1
        for p in def_list:
            if f"{p}_def" in player_to_col:
                row[player_to_col[f"{p}_def"]] = 1
        
        # Meta features
        if "META_home" in meta_config:
            row[meta_to_col["META_home"]] = 1 if home_poss else 0
        
        if "META_rb_all" in meta_config:
            row[meta_to_col["META_rb_all"]] = margin_home
            
        if "META_rb_q1" in meta_config and period == 1:
            row[meta_to_col["META_rb_q1"]] = margin_home
        if "META_rb_q2" in meta_config and period == 2:
            row[meta_to_col["META_rb_q2"]] = margin_home
        if "META_rb_q3" in meta_config and period == 3:
            row[meta_to_col["META_rb_q3"]] = margin_home
        if "META_rb_q4" in meta_config and period >= 4:
            row[meta_to_col["META_rb_q4"]] = margin_home
            
        if "META_fatigue" in meta_config:
            fatigue = ((period - 1) / 3.0) * fatigue_period_weight + progress * fatigue_progress_weight
            row[meta_to_col["META_fatigue"]] = fatigue
            
        if "META_close_game" in meta_config:
            row[meta_to_col["META_close_game"]] = 1 if abs(margin_home) <= close_game_threshold else 0
            
        if "META_blowout" in meta_config:
            row[meta_to_col["META_blowout"]] = 1 if abs(margin_home) > blowout_threshold else 0
            
        if "META_late_game" in meta_config:
            row[meta_to_col["META_late_game"]] = 1 if (period >= 4 and progress > 0.75) else 0
        
        rows.append(row)
        y.append(pts)
        
        # Update score
        if home_poss:
            game_score[gameid][0] += pts
        else:
            game_score[gameid][1] += pts
    
    return np.array(rows), np.array(y)


def test_individual_meta_columns(data):
    """Test each meta column individually"""
    
    candidates = [
        "META_home",
        "META_rb_all", 
        "META_rb_q1",
        "META_rb_q2", 
        "META_rb_q3",
        "META_rb_q4",
        "META_fatigue",
        "META_close_game",
        "META_blowout",
        "META_late_game"
    ]
    
    print("Testing individual meta columns...")
    results = {}
    
    # Baseline (no meta columns)
    X_base, y = build_features_fast(data, [])
    baseline_score = simple_ridge_score(X_base, y)
    print(f"Baseline (no meta): {baseline_score:.6f}")
    
    for col in candidates:
        X, y = build_features_fast(data, [col])
        score = simple_ridge_score(X, y)
        improvement = score - baseline_score
        results[col] = {"score": score, "improvement": improvement}
        print(f"{col}: {score:.6f} (improvement: {improvement:.6f})")
    
    return results, baseline_score


def optimize_parameters(data, meta_config):
    """Optimize parameters for given meta configuration"""
    
    print(f"Optimizing parameters for: {meta_config}")
    
    # Parameter ranges to test
    param_tests = [
        # Garbage time thresholds
        {'gt_base': {1: 20, 2: 15, 3: 12, 4: 8}},  # Stricter
        {'gt_base': {1: 30, 2: 25, 3: 20, 4: 15}}, # More lenient
        
        # Fatigue weights
        {'fatigue_period_weight': 0.7, 'fatigue_progress_weight': 0.3},
        {'fatigue_period_weight': 0.5, 'fatigue_progress_weight': 0.5},
        {'fatigue_period_weight': 0.8, 'fatigue_progress_weight': 0.2},
        
        # Game situation thresholds
        {'close_game_threshold': 3, 'blowout_threshold': 20},
        {'close_game_threshold': 7, 'blowout_threshold': 12},
        {'close_game_threshold': 10, 'blowout_threshold': 18},
    ]
    
    # Baseline
    X_base, y = build_features_fast(data, meta_config)
    baseline_score = simple_ridge_score(X_base, y)
    print(f"Baseline parameters: {baseline_score:.6f}")
    
    best_params = {}
    best_score = baseline_score
    
    for params in param_tests:
        X, y = build_features_fast(data, meta_config, params)
        score = simple_ridge_score(X, y)
        improvement = score - baseline_score
        
        print(f"  {params}: {score:.6f} (improvement: {improvement:.6f})")
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score


def main():
    print("Fast meta column optimization...")
    
    seasons = [2020, 2021]  # Just 2 seasons for speed
    print(f"Loading sample data for seasons {seasons}...")
    
    data = fetch_small_sample(seasons, limit=30000)  # Smaller sample
    print(f"Loaded {len(data)} possessions")
    
    # Test individual columns
    individual_results, baseline = test_individual_meta_columns(data)
    
    # Sort by improvement
    sorted_individual = sorted(individual_results.items(), 
                             key=lambda x: x[1]["improvement"], reverse=True)
    
    print(f"\n=== Top individual performers ===")
    top_5 = []
    for i, (col, result) in enumerate(sorted_individual[:5]):
        print(f"{i+1}. {col}: +{result['improvement']:.6f}")
        top_5.append(col)
    
    # Test combinations of top performers
    print(f"\n=== Testing combinations ===")
    
    test_combos = [
        top_5[:2],  # Top 2
        top_5[:3],  # Top 3
        ["META_home", "META_rb_all", "META_fatigue"],  # Logical combo
        ["META_home", "META_rb_q1", "META_rb_q2", "META_rb_q3", "META_rb_q4", "META_fatigue"],  # Current
    ]
    
    combo_results = []
    for combo in test_combos:
        X, y = build_features_fast(data, combo)
        score = simple_ridge_score(X, y)
        improvement = score - baseline
        combo_results.append({
            "config": combo,
            "score": score, 
            "improvement": improvement
        })
        print(f"{combo}: {score:.6f} (+{improvement:.6f})")
    
    # Find best combo
    best_combo = max(combo_results, key=lambda x: x["improvement"])
    print(f"\nBest combination: {best_combo['config']}")
    print(f"Best improvement: +{best_combo['improvement']:.6f}")
    
    # Optimize parameters for best combo
    print(f"\n=== Parameter optimization ===")
    best_params, optimized_score = optimize_parameters(data, best_combo["config"])
    
    final_improvement = optimized_score - baseline
    param_improvement = optimized_score - best_combo["score"]
    
    print(f"\nFinal optimized score: {optimized_score:.6f}")
    print(f"Total improvement: +{final_improvement:.6f}")
    print(f"Parameter improvement: +{param_improvement:.6f}")
    
    # Save results
    results = {
        "seasons": seasons,
        "sample_size": len(data),
        "baseline_score": baseline,
        "individual_results": individual_results,
        "combo_results": combo_results,
        "best_combination": best_combo,
        "best_parameters": best_params,
        "final_score": optimized_score,
        "total_improvement": final_improvement
    }
    
    with open(FAST_OPT_RESULTS, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n=== RECOMMENDATIONS ===")
    print(f"Optimal META_COLS: {best_combo['config']}")
    if best_params:
        print(f"Optimal parameters: {best_params}")
    
    # Generate code
    print(f"\nCode for rapm.py:")
    print("META_COLS = [")
    for col in best_combo['config']:
        print(f'    "{col}",')
    print("]")
    
    if best_params:
        print("\n# Optimal parameters:")
        for key, value in best_params.items():
            print(f"# {key} = {value}")


if __name__ == "__main__":
    main()