import os
#!/usr/bin/env python3
"""
Automated meta-column optimization for RAPM.

This script systematically tests different combinations of meta columns
to find the optimal set that maximizes model performance.
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
import pandas as pd

from paths import META_OPT_RESULTS, ensure_dirs

ensure_dirs()
from collections import defaultdict
import itertools
from typing import List, Dict, Tuple, Set
import json


class MetaColumnOptimizer:
    def __init__(self, seasons: List[int]):
        self.seasons = seasons
        self.data = None
        self.player_to_col = {}
        self.col_to_key = {}
        self.all_players = {}
        
        # Base garbage time thresholds
        self.GT_BASE = {1: 25, 2: 20, 3: 17, 4: 12}
        
    def fetch_data(self):
        """Fetch possession data from database"""
        cur = MySQLdb.connect(
            host="localhost", 
            user="root", 
            password=os.environ.get("NBA_DB_PASSWORD", ""), 
            db="nba_api", 
            unix_socket='/tmp/mysql.sock'
        )
        cursor = cur.cursor()
        placeholders = ','.join(['%s'] * len(self.seasons))
        
        query = f"""
        SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5,
               season, date, period, num, gameid
        FROM matchups
        WHERE season IN ({placeholders})
        ORDER BY gameid, period, num
        """
        
        cursor.execute(query, self.seasons)
        self.data = cursor.fetchall()
        cursor.close()
        cur.close()
        
        # Collect all players
        self.all_players = {}
        for item in self.data:
            for i in range(2, 12):  # a1..a5, h1..h5
                self.all_players[item[i]] = 1
                
    def setup_player_columns(self):
        """Setup player column mappings"""
        self.player_to_col = {}
        self.col_to_key = {}
        
        for p in self.all_players:
            for side in ['off', 'def']:
                key = f"{p}_{side}"
                if key not in self.player_to_col:
                    idx = len(self.player_to_col)
                    self.player_to_col[key] = idx
                    self.col_to_key[idx] = key
    
    def gt_threshold(self, period: int, progress: float) -> int:
        """Dynamic garbage-time margin threshold"""
        base = self.GT_BASE.get(period, 12)
        dynamic = base + int(8 * (1.0 - max(0.0, min(1.0, progress))))
        return dynamic
    
    def calculate_time_decay_weights(self, base_weight=1.0, decay_rate=1.0):
        """Calculate time decay weights for seasons"""
        current_season = max(self.seasons)
        return {season: base_weight * (decay_rate ** (current_season - season)) 
                for season in self.seasons}
    
    def get_meta_column_candidates(self) -> Dict[str, callable]:
        """Return all possible meta column calculators"""
        
        def home_advantage(home_poss, **kwargs):
            return 1 if home_poss else 0
            
        def away_advantage(home_poss, **kwargs):
            return 1 if not home_poss else 0
            
        def back_to_back_home(b2b_home, **kwargs):
            return 1 if b2b_home else 0
            
        def back_to_back_away(b2b_away, **kwargs):
            return 1 if b2b_away else 0
            
        def rest_advantage_home(rest_diff, **kwargs):
            # Positive when home has more rest
            return max(0, rest_diff)
            
        def rest_advantage_away(rest_diff, **kwargs):
            # Positive when away has more rest
            return max(0, -rest_diff)
            
        def rest_difference(rest_diff, **kwargs):
            # Raw rest difference (home - away)
            return rest_diff
            
        def rest_difference_abs(rest_diff, **kwargs):
            # Absolute rest difference
            return abs(rest_diff)
        
        def quarter_1(period, **kwargs):
            return 1 if period == 1 else 0
            
        def quarter_2(period, **kwargs):
            return 1 if period == 2 else 0
            
        def quarter_3(period, **kwargs):
            return 1 if period == 3 else 0
            
        def quarter_4(period, **kwargs):
            return 1 if period >= 4 else 0
            
        def overtime(period, **kwargs):
            return 1 if period > 4 else 0
        
        def rubberband_q1(period, margin_home, **kwargs):
            return margin_home if period == 1 else 0
            
        def rubberband_q2(period, margin_home, **kwargs):
            return margin_home if period == 2 else 0
            
        def rubberband_q3(period, margin_home, **kwargs):
            return margin_home if period == 3 else 0
            
        def rubberband_q4(period, margin_home, **kwargs):
            return margin_home if period >= 4 else 0
            
        def rubberband_all(margin_home, **kwargs):
            return margin_home
            
        def rubberband_squared(margin_home, **kwargs):
            return margin_home ** 2
            
        def rubberband_abs(margin_home, **kwargs):
            return abs(margin_home)
        
        def fatigue_linear(period, progress, **kwargs):
            return ((period - 1) / 3.0) * 0.6 + progress * 0.4
            
        def fatigue_quadratic(period, progress, **kwargs):
            base = ((period - 1) / 3.0) * 0.6 + progress * 0.4
            return base ** 2
            
        def fatigue_exponential(period, progress, **kwargs):
            return 1 - np.exp(-2 * (((period - 1) / 3.0) * 0.6 + progress * 0.4))
        
        def possession_number(num, **kwargs):
            return num / 100.0  # normalize
            
        def possession_squared(num, **kwargs):
            return (num / 100.0) ** 2
            
        def late_game(period, progress, **kwargs):
            return 1 if (period >= 4 and progress > 0.75) else 0
            
        def close_game(margin_home, **kwargs):
            return 1 if abs(margin_home) <= 5 else 0
            
        def blowout_game(margin_home, **kwargs):
            return 1 if abs(margin_home) > 15 else 0
            
        def margin_times_quarter(period, margin_home, **kwargs):
            return period * margin_home
            
        def progress_interaction(period, progress, **kwargs):
            return period * progress
            
        def rest_advantage(period, **kwargs):
            # Proxy for rest - higher in later quarters assuming fatigue
            return (5 - period) if period <= 4 else 1
        
        return {
            "META_home": home_advantage,
            "META_away": away_advantage,
            "META_b2b_home": back_to_back_home,
            "META_b2b_away": back_to_back_away,
            "META_rest_home": rest_advantage_home,
            "META_rest_away": rest_advantage_away,
            "META_rest_diff": rest_difference,
            "META_rest_abs": rest_difference_abs,
            "META_q1": quarter_1,
            "META_q2": quarter_2, 
            "META_q3": quarter_3,
            "META_q4": quarter_4,
            "META_overtime": overtime,
            "META_rb_q1": rubberband_q1,
            "META_rb_q2": rubberband_q2,
            "META_rb_q3": rubberband_q3,
            "META_rb_q4": rubberband_q4,
            "META_rb_all": rubberband_all,
            "META_rb_squared": rubberband_squared,
            "META_rb_abs": rubberband_abs,
            "META_fatigue": fatigue_linear,
            "META_fatigue_quad": fatigue_quadratic,
            "META_fatigue_exp": fatigue_exponential,
            "META_poss_num": possession_number,
            "META_poss_squared": possession_squared,
            "META_late_game": late_game,
            "META_close_game": close_game,
            "META_blowout": blowout_game,
            "META_margin_quarter": margin_times_quarter,
            "META_progress_int": progress_interaction,
            "META_rest": rest_advantage,
        }
    
    def build_X_y(self, meta_columns: List[str]) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """Build feature matrix and target vector with specified meta columns"""
        
        # Get meta column calculators
        meta_calculators = self.get_meta_column_candidates()
        selected_calculators = {col: meta_calculators[col] for col in meta_columns}
        
        # Setup columns
        self.setup_player_columns()
        
        # Add meta columns at the end
        meta_to_col = {}
        for m in meta_columns:
            idx = len(self.col_to_key)
            meta_to_col[m] = idx
            self.col_to_key[idx] = m
        
        # Precompute max possession index per (game, period) 
        max_num = defaultdict(int)
        for item in self.data:
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
        season_weights = self.calculate_time_decay_weights(decay_rate=1.0)
        
        # Running score by game
        game_score = defaultdict(lambda: [0, 0])  # (home_pts, away_pts)
        
        for item in self.data:
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
            if abs(margin_home) >= self.gt_threshold(period, progress):
                # Update running score and skip
                if home_poss:
                    game_score[gameid][0] += pts
                else:
                    game_score[gameid][1] += pts
                continue
            
            # Build sparse row
            row = lil_matrix((1, len(self.col_to_key)))
            
            # Player indicators
            off_list, def_list = (h, a) if home_poss else (a, h)
            for p in off_list:
                row[0, self.player_to_col[f"{p}_off"]] = 1
            for p in def_list:
                row[0, self.player_to_col[f"{p}_def"]] = 1
            
            # Meta columns
            meta_context = {
                'home_poss': home_poss,
                'period': period,
                'progress': progress,
                'margin_home': margin_home,
                'num': num,
                'season': season
            }
            
            for meta_col in meta_columns:
                if meta_col in selected_calculators:
                    value = selected_calculators[meta_col](**meta_context)
                    row[0, meta_to_col[meta_col]] = value
            
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
    
    def evaluate_meta_columns(self, meta_columns: List[str]) -> float:
        """Evaluate a combination of meta columns using cross-validation"""
        try:
            X, y, weights = self.build_X_y(meta_columns)
            
            # Subtract mean for centering
            y_centered = y - np.average(y)
            
            # Use Ridge with fixed alpha for speed
            clf = linear_model.Ridge(alpha=10000)
            
            # Cross-validation score (negative MSE, higher is better)
            scores = cross_val_score(clf, X, y_centered, cv=3, 
                                   scoring='neg_mean_squared_error',
                                   fit_params={'sample_weight': weights})
            
            return scores.mean()
            
        except Exception as e:
            print(f"Error evaluating {meta_columns}: {e}")
            return -np.inf
    
    def search_optimal_meta_columns(self, max_combinations: int = 100) -> Dict:
        """Search for optimal meta column combinations"""
        
        print("Fetching data...")
        self.fetch_data()
        print(f"Loaded {len(self.data)} possessions")
        
        # Get all possible meta columns
        all_meta_cols = list(self.get_meta_column_candidates().keys())
        print(f"Testing {len(all_meta_cols)} possible meta columns:")
        for col in all_meta_cols:
            print(f"  - {col}")
        
        results = []
        
        # Test individual columns first
        print("\n=== Testing individual columns ===")
        individual_scores = {}
        for col in all_meta_cols:
            score = self.evaluate_meta_columns([col])
            individual_scores[col] = score
            results.append({
                'columns': [col],
                'score': score,
                'size': 1
            })
            print(f"{col}: {score:.6f}")
        
        # Sort by individual performance
        sorted_cols = sorted(individual_scores.items(), key=lambda x: x[1], reverse=True)
        top_cols = [col for col, score in sorted_cols[:10]]  # Top 10 individual performers
        
        print(f"\n=== Top 10 individual performers ===")
        for i, (col, score) in enumerate(sorted_cols[:10]):
            print(f"{i+1}. {col}: {score:.6f}")
        
        # Test combinations of top performers
        print(f"\n=== Testing combinations of top performers ===")
        
        tested_combinations = set()
        combination_count = 0
        
        # Test pairs
        for size in [2, 3, 4, 5]:
            if combination_count >= max_combinations:
                break
                
            print(f"\nTesting size {size} combinations...")
            
            for combo in itertools.combinations(top_cols, size):
                if combination_count >= max_combinations:
                    break
                    
                combo_tuple = tuple(sorted(combo))
                if combo_tuple in tested_combinations:
                    continue
                    
                tested_combinations.add(combo_tuple)
                score = self.evaluate_meta_columns(list(combo))
                results.append({
                    'columns': list(combo),
                    'score': score,
                    'size': size
                })
                
                combination_count += 1
                print(f"  {combo}: {score:.6f}")
        
        # Sort all results
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n=== TOP 10 COMBINATIONS ===")
        for i, result in enumerate(results[:10]):
            print(f"{i+1}. Score: {result['score']:.6f}, "
                  f"Size: {result['size']}, "
                  f"Columns: {result['columns']}")
        
        return {
            'best_combination': results[0] if results else None,
            'all_results': results[:20],  # Top 20
            'individual_scores': individual_scores
        }


def main():
    # Test on a smaller window first
    seasons = [2019, 2020, 2021]  # 3-year window for faster testing
    
    print(f"Optimizing meta columns for seasons {seasons}")
    
    optimizer = MetaColumnOptimizer(seasons)
    results = optimizer.search_optimal_meta_columns(max_combinations=50)
    
    # Save results
    with open(META_OPT_RESULTS, 'w') as f:
        json.dump({
            'seasons': seasons,
            'best_combination': results['best_combination'],
            'top_results': results['all_results'][:10],
            'individual_scores': results['individual_scores']
        }, f, indent=2)
    
    print(f"\nResults saved to {META_OPT_RESULTS}")
    
    if results['best_combination']:
        best = results['best_combination']
        print(f"\nOPTIMAL META COLUMNS:")
        print(f"Score: {best['score']:.6f}")
        print(f"Columns: {best['columns']}")
        
        # Generate code snippet
        print(f"\nCode to update META_COLS in rapm.py:")
        print("META_COLS = [")
        for col in best['columns']:
            print(f'    "{col}",')
        print("]")


if __name__ == "__main__":
    main()