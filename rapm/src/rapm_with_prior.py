import os
from sklearn import linear_model
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
try:
    import MySQLdb
except ImportError:
    try:
        import pymysql
        pymysql.install_as_MySQLdb()
        import MySQLdb
    except ImportError:
        print("Error: MySQLdb module not found. Please install it using 'pip install mysqlclient' or 'pip install pymysql'.")
        raise
import csv
import pandas as pd

from paths import ALL_NAMES_CSV, PRIOR_CSV, OUTPUTS, ensure_dirs

ensure_dirs()


def format_season_list(seasons):
    start_year = str(seasons[0])
    end_years = '-'.join([str(year)[-2:] for year in seasons[1:]])
    return f"{start_year}-{end_years}"

def run_bayes_model(X, y):
    import pymc as pm
    basic_model = pm.Model()
    with basic_model:
        alpha = pm.Normal("alpha", mu=1.1, sigma=0.1)
        beta = pm.Normal("beta", 0, 0.02, shape=(np.shape(X)[1],))
        mu = alpha + pm.math.dot(X, beta)
        Y_obs = pm.Normal("Y_obs", mu=mu, observed=y)
        idata = pm.find_MAP()
    return idata['beta']

def calculate_time_decay_weights(seasons, base_weight=1.0, decay_rate=0.5):
    """
    Returns a dict: {season: weight}, applying exponential decay 
    from the max season in 'seasons' downward.
    """
    current_season = max(seasons)
    weights = {}
    for season in seasons:
        years_ago = current_season - season
        weights[season] = base_weight * (decay_rate ** years_ago)
    return weights

def build_prior_dict(season, filename=PRIOR_CSV):
    """
    Reads a CSV with priors for all seasons and filters for the specified season.
    Fills missing OffPrior and DefPrior with zero.
    
    CSV Format:
        Season, PLAYER_ID, OffPrior, DefPrior
    """
    df = pd.read_csv(filename)
    df_season = df[df['Season'] == season].fillna({'SPM_O': 0.0, 'SPM_D': 0.0})
    
    if df_season.empty:
        print(f"No prior data found for season {season}. Using zero priors.")
    
    prior_dict = {}
    for _, row in df_season.iterrows():
        pid_str = str(int(row["PLAYER_ID"]))
        prior_dict[pid_str + "_off"] = row["SPM_O"]
        prior_dict[pid_str + "_def"] = -row["SPM_D"]
    
    return prior_dict

def build_prior_vector(col_to_player, prior_dict):
    """
    Creates a NumPy vector matching the columns in X. 
    If col_to_player[i] = '201939_off' and prior_dict['201939_off'] = 0.25, 
    then prior_vec[i] = 0.25. Otherwise 0.0 if missing from prior_dict.
    """
    prior_vec = np.zeros(len(col_to_player))
    for col_idx, player_side in col_to_player.items():
        if player_side in prior_dict:
            prior_vec[col_idx] = prior_dict[player_side]
        else:
            prior_vec[col_idx] = 0.0
    # Replace any NaN with zero
    prior_vec = np.nan_to_num(prior_vec, nan=0.0)
    return prior_vec

def run_ridge_model(X, y, sample_weights):
    # Modify alpha list as you see fit
    alphas = [4000]  
    clf = linear_model.RidgeCV(alphas, cv=4)
    clf.fit(X, y, sample_weight=sample_weights)
    print('ALPHA:', clf.alpha_)
    return clf.coef_

def run_ridge_model_with_prior(X, y, prior_coefs, sample_weights):
    """
    1) offset = X * prior_coefs
    2) y_adj = y - offset
    3) run normal ridge on y_adj
    4) final_coefs = raw_coefs + prior_coefs
    Returns both raw_coefs and final_coefs.
    """
    offset = X.dot(prior_coefs)
    y_adj = y - offset
    
    # Debugging: Check for NaNs in y_adj
    if np.isnan(y_adj).any():
        print("y_adj contains NaN values. Handling NaNs...")
        # Option 1: Remove samples with NaN
        # valid_indices = ~np.isnan(y_adj)
        # X = X[valid_indices]
        # y_adj = y_adj[valid_indices]
        # sample_weights = [w for i, w in enumerate(sample_weights) if valid_indices[i]]
        
        # Option 2: Replace NaNs with zero
        y_adj = np.nan_to_num(y_adj, nan=0.0)
    
    raw_coefs = run_ridge_model(X, y_adj, sample_weights)
    final_coefs = raw_coefs + prior_coefs
    return run_ridge_model(X, y, sample_weights), final_coefs  # <-- Modified to return both

def process_single_season(season, prior_filename=PRIOR_CSV):
    """
    Processes RAPM with prior for a single season.
    
    Returns a DataFrame with columns:
        ["Player", "Prior_RAPM", "Raw_RAPM", "Final_RAPM", "Season"]
    """
    # Database connection
    try:
        cur = MySQLdb.connect(
            host="localhost", 
            user='root',
            password=os.environ.get("NBA_DB_PASSWORD", ""), 
            db="nba_api", 
            unix_socket='/tmp/mysql.sock'
        )
    except MySQLdb.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return pd.DataFrame()
    
    cursor = cur.cursor()
    
    # Query for the specific season
    seasons_list = [season]
    placeholders = ','.join(['%s'] * len(seasons_list))
    query = f"""
        SELECT 
            home_poss, pts, a1, a2, a3, a4, a5, 
            h1, h2, h3, h4, h5, season, date
        FROM matchups
        WHERE season IN ({placeholders})
          AND date NOT BETWEEN CONCAT(season, '-04-12') AND CONCAT(season, '-06-30')
          AND pts IS NOT NULL
    """
    cursor.execute(query, seasons_list)
    data = cursor.fetchall()
    
    # Close the database connection
    cursor.close()
    cur.close()
    
    if not data:
        print(f"No data found for season {season}.")
        return pd.DataFrame()
    
    # Gather all players
    all_players = {}
    for item in data:
        for i in range(2, 12):  # a1..a5, h1..h5
            all_players[item[i]] = 1
    
    all_players = {}
    for item in data:
        for i in range(2, 12):
            all_players[item[i]] = 1
    player_to_col = {}
    col_to_player = {}
    for p in all_players:
        for side in ['off', 'def']:
            p_side = str(p) + '_' + side
            if p_side not in player_to_col:
                number = len(player_to_col)
                player_to_col[p_side] = number
                col_to_player[number] = p_side

    X = lil_matrix((len(data), len(col_to_player)))
    y = np.zeros(len(data))
    sample_weights = []

    season_weights = calculate_time_decay_weights(seasons_list, decay_rate=1)
    counter = 0
    for item in data:
        home_poss = item[0]
        pts = item[1]
        season = item[12]
        home_list = []
        away_list = []
        for i in range(2, 7):
            away_list.append(item[i])
        for i in range(7, 12):
            home_list.append(item[i])
        if home_poss:
            [off_list, def_list] = home_list, away_list
        else:
            [off_list, def_list] = away_list, home_list
        for p in off_list:
            off_p = str(p) + '_off'
            X[counter, player_to_col[off_p]] = 1
        for p in def_list:
            def_p = str(p) + '_def'
            X[counter, player_to_col[def_p]] = 1
        y[counter] = pts
        sample_weights.append(season_weights[season])
        counter += 1
    y_av = np.average(y)
    y_centered = y - y_av
    
    # Build & apply the prior for the current season
    prior_dict = build_prior_dict(season, prior_filename)
    prior_vector = build_prior_vector(col_to_player, prior_dict)
    
    # Fit ridge with prior
    raw_coefs, final_coefs = run_ridge_model_with_prior(
        X.tocsr(),
        y_centered,
        prior_vector,
        sample_weights
    )
    
    # Build a final DataFrame
    # player_side might look like "201939_off" or "201939_def"
    results_list = []
    for i in range(len(final_coefs)):
        player_side = col_to_player[i]
        prior_val = prior_vector[i]
        raw_val = raw_coefs[i]
        final_val = final_coefs[i]
        results_list.append((player_side, prior_val, raw_val, final_val, season))
    
    df_out = pd.DataFrame(results_list, columns=["Player", "Prior_RAPM", "Raw_RAPM", "Final_RAPM", "Season"])
    return df_out

def process_multiple_seasons(seasons_list, prior_filename=PRIOR_CSV):
    """
    Processes RAPM with prior for multiple seasons.
    
    Returns a single merged DataFrame containing all seasons' results.
    """
    all_dfs = []
    for s in seasons_list:
        print(f"Processing season {s} ...")
        df_s = process_single_season(s, prior_filename=prior_filename)
        if not df_s.empty:
            all_dfs.append(df_s)
        else:
            print(f"Skipping season {s} due to lack of data.")
    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
    else:
        merged = pd.DataFrame()
    return merged

def write_human_readable_results_custom(seasons, results_df):
    """
    Converts the raw RAPM_with_Prior results into a human-readable format.
    
    Parameters:
        seasons (list): List of seasons processed.
        results_df (DataFrame): DataFrame containing RAPM_with_Prior results.
    """
    if results_df.empty:
        print("No results to process.")
        return
    
    # Read player names
    player_names = pd.read_csv(ALL_NAMES_CSV)
    
    # Extract player_id and role
    results_df['player_id'] = results_df['Player'].apply(lambda x: x.split('_')[0])
    results_df['role'] = results_df['Player'].apply(lambda x: x.split('_')[1])
    
    # Map player_id to player names
    id_to_name = dict(zip(player_names['PLAYER_ID'].astype(str), player_names['PLAYER_NAME']))
    results_df['Name'] = results_df['player_id'].map(id_to_name)
    
    # Handle missing names
    results_df['Name'] = results_df['Name'].fillna(results_df['player_id'])
    
    # Separate Off and Def coefficients
    results_df['Off_Prior_RAPM'] = results_df.apply(lambda row: row['Prior_RAPM'] if row['role'] == 'off' else 0.0, axis=1)*100
    results_df['Def_Prior_RAPM'] = results_df.apply(lambda row: row['Prior_RAPM'] if row['role'] == 'def' else 0.0, axis=1)*100
    
    results_df['Off_Raw_RAPM'] = results_df.apply(lambda row: row['Raw_RAPM'] if row['role'] == 'off' else 0.0, axis=1)*100
    results_df['Def_Raw_RAPM'] = results_df.apply(lambda row: row['Raw_RAPM'] if row['role'] == 'def' else 0.0, axis=1)*100
    
    results_df['Off_Final_RAPM'] = results_df.apply(lambda row: row['Final_RAPM'] if row['role'] == 'off' else 0.0, axis=1)*100
    results_df['Def_Final_RAPM'] = results_df.apply(lambda row: row['Final_RAPM'] if row['role'] == 'def' else 0.0, axis=1)*100
    
    # Aggregate per player per season
    aggregated_df = results_df.groupby(['Name', 'Season']).agg({
        'Off_Prior_RAPM': 'sum',
        'Def_Prior_RAPM': 'sum',
        'Off_Raw_RAPM': 'sum',
        'Def_Raw_RAPM': 'sum',
        'Off_Final_RAPM': 'sum',
        'Def_Final_RAPM': 'sum'
    }).reset_index()
    
    # Calculate RAPM
    aggregated_df["Rapm"] = aggregated_df["Off_Final_RAPM"] - aggregated_df["Def_Final_RAPM"]
    
    # Sort by RAPM
    aggregated_df.sort_values(by="Rapm", ascending=False, inplace=True)
    
    # Create a Season string
    aggregated_df["Season"] = aggregated_df["Season"].astype(str)
    
    # Reorder columns for clarity
    aggregated_df = aggregated_df[[
        "Name", "Season",
        "Off_Prior_RAPM", "Def_Prior_RAPM",
        "Off_Raw_RAPM", "Def_Raw_RAPM",
        "Off_Final_RAPM", "Def_Final_RAPM",
        "Rapm"
    ]]
    
    # Rename columns for better readability
    aggregated_df.rename(columns={
        "Off_Prior_RAPM": "Offensive Prior RAPM",
        "Def_Prior_RAPM": "Defensive Prior RAPM",
        "Off_Raw_RAPM": "Offensive Raw RAPM",
        "Def_Raw_RAPM": "Defensive Raw RAPM",
        "Off_Final_RAPM": "Offensive Final RAPM",
        "Def_Final_RAPM": "Defensive Final RAPM",
        "Rapm": "RAPM"
    }, inplace=True)

    # Round numeric columns to 2 decimals
    numeric_cols = aggregated_df.select_dtypes(include=[np.number]).columns
    aggregated_df[numeric_cols] = aggregated_df[numeric_cols].round(2)
    
    # Save the aggregated results
    output_filename = OUTPUTS / "RAPM_with_prior_all_seasons.csv"
    aggregated_df.to_csv(output_filename, index=False)
    print(f"Human readable results with prior saved to '{output_filename}'")
    
    # Print Top 15 for the last season in the list
    if seasons:
        last_season = str(max(seasons))
        print(f"\n---------------------------------------------------")
        print(f"Top 15 Players for {last_season} Season")
        print(f"---------------------------------------------------")
        last_season_df = aggregated_df[aggregated_df["Season"] == last_season]
        # Sort by RAPM just in case
        last_season_df = last_season_df.sort_values(by="RAPM", ascending=False)
        print(last_season_df[["Name", "RAPM", "Offensive Final RAPM", "Defensive Final RAPM"]].head(15).to_string(index=False))
        print(f"---------------------------------------------------\n")

def process_and_save_all_seasons(seasons_to_run, prior_filename=PRIOR_CSV):
    """
    Processes multiple seasons and saves the human-readable RAPM with prior results.
    
    Parameters:
        seasons_to_run (list): List of seasons to process.
        prior_filename (str): Filename of the priors CSV.
    """
    final_df = process_multiple_seasons(seasons_to_run, prior_filename=prior_filename)
    write_human_readable_results_custom(seasons_to_run, final_df)
    print("Merged RAPM with Prior for all seasons completed.")

def main():
    # Define the list of seasons you want to process
    seasons_to_run = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]  # example list
    # Process all seasons and save the results
    process_and_save_all_seasons(seasons_to_run, prior_filename=PRIOR_CSV)

if __name__ == '__main__':
    main()
 