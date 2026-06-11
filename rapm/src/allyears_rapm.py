from sklearn import linear_model
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
import MySQLdb
import csv
import pandas as pd

from paths import ALL_NAMES_CSV, DUMP, RAPM_RESULTS, ensure_dirs

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
    current_season = max(seasons)
    weights = {}
    for season in seasons:
        years_ago = current_season - season
        weights[season] = base_weight * (decay_rate ** years_ago)
    return weights

def run_ridge_model(X, y, sample_weights):
    alphas=[1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000]
    clf = linear_model.RidgeCV(alphas, cv=4)
    clf.fit(X, y, sample_weight=sample_weights)
    print('ALPHA:', clf.alpha_)
    return clf.coef_
def process_all_data():
    cur = MySQLdb.connect(host="localhost", user='root', db="nba_api", unix_socket='/tmp/mysql.sock')
    cursor = cur.cursor()

    query = """SELECT home_poss, pts, a1, a2, a3, a4, a5, h1, h2, h3, h4, h5, season, date
                FROM matchups"""
    cursor.execute(query)
    data = cursor.fetchall()

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

    # Assuming no decay in weights for the full dataset
    for counter, item in enumerate(data):
        home_poss = item[0]
        pts = item[1]
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
        sample_weights.append(1)  # Uniform weight

    y_av = np.average(y)

    beta_ridge = run_ridge_model(X.tocsr(), y - y_av, sample_weights)
    results = []
    for i in range(len(beta_ridge)):
        results.append([col_to_player[i], beta_ridge[i], 0])

    output_filename = DUMP / 'results_full_dataset.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Player', 'Ridge Coefficient', 'Bayesian Coefficient'])
        csvwriter.writerows(results)

    print(f"Results saved to {output_filename}")
def write_human_readable_results_custom(dataset_label):
    results = pd.read_csv(DUMP / "results_full_dataset.csv")
    all_names = pd.read_csv(ALL_NAMES_CSV)

    id_to_name = dict(zip(all_names['PLAYER_ID'].astype(str), all_names['PLAYER_NAME']))

    results['player_id'] = results['Player'].apply(lambda x: x.split('_')[0])
    results['role'] = results['Player'].apply(lambda x: x.split('_')[1])

    results['Off'] = results.apply(lambda row: row['Ridge Coefficient'] * 1 if row['role'] == 'off' else None, axis=1)
    results['Def'] = results.apply(lambda row: row['Ridge Coefficient'] * 1 if row['role'] == 'def' else None, axis=1)

    results['Name'] = results['player_id'].map(id_to_name)

    new_df = results[['Name', 'Off', 'Def']].copy()
    new_df.drop(['player_id', 'role'], axis=1, inplace=True, errors='ignore')
    merged_df = new_df.groupby('Name').agg({'Off': 'sum', 'Def': 'sum'}).reset_index()
    merged_df["Rapm"] = merged_df["Off"] - merged_df["Def"]
    merged_df.sort_values(by="Rapm", ascending=False, inplace=True)
    merged_df[["Off", "Def", "Rapm"]] = merged_df[["Off", "Def", "Rapm"]].mul(100)
    # merged_df["Dataset"] = dataset_label
    output_filename = RAPM_RESULTS / f"Full_Rapm_{dataset_label}.csv"
    merged_df.to_csv(output_filename, index=False)
    print(merged_df.head(50))
    print(f"Human readable results saved to {output_filename}")

def main():
    print("Processing the full dataset...")
    process_all_data()
    write_human_readable_results_custom(['all_data'])
    print("Done processing the full dataset!")

if __name__ == '__main__':
    main()