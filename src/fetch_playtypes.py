import pandas as pd
import numpy as np

from paths import PLAYTYPE_POE_FEATURES, ensure_dirs

ensure_dirs()

PLAYTYPE_URL = "https://raw.githubusercontent.com/gabriel1200/site_Data/master/playtype.csv"

def fetch_and_prepare_playtypes():
    print(f"Downloading playtype data from {PLAYTYPE_URL}...")
    try:
        df = pd.read_csv(PLAYTYPE_URL)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # Keep only relevant columns
    cols_to_keep = ['PLAYER_ID', 'year', 'playtype', 'Poss', 'Points', 'PPP']
    # Some older files might have different column names, let's make sure we have them
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols].copy()
    
    # Drop rows with missing crucial identifiers
    df = df.dropna(subset=['PLAYER_ID', 'year', 'playtype', 'Poss', 'Points'])
    
    df['Poss'] = pd.to_numeric(df['Poss'], errors='coerce')
    df['Points'] = pd.to_numeric(df['Points'], errors='coerce')
    df['PLAYER_ID'] = pd.to_numeric(df['PLAYER_ID'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    df = df.dropna(subset=['PLAYER_ID', 'year', 'Poss', 'Points'])
    
    df['PLAYER_ID'] = df['PLAYER_ID'].astype(int)
    df['year'] = df['year'].astype(int)
    
    print(f"Processing {len(df)} playtype records...")

    # Calculate League Average PPP by year and playtype
    league_averages = df.groupby(['year', 'playtype']).agg(
        Total_Points=('Points', 'sum'),
        Total_Poss=('Poss', 'sum')
    ).reset_index()
    
    # League PPP
    league_averages['League_PPP'] = np.where(
        league_averages['Total_Poss'] > 0, 
        league_averages['Total_Points'] / league_averages['Total_Poss'], 
        0
    )
    
    # Merge league PPP back to main dataframe
    df = pd.merge(df, league_averages[['year', 'playtype', 'League_PPP']], on=['year', 'playtype'], how='left')
    
    # Calculate Playtype Points Over Expectation (POE) for each row
    df['Player_PPP'] = np.where(df['Poss'] > 0, df['Points'] / df['Poss'], 0)
    df['Playtype_POE'] = (df['Player_PPP'] - df['League_PPP']) * df['Poss']
    
    # Aggregate back to Player-Year level
    player_agg = df.groupby(['PLAYER_ID', 'year']).agg(
        Total_Playtype_POE=('Playtype_POE', 'sum'),
        Total_Synergy_Poss=('Poss', 'sum')
    ).reset_index()
    
    # Calculate Per 75 Metric
    # If Total_Synergy_Poss is 0, POE is 0
    player_agg['Playtype_POE_per_75'] = np.where(
        player_agg['Total_Synergy_Poss'] > 0,
        player_agg['Total_Playtype_POE'] / (player_agg['Total_Synergy_Poss'] / 75.0),
        0
    )
    
    # Keep only the final feature
    final_df = player_agg[['PLAYER_ID', 'year', 'Playtype_POE_per_75']]
    
    print(f"Saving {len(final_df)} player-seasons to {PLAYTYPE_POE_FEATURES}...")
    final_df.to_csv(PLAYTYPE_POE_FEATURES, index=False)
    print("Done! Example output:")
    print(final_df.head())

if __name__ == "__main__":
    fetch_and_prepare_playtypes()
