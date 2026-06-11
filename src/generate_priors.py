import pandas as pd
import numpy as np
import joblib

from paths import (
    INFERENCE_DATA,
    PLAYTYPE_POE_FEATURES,
    PRIOR_CSV,
    SPM_OFF_MODEL,
    SPM_DEF_MODEL,
    SPM_SCALER,
    ensure_dirs,
)

ensure_dirs()

INF_SEASON_COL = "year"
INF_PLAYER_ID_COL = "PLAYER_ID"

print(f"Loading Inference Data: {INFERENCE_DATA}...")
df_inf = pd.read_csv(INFERENCE_DATA)
df_inf.columns = df_inf.columns.str.replace('\xa0', ' ')

# Features
base_features = [
    "Points_per100_off", "FtPoints_per100_off", "AFGM_per100_off", 'DRIVES_per100_off',
    "3PtP", 'FG3A_per100_off',
    "ThreePtAssists_per100_off", "cTOV", 'AtRimAssists_per100_off', 'Net Passes_per100_off', "on-ball-time%",
    "DREB_CONTEST_per100_def", "OREB_CONTEST_per100_off", "DREB_UNCONTEST_per100_def", "OREB_UNCONTEST_per100_off", 'SelfOReb_per100_off',
    "rSTOP%", "RimPointsSaved", "PF_per100_def", 'Contested3PT Shots_per100_def',
    'Loose BallsRecovered_per100_def', 'Deflections_per100_def',
    'RecoveredBlocks_per100_def', 'Steals_per100_def', 'DFGA_rim_defense_per100_def', 'ChargesDrawn_per100_def'
]

engineer_cols = ['ShotQualityAvg', 'TS_PCT', 'UAPTS', 'PTS']

# Ensure required columns
missing_feats = [f for f in base_features + engineer_cols if f not in df_inf.columns]
if missing_feats:
    print(f"Warning: Missing features in inference file: {missing_feats}")
    for f in missing_feats:
        df_inf[f] = 0

# Load POE features and merge
try:
    df_poe = pd.read_csv(PLAYTYPE_POE_FEATURES)
    df_poe = df_poe.rename(columns={'year': INF_SEASON_COL}) if 'year' != INF_SEASON_COL else df_poe
    df_inf = pd.merge(df_inf, df_poe, left_on=[INF_PLAYER_ID_COL, INF_SEASON_COL], right_on=['PLAYER_ID', INF_SEASON_COL], how='left')
    df_inf['Playtype_POE_per_75'] = df_inf['Playtype_POE_per_75'].fillna(0)
    print(f"Merged Playtype_POE_per_75. Missing filled with 0.")
except Exception as e:
    print(f"Error loading POE features: {e}. 'Playtype_POE_per_75' will be 0.")
    df_inf['Playtype_POE_per_75'] = 0

# Feature Engineering
print("Engineering new features...")
df_inf['Self_Creation_Ratio'] = df_inf['UAPTS'] / (df_inf['PTS'] + 0.1)

new_engineered = ['Self_Creation_Ratio', 'Playtype_POE_per_75']

features = base_features + new_engineered

# Filter Years 2017-2024
df_inf[INF_SEASON_COL] = pd.to_numeric(df_inf[INF_SEASON_COL], errors='coerce')
df_inf = df_inf[(df_inf[INF_SEASON_COL] >= 2017) & (df_inf[INF_SEASON_COL] <= 2024)]

if INF_PLAYER_ID_COL not in df_inf.columns:
    cols = [c for c in df_inf.columns if "PLAYER_ID" in c or "PlayerID" in c]
    if cols:
        INF_PLAYER_ID_COL = cols[0]
        print(f"Using {INF_PLAYER_ID_COL} as Player ID.")

df_inf = df_inf.dropna(subset=features)
X_inf = df_inf[features]

print("Loading global serialized models (scaler, spm_off, spm_def)...")
try:
    scaler = joblib.load(SPM_SCALER)
    spm_off = joblib.load(SPM_OFF_MODEL)
    spm_def = joblib.load(SPM_DEF_MODEL)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing model artifact: {e}")

print("Scaling inference data...")
X_inf_scaled = scaler.transform(X_inf)
X_inf_scaled = pd.DataFrame(X_inf_scaled, columns=features)

print("Predicting for Historical Data...")
spm_o_pred = spm_off.predict(X_inf_scaled)
spm_d_pred = spm_def.predict(X_inf_scaled)

# Since we trained D directly on target output (where positive is better in the smaller_stats dataset)
# but the RAPM pipeline priors explicitly use `-spm_d/100`, we mirror what was originally expected.
prior_df = pd.DataFrame({
    'Season': df_inf[INF_SEASON_COL].astype(int),
    'PLAYER_ID': df_inf[INF_PLAYER_ID_COL].astype(int, errors='ignore'),
    'SPM_O': spm_o_pred / 100.0,
    'SPM_D': -spm_d_pred / 100.0
})

prior_df.dropna(subset=['SPM_O', 'SPM_D'], inplace=True)

print(f"Saving {len(prior_df)} rows to {PRIOR_CSV}...")
prior_df.to_csv(PRIOR_CSV, index=False)
print("Done.")
print(prior_df.head())
