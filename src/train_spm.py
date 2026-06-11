import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from paths import (
    TRAINING_DATA,
    PLAYTYPE_POE_FEATURES,
    TRAIN_PREDICTIONS,
    SPM_OFF_MODEL,
    SPM_DEF_MODEL,
    SPM_SCALER,
    ensure_dirs,
)

# ------------------------------------------------------------------------------
# 1. Configuration and Data Loading
# ------------------------------------------------------------------------------

ensure_dirs()

print(f"Loading data from {TRAINING_DATA} ...")
df = pd.read_csv(TRAINING_DATA)

# Load POE features and merge
try:
    df_poe = pd.read_csv(PLAYTYPE_POE_FEATURES)
    
    # df['year'] looks like '2018-22'. We need to extract start and end years.
    # We will build a mapping of (PLAYER_ID, year_string) -> Mean POE
    poe_mapping = []
    
    # Extract unique (PLAYER_ID, year_string) from df
    unique_rows = df[['PLAYER_ID', 'year']].drop_duplicates()
    
    for _, row in unique_rows.iterrows():
        pid = row['PLAYER_ID']
        yr_str = str(row['year'])
        try:
            # Parse '2018-22' -> 2018, 2022
            parts = yr_str.split('-')
            if len(parts) == 2:
                start_yr = int(parts[0])
                # parts[1] is '22', need to make it '2022'
                end_yr = int("20" + parts[1]) if len(parts[1]) == 2 else int(parts[1])
                
                # Get POE for this player between start_yr and end_yr inclusive
                player_poe = df_poe[(df_poe['PLAYER_ID'] == pid) & (df_poe['year'] >= start_yr) & (df_poe['year'] <= end_yr)]
                
                if len(player_poe) > 0:
                    mean_poe = player_poe['Playtype_POE_per_75'].mean()
                else:
                    mean_poe = 0
            else:
                mean_poe = 0
        except:
            mean_poe = 0
            
        poe_mapping.append({'PLAYER_ID': pid, 'year': yr_str, 'Playtype_POE_per_75': mean_poe})
        
    poe_df_mapped = pd.DataFrame(poe_mapping)
    df = pd.merge(df, poe_df_mapped, on=['PLAYER_ID', 'year'], how='left')
    df['Playtype_POE_per_75'] = df['Playtype_POE_per_75'].fillna(0)
    print(f"Merged Playtype_POE_per_75 via 5-year rolling windows. Missing filled with 0.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error loading POE features: {e}. 'Playtype_POE_per_75' will be 0.")
    df['Playtype_POE_per_75'] = 0

# Fix non-breaking spaces in column names
df.columns = df.columns.str.replace('\xa0', ' ')

# Define Base Features
base_features = [
    "Points_per100_off", "FtPoints_per100_off", "AFGM_per100_off", 'DRIVES_per100_off',
    "3PtP", 'FG3A_per100_off',
    "ThreePtAssists_per100_off", "cTOV", 'AtRimAssists_per100_off', 'Net Passes_per100_off', "on-ball-time%",
    "DREB_CONTEST_per100_def", "OREB_CONTEST_per100_off", "DREB_UNCONTEST_per100_def", "OREB_UNCONTEST_per100_off", 'SelfOReb_per100_off',
    "rSTOP%", "RimPointsSaved", "PF_per100_def", 'Contested3PT Shots_per100_def',
    'Loose BallsRecovered_per100_def', 'Deflections_per100_def',
    'RecoveredBlocks_per100_def', 'Steals_per100_def', 'DFGA_rim_defense_per100_def', 'ChargesDrawn_per100_def'
]

target_off = 'Off'
target_def = 'Def'
weight_col = 'MIN'
group_col = 'PLAYER_ID'

engineer_cols = ['ShotQualityAvg', 'TS_PCT', 'UAPTS', 'PTS']

# Filtering
df.dropna(subset=base_features + engineer_cols + [target_off, target_def], inplace=True)
if weight_col not in df.columns:
    if 'Min' in df.columns:
        weight_col = 'Min'
        df['MIN'] = df['Min']
    else:
         raise ValueError("Weight column 'MIN' or 'Min' not found.")
df['Weight'] = df[weight_col]

# Feature Engineering
print("Engineering new features...")
df['Self_Creation_Ratio'] = df['UAPTS'] / (df['PTS'] + 0.1)

new_engineered = ['Self_Creation_Ratio', 'Playtype_POE_per_75']

features = base_features + new_engineered

# Identify groups for GroupKFold
# If PLAYER_ID is missing, use Player Name
if group_col not in df.columns:
    if 'Player' in df.columns:
        group_col = 'Player'
    else:
        raise ValueError("Group column 'PLAYER_ID' or 'Player' not found.")

groups = df[group_col].values

print(f"Data Loaded. Shape: {df.shape}")
print(f"Features: {len(features)}")
print(f"Unique Groups (Players): {len(np.unique(groups))}")

# ------------------------------------------------------------------------------
# 2. Preprocessing & GroupKFold Split
# ------------------------------------------------------------------------------

X = df[features]
# We use GroupKFold directly in GridSearchCV/RandomizedSearchCV to prevent leakage.
gkf = GroupKFold(n_splits=5)

# ------------------------------------------------------------------------------
# 3. Model Definition & Tuning Pipeline
# ------------------------------------------------------------------------------

models_config = {
    "XGBoost": {
        "model": XGBRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 5, 10]
        }
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "num_leaves": [31, 63],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 5, 10]
        }
    }
}

def optimize_target(target_name, X_train, y_train, train_groups, train_weights):
    print(f"\n{'='*41}")
    print(f"Optimizing SPM for: {target_name}")
    print(f"{'='*41}\n")
    
    best_model_name = None
    best_model_instance = None
    best_score = float('inf')
    
    for name, config in models_config.items():
        print(f"Training {name} for {target_name}...")
        
        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=15,
            scoring='neg_root_mean_squared_error',
            cv=gkf,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        # Fit with weights and groups
        search.fit(
            X_train, y_train,
            groups=train_groups,
            sample_weight=train_weights
        )
        
        rmse = -search.best_score_
        print(f"{name} Best RMSE: {rmse:.4f}")
        print(f"{name} Best Params: {search.best_params_}")
        
        if rmse < best_score:
            best_score = rmse
            best_model_name = name
            best_model_instance = search.best_estimator_
            
    print(f"\n*** Winner for {target_name}: {best_model_name} (RMSE: {best_score:.4f}) ***")
    
    # Feature Importances (if applicable)
    if hasattr(best_model_instance, "feature_importances_"):
        importances = best_model_instance.feature_importances_
        imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=False)
        print(f"\nTop 10 Features ({best_model_name}):")
        print(imp_df.head(10).to_string(index=False))
        
    return best_model_instance



models_config = {
    "XGBoost": {
        "model": XGBRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 5, 10]
        }
    },
    "LightGBM": {
        "model": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 4, 5, -1],
            "num_leaves": [15, 31, 63],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_alpha": [0, 0.1, 1],
            "reg_lambda": [1, 5, 10]
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4]
        }
    }
}

def optimize_target(target_name, X_train, y_train, train_groups, train_weights):
    print(f"\n=========================================")
    print(f"Optimizing SPM for: {target_name}")
    print(f"=========================================")
    
    best_cv_rmse = float("inf")
    best_model_name = None
    best_model_instance = None
    best_params = None
    
    for name, config in models_config.items():
        print(f"\nTraining {name} for {target_name}...")
        
        search = RandomizedSearchCV(
            config["model"], 
            param_distributions=config["params"], 
            n_iter=15, 
            cv=gkf, 
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X_train, y_train, groups=train_groups, sample_weight=train_weights)
        
        cv_rmse = -search.best_score_
        
        print(f"  Best Params: {search.best_params_}")
        print(f"  CV RMSE (Grouped): {cv_rmse:.4f}")
        
        if cv_rmse < best_cv_rmse:
            best_cv_rmse = cv_rmse
            best_model_name = name
            best_model_instance = search.best_estimator_
            best_params = search.best_params_

    print("\n---------------------------------------------------")
    print(f"WINNER for {target_name}: {best_model_name} (RMSE: {best_cv_rmse:.4f})")
    print("---------------------------------------------------")
    
    best_model_instance.fit(X_train, y_train, sample_weight=train_weights)
    
    # Feature Importances
    if hasattr(best_model_instance, "feature_importances_"):
        importances = best_model_instance.feature_importances_
        imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        print(f"\nAll Feature Importances for {target_name}:")
        print(imp_df.sort_values(by="Importance", ascending=False).to_string(index=False))
        
    return best_model_instance

# ------------------------------------------------------------------------------
# 4. Global Model Training & Season-by-Season Evaluation
# ------------------------------------------------------------------------------

X_global = df[features]
y_off = df[target_off]
y_def = df[target_def]
weights = df['Weight']
groups = df[group_col].values

scaler_global = StandardScaler()
X_scaled = scaler_global.fit_transform(X_global)
X_scaled = pd.DataFrame(X_scaled, columns=features)
joblib.dump(scaler_global, SPM_SCALER)

# Run Optimization for Offense and Defense globally on all years
best_off_model = optimize_target("Offensive SPM", X_scaled, y_off, groups, weights)
best_def_model = optimize_target("Defensive SPM", X_scaled, y_def, groups, weights)

# ------------------------------------------------------------------------------
# 4. Final Evaluation, Output, and Serialization
# ------------------------------------------------------------------------------

print("\nSaving Global Models...")
joblib.dump(best_off_model, SPM_OFF_MODEL)
joblib.dump(best_def_model, SPM_DEF_MODEL)
print(f"Saved global models to {SPM_OFF_MODEL.parent}/")

# Generate Predictions on training set to verify
df['SPM_O_Predicted'] = best_off_model.predict(X_scaled)
df['SPM_D_Predicted'] = best_def_model.predict(X_scaled)
df['SPM_Total_Predicted'] = df['SPM_O_Predicted'] - df['SPM_D_Predicted']

if 'Rapm' not in df.columns:
    df['Rapm'] = df[target_off] - df[target_def]

print("\n=========================================")
print("Evaluating Model By Season (Top 5 per 5-year window)")
print("=========================================")

# Ensure we have Player names for rendering
display_cols = ['Player', 'year', 'SPM_Total_Predicted', 'Rapm', 'SPM_O_Predicted', 'SPM_D_Predicted'] if 'Player' in df.columns else ['PLAYER_ID', 'year', 'SPM_Total_Predicted']

# Group by unique year window and print the head(5) of each
for u_year in sorted(df['year'].unique()):
    df_yr = df[df['year'] == u_year]
    print(f"\n--- Season Window: {u_year} ---")
    print(df_yr[display_cols].sort_values(by="SPM_Total_Predicted", ascending=False).head(5).to_string(index=False))

# Re-sort full dataset globally to show absolute Top 10
print("\nTop 10 Players Overall (All Seasons combined):")
print(df[display_cols].sort_values(by="SPM_Total_Predicted", ascending=False).head(10).to_string(index=False))

# Save predictions
print(f"\nSaving global predictions to {TRAIN_PREDICTIONS}...")
df.to_csv(TRAIN_PREDICTIONS, index=False)
print("Done.")
