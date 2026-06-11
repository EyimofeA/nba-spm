import pandas as pd
import numpy as np

from paths import TRAIN_PREDICTIONS, POSTERIOR_CSV, ensure_dirs

ensure_dirs()

PRIOR_WEIGHT = 1000  # Equivalent minutes for the prior "dummy observations"
TARGET_COL = "Rapm"              # Observed RAPM column
PRED_COL = "SPM_Total_Predicted" # Predicted SPM column
WEIGHT_COL = "Weight"            # Minutes

print(f"Loading {TRAIN_PREDICTIONS}...")
df = pd.read_csv(TRAIN_PREDICTIONS)

# Ensure required columns exist
required_cols = [TARGET_COL, PRED_COL, WEIGHT_COL, 'Player', 'year']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column '{col}' in input file.")

print(f"Applying Bayesian Prior with Weight = {PRIOR_WEIGHT} minutes...")

# Formula: (Minutes * Observed + PriorWeight * Prior) / (Minutes + PriorWeight)
df['Posterior_Rapm'] = (
    (df[WEIGHT_COL] * df[TARGET_COL]) + (PRIOR_WEIGHT * df[PRED_COL])
) / (df[WEIGHT_COL] + PRIOR_WEIGHT)

# Calculate difference
df['Prior_Impact'] = df['Posterior_Rapm'] - df[TARGET_COL]

print("\n---------------------------------------------------")
print("Top 20 Players by Posterior Total RAPM")
print("---------------------------------------------------")
cols_to_show = ['Player', 'year', 'Weight', TARGET_COL, PRED_COL, 'Posterior_Rapm', 'Prior_Impact']
print(df[cols_to_show].sort_values(by='Posterior_Rapm', ascending=False).head(20).to_string(index=False))

print("\n---------------------------------------------------")
print("Top 20 Players where Prior Increased RAPM Most (Undervalued by raw RAPM)")
print("---------------------------------------------------")
print(df[cols_to_show].sort_values(by='Prior_Impact', ascending=False).head(20).to_string(index=False))

print("\n---------------------------------------------------")
print("Top 20 Players where Prior Decreased RAPM Most (Overvalued by raw RAPM)")
print("---------------------------------------------------")
print(df[cols_to_show].sort_values(by='Prior_Impact', ascending=True).head(20).to_string(index=False))

df.to_csv(POSTERIOR_CSV, index=False)
print(f"\nSaved posterior results to {POSTERIOR_CSV}")
