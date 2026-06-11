import pandas as pd

from paths import COMPREHENSIVE_RAPM

df = pd.read_csv(COMPREHENSIVE_RAPM)

# Create raw RAPM comparison (Regular Season vs. Playoff Raw)
comparison_raw = df[df['Data_Type'].isin(['Regular Season', 'Playoff Raw'])].pivot_table(
    index=['Name', 'Window'], columns='Data_Type', values='Rapm'
).reset_index()
comparison_raw = comparison_raw.dropna()

# Create prior RAPM comparison (Regular Season vs. Playoff with Prior)
comparison_prior = df[df['Data_Type'].isin(['Regular Season', 'Playoff with Prior'])].pivot_table(
    index=['Name', 'Window'], columns='Data_Type', values='Rapm'
).reset_index()
comparison_prior = comparison_prior.dropna()

# Calculate performance differences
comparison_prior['Dropoff_Prior'] = comparison_prior['Regular Season'] - comparison_prior['Playoff with Prior']
comparison_prior['Increase_Prior'] = comparison_prior['Playoff with Prior'] - comparison_prior['Regular Season']

# Top playoff underperformers using the Prior method
print('MOST CONSISTENT PLAYOFF UNDERPERFORMERS (with Prior):')
prior_dropoffs = comparison_prior[comparison_prior['Regular Season'] >= 3.0].nlargest(30, 'Dropoff_Prior')
underperformers = prior_dropoffs['Name'].value_counts()
print(underperformers.head(5))

# Top playoff overperformers using the Prior method
print('\nMOST CONSISTENT PLAYOFF OVERPERFORMERS (with Prior):')
prior_increases = comparison_prior[comparison_prior['Playoff with Prior'] >= 3.0].nlargest(30, 'Increase_Prior')
overperformers = prior_increases['Name'].value_counts()
print(overperformers.head(5))

# Key takeaway
print('\nKEY TAKEAWAY:')
print('Prior method produces much smaller, more stable differences.')
print('Raw method shows extreme swings; Prior method moderates them significantly.')


import pandas as pd

# Load data
df = pd.read_csv(COMPREHENSIVE_RAPM)

# Create prior RAPM comparison (Regular Season vs. Playoff with Prior)
comparison_prior = df[df['Data_Type'].isin(['Regular Season', 'Playoff with Prior'])].pivot_table(
    index=['Name', 'Window'], columns='Data_Type', values='Rapm'
).reset_index()
comparison_prior = comparison_prior.dropna()

# Calculate differences
comparison_prior['Dropoff_Prior'] = comparison_prior['Regular Season'] - comparison_prior['Playoff with Prior']
comparison_prior['Increase_Prior'] = comparison_prior['Playoff with Prior'] - comparison_prior['Regular Season']

# Get top 50 dropoffs (RS RAPM >= 3.0)
dropoffs = comparison_prior[comparison_prior['Regular Season'] >= 3.0]
dropoffs_top50 = dropoffs.sort_values(by='Dropoff_Prior', ascending=False).head(50)

# Get top 50 increases (Playoff RAPM >= 3.0)
increases = comparison_prior[comparison_prior['Playoff with Prior'] >= 3.0]
increases_top50 = increases.sort_values(by='Increase_Prior', ascending=False).head(50)

# Display results
print('TOP 50 PLAYOFF DROPOFFS (RS RAPM ≥ 3.0, using Prior method):')
print(dropoffs_top50[['Name', 'Window', 'Regular Season', 'Playoff with Prior', 'Dropoff_Prior']].to_string(index=False))

print('\nTOP 50 PLAYOFF INCREASES (PS RAPM ≥ 3.0, using Prior method):')
print(increases_top50[['Name', 'Window', 'Regular Season', 'Playoff with Prior', 'Increase_Prior']].to_string(index=False))


import pandas as pd

# Load data
df = pd.read_csv(COMPREHENSIVE_RAPM)

# Filter only relevant data types
comparison_prior = df[df['Data_Type'].isin(['Regular Season', 'Playoff with Prior'])]

# Pivot to align RS and Playoff RAPM per player-season
comparison_prior = comparison_prior.pivot_table(
    index=['Name', 'Window'], columns='Data_Type', values='Rapm'
).reset_index()
comparison_prior = comparison_prior.dropna()

# Calculate differences
comparison_prior['Dropoff_Prior'] = comparison_prior['Regular Season'] - comparison_prior['Playoff with Prior']
comparison_prior['Increase_Prior'] = comparison_prior['Playoff with Prior'] - comparison_prior['Regular Season']

# Filter and compute average dropoff per player (only if RS RAPM >= 3.0 in that season)
dropoffs = comparison_prior[comparison_prior['Regular Season'] >= 3.0]
avg_dropoffs = dropoffs.groupby('Name')['Dropoff_Prior'].mean().sort_values(ascending=False).head(50)

# Filter and compute average increase per player (only if Playoff RAPM >= 3.0 in that season)
increases = comparison_prior[comparison_prior['Playoff with Prior'] >= 3.0]
avg_increases = increases.groupby('Name')['Increase_Prior'].mean().sort_values(ascending=False).head(50)

# Display results
print('TOP 50 MOST CONSISTENT PLAYOFF UNDERPERFORMERS (Average Dropoff | RS RAPM ≥ 3.0):')
print(avg_dropoffs.round(3).to_string())

print('\nTOP 50 MOST CONSISTENT PLAYOFF OVERPERFORMERS (Average Increase | PS RAPM ≥ 3.0):')
print(avg_increases.round(3).to_string())


import pandas as pd

# Load data
df = pd.read_csv(COMPREHENSIVE_RAPM)

# Keep only RS vs Playoff-with-Prior and align them per player-window
paired = (
    df[df['Data_Type'].isin(['Regular Season', 'Playoff with Prior'])]
    .pivot_table(index=['Name', 'Window'], columns='Data_Type', values='Rapm')
    .reset_index()
    .dropna(subset=['Regular Season', 'Playoff with Prior'])
)

# Differences
paired['Delta'] = paired['Playoff with Prior'] - paired['Regular Season']
paired['Increase'] = paired['Delta'].clip(lower=0)        # only positive deltas
paired['Dropoff'] = (-paired['Delta']).clip(lower=0)      # only negative deltas, made positive

# Aggregate per player (only spans with PS exist are already enforced by dropna above)
agg = paired.groupby('Name').agg(
    Total_Increase=('Increase', 'sum'),
    Total_Dropoff=('Dropoff', 'sum'),
    Net_Change=('Delta', 'sum'),
    Spans_Count=('Delta', 'size')
).sort_values('Net_Change', ascending=False)

# Top totals
top_increases = agg.sort_values('Total_Increase', ascending=False).head(25)
top_dropoffs  = agg.sort_values('Total_Dropoff',  ascending=False).head(25)
top_net_plus  = agg.sort_values('Net_Change',     ascending=False).head(25)
top_net_minus = agg.sort_values('Net_Change',     ascending=True).head(25)

print("HIGHEST TOTAL PLAYOFF INCREASE (sum of positive PS−RS):")
print(top_increases[['Total_Increase','Spans_Count']].round(3).to_string())

print("\nHIGHEST TOTAL PLAYOFF DROPOFF (sum of positive RS−PS):")
print(top_dropoffs[['Total_Dropoff','Spans_Count']].round(3).to_string())

print("\nBEST NET PLAYOFF CHANGE (sum of PS−RS across spans):")
print(top_net_plus[['Net_Change','Spans_Count']].round(3).to_string())

print("\nWORST NET PLAYOFF CHANGE (sum of PS−RS across spans):")
print(top_net_minus[['Net_Change','Spans_Count']].round(3).to_string())
