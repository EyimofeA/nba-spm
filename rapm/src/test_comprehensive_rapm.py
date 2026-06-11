#!/usr/bin/env python3
"""
Test version of comprehensive RAPM - just a few windows to validate approach
"""

# Import everything from the main file
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
from playoff_rapm_with_prior import *
from paths import OUTPUTS

def main():
    """Test on just 3 windows to validate the approach"""
    
    print("="*80)
    print("TEST: 3-YEAR RAPM ANALYSIS (LIMITED WINDOWS)")
    print("="*80)
    print("Testing on windows: 2019-2021, 2020-2022, 2021-2023")
    print("Process: Regular Season → Playoff Raw → Playoff with Prior")
    print()
    
    # Test with just 3 windows
    test_windows = [[2019, 2020, 2021], [2020, 2021, 2022], [2021, 2022, 2023]]
    
    # Fetch data for test years only
    all_regular_data, all_playoff_data = fetch_all_data_optimized(2019, 2023)
    
    # Store all results
    all_results = []
    
    for i, seasons in enumerate(test_windows):
        print(f"\n{'='*60}")
        print(f"Processing window {i+1}/{len(test_windows)}: {format_season_list(seasons)}")
        print(f"{'='*60}")
        
        # Filter data for current window
        reg_data = filter_data_by_seasons(all_regular_data, seasons)
        playoff_data = filter_data_by_seasons(all_playoff_data, seasons)
        
        # Step 1: Regular Season RAPM (find optimal alpha)
        print("\n--- Step 1: Regular Season RAPM ---")
        reg_coefs, reg_col_to_key, reg_poss = build_and_fit_from_data(reg_data, seasons, "Regular Season")
        
        if reg_coefs is None:
            print(f"Skipping {format_season_list(seasons)} - no regular season data")
            continue
        
        # Use the alpha that was found during regular season fitting
        selected_alpha = 3000  # Default for testing
            
        reg_dict = coefficients_to_dict(reg_coefs, reg_col_to_key)
        reg_readable = create_human_readable_from_coefs(reg_coefs, reg_col_to_key, seasons)
        
        # Add metadata to results
        reg_readable['Data_Type'] = 'Regular Season'
        reg_readable['Window'] = format_season_list(seasons)
        reg_readable['Possessions'] = reg_poss
        reg_readable['Alpha'] = selected_alpha
        
        all_results.append(reg_readable)
        print(f"Regular season: {len(reg_readable)} players, {reg_poss} possessions")
        
        # Step 2: Raw Playoff RAPM (use same alpha)
        print("\n--- Step 2: Raw Playoff RAPM ---") 
        playoff_coefs, playoff_col_to_key, playoff_poss = build_and_fit_from_data(
            playoff_data, seasons, "Playoffs", fixed_alpha=selected_alpha)
        
        if playoff_coefs is not None:
            playoff_readable = create_human_readable_from_coefs(playoff_coefs, playoff_col_to_key, seasons)
            playoff_readable['Data_Type'] = 'Playoff Raw'
            playoff_readable['Window'] = format_season_list(seasons)
            playoff_readable['Possessions'] = playoff_poss
            playoff_readable['Alpha'] = selected_alpha
            all_results.append(playoff_readable)
            print(f"Playoff raw: {len(playoff_readable)} players, {playoff_poss} possessions")
        
            # Step 3: Playoff RAPM with Prior (use same alpha)
            print("\n--- Step 3: Playoff RAPM with Prior ---")
            playoff_with_prior = calculate_playoff_with_prior(
                playoff_data, seasons, reg_dict, playoff_col_to_key, fixed_alpha=selected_alpha)
            
            if playoff_with_prior is not None:
                playoff_with_prior['Data_Type'] = 'Playoff with Prior'
                playoff_with_prior['Window'] = format_season_list(seasons)
                playoff_with_prior['Possessions'] = playoff_poss
                playoff_with_prior['Alpha'] = selected_alpha
                all_results.append(playoff_with_prior)
                print(f"Playoff with prior: {len(playoff_with_prior)} players, {playoff_poss} possessions")
        
        print(f"Completed {format_season_list(seasons)}")
    
    # Combine all results
    print(f"\n{'='*60}")
    print("COMBINING ALL RESULTS")
    print(f"{'='*60}")
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns for better readability
        available_columns = combined_df.columns.tolist()
        column_order = ['Window', 'Data_Type', 'Name', 'Off', 'Def', 'Rapm', 'Season', 'Possessions', 'Alpha']
        # Only include columns that exist
        column_order = [col for col in column_order if col in available_columns]
        combined_df = combined_df[column_order]
        
        # Save test results
        output_filename = OUTPUTS / "test_comprehensive_rapm.csv"
        combined_df.to_csv(output_filename, index=False)
        
        print(f"Test results saved to {output_filename}")
        print(f"Total records: {len(combined_df)}")
        print(f"Windows processed: {len(test_windows)}")
        
        # Show summary statistics
        summary = combined_df.groupby(['Data_Type']).agg({
            'Name': 'count',
            'Rapm': ['mean', 'std'],
            'Possessions': 'sum'
        })
        
        print(f"\nSummary by Data Type:")
        print(summary)
        
        # Show top players by category
        print(f"\nTop 5 players by category:")
        for data_type in ['Regular Season', 'Playoff Raw', 'Playoff with Prior']:
            subset = combined_df[combined_df['Data_Type'] == data_type]
            if len(subset) > 0:
                top5 = subset.nlargest(5, 'Rapm')[['Name', 'Window', 'Rapm']]
                print(f"\n{data_type}:")
                print(top5.to_string(index=False))
        
        return combined_df
    else:
        print("No results generated!")
        return None


if __name__ == "__main__":
    main()