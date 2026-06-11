#!/usr/bin/env python3
"""
Batch runner for historical RAPM analysis.
Runs all rolling windows from 2001-2024 using both Dummy and Offset methods.
"""

from playoff_rapm_with_prior import run_playoff_rapm_with_prior
import time

def generate_rolling_windows(start_year, end_year, window_size):
    """Generate all rolling windows of given size"""
    windows = []
    for start in range(start_year, end_year - window_size + 2):
        window = list(range(start, start + window_size))
        windows.append(window)
    return windows


def run_batch():
    start_time = time.time()
    
    # Configuration
    START_YEAR = 2001
    END_YEAR = 2024
    
    # Generate all window configurations
    windows_3yr = generate_rolling_windows(START_YEAR, END_YEAR, 3)
    windows_5yr = generate_rolling_windows(START_YEAR, END_YEAR, 5)
    full_period = [list(range(START_YEAR, END_YEAR + 1))]
    
    print("=" * 60)
    print("HISTORICAL RAPM BATCH RUN")
    print("=" * 60)
    print(f"3-Year Windows: {len(windows_3yr)} runs")
    print(f"5-Year Windows: {len(windows_5yr)} runs")
    print(f"Full Period: 1 run")
    print(f"Methods: Dummy + Offset = 2x each")
    print(f"Total Runs: {(len(windows_3yr) + len(windows_5yr) + 1) * 2}")
    print("=" * 60)
    
    completed = 0
    total = (len(windows_3yr) + len(windows_5yr) + 1) * 2
    
    # Run 3-year rolling windows
    print("\n### ROLLING 3-YEAR WINDOWS ###\n")
    for seasons in windows_3yr:
        print(f"\n[{completed+1}/{total}] 3yr: {seasons} - DUMMY METHOD")
        try:
            run_playoff_rapm_with_prior(
                seasons,
                use_dummy_method=True,
                off_conf=3000,
                def_conf=2000,
                time_decay=True,
                use_cache=True
            )
        except Exception as e:
            print(f"ERROR: {e}")
        completed += 1
        
        print(f"\n[{completed+1}/{total}] 3yr: {seasons} - OFFSET METHOD")
        try:
            run_playoff_rapm_with_prior(
                seasons,
                use_dummy_method=False,
                prior_weight=1.0,
                use_cache=True
            )
        except Exception as e:
            print(f"ERROR: {e}")
        completed += 1
    
    # Run 5-year rolling windows
    print("\n### ROLLING 5-YEAR WINDOWS ###\n")
    for seasons in windows_5yr:
        print(f"\n[{completed+1}/{total}] 5yr: {seasons} - DUMMY METHOD")
        try:
            run_playoff_rapm_with_prior(
                seasons,
                use_dummy_method=True,
                off_conf=3000,
                def_conf=2000,
                time_decay=True,
                use_cache=True
            )
        except Exception as e:
            print(f"ERROR: {e}")
        completed += 1
        
        print(f"\n[{completed+1}/{total}] 5yr: {seasons} - OFFSET METHOD")
        try:
            run_playoff_rapm_with_prior(
                seasons,
                use_dummy_method=False,
                prior_weight=1.0,
                use_cache=True
            )
        except Exception as e:
            print(f"ERROR: {e}")
        completed += 1
    
    # Run full period
    print("\n### FULL PERIOD (2001-2024) ###\n")
    for seasons in full_period:
        print(f"\n[{completed+1}/{total}] Full: {seasons[0]}-{seasons[-1]} - DUMMY METHOD")
        try:
            run_playoff_rapm_with_prior(
                seasons,
                use_dummy_method=True,
                off_conf=3000,
                def_conf=2000,
                time_decay=True,
                use_cache=True
            )
        except Exception as e:
            print(f"ERROR: {e}")
        completed += 1
        
        print(f"\n[{completed+1}/{total}] Full: {seasons[0]}-{seasons[-1]} - OFFSET METHOD")
        try:
            run_playoff_rapm_with_prior(
                seasons,
                use_dummy_method=False,
                prior_weight=1.0,
                use_cache=True
            )
        except Exception as e:
            print(f"ERROR: {e}")
        completed += 1
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE: {completed} runs in {elapsed/60:.1f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    run_batch()
