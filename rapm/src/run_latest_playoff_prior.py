from playoff_rapm_with_prior import run_playoff_rapm_with_prior

def main():
    seasons = [2022, 2023, 2024]
    print(f"Starting analysis for seasons: {seasons}")
    
    # FINAL RUN: Dummy Method with Split Off/Def Confidence + Time Decay
    # - Time decay: More recent playoffs weighted higher
    # - Cache: Uses cached Regular Season prior if available  
    # - Off Conf 3000: Trust the Regular Season Offensive prior heavily
    # - Def Conf 2000: Trust Defense prior slightly less (noisier measurement)
    print("\n>>> FINAL RUN: Dummy Method with Enhancements")
    run_playoff_rapm_with_prior(
        seasons, 
        use_dummy_method=True, 
        off_conf=3000, 
        def_conf=2000,
        time_decay=True,
        use_cache=True
    )
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()
