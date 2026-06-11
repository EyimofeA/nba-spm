# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a basketball analytics project that calculates RAPM (Regularized Adjusted Plus-Minus) statistics for NBA players. RAPM is an advanced basketball metric that measures a player's impact on team performance while controlling for teammates and opponents.

## Architecture

The project consists of several Python scripts that implement different variations of RAPM calculations:

### Core Scripts

- `rapm.py` - Experimental RAPM implementation with advanced features:
  - Garbage time filtering using dynamic thresholds
  - Rubberband adjustment (score margin covariates per quarter)
  - Home-court advantage covariate
  - Within-game fatigue modeling
  - Ridge regression with time decay weights

- `allyears_rapm.py` - Processes the complete dataset without time windows

- `rapm_with_prior.py` - RAPM calculation incorporating prior information (SPM - Statistical Plus-Minus)

### Data Structure

- Raw play-by-play data stored in MySQL database (`nba_api` database)
- Results saved to `dump/` directory as CSV files
- Human-readable outputs in `rapm_results/` directory
- Combined results aggregated across multiple time windows

### Key Features

1. **Time Window Analysis**: Calculates RAPM for rolling multi-year windows
2. **Regularization**: Uses Ridge regression with cross-validation for coefficient estimation
3. **Advanced Filtering**: 
   - Removes garbage time possessions using dynamic thresholds
   - Excludes playoff games for regular season analysis
4. **Covariate Modeling**:
   - Home court advantage
   - Score margin effects by quarter (rubberband adjustment)
   - Player fatigue within games
5. **Prior Integration**: Incorporates SPM priors to improve estimates for players with limited data

## Database Connection

Scripts connect to a local MySQL instance:
- Host: localhost
- User: root
- Database: nba_api
- Socket: /tmp/mysql.sock

The `matchups` table contains play-by-play data with columns:
- `home_poss`, `pts` - possession info and points scored
- `a1-a5`, `h1-h5` - away and home players on court
- `season`, `date`, `period`, `num` - game context

## Running the Scripts

Execute any of the main scripts directly:
```bash
python rapm.py           # Experimental RAPM with advanced features
python allyears_rapm.py  # Full dataset processing
python rapm_with_prior.py # RAPM with prior information
```

The scripts will prompt for season ranges and parameters as needed.

## Dependencies

- scikit-learn (Ridge regression)
- numpy, scipy (matrix operations)
- pandas (data manipulation)
- MySQLdb/PyMySQL (database connectivity)
- pymc (Bayesian modeling, optional)

## Output Files

- Raw coefficients: `dump/results_*.csv`
- Human-readable results: `rapm_results/*.csv`
- Combined datasets: `Combined_Rapm*.csv`

The human-readable outputs contain player names, offensive/defensive ratings, and overall RAPM scores multiplied by 100 for easier interpretation.