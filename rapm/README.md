# RAPM (Regularized Adjusted Plus/Minus)

Ridge regression on NBA play-by-play possessions to estimate each player's
offensive and defensive impact, with optional SPM priors and playoff variants.

Agent-oriented documentation lives in [`AGENTS.md`](./AGENTS.md). This file is
a quick-start for humans.

## Prerequisites

- Python 3.11+ (a local virtualenv is checked in at `venv/`, built against 3.14)
- MySQL with a `nba_api` database loaded from `data/Dump20240524.sql`:
  ```bash
  mysql -u root nba_api < data/Dump20240524.sql
  ```
- Socket expected at `/tmp/mysql.sock` (change in `src/*.py` if needed)

Install deps (from the repo root):

```bash
pip install -r ../requirements.txt
```

## Running

```bash
# Experimental ridge with meta columns + garbage-time filter
python src/rapm.py

# Per-season RAPM using SPM priors
python src/rapm_with_prior.py

# Latest (2022-24) playoff RAPM with prior
python src/run_latest_playoff_prior.py

# Full historical batch (long, writes comprehensive_rapm_1997_2024.csv)
python src/run_historical_batch.py
```

## Outputs

| Location | Content |
|----------|---------|
| `outputs/dump/` | Raw coefficient CSVs (one per run), plus prior caches for the playoff pipeline |
| `outputs/rapm_results/` | Human-readable per-player CSVs (Name, Off, Def, Rapm, Season) |
| `outputs/Combined_Rapm*.csv` | Aggregated across windows |
| `outputs/RAPM_with_prior_all_seasons.csv` | Output of `rapm_with_prior.py` |
| `outputs/comprehensive_rapm_1997_2024.csv` | Full batch: reg-season + playoff-raw + playoff-with-prior |

## Data

| Path | Role |
|------|------|
| `data/all_names.csv` | PLAYER_ID → PLAYER_NAME map (shared across scripts) |
| `data/prior.csv` | Snapshot of SPM prior consumed by the "with prior" variants |
| `data/Dump20240524.sql` | MySQL dump of the `matchups` possession table |

See [`AGENTS.md`](./AGENTS.md) for the method details and how priors flow in
from the parent SPM pipeline.
