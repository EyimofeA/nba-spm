# AGENTS.md — RAPM sub-project

Ridge regression on NBA play-by-play possessions to estimate each player's
offensive and defensive Regularized Adjusted Plus/Minus.

This sub-project consumes priors produced by the parent SPM pipeline
(`../src/generate_priors.py` → `../data/outputs/prior.csv`) but is otherwise
self-contained. Parent workspace charter: [`../AGENTS.md`](../AGENTS.md).

---

## Folder layout

```
rapm/
├── AGENTS.md                  ← this file
├── README.md                  ← quick-start for humans
├── CLAUDE.md                  ← legacy (kept for provenance; prefer AGENTS.md)
│
├── src/                       ← all Python scripts, run from here
│   ├── paths.py               ← central path constants (IMPORT THIS)
│   ├── rapm.py                ← experimental ridge w/ meta cols + garbage-time filter
│   ├── rapm_with_prior.py     ← per-season ridge with SPM prior (offset / residual method)
│   ├── allyears_rapm.py       ← single ridge on the full matchups table, no windows
│   ├── playoff_rapm_with_prior.py  ← 3yr windows: reg-season prior → playoff RAPM
│   ├── run_historical_batch.py     ← batch driver for playoff_rapm_with_prior
│   ├── run_latest_playoff_prior.py ← 2022-24 single run
│   ├── rapm_analysis.py            ← summary stats on comprehensive_rapm_1997_2024.csv
│   ├── prettyprintrapm.py          ← top-30 Excel/HTML/PNG report
│   ├── meta_column_optimizer.py    ← systematic meta-column search
│   ├── fast_meta_optimizer.py      ← faster heuristic search
│   ├── quick_meta_optimizer.py     ← smallest / fastest search
│   └── test_comprehensive_rapm.py  ← 3-window smoke test for playoff pipeline
│
├── data/                      ← inputs
│   ├── all_names.csv          ← PLAYER_ID → PLAYER_NAME
│   ├── prior.csv              ← snapshot of SPM prior (see "Priors" below)
│   └── Dump20240524.sql       ← MySQL dump of the `matchups` table (670 MB)
│
├── outputs/                   ← every produced artifact
│   ├── dump/                  ← raw coefficient CSVs (one per run)
│   ├── rapm_results/          ← human-readable CSVs (player + Off/Def/Rapm)
│   ├── Combined_Rapm.csv                 ← dummy-method combined across windows
│   ├── Combined_Rapm_experimental.csv    ← rapm.py experimental variant
│   ├── RAPM_with_prior_all_seasons.csv   ← rapm_with_prior.py output
│   ├── comprehensive_rapm_1997_2024.csv  ← full playoff-with-prior history
│   ├── results.csv                       ← legacy snapshot
│   ├── batch_output.log                  ← output of last run_historical_batch
│   └── {fast,quick,meta}_optimization_results.json
│
└── venv/                      ← project-local Python 3.14 virtualenv
```

---

## Data source: MySQL `matchups` table

All ridge scripts connect to a local MySQL via unix socket:

```python
MySQLdb.connect(host="localhost", user="root",
                db="nba_api", unix_socket="/tmp/mysql.sock")
```

Load the database once from the dump:

```bash
mysql -u root nba_api < rapm/data/Dump20240524.sql
```

The `matchups` table stores one row per possession with columns:
- `home_poss` (1 if home team is offense), `pts` scored on that possession
- `a1..a5`, `h1..h5` — PLAYER_IDs of the ten players on court
- `season`, `date`, `period`, `num`, `gameid`

The secret hardcoded into the scripts (`password="41rApm_@02"`) needs to move
to an env var before this ever hits version control.

---

## Method overview

Each script implements a variant of ridge regression:

```
X  = sparse (n_possessions × 2·n_players + n_meta)   # off/def indicator per player
y  = points scored on the possession (centered)
β  = RidgeCV(alphas=[…]).fit(X, y, sample_weight=…).coef_
```

The final RAPM per player is the offensive coefficient minus the defensive one,
scaled by 100.

### Variants

| Script | What it adds |
|--------|--------------|
| `rapm.py` | Meta columns (home, rubberband margin per quarter, fatigue), dynamic garbage-time filter, optimized alpha search. |
| `rapm_with_prior.py` | Per-season ridge. Reads `prior.csv`, runs offset method: fit ridge on residuals `y − X·prior`, add prior back to raw coefficients. |
| `playoff_rapm_with_prior.py` | 3-year windows. Calculates reg-season RAPM first (cached to `outputs/dump/prior_cache_*.csv`), then playoff RAPM using either the *offset* method or the *dummy-game* method (adds synthetic possessions anchoring each player to their prior with configurable off/def confidence weights). |
| `allyears_rapm.py` | Single ridge on all matchups, uniform weights, no prior. |
| `meta_column_optimizer.py` / `fast_meta_optimizer.py` / `quick_meta_optimizer.py` | Grid-search different meta-column subsets to maximize held-out score. |

### Meta columns (in rapm.py and playoff_rapm_with_prior.py)

```python
META_COLS = [
    "META_home",      # 1 if home-possession offense
    "META_rb_q1..q4", # rubberband: score margin at possession start, per quarter
    "META_fatigue",   # 0.8·(period-1)/3 + 0.2·progress_within_quarter
]
```

Garbage-time filter drops possessions where `|margin|` exceeds a per-quarter
threshold (`{1:25, 2:20, 3:17, 4:12}` widened by up to +8 early in the quarter).

---

## Priors (flow from the SPM pipeline)

`rapm_with_prior.py` and `playoff_rapm_with_prior.py` read `prior.csv` with
columns `Season, PLAYER_ID, SPM_O, SPM_D`. Currently `rapm/src/paths.py` has
two constants:

- `PRIOR_CSV = rapm/data/prior.csv` — local snapshot used by the scripts.
- `SPM_PRIOR_CSV = ../data/outputs/prior.csv` — fresh output from SPM.

After the SPM pipeline regenerates priors, sync the local snapshot:

```bash
cp ../data/outputs/prior.csv rapm/data/prior.csv
```

Or flip the scripts to read `SPM_PRIOR_CSV` directly. The default was kept as
a local snapshot so RAPM runs are reproducible even if SPM is retrained.

---

## Running things

From the repo root:

```bash
cd rapm

# One-off experimental ridge on a 5-year window
python src/rapm.py

# Per-season ridge with SPM prior (defaults to seasons 2017–2024)
python src/rapm_with_prior.py

# 3-year playoff windows with prior (tune off_conf / def_conf inside)
python src/run_latest_playoff_prior.py

# Full historical batch (23 × 3-yr + 21 × 5-yr + 1 × full-period, 2 methods each)
python src/run_historical_batch.py    # long-running, produces comprehensive_rapm_1997_2024.csv
```

All outputs land under `outputs/` — no file is written outside the `rapm/` tree.

---

## Rules for future agents

1. Every new script goes under `rapm/src/`.
2. Every new input goes under `rapm/data/` (or `../data/` if shared with SPM).
3. Every new output goes under `rapm/outputs/` via a constant in `src/paths.py`.
4. Don't hardcode paths — extend `src/paths.py` instead.
5. Don't touch `outputs/dump/` or `outputs/rapm_results/` contents manually;
   they are produced by scripts and reproduced on demand.
6. The MySQL password is embedded in several scripts. Treat every one of them
   as "do not commit" until that's fixed.
