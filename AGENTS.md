# AGENTS.md — New SPM Workspace

This file orients any future AI agent (or human) working in this repo. Read it
before making changes. Keep it short; update it whenever the high-level layout
or pipeline shifts.

---

## What lives here

Three related NBA analytics projects share this workspace:

| Sub-project        | Purpose                                                                                                                                             | Entry points                                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **SPM** (root)     | Train gradient-boosted models that predict offensive / defensive RAPM from box + tracking features. Output is a per-player "prior" used downstream. | `src/train_spm.py`, `src/generate_priors.py`, `src/apply_prior.py`                       |
| **RAPM** (`rapm/`) | Ridge regression on play-by-play possessions from a MySQL `matchups` table, with optional SPM priors and playoff/meta adjustments.                  | `rapm/src/rapm.py`, `rapm/src/rapm_with_prior.py`, `rapm/src/playoff_rapm_with_prior.py` |
| **zTS** (`zts/`)   | Playtype-adjusted relative True Shooting. Has its own charter — see `zts/AGENTS.md`.                                                                | `zts/compute_zts.ipynb`                                                                  |

The SPM and RAPM projects exchange a single artifact: `data/outputs/prior.csv`
(produced by SPM, consumed by the RAPM "with prior" variants).

---

## Folder layout

```
New SPM/
├── AGENTS.md                         ← this file
├── README.md                         ← quick-start for humans
├── requirements.txt                  ← SPM + RAPM deps (zts has its own)
│
├── src/                              ← SPM pipeline (Python scripts, run directly)
│   ├── paths.py                      ← central path constants (IMPORT THIS)
│   ├── fetch_playtypes.py            ← downloads Synergy playtype CSV → POE features
│   ├── train_spm.py                  ← GBM training (was bpm_optimized.py)
│   ├── generate_priors.py            ← inference → data/outputs/prior.csv
│   └── apply_prior.py                ← Bayesian blend of SPM prior with observed RAPM
│
├── notebooks/
│   └── bpm.ipynb                     ← exploratory training notebook
│
├── models/
│   ├── current/                      ← latest global artifacts (used by generate_priors)
│   │   ├── spm_off_model.pkl
│   │   ├── spm_def_model.pkl
│   │   ├── spm_scaler.pkl
│   │   └── model_features.pkl
│   └── rolling/{2018-22,2019-23,2020-24}/
│       ├── off_model.pkl             ← rolling-window models (not wired into main pipeline)
│       ├── def_model.pkl
│       └── scaler.pkl
│
├── data/
│   ├── raw/                          ← scraped inputs, don't modify
│   │   ├── site_Data/                ← mirror of github.com/gabriel1200/site_Data
│   │   ├── playersheets/year_totals/ ← per-season box score totals
│   │   └── 2025/                     ← loose 2025 stats
│   ├── processed/                    ← derived, hand-fed to training/inference
│   │   ├── smaller_player_stats_with_rapm.csv   ← TRAINING DATA (target = Off, Def)
│   │   ├── smaller_player_stats_with_SPM.csv
│   │   ├── smaller_player_stats_with_SPM_O.csv
│   │   ├── merged_dataset.csv
│   │   ├── merged_per100_dataset.csv            ← INFERENCE DATA (for generate_priors)
│   │   ├── merged_per100_with_rTS_AuPM.csv
│   │   ├── merged_dataset_inference.csv
│   │   ├── all_years_data.csv
│   │   ├── playtype_poe_features.csv            ← produced by fetch_playtypes.py
│   │   └── per_year/                            ← legacy YYYY_inference.csv snapshots
│   └── outputs/                      ← produced by the SPM pipeline
│       ├── prior.csv                 ← consumed by rapm/src/rapm_with_prior.py
│       ├── bpm_optimized_predictions.csv
│       └── rapm_posterior.csv
│
├── rapm/                             ← RAPM sub-project, see rapm/AGENTS.md
│   ├── AGENTS.md
│   ├── README.md
│   ├── src/                          ← run from this directory
│   ├── data/                         ← all_names.csv, prior.csv snapshot, Dump20240524.sql
│   ├── outputs/                      ← dump/, rapm_results/, Combined_Rapm_*.csv
│   └── venv/                         ← project-local virtualenv (gitignored in spirit)
│
├── zts/                              ← independent zTS project (see zts/AGENTS.md)
│
├── random research/                  ← ad hoc analysis scripts + generated research outputs
│   └── age_adjusted_rookie_impact.py ← rookie age/impact trend study
│
└── archive/                          ← legacy scripts + old output CSVs, safe to ignore
    ├── scripts/                      ← bpm2.py, bpmyear.py, combine_seasons.py, etc.
    ├── data/                         ← old Combined_Rapm_*.csv, AIAGENT.md, …
    └── AIAGENT.md                    ← previous agent charter (superseded by this file)
```

---

## The SPM pipeline (root)

```
data/raw/ ─► (manual / upstream) ─► data/processed/smaller_player_stats_with_rapm.csv
                                                         │
                                                         ▼
                      src/fetch_playtypes.py ─► data/processed/playtype_poe_features.csv
                                                         │
                                                         ▼
                       src/train_spm.py  ─► models/current/{spm_off, spm_def, spm_scaler}.pkl
                                           + data/outputs/bpm_optimized_predictions.csv
                                                         │
                   data/processed/merged_per100_dataset.csv
                                                         │
                                                         ▼
                        src/generate_priors.py  ─► data/outputs/prior.csv
                                                         │
                        src/apply_prior.py  ─► data/outputs/rapm_posterior.csv
                                                         │
                                                         ▼
                                         rapm/src/rapm_with_prior.py
                                         rapm/src/playoff_rapm_with_prior.py
```

### Model architecture

| Target        | Winner   | Key hyperparams                                                               |
| ------------- | -------- | ----------------------------------------------------------------------------- |
| Offensive SPM | LightGBM | `lr=0.05`, `n_estimators=100`, `max_depth=4`, `num_leaves=63`, `reg_lambda=5` |
| Defensive SPM | XGBoost  | `lr=0.05`, `n_estimators=100`, `max_depth=4`, `reg_alpha=1`, `reg_lambda=1`   |

Training uses `GroupKFold(n_splits=5)` grouped by `PLAYER_ID` with `sample_weight=MIN`
and a per-feature `StandardScaler`. Search strategy is `RandomizedSearchCV(n_iter=15)`.

### Feature set (28 total)

Base features (26):
`Points_per100_off`, `FtPoints_per100_off`, `AFGM_per100_off`, `DRIVES_per100_off`,
`3PtP`, `FG3A_per100_off`, `ThreePtAssists_per100_off`, `cTOV`,
`AtRimAssists_per100_off`, `Net Passes_per100_off`, `on-ball-time%`,
`DREB_CONTEST_per100_def`, `OREB_CONTEST_per100_off`, `DREB_UNCONTEST_per100_def`,
`OREB_UNCONTEST_per100_off`, `SelfOReb_per100_off`, `rSTOP%`, `RimPointsSaved`,
`PF_per100_def`, `Contested3PT Shots_per100_def`,
`Loose BallsRecovered_per100_def`, `Deflections_per100_def`,
`RecoveredBlocks_per100_def`, `Steals_per100_def`,
`DFGA_rim_defense_per100_def`, `ChargesDrawn_per100_def`.

Engineered (2):

- `Playtype_POE_per_75` — Synergy playtype Points Over Expectation / 75 poss
- `Self_Creation_Ratio` — `UAPTS / (PTS + 0.1)`

### Sign conventions

`generate_priors.py` emits `SPM_O = off_pred / 100` and `SPM_D = -def_pred / 100`.
The negation matches the RAPM "defense" convention used in `rapm_with_prior.py`
(a _higher_ `SPM_D` column in `prior.csv` means a _worse_ defender, so subtracting
it in the final RAPM sum gives the intuitive sign).

### Paths

All SPM scripts import from `src/paths.py`:

```python
from paths import TRAINING_DATA, PRIOR_CSV, SPM_OFF_MODEL  # etc.
```

When you add a new input/output, put the constant in `src/paths.py` rather than
hardcoding a string. Scripts are run directly (`python src/train_spm.py`) — no
package layout, so `from paths import …` is fine because Python adds the script
directory to `sys.path`.

---

## The RAPM sub-project

See `rapm/AGENTS.md` for details. Short version:

- Connects to a local MySQL `nba_api` database (`matchups` table of possessions).
- `rapm.py` — ridge model with meta columns (home, rubberband per quarter, fatigue) and garbage-time filtering.
- `rapm_with_prior.py` — per-season ridge with SPM priors from `data/outputs/prior.csv`.
- `playoff_rapm_with_prior.py` — 3-year windows, regular-season → playoff prior, with dummy-game and offset methods.
- `allyears_rapm.py` — single full-history ridge (no time decay).

All RAPM paths route through `rapm/src/paths.py`.

---

## Ad hoc research

`random research/age_adjusted_rookie_impact.py` studies whether NBA rookies have
become more impactful over time. It builds rookie cohorts from Basketball
Reference debut seasons, joins `rapm/outputs/aging/age_adjusted_rapm.csv`, and
uses a Basketball Reference advanced scrape at:

`/Users/eadebayo/Documents/Projects/Sports Analytics/mof's spm project/archive/Advanced.csv`

That external advanced file contains `obpm`, `dbpm`, `bpm`, `vorp`, `per`,
`ws`, `ws_48`, age, and experience. The script writes charts/CSVs to
`random research/outputs/age_adjusted_rookie_impact/`, including all-rookie,
top-20-by-minutes, standardized context, and within-age-group views.

Run from repo root with:

```bash
rapm/venv/bin/python "random research/age_adjusted_rookie_impact.py"
```

---

## Running anything

From repo root:

```bash
# Environment — Python 3.11+ is assumed
pip install -r requirements.txt

# SPM training
python src/fetch_playtypes.py         # refresh playtype POE features
python src/train_spm.py               # fit Off/Def GBMs, write models + predictions
python src/generate_priors.py         # write data/outputs/prior.csv
python src/apply_prior.py             # blend prior with observed RAPM → posterior

# RAPM ridge
cd rapm
python src/rapm.py                    # experimental meta-column variant
python src/rapm_with_prior.py         # with SPM prior
python src/playoff_rapm_with_prior.py # playoff / prior variants
```

The RAPM scripts require a running MySQL socket at `/tmp/mysql.sock` with the
`nba_api` database loaded from `rapm/data/Dump20240524.sql`.

---

## Rules for future agents

1. **Don't restore the root-level dumping ground.** Every new script goes in
   `src/` or `rapm/src/`. Every new input goes in `data/raw/` or `data/processed/`.
   Every new output goes in `data/outputs/` (SPM) or `rapm/outputs/` (RAPM).
   Exception: self-contained exploratory studies may live under `random research/`.
2. **Use `paths.py` constants instead of hardcoded strings.** Add new constants
   there when introducing files.
3. **The `archive/` directory is read-only history.** Don't depend on anything
   inside it. If you truly need an archived artifact, promote a single clean copy
   to `data/processed/` or `data/outputs/` and document it here.
4. **Don't commit a working `.DS_Store` / `__pycache__` / `venv/`** into git
   (this repo isn't under git yet, but when it is add a `.gitignore`).
5. **zts/ is self-contained.** Don't fold SPM or RAPM code into it; see its
   own charter for rules.
6. **Keep this file under ~400 lines.** If it grows past that, split into topic-
   specific docs under `docs/`.

---

## Known issues / TODO

- `rapm/data/prior.csv` is a stale snapshot from Feb 3. After regenerating SPM
  priors, copy `data/outputs/prior.csv` over it (or change `rapm/src/paths.py`
  `PRIOR_CSV` to point at `SPM_PRIOR_CSV`).
- `rapm/src/fast_meta_optimizer.py`, `meta_column_optimizer.py`, and
  `quick_meta_optimizer.py` each hard-code the MySQL password. Move to an env
  var before anything approaches version control.
- `models/rolling/` artifacts exist but aren't plumbed into the active pipeline;
  `generate_priors.py` only uses `models/current/`.
- `notebooks/bpm.ipynb` likely contains stale paths from the pre-reorg layout.
  Either update or retire it.
