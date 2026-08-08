# AGENTS.md — New SPM Workspace

This file orients any future AI agent (or human) working in this repo. Read it
before making changes. Keep it short; update it whenever the high-level layout
or pipeline shifts.

For live priorities, read `ROADMAP.md`. For the detailed impact plan, read
`IMPACT_MODEL_ROADMAP.md`; for shared model rules, read `MODELING_PLAYBOOK.md`.
For WP inputs, evidence, and rejected
variants, read `WIN_PROBABILITY.md`; for nonlinear candidates and their promotion
gates, read `WP_ARCHITECTURES.md`. Treat the older untracked `PROJECT.md` and
`IDEAS.md` as historical RAPM context, not the current task queue.

---

## Working style

- Lead with the outcome and next action. Be concise, direct, and candid; clearly
  separate verified facts, inferences, and unresolved uncertainty.
- Use an ASD-STE100-inspired technical style for instructions, reports, and user
  explanations. Use short active sentences, one main idea per sentence, one term
  for one concept, and consistent NBA/model terminology. Define uncommon terms.
  Keep mathematical names and domain terms when they improve accuracy. Do not
  claim formal ASD-STE100 compliance unless the text is checked against the
  official standard and dictionary.
- Use the Visualize skill when a chart, diagram, simulator, or interface view
  materially improves an explanation. Do not add decorative visuals.
- Ground research in authoritative, current sources and link important evidence.
- Preserve the user's original goal and constraints. Complete authorized work
  end to end and verify the observable result before claiming success.
- Ask only when a decision is materially ambiguous, risky, or needs approval.
- Subagents are optional and cost-aware. Use them only when independent work
  materially reduces wall time or adds a real verification path; default to
  local execution for small or sequential work. Never exceed four concurrent
  threads, give delegated work non-overlapping scopes, and keep critical-path
  integration with the primary agent. Synthesize and verify every result.
- Keep changes focused and simple. Avoid unrelated edits, unnecessary
  abstractions, and low-signal tests.
- Test observable behavior and validate user-facing work in the real interface
  when one exists. Preserve unrelated work and avoid destructive, production,
  or external actions beyond the granted scope.
- Report meaningful blockers, outcomes, and evidence without noisy narration.

## Model-building discipline

Use `MODELING_PLAYBOOK.md`, which adapts Karpathy's neural-network recipe to NBA
statistical modeling. Use the original recipe as an engineering and debugging
checklist, not as the scientific evaluation design:

1. Inspect raw examples, labels, joins, missingness, duplicates, and outliers
   before model code.
2. Establish the full data-to-metric path and a simple baseline first.
3. For trainable nonlinear models, verify initial loss, overfit a tiny batch, and
   inspect the exact tensor or table that enters the model.
4. Add one feature family or complexity change at a time. State the expected
   effect before the run.
5. Record train and validation curves, fixed seeds, convergence state, data and
   code hashes, and the complete configuration.
6. Tune only inside the training period. Keep chronological outer seasons frozen.
7. Compare identical rows. Resample whole games. Report calibration, uncertainty,
   important subgroups, and null results.
8. Prefer more valid data and a smaller model before ensembles or architecture
   escalation. Use cloud compute for neural training; do not train it on this Mac.

Source: Andrej Karpathy, "A Recipe for Training Neural Networks":
https://karpathy.github.io/2019/04/25/recipe/

Fractional RAPM provenance: fractional within-possession exposure was created in
this repository in commit `db4cb02` as a sensitivity analysis for substitutions
inside possessions. It was not taken from a paper and is not an established RAPM
standard. The implementation is `src/nba_impact/models/rapm_lineup_policy.py`.

---

## What lives here

Four related NBA analytics projects share this workspace. The clean NBA Impact
package is the active integration path; the older SPM/RAPM scripts remain useful
research references until intentionally migrated.

| Sub-project        | Purpose                                                                                                                                             | Entry points                                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **NBA Impact** (`src/nba_impact/`) | Versioned ingestion, canonical games/events/player-games/lineups, model registry, clean RAPM, and win probability. | `nba-impact`, `pyproject.toml`, `configs/ingest/` |
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
├── pyproject.toml                     ← clean NBA Impact package + CLI dependencies
├── configs/ingest/                    ← pinned, resumable source manifests
├── artifacts/                        ← model runs + DuckDB registry (generated)
│
├── src/
│   ├── nba_impact/                   ← active clean ingestion/model package
│   │   ├── data/                     ← canonical builders + QA/quarantine gates
│   │   └── models/                   ← clean RAPM and win-probability baselines
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
│   ├── lake/                         ← bronze/silver/manifests for NBA Impact
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

## Active NBA Impact path

The current clean pipeline is:

```text
pinned source manifests
  → data/lake/bronze
  → game_dim + event_states + player_games + lineup_stints
  → possessions + possession_lineup_segments (CDN orderNumber, not actionNumber)
  → chronological model evaluation
  → artifacts/models + artifacts/registry/nba_impact.duckdb
```

Current validated scope is 2023–24 through 2025–26. `lineup_stints.parquet`
emits only minute-reconciled games; quarantined games remain visible in
`lineup_game_quality.parquet`. Current RAPM must use `possessions.parquet` plus
the ordinal `possession_lineup_segments.parquet`; a clock-only lineup join is
not safe at same-clock substitutions. ESPN Net Points/WPA mirrors are research
benchmarks only because the upstream repository declares no license.
The 2023–24 player-game layer uses provenance-marked ESPN fallback rows where
the primary NBA box cache is absent. Quarantined games are repaired only through
immutable official BoxScoreTraditionalV3 JSON; never relax lineup minute gates
to hide incomplete boxes.

Win-probability research must use chronological seasons and post-action states.
External comparisons use the resumable `ingest-espn-win-probability` command,
then `benchmark-win-probability`, which scores ESPN and the local model only on
the same game/period/score/clock states and reports join coverage plus a paired
game bootstrap. Do not compare headline metrics from unmatched state samples.
The verified 2025–26 benchmark is `wp_espn_benchmark_v1_ca79cde82d`: 631,380
matched nonterminal plays across 1,313 games, 99.26% raw play-match coverage.
ESPN Brier is 0.14759 versus 0.14883 locally; the equal-game bootstrap interval
crosses zero. ESPN's clear advantage is at game start, so improve pregame team
and expected-lineup strength before adding more in-game complexity.
The first leakage-safe starter challenger is `wp_lineup_ablation_v1_7570ad01c9`.
Prior-season starter RAPM slightly improves tipoff Brier (0.21181 → 0.21057),
but its paired interval crosses zero and it remains materially behind ESPN
(0.20210). Keep it as a documented null/inconclusive result, not production.
Rolling margin plus rest is the stronger challenger: run
`wp_pregame_ablation_v3_30ab68d381` and
`wp_pregame_ablation_v3_cdbcea84ee` confirm starter-free rolling context against
Elo in both outer folds. Adding starter RAPM to context is unresolved and moves
the two fold point estimates in opposite directions. Freeze the smaller
starter-free Stage 0 model; keep the starter variant as research only.
Run `wp_stage1_v1_7e6c77d51a` rejects both the five-knot additive spline model
and the bounded histogram GBM on identical inputs: each loses Brier and AUC in
both folds. Do not retune them on the outer seasons. The next fixed parity test
was a 2×64 feed-forward proxy with five seeds for optimizer variance.
Run `wp_mlp_v1_7a7825bf09` rejects the available five-seed 64×64 feed-forward
MLP: pooled Brier is worse by 0.03171 and ranking/calibration both regress. It is
not a residual network because PyTorch is unavailable. Do not retune it on the
outer folds; build prefix-invariant causal sequence tokens before TCN/GRU/
transformer work.
WP is now frozen as good enough. Regular-season evidence is strong across two
outer folds. Playoff slices contain only 84–85 games; one has poor calibration
and neither identifies a context-versus-Elo gain. Do not fit a playoff-specific
model or continue neural work on the local Mac. Reopen WP only for new causal
inputs, historical event data, or a concrete product failure.

Possession/control must enter WP only at causal possession starts. Run
`wp_possession_start_v2_1db472e450` confirms
`wp_possession_start_v2_0a5d626234`: time-interacted control improves Brier and
late-game accuracy in both outer folds on the starter-free baseline. Each run
reconstructs score from completed prior possessions; raw CDN possession tags on
arbitrary action rows remain forbidden because they can reveal rebound/control
outcomes. For current
RAPM, run `rapm_lineup_policy_v1_23149bbb29` rejects
start-lineup attribution. Use terminal lineup as the provisional simple policy;
keep fractional segment exposure as a research challenger because its small
numerical advantage over terminal is not statistically resolved in one fold.

Run from the repository root with `uv run python -m nba_impact.cli …` so saved
scikit-learn artifacts use the locked runtime, or install the `nba-impact`
entry point from `pyproject.toml` in an equivalently locked environment.

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
