# RAPM pipeline

START HERE if you are an agent: `../PROJECT.md` is the master state document
(state of the world, gate rules, conventions, ranked TODO stack, handoff protocol).
This README maps the scripts in `src/` only.

Ridge-regression RAPM from possession-level NBA data (MySQL `matchups` table,
cached locally). Everything runs from `src/`, all paths resolve via `src/paths.py`.

## Scripts (src/)

| script | role |
|---|---|
| `standard_rapm.py` | The engine. Fetch/cache possessions, build sparse design, solve penalized ridge, write player tables. All other scripts import from it. |
| `evaluate_rapm_models.py` | Ablation scorecard: spec variants × (game-grouped CV, chronological retrodiction), scored on game-margin RMSE/corr on a common non-garbage validation set. |
| `validate_ratings.py` | Rating-level validation: split-half reliability + next-season retrodiction from frozen ratings. This is the model-selection gate. |
| `sweep_lambdas.py` | Lambda sweep on rating-level metrics. (Verdict: λ barely matters; fixed 3000.) |
| `nonlinear_probe.py` | LightGBM vs linear ridge on next-season margins; tests a learned (nonlinear) rubberband. |
| `spm_v2.py` | Pooled-window SPM, CV alpha, heteroskedastic tau, residual lane. |
| `feature_eval.py` | RAPM×SPM splice harness — single entrypoint for gate eval. |
| `feature_foundry.py` | Foundry generation runner (single-process lock). |
| `build_human_viewer.py` | Static human + agent HTML viewers. |
| `log_run_greps.py` | Append status greps to `outputs/grep_digest.log`. |

See also `OPERATOR_GREPS.md` (status commands) and `features/program.md` (foundry rules).

## Data

- `data/possession_cache/matchups_<season>.parquet` — per-season snapshots of the DB.
  Delete a file to force a re-fetch for that season.

## Outputs

- `outputs/rapm_results/` — human-readable player tables (final product).
- `outputs/dump/` — raw coefficients.
- `outputs/diagnostics/standard_rapm/` — eval scorecards, validation CSVs.
  Files are stamped with spec + seasons + run id; safe to delete old ones.

## Decisions on record (2026-07-03, final)

- PRODUCTION CONFIG (updated after decay session): players + home effect,
  garbage time filtered, 250-day recency half-life (two-fold champion),
  symmetric λ = 3000, zero prior.
  Shipped: `outputs/rapm_results/final_20260703_hl250/` (26 windows, 1997–2024).
  The hl365 panel (`final_20260703/`) is retained for comparison.
- No rubberband column (kills forward prediction, r 0.63 → 0.42; endogeneity).
- No season dummies on 3-year windows (redundant with intercept).
- No stale priors: previous-window and iterated ("infinite") priors both LOSE
  to zero prior. Same-window SPM prior is the open next-project lane.
- Asymmetric lambdas help alone but don't stack with decay; not used.
- All periods run as 3-year rolling windows — never single full-history sweeps.
- Model-selection gate: next-season game-margin retrodiction + Gobert sign
  anchors + ESS check; split-half reliability for finalists. Possession-level
  RMSE is banned as a selection metric.
- Full experiment record: `outputs/diagnostics/experiments.csv` + `../IDEAS.md`
  + `../RESEARCH_LOG.md`.
