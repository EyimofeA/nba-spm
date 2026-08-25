# CourtSignal external-reproduction audit

Read-only statistical and code audit. Do not edit files or rerun expensive RAPM
fits. Inspect:

- `research/rapm_lab/run_external_reproduction_benchmark.py`
- `tests/test_external_reproduction_benchmark.py`
- `research/rapm_lab/outputs/external_reproduction_benchmark/external_reproduction_benchmark_v1_cb04182717/run.json`
- `RESEARCH_LOG.md`, final external-reproduction entry

The saved run reports:

- Ryan Davis annual normal net RAPM, 2014-2023 pooled: n=5,217,
  Pearson=.9672, Spearman=.9615, CourtSignal-on-reference slope=1.3907.
- Ryan Davis exact six-year windows pooled: n=4,768, Pearson=.9262,
  Spearman=.8994, slope=1.3622.
- xRAPM 2024-2026: n=687, Pearson=.8966, Spearman=.8881,
  slope=.9983. xRAPM uses unequal season weights; CourtSignal does not.
- DARKO WOWY season averages, 2017-2026 pooled: net Pearson=.5737.
- RAPTOR raw on/off, 2014-2022 pooled: net Pearson=.4360,
  Spearman=.8290, slope=.0797.
- Reproduced local legacy AuPM, 2014-2024 pooled: net Pearson=.6336;
  the formula reproduces the stored column to 1.78e-15.
- PBPStats raw on-court net, 2024: Pearson=.7114.
- Reproduced game-level PM ridge, 2024-2026: player-rating Pearson=.6211
  with three-year possession RAPM. Lambda 10 was selected on 2024 -> 2025;
  refit 2024-2025 scores 15.4126 RMSE and .3336 correlation on reused 2026.

Audit priorities:

1. Find concrete P0/P1/P2 defects in parsing, season/window alignment, joins,
   sign conventions, possession/minute construction, correlation calculations,
   leakage, source characterization, or interpretation.
2. Check the surprising RAPTOR Pearson/Spearman divergence and the near-zero
   fitted game-level home coefficient. Say whether either implies a bug.
3. Decide whether the three headline RAPM agreement claims are safe:
   Ryan annual .967, Ryan six-year .926, xRAPM .897.
4. Check that labels `exact_key_scope`, `same_window_weight_mismatch`,
   `different_estimand`, and `invalid_direct` are assigned honestly.
5. Identify the smallest decisive fixes or tests. Do not propose a broad
   rewrite.

Return a terse report: verdict, findings ordered by severity with file/line
evidence, safe claims, unsafe claims, and at most five next actions. Explicitly
state when no P0 or P1 issue exists.
