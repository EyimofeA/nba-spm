# Documentation Index

This index separates current sources of truth from supporting reports and old
plans. When two documents conflict, use the root `ROADMAP.md`, the current model
contract, and the newest pinned artifact manifest in that order.

## Canonical entry points

Start at the repository root:

- [`AGENTS.md`](../AGENTS.md) — operating rules for agents;
- [`ROADMAP.md`](../ROADMAP.md) — current short queue;
- [`RESEARCH_LOG.md`](../RESEARCH_LOG.md) — append-only experiment record;
- [`README.md`](../README.md) — commands and repository overview.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — production, research, and legacy
  boundaries plus the current RAPM/SPM/AIO contracts;
- [`product/UI_GUIDE.md`](product/UI_GUIDE.md) — CourtSignal product, visual,
  interaction, and copy rules.

## Current contracts and reports

The list below contains the main entry points. Other files under `docs/impact/`
are supporting experiment reports. `RESEARCH_LOG.md` and immutable run manifests
hold the full experiment history.

- [`data/MATCHUP_SOURCE.md`](data/MATCHUP_SOURCE.md) — licensed player-matchup
  archive contract and local coverage;
- [`data/IDENTITY_DIMENSIONS.md`](data/IDENTITY_DIMENSIONS.md) — canonical
  player/team IDs, aliases, and observed player-team stints;
- [`data/ROLE_CONTEXT_SOURCE.md`](data/ROLE_CONTEXT_SOURCE.md) — pinned
  dribble-context inputs and their research-only contract;
- [`data/EVENT_SOURCE_COVERAGE.md`](data/EVENT_SOURCE_COVERAGE.md) — current
  event-source coverage and downstream source-selection guards;
- [`data/HISTORICAL_PLAYER_GAME_SOURCE_AUDIT.md`](data/HISTORICAL_PLAYER_GAME_SOURCE_AUDIT.md) —
  local 2017-23 player-box coverage, validation, provenance, and safe
  historical-backfill boundary;
- [`impact/CURRENT_FEATURE_QUALITY.md`](impact/CURRENT_FEATURE_QUALITY.md) —
  2025/2026 statistical-feature coverage, drift, and frozen-SPM confirmation;
- [`impact/CURRENT_2026_REFRESH.md`](impact/CURRENT_2026_REFRESH.md) — pinned
  current data, the 2014--2026 SPM null result, and RAPM input readiness;
- [`impact/UNIFIED_TIMELINE_2014_2026.md`](impact/UNIFIED_TIMELINE_2014_2026.md) —
  unified 2014--26 terminal-lineup AIO run and chronological SPM train-window comparison;
- [`impact/LEGACY_POSSESSION_MIGRATION.md`](impact/LEGACY_POSSESSION_MIGRATION.md) —
  strict historical cache migration, coverage, and terminal-lineup boundary;
- [`impact/HISTORICAL_V3_POSSESSIONS.md`](impact/HISTORICAL_V3_POSSESSIONS.md) —
  validated V3 possession-owner state machine, frozen gates, and historical
  candidate boundary;
- [`impact/HISTORICAL_V3_LINEUPS.md`](impact/HISTORICAL_V3_LINEUPS.md) — strict
  ordinal lineup reconstruction, pre-backfill coverage, and promotion gates;
- [`impact/HISTORICAL_MATCHED_RAPM.md`](impact/HISTORICAL_MATCHED_RAPM.md) —
  research-only matched V3 versus legacy terminal-lineup RAPM comparison for
  2017–23;
- [`impact/FACTOR_DECOMPOSITION.md`](impact/FACTOR_DECOMPOSITION.md) — factor
  feature families, public benchmark formulas, and AIO explanation contract;
- [`impact/FACTOR_TARGET_SPM.md`](impact/FACTOR_TARGET_SPM.md) — sparse
  statistical models of shooting, turnover, and rebounding RAPM plus the
  leave-one-player-out teammate-context ablation and full-feature ceiling;
- [`impact/HISTORICAL_FACTOR_SPECIALISTS_V2.md`](impact/HISTORICAL_FACTOR_SPECIALISTS_V2.md) —
  2014--26 shooting, shot-volume, turnover, and opponent-OREB specialist targets
  plus the combined Box15 residual test;
- [`impact/FIVE_YEAR_SPM_CONTEXT.md`](impact/FIVE_YEAR_SPM_CONTEXT.md) —
  chronological teammate-context residual test on the frozen five-year SPM;
- [`impact/AIO_DIAGNOSIS_AND_FEATURE_BLUEPRINT.md`](impact/AIO_DIAGNOSIS_AND_FEATURE_BLUEPRINT.md) —
  current AIO audit, public-model feature map, interpretation, aging-safe
  validation, and execution order;
- [`impact/SPM_OLD_VS_NEW.md`](impact/SPM_OLD_VS_NEW.md) — published annual SPM
  versus five-year research SPM, exact targets, training splits, selected
  feature additions, and the AIO prior update;
- [`impact/AIO_PRIOR_BAKEOFF_V1.md`](impact/AIO_PRIOR_BAKEOFF_V1.md) — controlled
  five-year SPM versus BoxPIPM-style prior comparison inside the same
  single-season RAPM update;
- [`impact/AIO_PRIOR_CANONICAL_FOLLOWUP_V1.md`](impact/AIO_PRIOR_CANONICAL_FOLLOWUP_V1.md) —
  canonical 2025--26 follow-up that selects BoxPIPM-style as the frozen
  research AIO prior;
- [`impact/FINAL_BOX_FEATURE_LADDER_V1.md`](impact/FINAL_BOX_FEATURE_LADDER_V1.md) —
  final cumulative feature ladder, downstream AIO selection, Box15
  interpretability, and corrected active-player leaderboard;
- [`impact/HISTORICAL_BOX15_EXTENSION_V1.md`](impact/HISTORICAL_BOX15_EXTENSION_V1.md) —
  focused Box15 follow-up, 1997--2026 RAPM coverage, 2001--2026 five-year
  SPM/AIO coverage, modern retention test, and local site-bundle boundary;
- [`impact/IMPACT_VALIDATION_SUITE_V1.md`](impact/IMPACT_VALIDATION_SUITE_V1.md) —
  ordered same-season, forward, reverse, midseason, and next-season tests for
  the frozen statistical priors, with weighted and equal-weight summaries;
- [`impact/IMPACT_VALIDATION_V2.md`](impact/IMPACT_VALIDATION_V2.md) —
  separate retrospective-impact and current-strength validation contracts,
  with chronological cutoffs, a pre-fit data gate, and frozen promotion rules;
- [`impact/COMBINED_VALIDATION_AND_INTERPRETABILITY_V1.md`](impact/COMBINED_VALIDATION_AND_INTERPRETABILITY_V1.md) —
  exact row-aligned validation, the Full SPM versus Box15 diagnosis, additive
  2026 AIO accounting, factor skills, trajectories, and the future dynamic-AIO
  design;
- [`impact/AIO_PRIOR_CALIBRATION_AND_FEATURE_ATLAS_V1.md`](impact/AIO_PRIOR_CALIBRATION_AND_FEATURE_ATLAS_V1.md) —
  frozen prior mean-versus-precision diagnosis, target-free feature stability
  atlas, and the clean learner-screen feasibility boundary;
- [`impact/ANNUAL_SPM_LEARNER_SCREEN_V1.md`](impact/ANNUAL_SPM_LEARNER_SCREEN_V1.md) —
  chronological annual-RAPM learner and feature-family screen, frozen later
  diagnostics, and the boundary before another AIO comparison;
- [`impact/ANNUAL_RICH_FORWARD_AIO_V1.md`](impact/ANNUAL_RICH_FORWARD_AIO_V1.md) —
  aligned next-season comparison of the frozen rich annual SPM and five-year
  Box15, alone and after precision-aware one-season RAPM updates;
- [`impact/TARGET_WINDOW_SPM_AIO_V1.md`](impact/TARGET_WINDOW_SPM_AIO_V1.md) —
  five-, seven-, and nine-season normal versus age-adjusted RAPM targets for
  annual Box15 and rich SPM priors under the same one-season RAPM update;
- [`impact/BOX15_DEFENSE_EXTENSION_9Y_V1.md`](impact/BOX15_DEFENSE_EXTENSION_9Y_V1.md) —
  defense-only residual feature screen for the nine-year Box15 prior and the
  downstream AIO decision;
- [`impact/AIO_PRIOR_COMPLEMENTARITY_V1.md`](impact/AIO_PRIOR_COMPLEMENTARITY_V1.md) —
  target-overlap, defensive-residual, prior-trust, and player-precision tests
  that retain Box15 as the AIO prior and rich SPM as the standalone impact head;
- [`impact/SPM_CONSENSUS_COMPLEMENTARITY_V1.md`](impact/SPM_CONSENSUS_COMPLEMENTARITY_V1.md) —
  fold-local consensus feature selection, disjoint future-reference error
  correlations, and the held-out likelihood check;
- [`impact/SPM_FINAL_PRIOR_STACK_V1.md`](impact/SPM_FINAL_PRIOR_STACK_V1.md) —
  final combined-prior test, promotion-gate result, and stop decision for broad
  retrospective SPM feature research;
- [`impact/EXTERNAL_ALL_IN_ONE_BENCHMARK_V2.md`](impact/EXTERNAL_ALL_IN_ONE_BENCHMARK_V2.md) —
  complete available-coverage, strict 2017--20, train-through-2023, full DARKO
  history, and dated DARKO timing comparisons against public all-in-one metrics;
- [`impact/WALK_FORWARD_SPM_PILOT_V1.md`](impact/WALK_FORWARD_SPM_PILOT_V1.md) —
  one-cutoff Full SPM versus BoxSPM future-game implementation check;
- [`impact/BOX_VS_TRACKING_SPM_PILOT_V1.md`](impact/BOX_VS_TRACKING_SPM_PILOT_V1.md) —
  one-fold corrected-input BoxSPM versus tracking-only SPM diagnostic;
- [`impact/PUBLIC_AIO_BENCHMARK_V1.md`](impact/PUBLIC_AIO_BENCHMARK_V1.md) —
  pairwise public-metric agreement and the oracle-minutes team-win test;
- [`impact/MODEL_REPLICATION_SPEC.md`](impact/MODEL_REPLICATION_SPEC.md) — exact
  RAPM, annual SPM, centered AIO, role, zTS, feature, window, and promotion
  contracts from the checked-in code and pinned artifacts;
- [`impact/BEHAVIOR_ROLES.md`](impact/BEHAVIOR_ROLES.md) — behavior-only role
  contract, stability gates, provisional interpretations, and AIO boundary;
- [`impact/SIDE_ROLES_AND_DEFENSE.md`](impact/SIDE_ROLES_AND_DEFENSE.md) —
  separate offense/defense roles, scorer-adjusted matchup features, and the
  fixed chronological defense comparison;
- [`impact/SHOT_DEFENSE_MODEL.md`](impact/SHOT_DEFENSE_MODEL.md) — exact
  shot/lineup panel, identification boundary, and null team-defense pilot;
- [`impact/DYNAMIC_TRAJECTORIES.md`](impact/DYNAMIC_TRAJECTORIES.md) — filtered
  time-decay baseline, forward evaluation, and latent-state boundary;
- [`impact/EXPECTED_POSSESSION_RAPM.md`](impact/EXPECTED_POSSESSION_RAPM.md) —
  causal possession-start context and residual-RAPM challenger contract;
- [`impact/RUBBERBAND_ADJUSTMENT.md`](impact/RUBBERBAND_ADJUSTMENT.md) —
  actual-clock, cross-fitted pre-possession score-margin curve and the boundary
  before adjusted player RAPM;
- [`modeling/PLAYBOOK.md`](modeling/PLAYBOOK.md) — NBA statistical modeling procedure;
- [`modeling/EVIDENCE_POLICY.md`](modeling/EVIDENCE_POLICY.md) — evidence and claim standards;
- [`planning/RESEARCH_DIAGNOSIS_2026-08-11.md`](planning/RESEARCH_DIAGNOSIS_2026-08-11.md) — accepted GPT Pro findings, live-repo corrections, and staged roadmap;
- [`win_probability/MODEL_CARD.md`](win_probability/MODEL_CARD.md) — frozen WP specification and results;
- [`win_probability/ARCHITECTURES.md`](win_probability/ARCHITECTURES.md) — paused WP architecture research.

## Planning references

- [`planning/LONG_TERM_GOALS.md`](planning/LONG_TERM_GOALS.md) — long horizon;
- [`planning/NBA_IMPACT_BUILD.md`](planning/NBA_IMPACT_BUILD.md) — earlier clean-build plan;
- [`planning/RESEARCH_BACKLOG.md`](planning/RESEARCH_BACKLOG.md) — earlier broad research queue.

These files provide background. The root `ROADMAP.md` is the only active queue.

## Historical references

- [`impact/ROADMAP.md`](impact/ROADMAP.md) — RAPM and all-in-one plan through
  2026-08-17; preserved as a historical snapshot;
- [`historical/TODO_2026-06-11.md`](historical/TODO_2026-06-11.md);
- [`historical/CRITIQUE_2026-06-11.md`](historical/CRITIQUE_2026-06-11.md).

Root `PROJECT.md` and `IDEAS.md` are legacy working documents. Preserve them,
but do not treat them as the active task queue.
