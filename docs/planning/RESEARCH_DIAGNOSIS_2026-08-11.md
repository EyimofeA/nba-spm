# Research diagnosis — accepted decisions

Status: **share with caveats**. GPT Pro produced the source diagnosis from a
static repository audit. It did not mount local data, run tests, or recompute any
rating. Numerical results from that report are not treated as fresh validation.
The live repository was checked before the decisions below were accepted.

## Highest-impact findings

1. **Name the product before improving it.** The provisional flagship is
   retrospective single-season impact. Current latent strength, forecasts,
   playoff value, rolling peaks, and win-probability credit are separate
   estimands and must never share an unlabeled leaderboard.
2. **Keep the production reference simple.** Terminal-lineup, zero-prior normal
   RAPM remains the reference. Annual SPM and prior-informed AIO are research
   challengers. The frozen annual SPM failed its 2025 check.
3. **Treat evidence governance as model work.** The next gain in credibility
   comes from estimand IDs, season-exposure records, artifact lineage, identical
   scoring rows, and whole-game uncertainty—not another broad feature search.
4. **Do not solve defense with a larger learner alone.** The next defensible
   defense study is a two-part scorer-defender model: opportunity or zone first,
   shot outcome second, with scorer, team, lineup, role, and season context.
   Current aggregate matchup factors remain observational and non-additive.
5. **Reserve new evidence.** Seasons 2022–24 are reused, 2025 is an inspected
   failure, and 2026 is partial/exposed. Season 2027 is reserved for one frozen
   annual confirmation after the 2026–27 season is complete.

## Live-repository corrections

- The report correctly found a rolling-peak eligibility mismatch. The contract
  required 1,000 offensive and defensive possessions in every constituent
  season. The code checked only the multi-year total. The old peak artifact is
  retired and must be rebuilt.
- The report's local coverage counts are stale because it could not mount this
  checkout. Current counts and run statuses must come from local manifests and
  tests, not from its static prose.
- Infrastructure prices and source-use restrictions can change. Preserve them
  as planning risks and obtain a source-rights review before any public raw-data
  distribution. This document is not legal advice.

## Four-stage roadmap

1. **Scientific control plane:** estimand registry, season exposure, artifact
   audit, preregistration, and explicit production/research/null states.
2. **Baseline honesty:** rebuild peak eligibility, then add game-cluster RAPM
   uncertainty and peak-selection uncertainty.
3. **First model challenger:** run the precision-aware prior experiment only
   after its training-only uncertainty inputs exist. Keep the zero-prior model
   when the practical gate fails.
4. **Research expansion:** identity/provenance spine, two-part defense, dynamic
   strength, playoff portability, then conserved event credit. UI, contracts,
   draft, injury, and RL work stay behind these gates.

## Immediate stopping rule

Do not start another AIO feature search until every pinned API rating maps to an
estimand, an evidence status, a data/config/code identity, and a documented
uncertainty state. Do not inspect Season 2027 before the frozen confirmation.
