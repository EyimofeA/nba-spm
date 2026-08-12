# NBA Impact Lab — Long-Term Goals

## North star

Build one reproducible NBA research system that can explain player value, follow it
through time, forecast games and careers, assign event credit, and expose every
estimate with its data version and uncertainty.

## End products

1. **Impact suite:** current RAPM, offense/defense components, role-aware AIO,
   dynamic career ratings, peaks, and historical leaderboards.
2. **Game intelligence:** pregame forecasts, live win probability, lineup estimates,
   playoff translation, and injury-aware scenarios.
3. **Credit system:** Net Points-style conserved event value, WPA, WP-RAPM, and
   clearly separated descriptive versus causal claims.
4. **Decision layer:** contracts, surplus value, aging, draft projections, and
   roster-building research.
5. **Research platform:** versioned datasets, paper replications, model registry,
   API, and a fast website built only from promoted artifacts.

## Phase 1 — trustworthy current NBA core

Target: 4–6 focused weeks.

1. Finish event → lineup stint → possession conversion for 2024–25 and 2025–26.
2. Reconcile every game to scores, possessions, team minutes, and player minutes.
3. Produce current regular-season and playoff RAPM from the new canonical layer.
4. Upgrade win probability with pregame strength and possession state.
5. Serve versioned game, player, and model endpoints through a small API.

Exit gate: current data is complete, identity joins exceed 99.5% minute-weighted
coverage, and every published number resolves to a passing snapshot and model run.

## Phase 2 — all-in-one and roles

Target: following 6–10 weeks.

1. Build time-safe box-only, tracking-only, playtype, and role-feature baselines.
2. Compare ridge/GAM, boosted trees, and constrained stacking under identical folds.
3. Estimate offense and defense separately, then validate recombined team impact.
4. Publish role as context alongside value rather than adjusting difficult roles away.
5. Add game-cluster uncertainty and missing-tracking robustness.

Exit gate: a challenger improves out-of-time primary loss by at least 1%, has at
least 95% paired-bootstrap probability of improvement, clears subgroup guardrails,
and passes one untouched season.

## Phase 3 — time, playoffs, and decisions

Target: following 2–4 months.

1. Fit filtered and smoothed dynamic player states with aging and role transitions.
2. Create 1/3/5-year peaks, career trajectories, era views, and interpolation rules.
3. Separate playoff rotation effects from player per-possession translation.
4. Add dated injuries/availability, contracts, cap, draft, and roster-stint dimensions.
5. Evaluate player forecasts, team forecasts, and contract surplus independently.

Exit gate: every retrospective chart distinguishes information available then from
information learned later, and every deployable forecast uses cutoff-time inputs.

## Phase 4 — event credit and research frontier

Target: after the core survives production use.

1. Build a calibrated possession-aware win-probability model.
2. Allocate conserved event value with an explicit residual/unobserved bucket.
3. Compare Net Points, raw WPA-RAPM, and leverage-normalized WP-RAPM.
4. Replicate selected Sloan papers, including TD/Shapley valuation, before extending them.
5. Test RL or neural credit only against transparent accounting and predictive baselines.

Exit gate: credit sums to team value, baseline choices are visible, counterfactual
claims are tested separately, and complex methods add verified signal.

## Not now

- No frontend redesign before stable metric contracts and API payloads.
- No “perfect” composite chosen from one season or player-name sanity checks.
- No contracts/draft detour before canonical identities and player-team stints.
- No causal language for RAPM, Shapley allocation, or passive value-function fitting.
- No production label while current possessions or play-ins are incomplete.
