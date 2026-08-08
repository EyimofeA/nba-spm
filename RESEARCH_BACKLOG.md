# NBA Impact Lab — Research Backlog

Updated: 2026-08-08

This is the active planning page for the NBA project. `TODO.md`, `IDEAS.md`,
`PROJECT.md`, existing outputs, and earlier critiques are historical evidence and
hypothesis sources. Their claims are not considered independently validated.

## Working rule

Build one trustworthy data and evaluation spine, then attach many models to it.
Production stays deliberately small. Research keeps the interesting variants,
replications, diagnostics, and failed experiments.

Every promoted result must state:

- estimand: descriptive impact, current ability, forecast, event credit, leverage,
  or counterfactual value;
- information cutoff and dataset snapshot;
- lineup assumption: observed, estimated at the cutoff, or counterfactual;
- model/configuration version;
- chronological validation result and baselines;
- uncertainty, coverage, and known identification limits.

## What we are doing now

### P0 — Trustworthy foundation

1. Freeze and checksum the usable 1997–2024 possession snapshot and current
   outputs. Label every existing rating `stale / unverified`.
2. Repair the canonical data layer: games, game type, teams, players, aliases,
   player-team stints, rosters, possessions, events, and source manifests.
3. Remove the empty 2025 possession cache from the ingest path and acquire
   complete 2024-25 and 2025-26 event data.
4. Reimplement simple zero-prior RAPM independently and reproduce its mechanics.
5. Establish nested chronological folds and lock new 2025–26 data as the final
   one-shot confirmation set.
6. Add a DuckDB model registry containing the data snapshot, configuration,
   metrics, artifact location, intended use, and status of every run.

### P1 — First useful publications

1. Production RAPM: one simple current model with offense, defense, net,
   possessions, uncertainty, and freshness.
2. Research RAPM comparison: 1Y/2Y/3Y/5Y, uniform versus decay, penalty variants,
   and prior variants under identical folds.
3. Peak boards: 2Y–5Y peaks selected on one common net window; offense and defense
   displayed from that same window.
4. Continuous role profiles for 2014+ using touches, passing, playtype, shooting,
   hustle, and rim-defense aggregates.
5. A table-first publication schema and API contract. No new frontend until the
   values and definitions stabilize.

### P2 — Event valuation and playoff translation

1. Transparent event ledger / Net Points baseline.
2. Expected-possession-value credit model.
3. Six-factor and multi-target RAPM.
4. State-only win probability, then cross-fitted WP-RAPM.
5. Regular-season-to-playoff translation study using forecasted rotations.

### Later, not forgotten

- injury incidence, playing-through-injury effects, return-to-play curves;
- contracts, salary-cap value, surplus value, aging and availability risk;
- draft and pre-NBA translation from college, G League, and international data;
- lineup optimization, synergy, role portability, and roster-fit simulations;
- replication of selected Sloan and peer-reviewed papers;
- richer RL / temporal-difference and Shapley attribution;
- causal designs around trades, injuries, rule changes, and role shocks.

## Product map

| Product | Question | Primary output | Promotion target |
|---|---|---|---|
| RAPM Current | What happened after lineup adjustment? | Off/def/net per 100 | Simple production model |
| AIO Current | What is the best current value estimate? | Off/def/net plus uncertainty | Production challenger |
| AIO Forecast | What should we expect after the cutoff? | Future impact using estimated minutes | Separate forecast product |
| Dynamic AIO | How did latent ability change? | Filtered and smoothed trajectory | Research, then product |
| Event Credit | Who receives credit for observed events? | Conserved point ledger | Descriptive product |
| WP-RAPM | Who changed winning odds? | Raw and leverage-normalized impact | Separate research metric |
| Role & Portability | How does value change with deployment? | Role vector and response surface | Research product |
| Playoff Translation | What survives playoff environments? | Shrunk player/role translation | Forecasting research |

## Net Points / credit assignment track

ESPN's Net Points is an event-accounting system, not RAPM. It allocates credit and
blame for shots, passes, spacing estimates, rebounds, turnovers, fouls, and defense,
and approximately reconciles player totals to team scoring margin. The published
version is explicitly descriptive rather than predictive.

Our implementation should have three layers:

1. **Conservation ledger:** every event creates a known amount of team value and
   player credits sum exactly to it. Residual/unobserved credit must remain visible.
2. **Opportunity model:** estimate the difficulty and expected value of the action
   or state before observing its result.
3. **Attribution model:** allocate the realized change among shooter, passer,
   rebounder, lineup spacers, defenders, and an explicit residual.

Required comparisons:

- fixed transparent weights;
- learned expected-possession-value changes;
- Shapley/coalition attribution with declared replacement baseline;
- hybrid event weights plus lineup residual;
- RAPM and AIO as separate benchmarks.

Validation is not just correlation with RAPM. Require accounting conservation,
out-of-time action/state calibration, split-sample player stability, incremental
game/season prediction, and reviewed game clips or hand-coded events for face
validity. A decomposition that does not reconcile to its total is not published as
accounting.

Sources: [ESPN Net Points explainer](https://www.espnanalytics.com/net-pts-explainer),
[NBA Net Points](https://www.espnanalytics.com/nba-net-pts), and
[Piece of Harmony](https://ldeano.substack.com/p/piece-of-harmony).

## Playoff predictiveness track

Do not begin with a separate noisy "playoff RAPM." Separate rotation effects from
per-possession translation:

1. Regular-season ratings + actual playoff lineups: retrospective upper-bound
   lineup test.
2. Regular-season ratings + cutoff-time projected playoff rotations: deployable
   forecast baseline.
3. Add global playoff environment and team/style interactions.
4. Add heavily shrunk player-by-playoff deviations.
5. Add role/context interactions: creation burden, spacing, transition dependence,
   rim pressure, switchability, size, and opponent quality.

Report the difference between:

- value gained from tighter minutes distributions;
- value gained or lost per possession;
- opponent/matchup selection;
- uncertainty caused by small postseason samples.

The primary gate is future playoff game margin and win probability using only
information available before each series. Player-level playoff coefficients are a
secondary descriptive output.

## Offense/defense model question

Keep offense and defense because they answer real basketball questions, but do not
select models using component aesthetics.

- RAPM: use one lineup design with separate offensive and defensive player columns
  against net possession outcome, plus a net-only comparator.
- AIO: train offense and defense component models separately when their features
  and noise differ, then recombine them into net lineup/team predictions.
- Selection: score offense, defense, and net separately, with net held-out team-game
  prediction as the primary product gate.
- Calibration: the recombined components must reproduce the net scale; a model
  cannot win because its offense looks plausible while its defense is uncalibrated.

## RL / Shapley replication track

Replication target:
[Deep Reinforcement Learning for NBA Player Valuation](https://www.sloansportsconference.com/research-papers/deep-reinforcement-learning-for-nba-player-valuation-a-temporal-difference-approach-with-shapley-attribution).

The paper combines a distributional temporal-difference value network, player
embeddings, replacement-player coalitions, neural Shapley approximation, event
credit, and pair synergies. It is interesting enough to replicate, but not a trusted
benchmark yet.

Replication ladder:

1. State-only expected final margin and win-probability baselines.
2. Supervised next-state value model before calling anything reinforcement learning.
3. Temporal-difference value model with identical state and data.
4. Exact/Monte-Carlo Shapley on small lineups and states.
5. Neural Shapley approximation checked against those ground truths.
6. Event credit, player aggregation, synergy, and only then forecast comparison.

Audit questions:

- Does per-second discounting create the reported leverage effect by construction?
- Does the coalition result change materially with the replacement embedding?
- Are player embeddings being explained by Shapley after already learning outcome
  associations from the same players?
- Do attribution components reconcile without unexplained residuals?
- Are expected minutes and starters genuinely known at the prediction cutoff?
- Does synergy persist for team changers, new seasons, and matched contexts?
- Does the full model beat a comparably tuned supervised sequence model and RAPM?

Treat Shapley as a division of a model prediction, not proof of causal credit.

## Role research track

Publish observed impact and role-conditioned impact beside each other. Do not
regress role away from raw impact because high-value players partly create their own
responsibilities.

Priority questions:

1. How much predictive value do continuous roles add over age, minutes, box, and
   RAPM baselines?
2. Which players retain value through team changes and large role changes?
3. What is value above a role-matched replacement rather than generic replacement?
4. Can teammate burden relief be measured through changed creation, shot quality,
   turnover, rim-help, and rebounding responsibilities?
5. Which players provide lineup option value by covering several responsibilities?

## Principal's RAPM question inventory

These questions came from the principal's cross-sport RAPM note. They remain open
unless a new independent run closes them; statuses in the older `IDEAS.md` are not
carried forward as verdicts.

| Lane | Questions to preserve | First fair test |
|---|---|---|
| Alternative targets | Rim FGA deterrence, rim FG%, shot quality, four/six factors, multi-target RAPM | Event outcomes reconciled per possession; multi-output ridge versus independent heads |
| On/off–RAPM spectrum | What useful estimators exist between raw on/off and full RAPM? | Partial pooling by lineup/team, varying penalty strength, reliability/prediction frontier |
| Observation grain | Possession, minute, stint, aggregated identical lineup; short-stint filtering/downweighting | Same games and outcome, chronological folds, cluster-aware uncertainty |
| Priors | What information, center, strength, and player-specific uncertainty make a good prior? | Zero/minutes/box/role priors, all generated time-safely inside folds |
| Separability | Which players cannot be distinguished because they rarely separate? | Co-appearance graph, condition diagnostics, ridge paths, pairwise coefficient covariance |
| Penalties | Ridge versus elastic net/LASSO, asymmetric off/def shrinkage, hierarchical pooling | Proper solvers and nested chronological tuning; do not infer from one failed optimizer |
| Nonlinearity | Neural/player embeddings, boosted models, lineup interactions | Additive ridge baseline, comparable information, ablations, unseen-lineup test |
| Synergy | Pair/trio effects, double-big lineups, role complementarity, lineup optimization | Hierarchical interactions with together/apart support and future-season replication |
| Time | Convergence, luck adjustment, decay, aging, breaks, career trajectories, iterated priors | Dynamic latent-state model versus rolling/decay baselines |
| Context | Home, score state, garbage time, coaches, rest, travel, altitude, officials | Pre-specified covariates; exclude variables caused by the outcome being modeled |
| Player state | Fatigue, foul trouble, injury, changing minutes and roles | Time-varying state/role model with availability and return-to-play data |
| Game phase | Regular season versus playoffs; stint-entry/exit mechanisms | Forecasted rotations, shrunk phase interactions, start/end selection diagnostics |
| Spatial | Spacing, matchup, rim deterrence, off-ball defense | Defer true claims until event/matchup or optical data supports the estimand |

## Full NBA data roadmap

"All NBA data" is a catalog and ingestion program, not one scrape. Public access
cannot provide complete optical tracking, private matchup labels, medical records,
or every contract term. The attainable data lake is staged as follows:

| Tier | Data | Why first / later |
|---|---|---|
| 1 | Games, schedule, teams, players, aliases, rosters, transactions | Keys every other table |
| 2 | Event play-by-play, possessions, substitutions, lineups, box scores | RAPM, WP, factors, playoffs |
| 3 | Shots, playtypes, tracking aggregates, hustle, matchup aggregates | Roles, skills, event credit |
| 4 | Injuries/availability, rest/travel, officials, coaches | Forecasts and research |
| 5 | Salaries, contracts, cap, options, trade restrictions | Contract/surplus value |
| 6 | Draft, combine, college, G League, international statistics | Draft projection |
| 7 | Odds and projections | Forecast benchmarking only |
| Blocked/private | Optical XY tracking, detailed medical data, internal matchup labels | Do not pretend public proxies are equivalent |

Every snapshot records provider, source URL, retrieved time, coverage, license or
redistribution constraint, checksum, schema version, and join-quality report.

## Cloud and overnight execution

- R2 is object storage, not model-training compute. It will not make a laptop upload
  faster than the home connection.
- A scraper running in a cloud container or VM downloads source data over datacenter
  bandwidth and writes directly to R2, avoiding the slow home link.
- A local 02:00–07:00 lane remains useful for sources requiring a residential IP,
  browser session, or access pattern that cloud hosts reject.
- Jobs are season/source batches with checkpoints, retry/backoff, rate limits,
  checksums, and atomic completion. No batch restart should lose completed work.
- Canonical Parquet is stored in R2; DuckDB queries only the required partitions.
  Raw licensed data remains private; only permitted derived tables are published.

## Replication program

Each replication receives a short preregistration before code:

- exact claim and estimand;
- paper data versus our closest available data;
- faithful implementation versus necessary deviation;
- baselines and chronological split;
- success/failure criteria;
- final result including non-reproduction.

Initial queue:

1. Sill-style RAPM baseline and regularization study.
2. Win-probability player impact.
3. ESPN-style conserved event-credit baseline.
4. Distributional TD + Shapley paper, through the ladder above.
5. EPV/action valuation only when the required event/tracking state exists.

## Idea intake template

Add new ideas here without promoting them directly into the sprint:

```text
ID / title:
Question:
Estimand:
Minimum data:
Simplest baseline:
Main identification/leakage risk:
Chronological validation:
Product if it works:
Status: inbox | specified | ready | running | replicated | failed | blocked
```
