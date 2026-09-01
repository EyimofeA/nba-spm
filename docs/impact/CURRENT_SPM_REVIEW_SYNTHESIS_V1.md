# Current SPM review synthesis

## Decision

Close broad retrospective SPM feature research. Keep two retrospective heads:

- rich annual SPM for standalone statistical impact;
- nine-year normal Box15 for the prior to the one-season RAPM update.

The next research lane estimates current player strength at dated cutoffs. Its
first baseline uses time-decayed Box15 rates and time-decayed possession
evidence. Availability and minutes remain separate.

## Independent proposal before external review

The project proposal was frozen before asking external models. It specified:

- Monday cutoffs;
- source rows strictly before the cutoff;
- player-game Box15 inputs with day-grain decay;
- a separate day-decayed possession update;
- one-game-one-cutoff scoring;
- oracle-exposure and deployable-minutes lanes;
- separate reliability for the statistical center and the possession
  likelihood;
- explicit rookie and long-absence rules;
- a small ridge-only search before any rich tracking features.

The full pre-review proposal is `docs/impact/CURRENT_SPM_PROPOSAL_V1.md`.

## External review

Fable 5.1 independently recommended the same core design:

1. stop broad retrospective feature search;
2. test Box15 in the current lane because the existing preseason current AIO
   uses a rich predictive SPM center;
3. assign every game to its latest preceding Monday exactly once;
4. use dated Box15 rates with empirical-Bayes stabilization;
5. add dated possession evidence;
6. separate strength from projected availability and minutes;
7. treat rookie priors and absence-related uncertainty as first-class model
   components.

The review also identified useful remaining estimator checks:

- estimate total AIO precision beyond the existing grid boundary;
- derive player-specific prior precision from cross-fitted prior errors;
- inspect rich-versus-Box prior errors by exposure and source availability;
- inspect within-team correlation of prior errors;
- test post-update calibration or stacking rather than another broad prior
  feature search.

One Fable claim was incorrect. The target-window Box15 label weight uses the
square root of the smaller offensive or defensive possession total from the
multi-season RAPM target window. It does not use only rating-season exposure.

GPT Pro independently found two current-code problems during its audit:

- players without an SPM prior were filled with zero before the available
  prior population was recentered, which moved the missing players away from
  zero;
- the existing current-AIO model is a preseason model and does not apply dated
  in-season cutoffs, while its older 14-day weekly display ledger overlaps
  games.

The missing-prior recentering bug is fixed. The center now recenters only
players with observed priors and leaves missing players at zero. A regression
test covers the rule. A new seven-day partitioned ledger assigns each included
game once. The full dated AIO update remains the next implementation stage.

## Verified retrospective interpretation

Box15 is the current retrospective AIO prior winner, not the best standalone
SPM. The rich model predicts the stable RAPM label better but produces a worse
posterior after the single-season possession update. This reversal survives
target exclusion, lagging, direct prior blends, outcome censoring, factor
specialists, player-specific precision proxies, and a final combined stack.

The evidence does not establish one causal explanation. The most defensible
interpretation is:

- rich SPM contains useful ordering information;
- the one-season RAPM update does not use its scale cleanly;
- Box15 supplies a smaller and more homogeneous correction;
- remaining complementary signal is mainly defensive and too small to clear
  the practical gate.

The calibration audit supports this reading. An affine calibration of finished
game predictions lets rich AIO slightly outperform Box15, but that adjustment
uses observed outcomes and does not define a player rating. It diagnoses
posterior dispersion.

## Current-SPM foundation built

The dated input build now covers 2021 through 2026:

| Check | Result |
| --- | ---: |
| Regular-season games | 7,230 |
| Player-game rows | 154,234 |
| Minimum lineup-to-box join coverage | 99.674% |
| 2026 games | 1,230 |
| 2026 join coverage | 100% |
| Official-box fallback games | 2 |

The first dated statistical foundation produces 188,892 player-cutoff ratings
across 109 Monday cutoffs and 1,090 players. It evaluates 365-day and 730-day
Box15 decay. Every rate receives a 500-possession league prior before the
Box15 ridge mapping. The target mapping uses only nine-year normal RAPM windows
ending before the rating season.

This artifact is a data and model foundation. It has not passed downstream
game validation. It does not yet include partial-season RAPM, rookie priors,
absence-dependent precision, or minutes forecasts.

## Smallest decisive current experiment

Score each game from the latest preceding Monday. Compare four arms on
identical games:

| Arm | Statistical center | Possession evidence |
| --- | --- | --- |
| A0 | frozen preseason rich predictive SPM | frozen preseason current AIO |
| A1 | zero | dated, day-decayed through cutoff |
| A2 | frozen preseason Box15 | dated, day-decayed through cutoff |
| A3 | dated, stabilized Box15 | dated, day-decayed through cutoff |

Use Box15 and possession half-lives of 365 and 730 days. Use stabilization
strengths of 500 and 1,500 possessions. Keep ridge as the only statistical
learner. Search prior trust in 0, 0.5, and 1 only inside earlier seasons.

The primary score remains equal-season mean whole-game margin MSE. Report RMSE,
correlation, calibration slope, week-of-season cuts, exposure cuts, rookies,
long absences, and source coverage. Oracle exposure is the first lane. A
deployable minutes lane follows only after rating quality is established.

## Research that remains closed

- another broad retrospective feature ladder;
- another generic learner tournament;
- more target-window selection on the same seasons;
- offense-side rich prior additions;
- role clusters as direct value features;
- factor specialists as a replacement for total-points AIO;
- season-end tracking tables in a dated current model.

## Research that remains open

- dated current SPM and current AIO;
- direct precision estimation and wider heterogeneous precision;
- rookie and long-absence uncertainty;
- availability and minutes forecasts;
- game-level defensive opportunities with verified timestamps;
- post-update calibration learned only from earlier folds.
