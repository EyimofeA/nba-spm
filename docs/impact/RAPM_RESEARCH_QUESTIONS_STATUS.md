# RAPM research questions: answers and project status

Updated 2026-08-25. This reconciles the GPT Pro critique and the research-team
brainstorm with the current CourtSignal repository. It is a decision ledger,
not a claim that every historical run is production evidence.

## Short answers

### What makes a good target?

A useful impact target needs all of the following:

1. a precise estimand: what quantity the coefficient means;
2. repeatable observations with a defensible exposure unit;
3. an attribution or conservation rule;
4. availability that does not condition away the impact being measured;
5. a chronological downstream validation test.

Points per possession satisfies these conditions for total retrospective impact.
Rim attempts, shot quality, turnovers, rebounds, and free throws are mechanism
targets. They can explain *how* value appears, but they do not become total
impact merely by being added together. A factor model needs a learned mapping
back to points and must beat direct points RAPM on future games.

### Which learner belongs where?

| Problem | Default | Reason |
|---|---|---|
| Pure lineup-adjusted impact | generalized ridge | additive player attribution with stable collinear estimates |
| Uncertainty reference | hierarchical Bayesian ridge or ridge covariance | makes shrinkage and uncertainty explicit |
| Statistical prior / SPM | ridge, HistGBM, or an interpretable additive model | nonlinear player-season features belong here |
| Current player strength | state-space model | evolves a latent value through time |
| Pair synergy | low-rank interaction model | avoids one coefficient for every pair |
| Style and similarity | embeddings or clustering | descriptive representation, not pure impact |

Player-ID trees are not a credible primary RAPM. Ridge remains the reference
until a challenger wins on identical possession inputs and chronological games.

### How should stints be weighted?

CourtSignal's current matrix has one row per possession, so every possession is
already one unit of evidence. There is no need to reweight an aggregated stint.
If stints are aggregated later, their loss should scale with possessions (or an
estimated outcome variance); weighting every stint equally is wrong. Arbitrary
short-stint deletion also throws away valid lineup-separation information. Test
robust weighting only if a documented data-quality mechanism motivates it.

### What does ridge's prior mean?

After the player blocks are centered, zero is league average, not replacement
level. In a Gaussian model, ridge can be written as a zero-mean prior, but the
prior variance depends on both the penalty and residual scale. An SPM-centered
RAPM is a different model: its cross-fitted SPM estimate is the prior mean.
Reusing an estimate from the same possessions as a prior double-counts evidence.

### How should collinearity be diagnosed?

The unregularized condition number is a warning, not a player-level diagnosis.
Use connectivity, exposure, coefficient covariance, split-window stability, and
a regularization-path plot. A player whose value changes sign under modest
penalty movement or teammate separation should not receive a precise hard rank.

### What is GCV, and why is it not the final selector?

Generalized cross-validation estimates training-row prediction error while
penalizing the model's effective degrees of freedom. It is fast because it does
not refit every held-out fold. That makes it useful for narrowing a penalty
search. It does not reproduce CourtSignal's real test: different future seasons,
temporal drift, repeated players, game-level aggregation, and paired game loss.
Our training-only GCV candidate landed near the existing penalty and did not
improve the later diagnostic seasons, which is exactly why GCV remains a
candidate generator rather than promotion evidence.

### Should we use time decay or longer windows?

They answer different questions. A fixed five-year window estimates average
retrospective impact across that window. Time decay estimates a more current
latent level. Weight normalization changes the effective amount of evidence,
so penalties must be reselected inside temporal training folds whenever decay
changes. Iteratively feeding the same RAPM back as a prior is not "infinite
information"; it repeatedly counts the same evidence. Use a state-space update
when new seasons arrive instead.

### Which context adjustments are safe?

Pregame variables such as home and rest can be nuisance controls. Live score
margin is partly caused by player quality, so conditioning on it changes the
estimand and can remove real impact. If a score-conditioned product is ever
useful, label it neutral-state or conditional impact and keep total impact
separate. Do not call team-home deviations altitude, travel, crowd, or referee
effects without a causal design.

### How do we compare two runs?

Use rolling-origin seasons, identical held-out games, game-level predictions,
equal-season aggregation, and paired whole-game loss differences. Primary loss
is game-margin RMSE; MAE, correlation, calibration, stability, and subgroup
failures are guardrails. Repeated random seeds are not independent evidence for
a deterministic ridge fit. Season 2027 remains untouched confirmation.

### How should playoffs be modeled?

A standalone playoff model is too noisy for most players. Start with a pooled
regular-season model plus a playoff deviation with heavy hierarchical shrinkage,
then test pre-series prediction on future playoff games. Rotation and opponent
selection must use only information available before the series.

### What should tracking data do?

Use tracking to estimate mechanisms such as shot quality, rim deterrence,
matchup difficulty, spacing, screen navigation, and pass quality. Feed stabilized
skills into SPM or a separately named mechanism model. Do not add downstream
tracking outcomes to pure RAPM if doing so conditions away the player's impact.

## Status against the two handoffs

### Completed or decisively tested

| Question | Current evidence |
|---|---|
| Unified 2014--26 lineup source | Canonical terminal-lineup adapter and QA are working; source transition is measured. |
| Start versus terminal versus fractional lineup | Tested on matched games. Terminal is the simple reference; fractional is parked sensitivity; start did not win. |
| Pure APM versus ridge | OLS/APM exists and demonstrates the expected extreme low-exposure variance; ridge remains reference. |
| Penalty search | Grid, Sobol, GCV, bivariate prior covariance, and empirical-Bayes families were tested. None cleared future-game gates over 3000/3000/300. |
| Sufficient statistics | Rolling five-year `X'X`, `X'y`, exposures, player order, and held-out game aggregates are stored. |
| Global home advantage | Improves held-out RMSE and remains in the reference. |
| Team-specific home deviations | A constrained 2014--26 test selected very strong shrinkage and then worsened 2024--26 RMSE. Keep one global home effect. |
| Garbage-time filtering | The current five-year test removed 9.84% of rows and worsened selection and diagnostics. Rejected for this estimand. |
| Score-margin/rubber-band control | The frozen five-year joint control worsened diagnostic RMSE. A later actual-clock study cross-fits lineup residuals by whole game and finds a stable, late-game-heavy score curve that transfers to 2026. A fixed 25-possession progress proxy reproduces the shape with 0.971 slope correlation. Refitting player RAPM on either adjusted target slightly worsened 2026 RMSE, so the context estimate is retained and the adjusted ratings are rejected. |
| Game-clock "fatigue" proxy | Numerically inert and is not observed fatigue. Rejected. |
| Ridge versus generic nonlinear models | Ridge beat completed LightGBM and bilinear lab challengers; Extra Trees lost its one costly fold and was stopped. |
| Elastic net | Tested in the statistical-model lane and lost; an earlier SGD RAPM probe also underfit. No reason to prioritize it. |
| Annual SPM | Cross-fitted annual offense/defense models exist. HistGBM offense plus ridge defense is the validated 2014--24 specification. |
| Annual prior-informed AIO | Cross-fitted SPM-centered RAPM exists; it is a research challenger, while zero-prior RAPM remains the reference. |
| SPM feature engineering | Empirical-Bayes rates, era-relative rates, box/playtype/tracking mechanisms, zTS, and feature-family ablations exist. |
| BPM/xRAPM comparison | Coverage and correlation comparisons exist; neither external metric is treated as ground truth. |
| Aging and fixed time decay | Year-over-year aging diagnostics and a fixed-decay trajectory exist; neither is the public retrospective RAPM. |
| Annual state-space model | A causal annual latent-state challenger exists and beats fixed decay on reused historical diagnostics. |
| RAPM uncertainty pilots | Analytic and whole-game bootstrap pilots exist for limited scopes. The expensive all-time peak bootstrap was stopped. |
| Role clustering | Stabilized descriptive offense/defense role maps exist; roles did not enter the selected SPM/AIO. |
| Previous-window and iterated priors | Stale and iterated RAPM priors lost. Repeatedly reusing the same evidence is rejected. |
| Win probability | The base WP model is frozen; this does not mean WP-RAPM or event credit is complete. |

### Partially complete

| Question | What exists | Missing decision |
|---|---|---|
| Feature selection | Grouped ablations and permutation diagnostics | versioned semantic feature registry, negative controls, and nested selection frequency |
| SPM target horizon | One- and three-year labels and long-window RAPM products exist | one frozen 1Y/3Y/5Y/latent-label comparison with identical features and downstream prior test |
| Uncertainty-aware SPM | possession reliability weights and RAPM uncertainty machinery exist | covariance-aware OFF/DEF label model and bounded inverse-variance experiment |
| Bivariate dynamics | separate offense/defense state work and a reference implementation exist | jointly estimated OFF/DEF innovation covariance, tests, and frozen forward gate |
| Factor decomposition | six/eight-factor feature design and direct OFF/DEF models exist | true shared-covariance multi-target lineup RAPM and learned reconstruction to points |
| Defensive mechanisms | matchup-adjusted features won historical diagnostics | clean prospective confirmation and complete current-season inputs |
| On-off spectrum | APM, RAPM, SPM, and prior-centered RAPM exist | RAPTOR-style courtmate-chain comparator and one coherent comparison table |
| Matchup impact | local exploratory Elo/matchup work and licensed pairs exist | frozen estimand, shot-quality target, and held-out validation; currently parked |
| Playoffs | WP has playoff audits and data contracts mention portability | a pooled/shrunk playoff impact experiment |
| Role change | annual soft roles and persistence exist | player-by-role impact with overlap/support checks |
| Shot-profile mechanisms | shot/playtype features exist for recent seasons | shot-profile RAPM heads and pre-2017 event-grade backfill |

### Not yet done

| Research question | Recommended first test |
|---|---|
| Regularization-path player diagnostics | trace selected high/medium/low-exposure players over a log penalty grid using stored matrices |
| Rest/back-to-back RAPM control | join schedule-derived pregame rest, then test global signed rest advantage on the same five-year folds |
| Season-varying home advantage | global home plus heavily pooled season deviations before adding more granular interactions |
| True joint four-/six-/eight-factor RAPM | begin with two correlated targets and prove held-out reconstruction gain before expanding |
| Low-rank teammate/opponent synergy | small rank path on residual outcomes with strict player main effects retained |
| Two- through five-player combination ratings | descriptive exposure-shrunk residual products; do not call them isolated causal effects |
| Event Points | publish a new additive event-credit contract; do not claim exact ESPN Net Points replication |
| WP-RAPM | fit only after defining whether the target is descriptive leverage credit or portable player value |
| Referee, coach, injury, foul-trouble effects | defer until the main model is stable and causal timing/source contracts exist |
| Spatial/spacing RAPM | start as stabilized tracking skills in SPM, not a kitchen-sink possession control |
| Exact constrained KKT solver | parked; first prove a material invariance or estimand problem in the centered ridge solver |
| Full public uncertainty/connectivity | complete scalable joint intervals and connectivity metadata before publishing hard ranks |

## Recommended order from here

1. Test schedule-derived rest/back-to-back as the next cheap, pregame nuisance.
2. Run the regularization-path diagnostic from the stored matrices.
3. Freeze the SPM target-horizon and uncertainty-weighting experiment.
4. Integrate a bivariate offense/defense state-space challenger.
5. Only then spend time on multi-target RAPM, low-rank synergy, and Event Points.

The production rule remains simple: a new idea does not enter RAPM because it
sounds realistic. It enters only when it improves future games on identical
rows without changing the stated estimand or failing a subgroup guardrail.
