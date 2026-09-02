# CourtSignal SPM external review packet

This packet combines the review brief, retrospective research ledger, independent current-state proposal, and review synthesis. The original documents remain the canonical editable sources.

## Included documents

1. CourtSignal retrospective SPM closeout and current-SPM design review
2. Retrospective SPM research ledger
3. Current SPM proposal
4. Current SPM review synthesis

## CourtSignal retrospective SPM closeout and current-SPM design review

### Project briefing

CourtSignal is an NBA player-impact research repository. Python builds
possession RAPM, statistical plus-minus models, prior-informed RAPM, and static
derived-data artifacts. The public site must not fit models or expose raw NBA
events.

The project has two separate estimands:

- retrospective impact during a completed season;
- current latent strength at a dated information cutoff.

This review should close the retrospective statistical-prior work and design
the next current-state model. Treat every numerical result as repository
evidence that still requires code and artifact verification.

### Retrospective model

The current research prior is Box15. It uses separate offense and defense ridge
models over 15 per-100 box rates: points, assists, turnovers, steals, blocks,
offensive rebounds, defensive rebounds, personal fouls, fouls drawn,
free-throw attempts, free throws made, two-point attempts, two-point makes,
three-point attempts, and three-point makes.

One season of Box15 inputs predicts nine-season zero-prior normal RAPM ending
in the rating season. The target RAPM uses terminal lineups and penalties
3000/3000/300. The predicted offense and defense center a one-season RAPM fit
using only the rating season's possessions and penalties 3000/4500/300.

The richer standalone SPM uses an audited annual pool drawn from 175 completed
box, play-by-play, tracking, shooting, passing, rebounding, matchup, and source
availability fields. Offense uses elastic net with alpha 0.03 and L1 ratio 0.1.
Defense uses ridge with alpha 3000. Each fold prunes correlations above 0.95
inside its training data. Possession exposure supplies square-root minimum-side
label weights and does not enter as a feature.

### Retrospective validation

For rating season `t`, each statistical model trains only on feature-target
pairs ending before `t`. The model predicts a season-`t` prior. AIO updates it
with season-`t` possessions. The resulting rating scores games in `t+1`.

The primary metric is equal-season mean next-season whole-game margin MSE.
RMSE, correlation, and calibration slope are secondary. Observed next-season
lineups supply exposure weights, so this is an oracle-exposure rating test and
not a deployable lineup forecast. Every comparison scores identical games and
common player coverage. Paired intervals resample whole games within season.

The practical gate requires at least 0.05 points-per-game RMSE improvement, a
paired MSE interval below zero, no more than 0.01 correlation loss, at least
three of five later-season wins, and no material source-era or exposure failure.

### Main findings

Across 12 Box15 folds, nine-season normal Box15 AIO scores MSE 183.885 and RMSE
13.560. Seven-season and five-season versions score 184.053 and 184.229 MSE.
Zero-prior one-season RAPM scores 188.369.

On common folds, rich SPM is better alone:

- Box15 prior MSE 195.301;
- rich SPM prior MSE 192.290.

The ordering reverses after the same one-season RAPM update:

- Box15 AIO MSE 186.697;
- rich SPM AIO MSE 188.599.

The forward annual comparison also finds that rich SPM wins standalone by
4.440 MSE points while Box15 wins after precision-aware RAPM updates by 2.855.

The project tested cumulative feature families, top individual features,
stabilization, teammate context, roles, tracking-only SPM, four factor
specialists, target exclusion, full lagging, direct side-specific blends,
defense residuals, outcome censoring, separate prior trust, player-specific
precision, stability selection, and a final combined stack.

The final stack improves later RMSE by only 0.0216. It passes its interval,
correlation, and fold-win checks but fails the 0.05 practical gate. Later folds
assign zero weight to rich consensus offense and defense increments and full
weight to the target-excluded defense residual.

The best remaining defense fields are rebound conversion above expected,
workload-adjusted shot suppression, and rim-protection workload value. Their
retrospective gain remains too small.

The evidence shows that Box15 combines better with the one-season possession
likelihood. It does not prove why. A shared-error diagnostic was invalidated by
its common future reference because target permutation reproduced large error
correlations.

### Existing current-strength baseline

The repository already has a preseason current-strength baseline. It uses five
completed seasons of possession data, a selected two-year half-life, and a raw
next-season predictive SPM prior. On 2020--24 development folds, its RMSE is
13.7122 versus 13.7429 for decayed zero-prior RAPM and 13.7681 for five-year
zero-prior RAPM. Reused 2025 and 2026 diagnostics retain the same ordering.

This model freezes its preseason rating. It does not yet rebuild time-decayed
statistical features and possession evidence at each in-season cutoff. A
mechanical weekly ledger exists, but a fully updated weekly current SPM does
not.

### Requested independent review

Do not treat the project's current proposal as authoritative. Produce your own
design from the evidence above.

1. Audit the retrospective closeout. Identify contradictions, missing tests,
   and claims that exceed the evidence.
2. Explain the most likely reasons rich SPM improves standalone RAPM
   reconstruction but loses as a RAPM prior. Separate proven findings from
   hypotheses.
3. Propose feature engineering that could add information independent of the
   possession likelihood. Rank ideas by expected value, data feasibility, and
   leakage risk.
4. Propose RAPM changes that could improve the statistical-prior update without
   reopening a broad retrospective penalty search. Consider orthogonalized or
   residual targets, factorized likelihoods, reliability, partial pooling, and
   chronology.
5. Propose modeling methods beyond the tested ridge, elastic net, and simple
   residual blends. Prefer methods that can survive a small number of NBA
   seasons.
6. Design a current SPM in the broad style of DARKO/DPM and predictive EPM. It
   must update at dated in-season cutoffs and use only information available at
   each cutoff.
7. Separate player strength from availability and minutes. Include rookies,
   traded players, long absences, missing tracking sources, and early-season
   behavior.
8. Define a rolling validation scheme that avoids double counting games across
   overlapping forecast horizons. Include oracle-exposure and deployable
   minutes lanes.
9. Give the smallest decisive first experiment, the exact baselines, a compact
   hyperparameter grid, failure diagnostics, and stop conditions.
10. State which retrospective research should remain closed and which ideas
    genuinely require new data rather than more tuning.

Do not answer by recommending a later season or a larger generic feature
search. Use the available 1997--2026 history and label the limits of reused
evidence. Return equations or pseudocode where they make the design precise.

### Repository map

- Model rules: `AGENTS.md`
- Active state: `ROADMAP.md`
- Estimands: `research/estimands.yml`
- Season-use policy: `research/season_exposure.yml`
- Retrospective ledger: `docs/impact/RETROSPECTIVE_SPM_LEDGER.md`
- Target-window report: `docs/impact/TARGET_WINDOW_SPM_AIO_V1.md`
- Complementarity report: `docs/impact/AIO_PRIOR_COMPLEMENTARITY_V1.md`
- Final-stack report: `docs/impact/SPM_FINAL_PRIOR_STACK_V1.md`
- External benchmark: `docs/impact/EXTERNAL_ALL_IN_ONE_BENCHMARK_V2.md`
- Current baseline: `docs/impact/PREDICTIVE_SPM_AND_AIO_2026.md`
- Box15 config: `configs/models/box15_spm_v1.json`
- Predictive SPM: `src/nba_impact/models/predictive_spm.py`
- Current AIO: `src/nba_impact/models/predictive_current_aio.py`

---

## Retrospective SPM research ledger

### Decision

CourtSignal uses two statistical heads for retrospective research.

- `spm_impact` is the rich statistical model. It reconstructs stable RAPM best
  without possession evidence.
- `spm_prior` is Box15. It produces the best downstream rating after a
  single-season RAPM update.

The current research winner maps one season of Box15 inputs to nine-season
normal RAPM ending in the rating season. The predicted offense and defense
then center a one-season terminal-lineup RAPM fit with penalties of 3000 for
offense, 4500 for defense, and 300 for home court.

This model does not replace the public SPM or AIO. The public contracts retain
their pinned 2017--24 scope. All modern retrospective comparisons use reused
historical evidence.

### Questions the research answered

The research separated four questions that earlier work had mixed together.

1. Which statistical model best reconstructs a stable retrospective RAPM
   target?
2. Which statistical estimate best centers a one-season RAPM update?
3. Which RAPM target horizon produces the best downstream game predictions?
4. Can rich statistical features add information that remains useful after
   the possession likelihood is observed?

The rich model wins the first question. Box15 wins the second and third. No
tested rich correction cleared the practical gate for the fourth.

### RAPM contracts

#### Stable target RAPM

One row represents one completed possession. The design contains five
offensive player indicators, five defensive player indicators, and one home
indicator. The response is possession points. The model uses terminal lineups,
season-centered scoring environments, a zero player prior, and penalties of
3000 for offense, 3000 for defense, and 300 for home court.

Positive defense means points prevented. Offense plus defense equals net.

Target-window research compared five, seven, and nine seasons. It also tested
categorical age-adjusted versions of each target. Nine-season normal RAPM was
the best tested Box15 target. Every age-adjusted target lost after the same
one-season update.

#### One-season AIO update

For rating season `t`, the statistical model supplies a prior mean in RAPM
coefficient units. Only season `t` possessions enter the likelihood.

```text
beta_hat = inverse(X'X + P) * [X'(y - intercept) + trust * P * prior]
```

The current research comparison uses total penalties of 3000 for offense,
4500 for defense, and 300 for home. Later folds selected full prior trust for
Box15. The experiment also searched general shrinkage and prior trust
separately. It did not force every model to accept the same amount of prior
information.

This update is one ridge fit. AIO does not add separate SPM and RAPM
leaderboards.

### Box15 statistical prior

Box15 contains these 15 per-100 box rates:

| Feature | Meaning |
| --- | --- |
| `PTS_p100` | points scored |
| `AST_p100` | assists |
| `TOV_p100` | turnovers |
| `STL_p100` | steals |
| `BLK_p100` | blocks |
| `OREB_p100` | offensive rebounds |
| `DREB_p100` | defensive rebounds |
| `PF_p100` | personal fouls committed |
| `PFD_p100` | personal fouls drawn |
| `FTA_p100` | free-throw attempts |
| `FTM_p100` | free throws made |
| `FG2A_p100` | two-point attempts |
| `FG2M_p100` | two-point makes |
| `FG3A_p100` | three-point attempts |
| `FG3M_p100` | three-point makes |

These are 15 reported rates, not 15 algebraically independent columns. Points
equal free throws made plus twice two-point makes plus three times three-point
makes. Ridge can fit this redundant representation, and the frozen model keeps
it for continuity with the BoxPIPM-style baseline.

Offense and defense use separate ridge models. The current target-window run
uses offense alpha 300 and defense alpha 1000. Each rating fold trains only on
feature-target pairs ending before that rating season. The label weight equals
the square root of the smaller offensive or defensive possession count in the
multi-season RAPM target window. It does not use only season `t` exposure.
Possessions are not predictors.

The checked-in `box15_spm_v1` contract describes the earlier five-year model
and a 3000/3000 one-season update. The target-window experiment is the newer
research result. It uses annual Box15 inputs, a nine-season target, and a
3000/4500 update. Reports must state which contract they use.

### Rich statistical impact model

The rich annual model uses the completed statistical feature panel. It fits
offense with elastic net and defense with ridge. The current frozen settings
are alpha 0.03 and L1 ratio 0.1 for offense, and alpha 3000 for defense. Each
fold prunes correlations above 0.95 using only its training data.

The candidate pool includes these feature families:

- scoring volume and efficiency;
- stabilized true shooting, zTS, and shot-zone profiles;
- assisted, catch-and-shoot, pull-up, rim, midrange, and three-point context;
- Box Creation, Offensive Load, passer, spacing, and possession-use ratios;
- turnover type, ball-security, drive, travel, and offensive-foul rates;
- offensive and defensive rebounding chances, contests, and conversion;
- steals, recovered blocks, deflections, charges, loose balls, and fouls;
- defended-shot and rim workload;
- expected-versus-actual defended-shot outcomes;
- scorer-adjusted matchup volume and suppression fields;
- source-availability indicators.

The completed annual panel contains 175 finite candidate inputs. The older
frozen five-year rich model retained 127 offense and 68 defense fields. The
newer annual rich learner uses the audited non-shifted pool and selects a
fold-specific subset after correlation pruning. It therefore does not have one
fixed 127/68 list.

The full feature lineage lives in:

- `docs/impact/AIO_PRIOR_CALIBRATION_AND_FEATURE_ATLAS_V1.md`;
- `docs/impact/FULL_SPM_FEATURES_2014_2026.md`;
- `artifacts/research/spm_feature_atlas/spm_feature_atlas_v1_6949ad7b60`;
- `artifacts/research/complete_feature_coverage/semantically_complete_spm_features_v1_fdee01ec4e`.

Age, height, position, minutes, games, on/off, plus-minus, BPM, xRAPM, and
external ratings do not enter the retrospective statistical models as general
features. Possession exposure supplies label weights.

### Validation scheme

#### Chronology

For rating season `t`, the statistical mapping trains only on target windows
ending before `t`. The model predicts a prior from season `t` statistics. AIO
then updates that prior with season `t` possessions. The resulting player
ratings score games in season `t+1`.

Box15 covers rating seasons 2014--25 and outcome seasons 2015--26. Rich models
start later when their historical feature requirements prevent a clean first
fold. Direct comparisons use only common folds.

#### Primary score

The primary score is equal-season mean next-season whole-game margin MSE. Each
outcome season receives equal weight. RMSE reports the square root in points per
game. Margin correlation and calibration slope are secondary.

The game predictions use observed next-season lineups as exposure weights.
This tests rating quality conditional on who played. It is not a deployable
pregame forecast because it does not predict availability or minutes.

Every comparison requires identical games and common player coverage. Missing
statistical priors receive the centered zero prior. Offense plus defense must
equal net at every stage.

#### Uncertainty and gates

Paired intervals resample whole games within season. The main complementarity
and final-stack runs use 5,000 draws. A challenger must satisfy every gate:

- improve later-period RMSE by at least 0.05 points per game;
- keep its paired MSE interval below zero;
- lose no more than 0.01 mean margin correlation;
- win at least three of five later diagnostic seasons;
- avoid an isolated source-era or exposure failure.

The bootstrap measures sampling variation inside the fixed historical design.
It does not turn reused seasons into new confirmation evidence.

#### Strict blocked-game sensitivity

The strict `impact_validation_v2` test removes each held game from the Box15
features and from the one-season RAPM likelihood. It only retains games whose
cached possession points reconcile exactly with official final scores. Box15
passes its four scored Gate A conditions on that subset, but the subset contains
43.2 percent of cache games and overrepresents larger margins. This result
supports further research. It does not establish a production win.

### Main results

#### Target horizon

Across 12 Box15 folds:

| Target | AIO MSE | AIO RMSE |
| --- | ---: | ---: |
| Nine-year normal RAPM | **183.885** | **13.560** |
| Seven-year normal RAPM | 184.053 | 13.567 |
| Five-year normal RAPM | 184.229 | 13.573 |
| Zero-prior one-season RAPM | 188.369 | 13.725 |

The nine-year advantage over seven years is only 0.006 RMSE. It is a research
selection, not evidence that nine years is universally optimal.

#### Rich model before and after RAPM

On the common 2016--25 rating folds and 2017--26 outcome folds:

| Model | Standalone prior MSE | AIO MSE |
| --- | ---: | ---: |
| Nine-year Box15 | 195.301 | **186.697** |
| Nine-year rich SPM | **192.290** | 188.599 |

Rich SPM improves standalone MSE by 3.011. Box15 improves downstream AIO MSE
by 1.902. The paired intervals exclude zero in opposite directions.

The forward annual comparison reaches the same conclusion. Rich annual SPM
beats Box15 alone by 4.440 MSE points. Box15 wins by 2.855 after both models
receive chronologically selected precision-aware one-season RAPM updates.

#### External comparison

On the strict 2017--20 common external panel:

| Model | RMSE |
| --- | ---: |
| Defense-residual AIO | 13.3745 |
| Box15 AIO | 13.3814 |
| MAMBA | 13.4526 |
| Rich-SPM AIO | 13.4569 |

The defense-residual advantage over Box15 remains below the practical gate.
The available public metrics have different timing, coverage, and information
contracts. These values compare aligned games. They do not establish that one
metric dominates under every use.

### Why Box15 currently wins

The evidence supports a narrow conclusion. Box15 combines better with the
one-season possession update. It does not show that 15 box rates measure
basketball better than the rich features.

Several mechanisms remain plausible:

- the rich model reconstructs the smooth RAPM target but repeats information
  that the one-season RAPM update observes directly;
- rich outcome features contain season-specific noise that helps target fit but
  does not transfer after the possession update;
- the small Box15 model leaves a larger, cleaner correction for RAPM;
- source-era shifts and missing-source indicators make rich predictions harder
  to calibrate across seasons;
- correlated rich features spread signal across unstable substitutes.

The shared-error diagnostic did not prove the first mechanism. Prior and RAPM
errors subtract the same future reference, and a permuted reference reproduced
large correlations. The causal explanation remains unresolved.

The strongest verified diagnosis is calibration. A game-level affine forecast
diagnostic gives rich AIO RMSE `14.3296` and Box15 AIO RMSE `14.3660` across
the four scored 2023--26 outcomes. Rich AIO also retains the higher margin
correlation. The diagnostic calibrates finished game predictions, not player
ratings, so it cannot define the retrospective metric. It shows that the rich
model retains ordering signal while its posterior magnitude is too dispersed.

The tested precision grids also end at 6000 for offense and 9000 for defense.
Later folds select those boundaries. The evidence rejects the tested rich
priors, but it does not prove that the total AIO penalty is fully optimized.
Any final estimator audit should estimate precision directly or include one
predeclared extension beyond those boundaries.

### Attempts to improve Box15

The project tested these approaches without clearing the gate:

- eight cumulative rich feature families;
- the five strongest individual additions on each side;
- stabilized versus raw feature pairs;
- teammate-context residuals;
- role features and role interactions;
- tracking-only SPM;
- shooting, shot-volume, turnover, and opponent-OREB specialists;
- target-excluded and fully lagged targets;
- direct Box15-rich offense and defense blends;
- activity-only and outcome-augmented defense residuals;
- removal of defended-shot outcome fields;
- separate total ridge strength and prior trust;
- player-specific offense and defense precision;
- fold-local stability selection with noise and permutation controls;
- a final combination of consensus increments and the defense residual.

The final stack improves later RMSE by 0.0216 points per game. Its paired MSE
interval, correlation, and season-win checks pass. It fails the required 0.05
RMSE improvement. Later folds assign zero weight to both rich consensus
increments and full weight to the defense residual.

The best remaining defensive mechanisms are rebound conversion above expected,
workload-adjusted shot suppression, and rim-protection workload value. Their
gain is too small for the retrospective prior. They remain candidates for a
current-state model where recency may make them more useful.

### Locked retrospective state

- Keep rich SPM as `spm_impact`.
- Keep nine-year normal Box15 as the research `spm_prior`.
- Keep the one-season 3000/4500/300 RAPM update for this research lane.
- Keep the target-excluded defensive residual as a frozen challenger.
- Stop broad retrospective feature and target searches on the same seasons.
- Keep the public model and API unchanged.

Two estimator checks remain open without reopening feature research:

- confirm the precision result beyond the current grid boundary or estimate it
  without a finite grid;
- test whether cross-fitted prior-error variance can support materially wider
  player-specific precision than the current clipped proxy model.

The next model should estimate current latent strength with timestamped inputs.
It should not reopen the retrospective estimand under a new name.

---

## Current SPM proposal

### Goal

Estimate a player's current offensive and defensive strength at an explicit
date. The rating should update during the season and predict near-future games.
It should not describe the completed season.

The first implementation should extend the existing predictive SPM and
two-year-half-life current AIO. It should not create a new modeling framework.

The existing preseason current AIO uses the rich predictive SPM prior. The
retrospective work later found that Box15 combines better with possession
evidence. The first current experiment must therefore compare both frozen
preseason centers. It must not assume that the existing rich center remains the
best current-state center.

### Outputs

The system should keep four outputs separate.

| Output | Meaning |
| --- | --- |
| Retrospective RAPM | Impact during a completed season |
| Retrospective SPM | Statistical reconstruction of completed-season impact |
| Current SPM | Statistical forecast of player strength at a dated cutoff |
| Current AIO | Current SPM combined with timestamped possession evidence |

Projected availability and minutes should remain separate from player
strength. Team forecasts need both, but they answer different questions.

### Rating timestamp

The first contract should produce ratings at the start of each Monday from
November 1 through April 1. Every source row must have a timestamp before the
cutoff. Monday games belong to the future scoring window.

Weekly cutoffs are frequent enough to test adaptation without treating daily
noise as a new player signal.

### Information state

#### Statistical state

Build the first current SPM from player-game box data. Use every past game with
exponential day decay. Aggregate the 15 Box15 concepts through the cutoff and
retain attempts and opportunities so low-volume rates can shrink toward a
season-relative prior.

The first challenger may add a small number of dated mechanisms that have
reliable game-level lineage. Season-end tracking tables must not enter a
midseason cutoff unless the source can be reconstructed through that date.

Candidate dated additions should start with:

- shooting efficiency and attempt mix;
- free-throw rate;
- turnover and assist creation;
- offensive and defensive rebounding;
- steals, blocks, and fouls;
- team-relative on-court residual only as a separate component;
- prior current-AIO offense and defense;
- age as a forecast trajectory term, not a retrospective value feature.

The baseline should use ridge. A histogram gradient-boosting challenger may
follow after the chronology and missing-player rules pass. The project should
not start with a large learner search.

#### Possession state

Reuse the current AIO sufficient-statistic solver. Weight every possession by
elapsed days, not by a hard season label. Start with the already selected
two-year half-life. Add the current season through the cutoff. Preserve the
terminal-lineup and score-conserving response contracts.

The statistical prior and possession state should have separate reliability.
General zero shrinkage must remain separate from trust in the current SPM
center.

### Statistical target

The primary objective should be future game prediction after the current AIO
update. Current SPM target fit is diagnostic.

For each cutoff:

1. freeze the statistical and possession state;
2. estimate player offense, defense, and net;
3. score games until the next Monday cutoff;
4. assign every game to its latest preceding cutoff exactly once.

Fourteen-day horizons can remain a secondary horizon analysis. They must use a
separate partitioned ledger and cannot count the same game from two adjacent
cutoffs in one aggregate.

The standalone current SPM should also predict a future player-impact proxy.
Use a future 60-day or rest-of-season RAPM only as an auxiliary diagnostic. Its
noise and lineup dependence make it a poor sole training objective.

### Validation

Use rolling-origin season folds. Every feature, decay rate, calibration, and
model choice must train on earlier cutoffs and seasons.

The first comparison should contain:

1. frozen preseason predictive SPM;
2. frozen preseason current AIO;
3. frozen preseason Box15-predictive prior with dated possessions;
4. decayed Box15 current SPM;
5. decayed Box15 current AIO with partial-season possessions;
6. zero-prior decayed RAPM;
7. a last-rating persistence baseline.

Run two scoring lanes.

- Oracle-exposure scoring uses observed future lineups to isolate player-rating
  quality.
- Deployable scoring uses a separate availability and minutes model built only
  from information available at the cutoff.

The primary metric is equal-season mean nonoverlapping future-game margin MSE.
Report RMSE, correlation, calibration slope, and win probability calibration as
secondary results. Resample whole games within season for paired intervals.

Report performance by:

- early, middle, and late season;
- rookies and returning players;
- low and high prior exposure;
- traded players;
- players returning from long absences;
- offense and defense disagreement;
- source era.

### Small frozen search

The first decisive run should search only:

- box half-life in 365 and 730 days;
- possession half-life in 365 and 730 days;
- prior trust in 0, 0.5, and 1;
- ridge as the only statistical learner.

Select inside earlier seasons. Freeze one candidate before reporting later
seasons. Do not add rich tracking features until the dated Box15 baseline and
partial-season update work end to end.

### Population rules

- Returning players receive their dated statistical and possession history.
- Players with box history but no possession history receive the statistical
  center plus general shrinkage.
- Players with no NBA history receive an explicit rookie prior estimated from
  earlier rookies by debut age. Its variance must remain wider than the veteran
  prior. Do not silently assign the league-average veteran prior at full trust.
- Players absent from the current roster can retain a rating but receive zero
  projected minutes.

Long absences should preserve the player's mean while reducing prior precision
as days since the last game increase. Trades require no strength reset because
the state is player-level. Team and role context belong in diagnostics or the
minutes model unless a dated, past-only source supports them.

### Promotion gate

A current-state challenger must:

- improve later-season RMSE by at least 0.05 points per game;
- have a paired MSE interval below zero;
- lose no more than 0.01 margin correlation;
- improve at least three later season folds;
- avoid a material early-season, rookie, or source-era failure;
- reproduce every cutoff from stored hashes and sufficient statistics.

The first run remains research-only. It should produce a dated local rating
table and a validation report. It should not modify the retrospective ratings,
public API, or website.

---

## Current SPM review synthesis

### Decision

Close broad retrospective SPM feature research. Keep two retrospective heads:

- rich annual SPM for standalone statistical impact;
- nine-year normal Box15 for the prior to the one-season RAPM update.

The next research lane estimates current player strength at dated cutoffs. Its
first baseline uses time-decayed Box15 rates and time-decayed possession
evidence. Availability and minutes remain separate.

### Independent proposal before external review

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

### External review

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

### Verified retrospective interpretation

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

### Current-SPM foundation built

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

### Smallest decisive current experiment

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

### Research that remains closed

- another broad retrospective feature ladder;
- another generic learner tournament;
- more target-window selection on the same seasons;
- offense-side rich prior additions;
- role clusters as direct value features;
- factor specialists as a replacement for total-points AIO;
- season-end tracking tables in a dated current model.

### Research that remains open

- dated current SPM and current AIO;
- direct precision estimation and wider heterogeneous precision;
- rookie and long-absence uncertainty;
- availability and minutes forecasts;
- game-level defensive opportunities with verified timestamps;
- post-update calibration learned only from earlier folds.
