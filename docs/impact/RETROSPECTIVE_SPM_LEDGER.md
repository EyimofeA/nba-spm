# Retrospective SPM research ledger

## Decision

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

## Questions the research answered

The research separated four questions that earlier work had mixed together.

1. Which statistical model best reconstructs a stable retrospective RAPM
   target?
2. Which statistical estimate best centers a one-season RAPM update?
3. Which RAPM target horizon produces the best downstream game predictions?
4. Can rich statistical features add information that remains useful after
   the possession likelihood is observed?

The rich model wins the first question. Box15 wins the second and third. No
tested rich correction cleared the practical gate for the fourth.

## RAPM contracts

### Stable target RAPM

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

### One-season AIO update

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

## Box15 statistical prior

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

## Rich statistical impact model

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

## Validation scheme

### Chronology

For rating season `t`, the statistical mapping trains only on target windows
ending before `t`. The model predicts a prior from season `t` statistics. AIO
then updates that prior with season `t` possessions. The resulting player
ratings score games in season `t+1`.

Box15 covers rating seasons 2014--25 and outcome seasons 2015--26. Rich models
start later when their historical feature requirements prevent a clean first
fold. Direct comparisons use only common folds.

### Primary score

The primary score is equal-season mean next-season whole-game margin MSE. Each
outcome season receives equal weight. RMSE reports the square root in points per
game. Margin correlation and calibration slope are secondary.

The game predictions use observed next-season lineups as exposure weights.
This tests rating quality conditional on who played. It is not a deployable
pregame forecast because it does not predict availability or minutes.

Every comparison requires identical games and common player coverage. Missing
statistical priors receive the centered zero prior. Offense plus defense must
equal net at every stage.

### Uncertainty and gates

Paired intervals resample whole games within season. The main complementarity
and final-stack runs use 5,000 draws. A challenger must satisfy every gate:

- improve later-period RMSE by at least 0.05 points per game;
- keep its paired MSE interval below zero;
- lose no more than 0.01 mean margin correlation;
- win at least three of five later diagnostic seasons;
- avoid an isolated source-era or exposure failure.

The bootstrap measures sampling variation inside the fixed historical design.
It does not turn reused seasons into new confirmation evidence.

### Strict blocked-game sensitivity

The strict `impact_validation_v2` test removes each held game from the Box15
features and from the one-season RAPM likelihood. It only retains games whose
cached possession points reconcile exactly with official final scores. Box15
passes its four scored Gate A conditions on that subset, but the subset contains
43.2 percent of cache games and overrepresents larger margins. This result
supports further research. It does not establish a production win.

## Main results

### Target horizon

Across 12 Box15 folds:

| Target | AIO MSE | AIO RMSE |
| --- | ---: | ---: |
| Nine-year normal RAPM | **183.885** | **13.560** |
| Seven-year normal RAPM | 184.053 | 13.567 |
| Five-year normal RAPM | 184.229 | 13.573 |
| Zero-prior one-season RAPM | 188.369 | 13.725 |

The nine-year advantage over seven years is only 0.006 RMSE. It is a research
selection, not evidence that nine years is universally optimal.

### Rich model before and after RAPM

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

### External comparison

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

## Why Box15 currently wins

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

## Attempts to improve Box15

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

## Locked retrospective state

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
