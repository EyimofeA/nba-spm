# CourtSignal retrospective SPM closeout and current-SPM design review

## Project briefing

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

## Retrospective model

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

## Retrospective validation

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

## Main findings

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

## Existing current-strength baseline

The repository already has a preseason current-strength baseline. It uses five
completed seasons of possession data, a selected two-year half-life, and a raw
next-season predictive SPM prior. On 2020--24 development folds, its RMSE is
13.7122 versus 13.7429 for decayed zero-prior RAPM and 13.7681 for five-year
zero-prior RAPM. Reused 2025 and 2026 diagnostics retain the same ordering.

This model freezes its preseason rating. It does not yet rebuild time-decayed
statistical features and possession evidence at each in-season cutoff. A
mechanical weekly ledger exists, but a fully updated weekly current SPM does
not.

## Requested independent review

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

## Repository map

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
