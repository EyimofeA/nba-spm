# Multi-season RAPM targets for SPM and AIO

## Decision

Keep normal RAPM targets. The nine-season Box15 AIO is the research leader in
this reused 2015--2026 comparison. It improves slightly over the seven- and
five-season versions. The age-adjusted targets lose for both Box15 and the rich
SPM before and after the same one-season RAPM update.

This result does not change the public model. The experiment compares several
targets on previously observed seasons.

## Question

Does target length or categorical lineup-age adjustment improve a statistical
prior and its downstream AIO?

The six target designs are:

| Target length | Target design |
|---:|---|
| 5 seasons | normal RAPM |
| 5 seasons | age-adjusted RAPM |
| 7 seasons | normal RAPM |
| 7 seasons | age-adjusted RAPM |
| 9 seasons | normal RAPM |
| 9 seasons | age-adjusted RAPM |

Each target design feeds two statistical models: Box15 and the audited rich
SPM. Each statistical prior also receives the same single-season RAPM update.

## Exact chronology

For rating season \(t\), one season of statistical inputs predicts the
multi-season RAPM target ending in \(t\):

\[
X_t \longrightarrow \widehat{\operatorname{RAPM}}_{t-H+1:t}.
\]

The model for season \(t\) trains only on feature-target pairs ending before
\(t\). It then predicts players from \(X_t\). The prior enters a RAPM fit that
uses only possessions from season \(t\). The resulting AIO predicts game
margins in season \(t+1\).

For example, the 2015 rating fold trains on pairs ending through 2014, predicts
the 2015 prior from 2015 statistics, updates that prior with 2015 possessions,
and scores 2016 games.

Box15 has historical features and covers all 12 rating folds from 2014 through
2025. Rich inputs begin in 2014. The rich model therefore starts with the 2015
rating and covers 11 folds. The experiment excludes an in-sample rich 2014
rating.

## Models

All multi-season targets use terminal-lineup possession RAPM with season-
centered points and penalties of 3000 offense, 3000 defense, and 300 home.

The age-adjusted fit adds separate offense and defense lineup counts for every
integer age from 19 through 43, with age 27 omitted. The age terms use a ridge
penalty of 100. The reported target adds the estimated age effect at each
player's observed age in the target-ending season. It does not rank every player
as if they were age 27.

The statistical learners and hyperparameters were fixed before this run:

| Prior | Offense | Defense |
|---|---|---|
| Box15 | ridge, alpha 300 | ridge, alpha 1000 |
| Rich SPM | elastic net, alpha 0.03, L1 ratio 0.1 | ridge, alpha 3000 |

The rich model applies fold-internal correlation pruning at 0.95. Both models
weight player-season labels by the square root of the smaller offensive or
defensive possession count. Possessions are weights, not predictors.

Every AIO uses one season of terminal-lineup possessions and fixed penalties
of 3000 offense, 4500 defense, and 300 home. The prior scale is one. This fixed
update isolates target design from posterior-strength tuning.

## Validation

The primary score is the equal-season mean of next-season whole-game margin
MSE. Each season receives equal weight. RMSE is the square root of that score.
Correlations and calibration slopes are secondary.

The game predictions use observed next-season lineups as exposure weights.
This evaluates rating quality conditional on who played. It is not a deployable
pregame forecast because it does not forecast availability or minutes.

The prior covers 86.8% to 90.3% of next-season lineup slots across folds.
Players without a statistical row receive the centered zero prior. Common-fold
Box15 and rich comparisons use the same player and game coverage.

Intervals use 2,000 paired whole-game bootstrap draws within each test season.
The comparison covers previously observed data and does not correct for testing
multiple target designs.

## Results

### Box15 AIO across all 12 folds

| Target | MSE | RMSE | Mean correlation | Mean calibration slope |
|---|---:|---:|---:|---:|
| 9-year normal | **183.885** | **13.560** | **0.370** | **0.852** |
| 7-year normal | 184.053 | 13.567 | 0.369 | 0.850 |
| 5-year normal | 184.229 | 13.573 | 0.368 | 0.848 |
| 9-year age-adjusted | 185.339 | 13.614 | 0.361 | 0.846 |
| 7-year age-adjusted | 185.428 | 13.617 | 0.361 | 0.843 |
| 5-year age-adjusted | 185.481 | 13.619 | 0.360 | 0.843 |
| Zero-prior RAPM | 188.369 | 13.725 | 0.337 | 0.911 |

The nine-year normal Box15 AIO lowers MSE by 0.167 against the seven-year
version. Its paired interval is [-0.223, -0.111]. It lowers MSE by 0.344 against
the five-year version, with interval [-0.451, -0.249]. These are small gains.

### Age adjustment

Negative values favor the normal target.

| Prior | Length | Stage | Normal minus age-adjusted MSE | 95% interval |
|---|---:|---|---:|---:|
| Box15 | 5 | AIO | -1.252 | [-1.415, -1.080] |
| Box15 | 7 | AIO | -1.376 | [-1.576, -1.185] |
| Box15 | 9 | AIO | -1.453 | [-1.682, -1.234] |
| Rich SPM | 5 | AIO | -1.419 | [-1.628, -1.206] |
| Rich SPM | 7 | AIO | -1.228 | [-1.450, -1.008] |
| Rich SPM | 9 | AIO | -1.095 | [-1.346, -0.842] |

The age-adjusted target loses in every tested combination. This rejects this
categorical age-target design under the current gate. It does not show that age
is irrelevant to forecasting or aging curves.

### Box15 versus rich SPM on the common 2016--2026 folds

| Target | Prior MSE: Box15 | Prior MSE: rich | AIO MSE: Box15 | AIO MSE: rich |
|---|---:|---:|---:|---:|
| 5-year normal | 195.567 | **191.676** | **187.013** | 188.707 |
| 7-year normal | 195.433 | **192.062** | **186.856** | 188.666 |
| 9-year normal | 195.301 | **192.290** | **186.697** | 188.599 |

The rich SPM is better alone. Box15 is better after the RAPM update. For the
nine-year target, Box15 minus rich MSE is +3.011 before the update and -1.902
after it. The paired intervals are [1.886, 4.032] and [-2.455, -1.316]. The rich
features explain the smooth target more closely, but they add less information
that complements current-season lineup evidence.

## Interpretation

Longer normal targets help Box15 modestly and consistently. The nine-year
target likely reduces target noise without making the prior less useful to the
single-season RAPM update. The categorical age adjustment changes the target in
a direction that hurts next-season game prediction.

The main result is not that nine years is universally optimal. The measured
advantage over seven years is only 0.006 RMSE. The result says nine-year normal
RAPM is the best research target among these six fixed designs on this reused
history.

## Artifacts

- Contract: `research/experiments/target_window_spm_aio_v1.yml`
- Runner: `research/run_target_window_spm_aio.py`
- Run: `artifacts/research/target_window_spm_aio/`
- Focused tests: `tests/test_target_window_spm_aio.py`
