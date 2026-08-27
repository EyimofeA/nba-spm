# SPM research sources

This is a methods ledger, not a claim that every linked idea works. Each entry states what is reusable and what experiment would be needed before promotion.

## Current experiment

### PIPM and the BoxPIPM base

- Local source: `PIPM Player Finder through 2021 - Database.csv` supplied by the user.
- Reusable idea: start with a small box-score model, then add an on/off residual instead of asking one large model to rediscover both signals.
- Current test: the same BoxPIPM feature bank and the same full CourtSignal feature bank are trained once on CourtSignal five-year RAPM labels and once on Ryan Davis five-year RAPM labels. Every prior then receives the same one-season RAPM update and scores the same next-season games.
- Corrected formula note: the published `G/GS%^2` label is backwards. The feature is `(games started / games played)^2`.
- Caution: the supplied PIPM database includes playoff minutes. It is an external agreement check, not a training target.

### Metric comparison by Dunks & Threes

- Source: https://dunksandthrees.com/blog/metric-comparison
- Reusable idea: translate player ratings into team strength using next-season minutes, then predict next-season results.
- Caution: actual next-season minutes make that exercise a retrodiction, not a deployable forecast. CourtSignal keeps actual future lineups only as fixed exposure weights in the research diagnostic and compares models on identical games.

### NBA metric stability notebook

- Source: https://github.com/903124/NBA_data_benchmark/blob/master/NBA_benchmark.ipynb
- Reusable idea: report year-to-year R-squared separately for offense, defense and net.
- Caution: stability can reward a metric for repeating a bias. It is secondary to future-game error.

## Feature candidates

### Offensive rebounding

- Local note: user-supplied `offensive rebounding.txt`.
- Source discussion: https://apbr.org/metrics/viewtopic.php?t=8025
- Reusable ideas: own-miss rebounds, team offensive-rebound change while on court, and adjustment for role or position.
- Current implementation: annual offensive-rebound percentage is shrunk toward that season's league mean, centered within guard/forward/center groups, then pooled across the five-year window with offensive possessions. No previous or future season enters an annual stabilized value.
- Next test: replace the broad position buckets with soft offensive-role interactions, but only after the role model is frozen out of sample.

### Spacing

- Current implementation: `3PA per 100 * (1.5 * stabilized 3P% - league eFG%)`.
- Why this version: it gives credit for both credible volume and shot value above the league's average field-goal attempt. The previous formula subtracted league eFG only once, rather than per unit of volume.
- Next test: add teammate shot-quality change and defender displacement when tracking coverage is complete.

### Cross-product and KIDD-style interactions

- Source: https://squared2020.com/2017/10/13/developing-a-cross-product-analytic-kidd-score/
- Reusable idea: some value comes from combinations of skills rather than isolated rates.
- Gate: interactions must improve future-game error after the corresponding main effects are already present.

## Predictive extensions parked for later

### DARKO/DPM-style time decay

- User-supplied summary of Kostya Medvedovsky's DPM 2.0 thread: use every prior game with exponential day decay, keep box and on/off components, and let possession evidence change their relative weight.
- Why parked: it estimates current latent strength, while the active SPM/AIO experiment estimates retrospective impact. Mixing those targets would make the present bake-off uninterpretable.

### Mechanical curves and aging

- Source: https://squared2020.com/2020/10/30/approximating-curves-i-mechanical-process/
- Reusable idea: estimate smooth change over time rather than hard age buckets.
- Gate: blocked future-season validation, with age information available at prediction time only.

### Luck-adjusted ratings

- Source: https://fansided.com/2018/01/08/nylon-calculus-calculating-luck-adjusted-ratings/
- Reusable idea: separate repeatable shot-quality effects from opponent free-throw and three-point variance.
- Gate: define the expected-shot model on earlier seasons and show that the adjustment improves future games rather than only making historical ratings look smoother.

### Broader play-by-play decomposition

- Source: https://squared2020.com/2015/10/30/nba-data-science-breaking-down-nba-data/
- Reusable idea: keep possession outcomes decomposable so a total rating can be audited through shooting, turnovers, rebounding and free throws.
- Gate: each component must conserve back to the total scoring outcome and use the same lineup contract.

## Decision rule

No source is adopted because its public leaderboard looks plausible. A candidate must:

1. use only information available at the rating date;
2. score the same future games as the baseline;
3. improve equal-season paired game-margin MSE;
4. avoid a material loss in margin correlation or a specific exposure group;
5. reproduce from a pinned data, feature and model manifest.
