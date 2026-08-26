# GPT Pro handoff: audit the RAPM closeout and specify the first SPM experiment

## Your role

Act as an independent statistical reviewer and research lead. Audit the work in
the repository; do not assume that summaries, labels such as `validated`, or
prior model judgments are correct. Separate:

1. results you recomputed;
2. repository claims supported by code and manifests;
3. plausible interpretations;
4. unresolved hypotheses.

The repository is <https://github.com/EyimofeA/nba-spm>. Audit branch
`codex/predictive-spm`. The RAPM Lab is deliberately localhost-only. Do not
recommend publishing an exploratory leaderboard merely because it exists.

This is primarily a read-only audit and design task. Inspect source, experiment
contracts, manifests, tests, and saved small outputs. Do not download large data
or launch a full RAPM/SPM fit. If a cheap static or unit-level check resolves a
question, run it. Label every numerical claim not recomputed by you as a
documented repository result.

## Decisions that are currently frozen

- Public retrospective estimand: single-season, possession-level RAPM.
- Stable rolling RAPM candidate: five seasons, terminal lineup, zero prior,
  regular season, penalties `3000 offense / 3000 defense / 300 home`.
- RAPM signs: offense positive is good; defense is converted from points allowed
  so positive defense is good; net equals offense plus defense.
- Season 2027 is untouched confirmation data. Do not use it.
- Score-state, age, coach, unit, multinomial, luck, factor, WP-credit, and
  teammate-event models are separate research estimands unless evidence supports
  promotion.
- Do not reopen a broad hyperparameter search. Name a small, predeclared test
  only if it could plausibly reverse a decision.

You may challenge any of these decisions, but show the exact evidence or defect
that warrants doing so.

## What RAPM is in this repository

One row is one possession outcome in points. The design has five `+1` offensive
player entries, five `+1` points-allowed defensive player entries, and a signed
home indicator. Game, season, period, possession, and lineup identifiers support
QA and grouped evaluation; they are not player features.

The ridge solution is equivalent to

\[
\hat\beta=(X^T X+P)^{-1}\{X^T(y-b)+sPc\},
\]

where `b` is mean possession scoring, `P` contains the side and home penalties,
`c` is a possible SPM center in coefficient units, and `s=0` for normal RAPM.
After fitting, offense and points-allowed defense blocks are exposure-weight
centered and the intercept is adjusted to preserve predictions. Published
ratings multiply coefficients by 100 and negate the points-allowed defense
coefficient.

Normal RAPM has no player-level box, tracking, role, age, position, minutes,
games, on/off, BPM, xRAPM, or SPM feature. The only model inputs are possession
points, the two lineups, and home status.

## What has been done

The principal implementation and result indexes are:

- `src/nba_impact/models/rapm.py`
- `research/experiments/`
- `research/rapm_lab/`
- `docs/impact/RAPM_FRONTIER_RESULTS_2026_08_25.md`
- `docs/impact/RUBBERBAND_ADJUSTMENT.md`
- `docs/impact/RAPM_RESEARCH_QUESTIONS_STATUS.md`
- `research/rapm_lab/audits/external_reproduction_audit_results.md`
- `research/rapm_lab/audits/wowy_raptor_saturation_audit_results.md`
- `RESEARCH_LOG.md`

Saved repository results say:

- Five years had the best equal-season next-year RMSE of the tested one-,
  three-, five-, and six-year target windows: `14.376`, `14.183`, `14.140`, and
  `14.177` respectively for the comparable 2020--26 evaluation set. Check the
  exact table and labels rather than relying on this sentence.
- Differential offense/defense penalties did not beat symmetric `3000/3000`
  for the frozen points target.
- Age-27 standardized ratings, coach controls, pair/trio/quartet/lineup-only
  models, multinomial possession points, garbage-time removal, and score-state
  controls failed their stated predictive gates.
- Actual lineup age can help a context-aware pregame prediction while the
  age-standardized player rating does not. That is not evidence for publishing
  an age-neutral player leaderboard.
- Luck adjustment and WP credit remain research challengers; their objectives
  differ from ordinary points RAPM.
- Six-sided factor, TS/turnover/rebound, teammate-event, shot-zone, and finish
  surfaces are useful decompositions but do not independently validate total
  RAPM because they reuse the same lineups and outcomes.

External agreement checks are strong but not ground truth:

- Ryan Davis normal RAPM: annual net Pearson/rank `.967/.962`, exact three-year
  `.980/.970`, exact five-year `.957/.948`.
- Current xRAPM 2024--26: `.897/.888`; the seasons are weighted differently.
- RAPTOR on/off at 1,000+ minutes: `.917/.912`; this is a different estimand.
- DARKO WOWY: `.574/.500`; this is also a different estimand and scope.
- CourtSignal's annual scale is about `1.391` times Ryan Davis's fitted scale.
  This leaves a cheap calibration question even though signs and joins look
  sound.
- DARKO's public season averages were exactly reconstructed from 432,853 public
  game rows (5,497 season aggregates). The official RAPTOR player CSV was also
  reproduced exactly. A RAPTOR-on/off-inspired proxy reached `.9658` Pearson
  correlation against its held-out regular-season team-stint target, but it is
  not an exact RAPTOR clone.

Four independent model audits (Grok 4.6, ox-alpha, Opus 5, and Kimi K3) voted to
freeze normal RAPM. Treat that as prior review coverage, not proof.

## Rubber-band problem to investigate

The empirical score-state association is real. It does not improve the player
rating under the current validation rule.

### Empirical curve

On 743,946 2024--26 regular-season possessions, lineup effects were estimated
out of fold by whole game. The residual was modeled against pre-possession
offense margin. The selected model uses eight six-minute actual-clock bins and
clips margin at 15. Its points-per-100 slope for each point of lead is:

| Minutes elapsed | Slope |
| ---: | ---: |
| 0--6 | +0.003 |
| 6--12 | -0.056 |
| 12--18 | -0.199 |
| 18--24 | -0.176 |
| 24--30 | -0.044 |
| 30--36 | -0.147 |
| 36--42 | -0.173 |
| 42--48 | -0.524 |

Thus a ten-point lead in the final six minutes is associated with `-5.24`
points per 100 relative to cross-fitted lineup expectation. A causal effort
interpretation is not claimed.

On reused 2026 data, possession-residual RMSE changed from `1.191592` to
`1.191427`; all 2,000 whole-game resamples favored the curve, but MSE improved
by only `0.028%`.

### Rating tests that failed

- Normal player RAPM: game-margin RMSE/correlation `15.473/.334`.
- Clock-adjusted target: `15.491/.344`; paired RMSE interval
  `[-.043,+.079]`.
- Possession-progress approximation: `15.499/.343`; interval
  `[-.036,+.087]`.
- Adjusted and normal descriptive ratings correlate `.991`; mean absolute net
  movement is `.239` points per 100.
- Adding the observed future score path back to predictions raises RMSE to
  `17.933` or worse. This is not forecast-valid because the path is endogenous.
- A 2014--26 J.E.-style exact-margin replication on 3,080,228 possessions finds
  the familiar asymmetric curve, but player-only RMSE worsens from `15.436` to
  `15.579`; paired interval `[+.043,+.244]`.
- Five-point signed buckets with separately tuned offense and defense penalties
  lose by `+.017` RMSE; interval `[-.021,+.055]`.
- Joint actual-clock columns select the strongest tested shrinkage and worsen
  neutral player-only RMSE by `+.031`.
- Signed buckets, clipped linear terms, and cubic splines all lose; their
  selected nuisance penalties sit at or near the shrink-to-zero boundary.

### Audit questions

1. Verify the pre-possession score reconstruction, offense-margin sign, per-100
   scaling, time buckets, centering, and use of only prior information.
2. Verify that the cross-fitting excludes the entire game, not merely a row.
3. Verify that score coefficients and player coefficients use compatible ridge
   scaling and that score-state recentering does not silently shift the player
   blocks.
4. Verify the target-adjustment sign and the neutral-context scoring rule.
5. Decide whether game-margin prediction from pregame lineups is the correct
   promotion gate for a retrospective player estimand. If not, specify a better
   estimand and gate without using post-treatment score paths.
6. Investigate the likely causal/statistical failure: player quality creates
   leads; strategy, bench substitution, and opponent response follow the lead.
   Score margin may be a mediator or collider, so controlling it can remove
   real impact even when the residual curve predicts possession scoring.
7. Decide whether the effect belongs in a separate live expected-points,
   coaching/strategy, or garbage-time context model instead of player RAPM.
8. If one unresolved rubber-band test is worth running, give its exact design,
   smallest decisive sample, split, metric, and stop rule. Otherwise say stop.

Do not answer “use splines”: splines, exact margin indicators, clipped linear
terms, signed five-point buckets, actual clock, possession progress, target
subtraction, and joint nuisance columns have already been tested.

## Data currently available locally

### Possessions and lineups

- Legacy terminal-lineup possession caches cover project seasons 1997 through
  2025 (`rapm/data/possession_cache/matchups_*.parquet`). The final cache label
  follows the project's season convention; verify the mapping before calling it
  the 2025--26 season.
- The long 1997--2026 research fit used 6,738,828 possessions and 35,532 games.
- The audited 2014--26 J.E. score-state panel used 3,080,228 possessions.
- The current canonical 2024--26 layer contains 3,941 game-dimension rows,
  1,950,498 event actions, player-game data for all 3,941 games, passed lineup
  data for 3,931 games, and possession data for 3,907 games.
- RAPM-ready regular-season games are 1,229/1,230 for 2024, 1,227/1,230 for
  2025, and 1,228/1,230 for 2026. Playoff coverage is 82/82, 84/84, and 85/85.
- The current research ledger has 743,946 regular-season possessions, 646,545
  shots, and 379,584 classified rebound opportunities. Eligible event mapping
  is 98.443%.

### Statistical and tracking features

- Base statistical panel: 6,942 player-seasons from 2014--26 and 97 features.
- Latest validated expanded feature artifact:
  `artifacts/research/current_feature_refresh/features/statistical_impact/statistical_features_v2_6bdb60a186`.
  It has 6,942 rows, 1,706 players, 289 total features, 192 engineered features,
  no duplicate keys, no infinite values, and no bounded-feature violations.
  It uses one-season windows.
- Its inputs include base box/play-by-play data, playtype features, official NBA
  defensive tracking, and official matchup defense. It does not include role,
  assist-quality, or player-skill feature artifacts.
- Current official defense sources include `LeagueDashPtDefend` DFG/rim-DFG
  data and `BoxScoreMatchupsV3` matchup data for all 1,230 2025--26 regular-season
  games. Inspect `docs/impact/CURRENT_FEATURE_QUALITY.md` and the source manifests.
- zTS is available and is selected in the current offense SPM feature list.
- Shot quality, empirical-Bayes rate stabilization, era-relative rates, Box
  Creation, Offensive Load, passer, spacing, tracking, and matchup-defense
  families are present.
- Role maps exist as descriptive artifacts, but roles are not SPM predictors in
  the pinned models.

### External comparison data

Local research references include Ryan Davis RAPM files, xRAPM panels,
Basketball Reference BPM, official FiveThirtyEight RAPTOR data, public DARKO
WOWY histories, and official NBA tracking/matchup sources. Some third-party
sources may be research-only. Audit source rights before proposing a public raw
data release.

## Current SPM contract and results

Annual SPM predicts the offense and defense sides of one-year, zero-prior,
terminal-lineup RAPM separately. It is not trained on one multi-year RAPM label.
Each row is a player-season. Training weight is

\[
\sqrt{\min(\text{offensive possessions},\text{defensive possessions})},
\]

which is not an input feature. Net equals predicted offense plus predicted
defense.

The present model uses a fixed 127-feature offense HistGradientBoosting model
and a 68-feature standardized ridge defense model. Offense uses learning rate
`.03`, 250 iterations, seven leaves, minimum 30 samples per leaf, and L2 `1`.
Defense uses ridge alpha `3000`.

Forbidden general predictive inputs are minutes, games, age, experience,
height, position, on/off, plus-minus, team ratings, BPM, xRAPM, and external
all-in-one ratings. Possession evidence belongs in the later lineup likelihood,
not as a statistical-prior feature.

The validated 2014--24 model produces public 2017--24 ratings. Documented
leave-one-season-out performance is:

| Side | Weighted RMSE | Correlation |
| --- | ---: | ---: |
| Offense | .9964 | .6303 |
| Defense | .9210 | .5526 |
| Net | 1.3556 | .6219 |

The later 2014--26 refresh used the same selected features and models. Its
documented ten-fold summary is `1.0108/.6317` offense, `.9783/.5090` defense,
and `1.4134/.6005` net. The 2025 defense correlation is `.3322`; 2026 is
`.3782`. Those seasons have already been inspected and cannot be treated as
untouched confirmation.

Important inconsistency: that refresh,
`single_season_spm_v1_47b3bd9b17`, consumed the older expanded feature artifact
`statistical_features_v2_b808fc1bf1`. The later official-defense artifact
`statistical_features_v2_6bdb60a186` passes current QA but has not been used in
a clean, predeclared SPM challenger. Several older docs still say 2026 lacks
DFG or matchup features; the latest superseding feature-quality record says the
data are now present. Reconcile manifests and dates instead of merging these
claims.

Historical leave-one-season-out SPM is suitable for retrospective ratings, but
it is optimistic for claims about predicting a future season because it trains
on seasons after the held-out season. Use chronological outer folds for model
selection and keep retrospective reconstruction results separately labeled.

The later AIO maps cross-fitted SPM offense to `c_off=SPM_offense/100` and
defense to `c_def=-SPM_defense/100`, exposure-centers each side, then fits the
same possession ridge with `s=1`. Roles and box features enter AIO only through
the SPM prior; they are not direct RAPM columns.

## The first SPM experiment I expect you to audit

The proposed first task is not a new model zoo. It is a source-transition and
defense-challenger experiment:

1. Build a field-level lineage table comparing the pinned public feature input,
   the 2014--26 refresh input, and `statistical_features_v2_6bdb60a186`.
2. On identical eligible player-season rows, verify season-by-season observed
   coverage, source identity, units, empirical-Bayes fallback rate, and
   distribution drift for the 68 selected defense features. Pay special
   attention to DFG, rim DFG, points saved, deflections/contests, and the eight
   matchup-defense fields.
3. Freeze the current defense ridge as control. Test one minimal challenger that
   changes only the repaired official-defense feature block; keep the offense
   model fixed.
4. Select using chronological outer folds ending no later than 2024. Report
   equal-season and possession-weighted RMSE, correlation, calibration slope,
   prediction spread, and low-/high-exposure slices. Do not use 2025 or 2026 to
   tune anything; show them only as already-reused failure diagnostics after the
   choice is frozen.
5. Only if the defense challenger transfers across multiple future-season folds,
   test whether its cross-fitted prior improves held-out game-margin prediction
   over zero-prior RAPM and the existing AIO on identical games.

Please decide whether this is the right first task. If not, replace it with one
smaller, more decisive experiment and explain why.

## Required deliverable

Return one report with these sections:

1. **Executive verdict:** `RAPM CORE FROZEN`, `RAPM BLOCKED`, or `RAPM REOPEN`;
   then the one first SPM action.
2. **Reproducibility audit:** defects and inconsistencies ranked P0--P3, with
   exact files and evidence.
3. **Rubber-band diagnosis:** mathematical, causal, implementation, and
   validation explanations for why the curve exists but the rating loses.
4. **Rubber-band decision:** stop, redirect to a context model, or run one
   precisely specified final test.
5. **Remaining RAPM research:** at most five ranked items; distinguish core
   estimator work from separate estimands.
6. **Data audit:** what the local data can support now, what is incomplete, and
   what cannot safely be published.
7. **SPM critique:** target, features, leakage risks, split design, weighting,
   calibration, offense/defense model choices, and the old/new defensive-source
   transition.
8. **Frozen first SPM experiment:** exact rows, inputs, exclusions, models,
   hyperparameters or search budget, folds, metrics, paired tests, promotion
   gate, artifacts, and stop rules.
9. **Four-week roadmap:** concrete dependencies, not a list of basketball ideas.
10. **Exactly five immediate actions.**

Search authoritative current statistical literature if it changes the audit,
and link sources. Prefer direct papers, official documentation, and original
method descriptions. Do not substitute citation volume for repository evidence.
