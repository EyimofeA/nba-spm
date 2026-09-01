# Current SPM proposal

## Goal

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

## Outputs

The system should keep four outputs separate.

| Output | Meaning |
| --- | --- |
| Retrospective RAPM | Impact during a completed season |
| Retrospective SPM | Statistical reconstruction of completed-season impact |
| Current SPM | Statistical forecast of player strength at a dated cutoff |
| Current AIO | Current SPM combined with timestamped possession evidence |

Projected availability and minutes should remain separate from player
strength. Team forecasts need both, but they answer different questions.

## Rating timestamp

The first contract should produce ratings at the start of each Monday from
November 1 through April 1. Every source row must have a timestamp before the
cutoff. Monday games belong to the future scoring window.

Weekly cutoffs are frequent enough to test adaptation without treating daily
noise as a new player signal.

## Information state

### Statistical state

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

### Possession state

Reuse the current AIO sufficient-statistic solver. Weight every possession by
elapsed days, not by a hard season label. Start with the already selected
two-year half-life. Add the current season through the cutoff. Preserve the
terminal-lineup and score-conserving response contracts.

The statistical prior and possession state should have separate reliability.
General zero shrinkage must remain separate from trust in the current SPM
center.

## Statistical target

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

## Validation

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

## Small frozen search

The first decisive run should search only:

- box half-life in 365 and 730 days;
- possession half-life in 365 and 730 days;
- prior trust in 0, 0.5, and 1;
- ridge as the only statistical learner.

Select inside earlier seasons. Freeze one candidate before reporting later
seasons. Do not add rich tracking features until the dated Box15 baseline and
partial-season update work end to end.

## Population rules

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

## Promotion gate

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
