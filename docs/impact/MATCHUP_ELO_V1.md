# Matchup Elo v1

## Status

**Research descriptive.** Do not add this output to RAPM, SPM, AIO, or the
public site. Do not describe it as primary-defender value or causal defense.

## Question

For a regular season, what scorer and listed-defender rate parameters best
describe the observed NBA matchup-assignment rows?

## Data and grain

One source row is one game, offensive player, and listed defender. It contains
assigned partial possessions and player points. The current v1 fit uses the
pinned licensed archive for 2018--25 and the official NBA Stats materialized
source for 2026.

The source is not shot-event tracking. A scorer can be assigned to several
defenders in the same game. It measures observed matchup exposure, not who
guarded a shot. See [Shot Defense](SHOT_DEFENSE_MODEL.md).

## Model

For scorer *i* and defender *j*:

\[
\log(PPP_{ij} / PPP_{league}) = o_i - d_j
\]

The fit is weighted by assigned partial possessions. It uses one L2 penalty for
all scorer and defender parameters. A three-possession league-rate smoother is
applied before the log transform. Each side is exposure-weight centered after
fitting.

The model reports an Elo-scale display value:

\[
E = 1500 + (400 / \ln 10) \times coefficient
\]

Thus a 400-point offense-minus-defense gap equals a tenfold **modelled** rate
ratio. This is an Elo *scale*, not sequential Elo. No game result, date order,
or K-factor update is used.

## Outputs

- `offense_elo`: scorer matchup rate parameter; 1500 is season average.
- `defense_elo`: listed-defender rate suppression parameter; 1500 is season
  average and higher is better.
- `net_elo`: offense plus defense, centered at zero.
- scorer and defender matchup exposures.

## Checks

- non-null player, defender, game, points, and exposure;
- no negative points or exposure;
- no duplicate game/scorer/defender source row;
- deterministic output under source-row reordering;
- weighted offensive and defensive scores center at 1500 separately;
- synthetic stronger scorer and stronger defender recover the correct order.

## Next decision

Inspect the fit against annual RAPM and the existing scorer-adjusted matchup
factors. A correlation is not a promotion result. Any predictive challenger
must use a predeclared chronological split and identical player-season rows.
