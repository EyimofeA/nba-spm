# Current possession quality gate

Intended use: regular-season 2023–24 through 2025–26 normal RAPM. Canonical
grain is one maximal same-offense action run; a child table preserves every
ordinal lineup segment inside the possession.

## Coverage

| Season | Source regular games | RAPM-ready | Coverage |
|---|---:|---:|---:|
| 2023–24 | 1,230 | 1,227 | 99.76% |
| 2024–25 | 1,230 | 1,226 | 99.67% |
| 2025–26 | 1,230 | 1,228 | 99.84% |
| Total | 3,690 | 3,681 | 99.76% |

Nine regular games are quarantined because exact lineup minutes or substitution
transitions do not reconcile. They are excluded, not guessed. The 25-game CDN
cutoff affects the 2025–26 playoffs, not this regular-season model.

## Passed checks

- 743,946 regular-season possessions across 3,681 games.
- Unique possession and lineup-segment IDs; zero null player lineup slots.
- Every offense team is the home or away team in its game.
- Per-game possession points equal official terminal scores.
- No negative possession points and no possession over seven points.
- Every segment-point total equals its parent possession total.
- Per-game possession counts range from 167 to 259, inside the frozen 150–300
  plausibility gate.

112,608 possessions contain multiple lineup segments. The active normal-RAPM
contract assigns each possession to its terminal lineup. Fractional exposure
remains parked as a research sensitivity by user direction.

## Frozen current baseline

Run `rapm_v0_01b5084f0a` fits zero-prior terminal-lineup ridge with penalties
3000/3000/300. It has 802 players, no missing names, and exact numerical
agreement with the previously validated frozen ratings. Start-versus-terminal
net ratings correlate 0.971; mean absolute difference is 0.358 points per 100,
which confirms that lineup policy is material enough to disclose.

The 2025–26 observed-lineup retrodiction has game-margin RMSE 15.473 and
correlation 0.334. Only 86.5% of test lineup slots refer to players seen in the
two earlier seasons, so this is a descriptive baseline, not a deployable
forecast or proof of player-level truth.
