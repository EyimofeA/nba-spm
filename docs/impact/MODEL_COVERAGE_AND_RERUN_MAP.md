# Model Coverage and Rerun Map

Date: 2026-08-18

This note is a code-and-contract audit. It does not recompute any rating.

## Current public and research scopes

| Output | Current scope | Status |
|---|---|---|
| Annual SPM, Normal RAPM, and AIO table | 2017--2024 | Published research table. It is not current through 2026. |
| Canonical current Normal RAPM | 2024--2026 | Validated possession/lineup run. It is separate from the annual AIO table. |
| Normal RAPM rolling peaks | 1997--2024 | Historical research endpoint. |
| Raw offense roles | 2017--2024 | Descriptive display only. |
| Raw defense roles | 2018--2024 | Descriptive display only. |
| Historical player projection backtests | 2019--2024 targets | Causal player-level folds; selection rows are marked as reused. |
| Player and team projection | 2027 from a 2026 origin | Research returning-minutes baseline, not a roster forecast. |

The first exportable player backtest target is 2019, from the first configured
walk-forward origin (2018). Target 2018 is not exported: it predates the
predeclared walk-forward evaluation window, and the selected aging method had
not yet been selected. Do not manufacture a 2018 row with a hindsight-selected
method. The site can show 2019--2024 backtests and the 2027 unscored forecast,
but must label selection rows as reused evidence. Season 2027 remains
unavailable for rating-model selection or confirmation.

## Current annual SPM

`configs/models/annual_spm_v1.json` freezes the annual SPM contract.

- Grain: one NBA regular-season player-season.
- Label: one-season, zero-prior, terminal-lineup Normal RAPM offense and
  defense; net is their sum.
- Training labels and features: 2014--2024.
- Descriptive evaluation: hold out one full season for each 2017--2024 fold.
- Final descriptive leaderboard: refit on all labeled seasons.
- Reliability weight: `sqrt(min(offensive possessions, defensive possessions))`.
  Possessions are a weight, not a player feature.
- Forbidden general features: on/off, plus-minus, team rating, games, minutes,
  age, experience, height, and listed position.
- Offense learner: histogram gradient boosting. The frozen addition is zTS.
- Defense learner: ridge. The frozen tracking block contains DFG, rim,
  deflections, charges, contested shots, and loose-ball fields. The newer
  documented annual challenger also adds eight scorer-adjusted matchup fields.
- Net model: no direct third learner. It is `offense + defense`.

The documented frozen baseline had mean held-out RMSE/correlation of
0.997/0.630 on offense, 0.960/0.496 on defense, and 1.386/0.599 on net.
These are historical descriptive-fold results, not a current-season forecast
claim. The first untouched 2025 diagnosis failed its gate, especially on
defense. Do not call the SPM current through 2025 or 2026.

## Roles

Roles do **not** enter the published SPM, Normal RAPM, or AIO rating.

- The role process excludes RAPM, SPM, on/off, team outcome, age, size,
  position, minutes, games, and efficiency outcomes.
- It selected six target-free offensive clusters and five target-free defensive
  clusters.
- A frozen test of defense role axes, affinities, and role-by-skill interactions
  lost to the matchup-only defense challenger on the two selection seasons.
- Therefore raw roles remain display, subgroup-diagnostic, and future-hypothesis
  inputs. Stabilized roles are also display only and are not exported publicly.

## Exact SPM-to-RAPM update

For a season, let `X` be the terminal-lineup possession matrix, `y` points per
possession, `b` the player and home coefficients, `P` the diagonal penalties,
and `c` the SPM coefficient center. The centered fit is

`argmin_b ||(y - mean(y)) - Xb||^2 + (b - c)' P (b - c)`.

The player penalties are 3000 for offense and 3000 for defense. The home term
penalty is 300. For a player, `c_off = SPM_off / 100` and
`c_def = -SPM_def / 100`, because the design's defense coefficient represents
points allowed. The two player blocks are re-centered to zero using their own
training possession weights. Missing prior values center at zero. The home
center is zero.

Normal RAPM uses exactly the same design and penalties with `c = 0`.

AIO is **not** `SPM + Normal RAPM`. It is one joint ridge fit around `c`:

`AIO = centered SPM + (centered-RAPM fit - centered SPM)`.

The displayed RAPM update is the second parenthesis. A rated season is excluded
from its own SPM label fit for the historical AIO priors. Later seasons can
still train an earlier historical rating, so that procedure is retrospective,
not a pre-season forecast.

## Projection method and limits

The current projection code first takes the filtered annual offense and defense
trajectory. It compares AR(1), linear-age, quadratic-age, spline-age,
spline-age-plus-minutes, and spline-age-plus-impact residual adjustments.

- Selection origins: 2018--2021.
- Diagnostic origins: 2022--2023.
- Selected method: spline-age-plus-minutes.
- Historical exported player backtests: 2019--2024 targets. They use each
  fold's state at the prior-season origin and only earlier transition rows in
  the fit. The 2019--2022 rows helped select the method, so they are not an
  independent validation sample; 2023--2024 are diagnostic reuse.
- Current forecast row: 2026 player state projected to 2027.
- Team net: five times the 2026-minute-weighted projected player net.
- Win pace: `clip(41 + 2.7 * team net, 0, 82)`.

It holds each 2026 roster and minutes distribution fixed. It has no trades,
rookies, injuries, schedule, lineup optimization, or availability model. It is
therefore a returning-minutes baseline, not a team forecast suitable for public
odds comparison. Historical team rows are intentionally not exported: using the
actual next-season roster or minutes would leak the answer, while an explicit
historical returning-roster assumption needs its own product contract.

## What the 2017--2026 possession and lineup backfill must rerun

Do this in order. Do not overwrite a frozen result in place.

1. Build and QA canonical `game_dim`, ordered event states, lineup stints,
   possessions, and possession-lineup segments for every 2017--2026 season.
   Require complete game coverage, five-man lineups, minute reconciliation,
   valid score progression, and a quarantine table.
2. Refit zero-prior terminal-lineup one-season Normal RAPM for 2017--2026.
   Publish a new versioned target panel; do not mix legacy 2017--2024 targets
   with canonical 2025--2026 targets in one label table.
3. Rebuild annual SPM targets and evaluate the unchanged frozen feature contract
   only where full feature panels exist. This requires complete player sheets,
   box, playtype, DFG/rim/hustle, and matchup coverage as applicable.
4. Recreate historical SPM priors with the same leakage rule, then run the
   fixed-center annual AIO comparison on exactly matched games. The prior is a
   research challenger until a predeclared confirmation succeeds.
5. Re-run raw role inference for 2025--2026 only after its behavioral feature
   inputs pass coverage QA. Keep it separate from ratings.
6. Rebuild the site snapshot only from the new pinned artifacts. Add explicit
   vintage labels so an annual 2017--2026 panel cannot be mistaken for the old
   2017--2024 table.
7. Do not extend 1997--2024 peaks from this backfill. It supplies 2017--2026,
   not the missing pre-2017 canonical inputs. Extend peaks only after all
   constituent seasons meet the same possession/lineup contract.

The 2026 annual feature panel is currently blocked because its exposure is only
81.8% of the preceding two-season median. Possession data alone does not make a
2026 AIO defensible. The published annual AIO should remain capped at 2024 until
the full feature and confirmation contracts pass.

## Evidence

- `AGENTS.md` (active-model and scope rules).
- `configs/models/annual_spm_v1.json` (annual SPM contract).
- `src/nba_impact/models/rapm.py` and
  `src/nba_impact/models/prior_informed_rapm.py` (centered-ridge mathematics).
- `src/nba_impact/models/annual_aio_ratings.py` and
  `src/nba_impact/models/annual_spm_priors.py` (annual AIO construction and
  retrospective prior rule).
- `docs/impact/SIDE_ROLES_AND_DEFENSE.md` and
  `configs/models/defense_role_challenger_v1.json` (roles result).
- `src/nba_impact/models/aging_projection.py` (projection method).
