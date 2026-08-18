# 2026 data and model refresh

Status: verified research refresh. Public promotion is rejected.

## What changed

- Pinned the 2025 and 2026 player sheets from `gabriel1200/player_sheets` at
  revision `a86cbe4c9cacff69906cb63600e76c558c83494e`.
- The 2026 sheet now has 582 unique player rows and full-season exposure.
- Rebuilt the one-season base feature panel for 2014--2026.
- Rebuilt playtype and player-skill features through 2026.
- Rebuilt defensive tracking through 2025. The source has no 2026 DFG or rim
  DFG rows.
- Downloaded the 10 targeted Gabriel team files for games that failed strict
  lineup QA. These files are raw repair inputs, not RAPM-ready rows.
- Refit the frozen single-season SPM on the 2014--2026 target panel.

## Data checks

| Check | Result |
| --- | ---: |
| 2025 player rows | 569 |
| 2026 player rows | 582 |
| Missing player IDs | 0 |
| Duplicate player-season keys | 0 |
| 2026 exposure / median 2024--2025 exposure | 1.005 |
| Base feature rows | 6,942 |
| Base features | 97 |
| Expanded feature rows | 6,942 |
| Expanded features | 300 |
| Invalid bounded feature values | 0 |

The expanded panel neutral-fills unavailable source fields. This is material
for current defense: matchup assignments end in 2024, and DFG/rim DFG end in
2025. Roles are not rating inputs.

## Frozen SPM design

- Target: one-season, terminal-lineup, zero-prior Normal RAPM.
- RAPM penalties: offense 3000, defense 3000, home 300.
- Training panel: 2014--2026.
- Evaluation: leave one complete season out, then train on every other season.
  This measures descriptive reconstruction. It is not a forecast test.
- Sample weight: square root of the smaller offensive or defensive possession
  count.
- Offense model: histogram gradient boosting with 127 fixed inputs.
- Defense model: ridge regression with 68 fixed inputs.
- Extra offense input: zTS.
- Extra defense inputs: DFG, rim DFG, hustle, and scorer-adjusted matchup
  aggregates.
- Excluded as general predictors: minutes, games, age, height, position,
  experience, team ratings, and on/off ratings.
- Role clusters and role coordinates are excluded.

The complete input list is stored in
`artifacts/models/single_season_spm/single_season_spm_v1_47b3bd9b17/run.json`.

## Result

The refresh did not improve the frozen public model on the same 2017--2024
folds.

| Component | Old RMSE | Refresh RMSE | Old correlation | Refresh correlation |
| --- | ---: | ---: | ---: | ---: |
| Offense | 0.9964 | 0.9971 | 0.6303 | 0.6312 |
| Defense | 0.9210 | 0.9267 | 0.5526 | 0.5475 |
| Net | 1.3556 | 1.3591 | 0.6219 | 0.6206 |

Current-season defense remained weak:

| Season | Offense correlation | Defense correlation | Net correlation |
| --- | ---: | ---: | ---: |
| 2025 | 0.6207 | 0.3322 | 0.5172 |
| 2026 | 0.6463 | 0.3782 | 0.5226 |

Decision: keep the 2017--2024 SPM and AIO as the public reference. Publish
Normal RAPM through 2026. Keep the refreshed 2025--2026 SPM as a research null.

## Possession and lineup readiness

The clean silver RAPM input is close to complete for the regular season:

| Season | Regular RAPM-ready games | Playoff RAPM-ready games |
| --- | ---: | ---: |
| 2024 | 1,228 / 1,230 | 82 / 82 |
| 2025 | 1,226 / 1,230 | 84 / 84 |
| 2026 | 1,228 / 1,230 | 60 / 85 |

The 2017--2023 legacy cache contains possessions and ten-player lineups under
its historical contract. It is not yet rebuilt into the clean ordinal silver
contract.

The strict Gabriel adapter repaired game `0022300535`. It rejected the other
nine targets because of unmappable event states, player-minute errors above
five seconds, or a missing canonical possession source. Do not relax the
existing QA gates. The repaired 2024 fit uses 1,228 regular-season games.

On the same 572 players, repaired and prior 2024 net RAPM have correlation
0.99975. The mean absolute change is 0.0079 points per 100 possessions. The
largest absolute change is 0.4982. The current three-season Normal RAPM artifact
is `current_single_season_rapm_targets_v1_b4cdb51de8`.

## AIO update

AIO is one centered ridge fit. It is not arithmetic addition of SPM and RAPM.
SPM supplies the coefficient center. Possessions supply the likelihood. The
current canonical 2025--2026 path supports zero-prior Normal RAPM, but it does
not yet support the centered-prior AIO adapter. Do not publish a 2025 or 2026
AIO until that adapter exists and the weak defensive prior is handled.
