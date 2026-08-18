# Historical V3 ordinal lineup candidates

## Status

Research candidate only. Do not merge these rows into the canonical current
lineup tables and do not publish a historical RAPM from them until the
matched legacy comparison passes.

## Contract

- NBA Stats V3 `actionId` is the only event order. Clock-only joins are banned.
- Starters, game rosters, and player minutes come from a separately validated
  player-game table.
- An incoming substitution name must resolve to exactly one player on the same
  game and team roster. Compound surnames, first-name abbreviations, and explicit
  suffixes are supported; fuzzy matching is not.
- Each period reset requires exactly five inferred players per team.
- Every V3 action must have one valid ten-player state.
- Every emitted game must have five unique players per team, no cross-team
  overlap, exact action-derived final points, and at most five seconds of
  player-minute error.
- Failed games remain in the quality ledger and emit no lineup rows.

Sparse V3 cumulative score fields are not a valid monotonicity check: some
historical games contain mid-game score resets. Final-score validation instead
uses made-shot and made-free-throw actions, matching the accepted historical
possession candidate contract.

## 2023 strict pilot

The first implementation passed 629 of 1,230 regular-season games. Conservative
support for compound surnames and explicit suffixes, plus action-derived score
conservation, raised that to 1,023 games without fuzzy matching or a relaxed
minute gate. After the official-box repair accepted every 2018--2023 game in the
separate player-game table, the same frozen lineup contract passed 1,085 games.
The remaining 145 games stay quarantined. Their overlapping current issues are:

- 114 exceed five seconds of player-minute error;
- 110 have an unresolved substitution name;
- 107 have an invalid inferred period start;
- 105 record a period-start inference failure;
- 76 have an invalid substitution transition.

These categories overlap. The official BoxScoreTraditionalV3 backfill is
resumable and should address missing or incomplete roster evidence. It cannot
by itself justify ambiguous substitutions or inferred period starts.

The strict possession-lineup attachment then emitted all 1,085 double-passing
games: 217,853 possessions, 263,954 ordinal lineup segments, and 500,709 owned
actions. Every owned action maps once, the official score and segment points
conserve, every segment has ten unique players, and the terminal-lineup RAPM
loader accepts all 217,853 rows with no missing values. These are still
separate candidate tables, not production inputs.

## Full 2017--23 regular-season build

The same frozen contract was run for every project season. Project season 2017
failed closed because the accepted player-game table contains no trustworthy
starter rows. Seasons 2018--23 emitted 5,569 games, 1,118,333 possessions,
1,356,240 ordinal lineup segments, and 2,575,531 owned actions. The attachment
stage rejected zero games that had already passed both the possession and lineup
contracts.

| Season | Lineup pass | Quarantine | Possessions | Segments | RAPM run |
|---:|---:|---:|---:|---:|---|
| 2017 | 0 | 1,230 | -- | -- | blocked: no accepted starters |
| 2018 | 529 | 701 | 104,292 | 125,749 | `current_single_season_rapm_targets_v1_d999837054` |
| 2019 | 1,117 | 113 | 226,110 | 273,711 | `current_single_season_rapm_targets_v1_56fc612455` |
| 2020 | 893 | 166 | 181,484 | 221,070 | `current_single_season_rapm_targets_v1_75183420c6` |
| 2021 | 890 | 190 | 178,777 | 216,424 | `current_single_season_rapm_targets_v1_fb085c13e8` |
| 2022 | 1,055 | 175 | 209,817 | 255,332 | `current_single_season_rapm_targets_v1_8f89e8ba21` |
| 2023 | 1,085 | 145 | 217,853 | 263,954 | `current_single_season_rapm_targets_v1_d0760ddd78` |

The six RAPM runs use one regular season, terminal lineups, a zero prior, and
the frozen `3000/3000/300` penalties. They are isolated research training-label
artifacts. They are not eligible for the public panel until the matched-game
comparison with the legacy terminal-lineup source passes.

## Next gate

1. Complete the immutable official box cache for all 2017--2023 games.
2. Rebuild the separate historical player-game table using official rows only
   where they pass; verify the 2018--23 accepted-game set is unchanged.
3. Audit a possible 2017 starter source independently. Do not infer starters
   from minutes or relax the five-starter gate.
4. Compare the six fitted candidates with legacy terminal-lineup RAPM on
   identical games and players.
5. Promote no historical target until that matched comparison passes.

Historical playoffs remain excluded from the first fit because the frozen
possession-owner rules miss the 2024 playoff count gate in two games.
