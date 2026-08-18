# Historical V3 ordinal lineup candidates

## Status

Research candidate only. Do not merge these rows into the canonical current
lineup tables. The matched legacy comparison passed the narrow source-
compatibility check, but it did not show a predictive improvement. Do not
replace the public historical RAPM until the full official player-game cache
is complete and the official-preferred rebuild reproduces the same gates.

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
Fixing pre-tokenization Unicode loss for names such as Jokić, Nurkić, and Šarić
raised the strict result to 1,124 games without fuzzy matching. The remaining
106 games stay quarantined. Their overlapping current issues are:

- 83 have an invalid inferred period start;
- 81 record a period-start inference failure;
- 75 exceed five seconds of player-minute error;
- 70 have an unresolved substitution name;
- 57 have an invalid substitution transition.

These categories overlap. The official BoxScoreTraditionalV3 backfill is
resumable and should address missing or incomplete roster evidence. It cannot
by itself justify ambiguous substitutions or inferred period starts.

The strict possession-lineup attachment then emitted all 1,124 double-passing
games: 225,784 possessions, 273,611 ordinal lineup segments, and 519,065 owned
actions. Every owned action maps once, the official score and segment points
conserve, every segment has ten unique players, and the terminal-lineup RAPM
loader accepts all 225,784 rows with no missing values. These are still
separate candidate tables, not production inputs.

## Full 2017--23 regular-season build

The same frozen contract was run for every project season. Project season 2017
uses the separately validated `official_box_first_five_v1` starter fallback:
first-five response order agreed with 10,073 independently marked later team
records and is permitted only for 2016--17 payloads. Seasons 2017--23 emitted
7,136 games, 1,425,380 possessions, 1,728,895 ordinal lineup segments, and
3,289,566 owned actions. The attachment stage rejected zero games that had
already passed both the possession and lineup contracts.

| Season | Lineup pass | Quarantine | Possessions | Segments | RAPM run |
|---:|---:|---:|---:|---:|---|
| 2017 | 941 | 289 | 183,171 | 222,326 | `current_single_season_rapm_targets_v1_c4512aa5c3` |
| 2018 | 1,014 | 216 | 199,696 | 241,438 | `current_single_season_rapm_targets_v1_64cdcaec78` |
| 2019 | 1,139 | 91 | 230,514 | 279,077 | `current_single_season_rapm_targets_v1_bd1209bc26` |
| 2020 | 949 | 110 | 192,896 | 234,941 | `current_single_season_rapm_targets_v1_80ad208f37` |
| 2021 | 892 | 188 | 179,181 | 216,916 | `current_single_season_rapm_targets_v1_10118df750` |
| 2022 | 1,077 | 153 | 214,138 | 260,586 | `current_single_season_rapm_targets_v1_c7b600f15e` |
| 2023 | 1,124 | 106 | 225,784 | 273,611 | `current_single_season_rapm_targets_v1_82f653ce02` |

The seven RAPM runs use one regular season, terminal lineups, a zero prior, and
the frozen `3000/3000/300` penalties. They are isolated research training-label
artifacts. The matched-game comparison passed a narrow compatibility check but
did not show predictive improvement. The artifacts remain research-only until
the official-preferred player-game rebuild reproduces the same gates.

## Next gate

1. Complete the immutable official box cache for all 2017--2023 games.
2. Rebuild the separate historical player-game table using official rows only
   where they pass; verify the 2018--23 accepted-game set is unchanged.
3. Retain the 2017 first-five fallback as a versioned exception; do not infer
   starters from minutes or extend the exception to another season.
4. Re-run the seven matched-source comparisons after the official-preferred
   rebuild and require the accepted-game and metric changes to be explained.
5. Keep the V3 targets research-only unless that reproducibility check passes.

Historical playoffs remain excluded from the first fit because the frozen
possession-owner rules miss the 2024 playoff count gate in two games.
