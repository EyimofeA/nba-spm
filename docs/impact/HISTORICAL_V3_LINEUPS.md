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
  suffixes are supported; fuzzy matching is not. Official-box DNP rows with zero
  minutes are excluded from substitution aliases because they cannot enter an
  observed game.
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

The official-only 2019 reproducibility pilot exposed one additional identity
edge case. Official boxes name player ID 2403 `Nene Hilario`; structured V3
actor rows use `Hilario`; substitution text uses `Nene`. All 103 new parse
failures across 39 games were this same player. The resolver now uses exact
same-game/team V3 actor aliases plus one versioned ID-specific alias
`2403 -> nene`; it still does not use fuzzy matching. The strict official-
preferred pilot then passed 1,141 games, restoring all 39 false losses and
adding two games that the mixed-source build had quarantined.

The official-only 2021 reproducibility pilot exposed a different exact-input
issue: official boxes retain zero-minute DNP rows, so surname-only substitution
text falsely made Grant/Robert Williams and Moses/Charlie Brown ambiguous.
Restricting substitution aliases to players with positive official minutes
restored the exact prior accepted set: 892 passes, 188 quarantines, no gained or
lost games. Attachment emitted all 892 games, 179,181 possessions, and 216,916
segments with zero rejects, invalid ten-player segments, or duplicate keys.

## Full 2017--23 regular-season build

The same frozen contract was run for every project season. Project season 2017
uses the separately validated `official_box_first_five_v1` starter fallback:
first-five response order agreed with 10,073 independently marked later team
records and is permitted only for 2016--17 payloads. The completed 8,871-game
official box-score cache then supported an official-only rebuild. Seasons
2017--23 emitted 7,250 games, 1,448,146 possessions, 1,756,230 ordinal lineup
segments, and 3,342,158 owned actions. The attachment stage rejected zero games that had
already passed both the possession and lineup contracts.

| Season | Lineup pass | Quarantine | Possessions | Segments | RAPM run |
|---:|---:|---:|---:|---:|---|
| 2017 | 996 | 234 | 194,364 | 235,649 | `current_single_season_rapm_targets_v1_c498e38e09` |
| 2018 | 1,051 | 179 | 206,981 | 250,166 | `current_single_season_rapm_targets_v1_54b97074d0` |
| 2019 | 1,135 | 95 | 229,686 | 278,065 | `current_single_season_rapm_targets_v1_31aa665391` |
| 2020 | 949 | 110 | 192,877 | 234,920 | `current_single_season_rapm_targets_v1_ba501c0ecd` |
| 2021 | 892 | 188 | 179,181 | 216,916 | `current_single_season_rapm_targets_v1_f4655e5d66` |
| 2022 | 1,084 | 146 | 215,543 | 262,288 | `current_single_season_rapm_targets_v1_f72bc5e5e4` |
| 2023 | 1,143 | 87 | 229,514 | 278,226 | `current_single_season_rapm_targets_v1_13d15a0a60` |

The seven RAPM runs use one regular season, terminal lineups, a zero prior, and
the frozen `3000/3000/300` penalties. They are isolated research training-label
artifacts. The matched-game comparison passed a narrow compatibility check but
did not show predictive improvement. The artifacts remain research-only.

## Next gate

1. Retain the 2017 first-five fallback as a versioned exception; do not infer
   starters from minutes or extend the exception to another season.
2. Explain the remaining season-level quarantine sets before investigating any
   coverage expansion.
3. Keep the V3 targets research-only unless a separately preregistered
   predictive promotion gate passes.

Historical playoffs remain excluded from the first fit because the frozen
possession-owner rules miss the 2024 playoff count gate in two games.
