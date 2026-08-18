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

## 2023 pre-backfill pilot

The first implementation passed 629 of 1,230 regular-season games. Conservative
support for compound surnames and explicit suffixes, plus action-derived score
conservation, raised that to 1,023 games without fuzzy matching or a relaxed
minute gate. The remaining 207 games stay quarantined:

- 111 have no accepted historical player-game rows;
- 75 have an invalid inferred period start;
- 73 record a period-start inference failure;
- 67 exceed five seconds of player-minute error;
- 64 have an unresolved substitution name;
- 52 have an invalid substitution transition.

These categories overlap. The official BoxScoreTraditionalV3 backfill is
resumable and should address missing or incomplete roster evidence. It cannot
by itself justify ambiguous substitutions or inferred period starts.

The strict possession-lineup attachment then emitted all 1,023 double-passing
games: 205,252 possessions, 248,766 ordinal lineup segments, and 471,674 owned
actions. Every owned action maps once, the official score and segment points
conserve, every segment has ten unique players, and the terminal-lineup RAPM
loader accepts all 205,252 rows with no missing values. These are still
separate candidate tables, not production inputs.

## Next gate

1. Complete the immutable official box cache for all 2017--2023 games.
2. Rebuild the separate historical player-game table, preferring official rows.
3. Rerun each regular season and retain only strict passes.
4. Attach V3 possessions to lineups by exact `actionId` intervals and prove
   action, point, score, and ten-player conservation.
5. Fit RAPM on regular-season passing games only and compare with the
   legacy terminal-lineup model on identical games.

Historical playoffs remain excluded from the first fit because the frozen
possession-owner rules miss the 2024 playoff count gate in two games.
