# Observed-Lineup Shot Defense

Status: **research null**. The validated event table is retained. No player
defense rating is promoted from this work.

## Decision

The originally proposed primary-defender model is not identified by the local
data. The licensed matchup source is aggregated at game, scorer, and defender
grain rather than attached to exact shot events:

- only about 36--37% of matched scorer-games exactly reconcile matchup makes
  and attempts with official shot detail;
- assigned matchup attempts exceed official shots by about 9--11%;
- 87% of shooter-games allocate attempts to more than one defender, with a
  median of five defenders.

Those records are useful exposure and matchup evidence. They are not a valid
primary-defender label for an exact shot. Therefore this project does not claim
primary-defender impact, causal individual defense, shot suppression,
possession defense, or an additive five-player defensive decomposition.

This boundary is consistent with the data requirements in
[Franks et al.'s tracking-based shot-defense work](https://www.lukebornn.com/papers/franks_aoas_2015.pdf):
identifying guarding responsibility requires spatial player tracking, not
game-level matchup totals.

## Defensible estimand

`observed_lineup_shot_defense_v1` measures observed defensive-unit association
with:

1. shot-zone mix conditional on a field-goal attempt;
2. make probability conditional on zone and pre-shot context;
3. standardized expected points per field-goal attempt.

It is descriptive. It does not assign causal credit to one defender.

## Validated event panel

Build command:

```bash
uv run python -m nba_impact.cli build-shot-defense-events
```

The canonical panel joins official shot detail to NBA Stats V3 actions, CDN
`orderNumber`, post-action state, exact ordinal possession-lineup segments, and
the game dimension. `actionNumber` is used only for source alignment;
`orderNumber` defines event order. Event state is joined by stable `actionId`
because multiple event records can share an action number.

Validated regular-season snapshot:

| Season | Shots | Games | Exact ordinal-lineup coverage |
|---|---:|---:|---:|
| 2023 | 217,570 | 1,227 | 99.754% |
| 2024 | 218,084 | 1,226 | 99.662% |
| 2025 | 218,722 | 1,228 | 99.839% |
| **Total** | **654,376** | **3,681** | **99.752%** |

Backcourt shots and fixed heaves beyond 35 feet with three seconds or less are
excluded. The output has no duplicate shot IDs, unknown zones, invalid lineup
identities, or non-finite model fields.

## Cost-bounded feasibility pilot

Run command:

```bash
uv run python -m nba_impact.cli run-shot-defense-pilot
```

Run `shot_defense_team_pilot_v1_a1d8880794` used a strict chronological split
within the reused 2024 season:

- training: 852 games and 151,423 shots through 2025-02-24;
- testing: 374 games and 66,661 shots from 2025-02-25;
- models: sparse L2 logistic zone and make models;
- comparison: offense-only context versus offense plus defense-team context.

The combined held-out log loss improved from 2.04882 to 2.04699, a relative
gain of **0.0893%**. Make-probability Brier score improved slightly, but the
gain missed the frozen 0.5% practical threshold. Only 41.5% of test shots used
a five-player defensive lineup seen in training, and only 23.7% of distinct
test lineups were previously seen.

## Consequence

Retain the event panel as a clean research asset. Classify the pilot as
`research_null`. Do not bootstrap it, fit individual defender coefficients,
merge it into the all-in-one, or publish rankings. Reopen this model family
only when exact event-level guarding assignments or spatial tracking data are
available. The active impact task returns to dynamic, time-decayed player
trajectories.
