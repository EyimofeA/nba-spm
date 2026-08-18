# Historical V3 possession candidates

## Purpose

Normal RAPM needs a possession owner, an outcome, and ten on-court players.
The local NBA Stats V3 archive covers project seasons 2017--2026. It has event
order and score actions, but it does not have the later CDN `possession` field.

`nba-impact build-historical-v3-possessions` creates a separate research
candidate. It does not replace canonical CDN possessions and it does not claim
that lineups are ready.

## Frozen owner rules

The builder orders V3 actions by `actionId`. It removes secondary credit rows
that have no action type. It then applies one forward state machine:

- a made shot, missed shot, regular free throw, turnover, or rebound uses the
  event team;
- a team rebound or team turnover uses the team ID stored in `personId`;
- a non-offensive foul inherits the current offense;
- an offensive foul ends the committing team's possession;
- a made shot keeps the same offense only when a same-clock free throw shows an
  and-one continuation;
- a made final regular free throw changes possession;
- technical and retained-ball free throws do not force a possession change;
- a jump ball uses the next unambiguous event-team anchor;
- every period is reset independently.

Points come from made-shot and made-free-throw actions. A made field goal is
three points only when the description identifies a three-point shot. This
action-based rule conserved both official team scores in every 2018--2023
regular-season game and every 2017--2023 playoff game in the pre-build audit.
Eight 2017 regular-season games differed by one point and remain rejected.

The completed 2017--2023 candidate has 8,863 accepted games and 1,768,472
possession rows. Only those eight 2017 regular-season games are rejected. All
other regular-season and playoff partitions conserve both official team scores,
stay inside the possession-count bounds, and keep the home-away possession
imbalance at five or less.

## Independent validation

The state machine was fixed on project season 2024 and then run unchanged on
project season 2025. The independent labels are the later CDN possession-owner
field. `actionNumber` is only the guarded cross-source validation key. It is
never the ordering field.

| Measure | 2024 development | 2025 untouched validation | Gate |
|---|---:|---:|---:|
| V3 actions mapped to CDN | 99.556% | 99.491% | at least 99% |
| Core action-owner agreement | 99.934% | 99.932% | at least 99.8% |
| Games with exact full owner sequence | 93.577% | 91.870% | at least 90% |
| Games with possession count within two | 99.756% | 99.106% | at least 99% |
| Mean count bias per game | -0.056 | -0.111 | absolute value at most 0.25 |

Both seasons pass the frozen gate. The remaining sequence errors prevent a
claim of exact historical ground truth. They are small enough to justify a
quarantined challenger and sensitivity analysis after lineups pass.

## Commands

Validate the frozen rules:

```bash
nba-impact validate-v3-possession-owners --project-season 2024
nba-impact validate-v3-possession-owners --project-season 2025
```

Build the separate historical candidates:

```bash
nba-impact build-historical-v3-possessions
```

Outputs are partitioned under
`data/lake/silver/historical_v3_possessions/`. Per-game acceptance is in
`data/lake/silver/historical_v3_possession_quality.parquet`.

## Promotion boundary

Do not fit or publish Normal RAPM from this table alone. A game becomes
RAPM-ready only after an exact ordinal lineup state covers its owned actions
and all official player minutes reconcile within five seconds. Compare any
historical V3 fit with the independent legacy terminal-lineup migration on the
same games before promotion.
