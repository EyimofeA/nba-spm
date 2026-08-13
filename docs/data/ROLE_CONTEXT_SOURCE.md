# Role-context source

This source is a versioned **bronze research input** for future continuous-role
work. It is not an AIO feature set, a player rating, or a canonical NBA truth
table.

## Snapshot

- Provider: [`gabriel1200/site_Data`](https://github.com/gabriel1200/site_Data)
- Pinned revision: `bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd`
- Ingest manifest: `configs/ingest/gabriel_role_context_v1.json`
- License: upstream does not declare one; do not redistribute the raw files in
  a release bundle.
- Bronze destination: `data/lake/bronze/gabriel_site_data/revision=bc583cb/role_context/`

| Input | Grain | Coverage | Validated use |
|---|---|---|---|
| `shooting_by_dribble_count.csv` | player-season-dribble bucket | 31,376 rows; 1,692 player IDs; 2014–26 | Shot volume and efficiency by dribble count. |
| `jump_shot_dribble_context.csv` | player-season-dribble bucket | 28,435 rows; 1,656 player IDs; 2014–26 | Jump-shot makes/attempts by dribble count. |

Both files have unique `(PLAYER_ID, year, dribbles)` keys and non-null IDs,
season labels, dribble buckets, and field-goal attempts.

## Known contract limitation

`jump_shot_dribble_context.csv` has blank supplied `FREQ%`, team, and several
display fields in 2014–24. Its `FGM` and `FGA` counts are populated throughout.
Any silver transform must derive within-player shares from counts, retain an
explicit missingness indicator for upstream display fields, and never coerce
their missing values to zero.

## Derived research table

`build-role-context-features` creates one research-only player-season artifact
with source `PLAYER_ID`, `Season`, count totals, zero/one-to-two/three-plus
dribble shares, and bucket field-goal percentages. It drops source age, games,
minutes, team display labels, and upstream percentage columns. The default
2014–25 scope deliberately excludes incomplete 2026 infrastructure data.

```bash
uv run python -m nba_impact.cli build-role-context-features \
  --shooting-by-dribble-source data/lake/bronze/gabriel_site_data/revision=bc583cb/role_context/shooting_by_dribble_count.csv \
  --jump-shot-by-dribble-source data/lake/bronze/gabriel_site_data/revision=bc583cb/role_context/jump_shot_dribble_context.csv
```

Evaluate this role block only in a preregistered chronological experiment
against the frozen AIO baseline. It must not be merged into the published
rating pipeline based on face validity or leaderboard appeal.
