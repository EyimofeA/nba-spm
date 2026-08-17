# NBA Impact web

The compact client has five views:

- About: casual overview plus the exact data, target, model, and validation
  specification needed to reproduce the public build.
- Ratings: lazy year tables for SPM, normal RAPM, and AIO. Sort by rating,
  player, team, year, or possessions; the full view compares all three models.
- Player: annual trajectories, the centered AIO decomposition, and raw or
  forward-stabilized role memberships.
- Roles: clickable offense and defense cluster maps. Position is excluded.
- Research: year-over-year RAPM/AIO aging curves.

Role stabilization is a descriptive display layer, not an SPM feature. It uses
70% current membership and 30% prior stable membership, resetting after gaps.

Regenerate the derived-data snapshot from the repository root:

```bash
uv run python -m nba_impact.cli build-web-snapshot
```

The export writes a small player index, one file per year, one role map per
side/year, and 32 player-detail shards under `web/public/data/`. The browser
loads only the index and catalog at startup. Season tables, role maps, and
player details load on demand. Player shards contain only annual ratings and
roles used by the client. It does not need the
Python API at runtime and contains no raw NBA event data.

Run and check the client:

```bash
cd web
npm install
npm run dev
npm test
npm run lint
```
