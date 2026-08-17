# NBA Impact web

The compact client has six views:

- About: short SPM, RAPM, and AIO definitions.
- Ratings: one lazy AIO table with offense, defense, net, team, year,
  possessions, and a configurable possession floor.
- Player: clickable annual trajectory, offense-plus-defense decomposition,
  skill radar, and role memberships.
- Roles: clickable offense and defense maps with selected-player highlight and
  nearest role matches. Position is excluded.
- Projections: research-only player and returning-minutes team baselines.
- Research: walk-forward/backward diagnostics, win probability, and aging.

Role stabilization is a descriptive display layer, not an SPM feature. It uses
70% current membership and 30% prior stable membership, resetting after gaps.

Regenerate the derived-data snapshot from the repository root:

```bash
uv run python -m nba_impact.cli build-web-snapshot
```

The export writes a small player index, one file per year, one role map per
side/year, and 128 player-detail shards under `web/public/data/`. The browser
loads only the index and catalog at startup. Season tables, role maps, and
player details load on demand. Player shards contain compact AIO ratings,
skill profiles, and roles used by the client. It does not need the
Python API at runtime and contains no raw NBA event data.

Run and check the client:

```bash
cd web
npm install
npm run dev
npm test
npm run lint
```
