# NBA Impact web

A static derived-data client. It loads small JSON files from
`web/public/data/` and needs no Python API at runtime.

Sections:

- Home: what a rating means, and the SPM / RAPM / AIO definitions.
- Ratings: one lazy table with a model selector, year, team, and possession
  floor. The floor defaults to All. Sorting uses the column headers.
- Player: annual trajectory, offense-plus-defense identity, the
  SPM-center-plus-RAPM-update decomposition for AIO, skill radar, and raw roles.
- Roles: clickable offense and defense maps with nearest role matches.
- Projections: research-only player backtests for 2019–24 and a 2027
  returning-minutes team forecast. The snapshot adds future artifacts only when
  they are exported.
- Research: build order, forward-only SPM accuracy, the next/previous season
  direction check, external agreement with BPM and xRAPM, aging, projection
  method selection, and the recorded limits.

Rules the client follows:

- Public roles are raw, single-season clusters. Stabilized role labels are a
  display experiment and stay out of the snapshot.
- Ratings stay pinned to the validated 2017–24 model. Season 2027 is reserved
  for annual confirmation and is never scored here.
- Every number comes from the snapshot or from a transcribed frozen run. The
  client never recomputes a model result.

The model selector reads `catalog.catalog.models`, which mirrors
`MODEL_CATALOG` in `src/nba_impact/api/web_snapshot.py`. A model can be selected
only when the loaded snapshot carries its columns (`aio_*`, `normal_rapm_*`,
`spm_*`). Snapshots built before those columns existed show AIO only; rebuild
them from the pinned runs to compare all three models.

Regenerate the derived-data snapshot from the repository root:

```bash
uv run python -m nba_impact.cli build-web-snapshot
```

The export writes a player index, one file per year, one role map per side and
year, and 128 player-detail shards. The browser loads only the index and catalog
at startup. It contains no raw NBA event data.

Run and check the client:

```bash
cd web
npm install
npm run dev
npm test    # builds, then runs the rendered-shell and data-contract tests
npm run lint
```

Snapshot generation is covered by `tests/test_web_snapshot.py` at the repository
root.
