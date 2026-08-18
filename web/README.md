# NBA Impact web

A static derived-data client. It loads small JSON files from
`web/public/data/` and needs no Python API at runtime.

Sections:

- Overview: what a rating means, the season's ten most valuable, the SPM / RAPM /
  AIO definitions, and the four rules for reading a rating.
- Ratings: ranked offense/defense bars over the full sortable board, with a
  model selector, season, team, and possession floor.
- Landscape: offense against defense on one plane, split at zero, with quadrant
  counts. Dot size is exposure; colour is the diverging net scale.
- Player: career trajectory, the SPM-center-plus-RAPM-update decomposition when
  the snapshot carries it, a skill-percentile wedge chart, role mix, and where
  the player sits in the season's distribution.
- Roles: the behavioural map with one role lifted at a time, cluster sizes that
  double as the picker, and nearest neighbours.
- Projections: research-only player and returning-minutes team baselines.
- Research: build order, forward-only SPM accuracy, the next/previous season
  direction check, external agreement with BPM and xRAPM, aging, projection
  method selection, the recorded limits, and the working queue of plans.

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

## Client structure

```
app/
  App.tsx          shell: hash routing, filters, search, theme
  lib/data.ts      snapshot types and cached loaders
  lib/viz.ts       palette roles, scales, formatting
  charts/frame.tsx Figure card, legend, tooltip, hover+focus state
  charts/*.tsx     bars, lines, scatter, pizza
  views/*.tsx      one file per section
```

The location hash is the single source of truth for the current view
(`#ratings`, `#player/2544`), so links, the back button, and in-page navigation
cannot disagree. Filters live on the shell, so the ratings board and the
landscape always describe the same slice.

## Chart rules this client follows

Colour does one job per chart, and the categorical palette is validated rather
than eyeballed:

- **Polarity** (a rating's side of zero) is the diverging blue/red pair with a
  neutral grey midpoint, so "about zero" reads as nothing.
- **Identity** (offense vs defense, the three components, skill groups) uses
  categorical slots in fixed order, assigned in sequence and never cycled.
  Colour follows the entity, so filtering never repaints the survivors.
- **Roles run to six categories**, which is past the three-colour cap for a
  scatter where any two dots can sit side by side. The role map therefore uses
  emphasis — one role lifted, the rest recessive — instead of six hues.
- Marks are thin, gridlines are hairlines, and separation comes from a surface
  gap or a surface ring rather than a stroke drawn around a mark.
- Every chart ships a **table view** beside it and direct labels on the
  extremes. That keeps every value reachable without hovering, and it is the
  relief for the light-mode slots that sit under 3:1 against the surface.
- Tooltips enhance and never gate: keyboard focus produces the same readout as
  the pointer, and hit targets are the whole row, a nearest-point layer, or a
  full wedge — never the painted pixels alone.
- Dark mode is a selected set of steps validated against the dark surface, not
  an automatic inversion. A blocking script stamps the saved theme before first
  paint so the page never flashes.
