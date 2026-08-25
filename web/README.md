# NBA Impact web

A static derived-data client. It loads small JSON files from
`web/public/data/` and needs no Python API at runtime.

Public sections:

- Overview: what a rating means, the season's ten most valuable, the SPM / RAPM /
  AIO definitions, and the four rules for reading a rating.
- Ratings: one sortable table with a table/chart switch, model selector, season,
  team, exposure floor, and role filter. The chart places offense against
  defense on one plane.
- Player: career trajectory, the SPM-center-plus-RAPM-update decomposition when
  the snapshot carries it, a skill-percentile wedge chart, role mix, and where
  the player sits in the season's distribution.
- Roles: the behavioural map with one role lifted at a time, cluster sizes that
  double as the picker, and nearest neighbours.
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
- Matchups is a localhost-only research view. Its payloads live in
  `web/local-data/` and are not copied into a production build.
- RAPM Lab is localhost-only. It shows the saved test, result, and decision for
  recent experiments, the actual-clock rubber-band estimate, and Matchups.

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

Refresh the local RAPM Lab payload after research runs:

```bash
uv run python web/scripts/build-rapm-lab-data.py
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
cannot disagree. Shared filters live in the shell. Ratings-specific team and
role filters live in the Ratings view.

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
