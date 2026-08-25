# CourtSignal UI guide

CourtSignal is a compact research product. The interface should make one answer
easy to find, then expose supporting context without turning every experiment
into navigation.

## Product rules

- Ratings is one view with a table and an optional offense-defense chart.
- The table is the default. Charts supplement it.
- Player pages show ratings, comparisons, skills, roles, and history.
- Roles remain descriptive. A role label is not an impact claim.
- Research explains evidence and limits. It does not act as a second backlog.
- Matchups and unfinished projections remain localhost-only.
- The client loads only derived data. It never recomputes a model result.

## Visual system

The code-level source of truth is `web/app/globals.css`.

- Use dark mode as the saved/default product theme, with a validated light mode.
- Use one color purpose per chart.
- Use the blue/red diverging scale only for rating polarity around zero.
- Keep offense, defense, and net series colors fixed across filters.
- For role maps, emphasize one selected role and recess the rest. Do not assign
  six competing categorical colors.
- Use thin marks, hairline grids, quiet surfaces, and direct labels on extremes.
- Avoid decorative cards, explanatory filler, and repeated caveats.
- Every chart needs an accessible table or equivalent exact-value view.
- Hover must not be the only way to read a value. Keyboard focus must work.

## Interaction rules

- The URL hash is the source of truth for the active section and player.
- Season, model, team, exposure, and role controls sit above the content they
  filter.
- Tables sort from their column headers and open a player from a row.
- A player comparison uses the same season on both sides.
- Load the catalog and player index once. Load annual tables, role maps, and
  player shards only when the active view needs them.
- Do not fetch local research datasets on a hosted page.

## Copy rules

- Use short, active sentences.
- Define RAPM, SPM, and AIO once on the overview page.
- Prefer labels over paragraphs. Remove text that only repeats a visible
  control, equation, or status.
- Do not use “normal RAPM” in the interface. Use “RAPM.” Technical documents can
  use the longer term when they must distinguish a prior-centered variant.

## Component map

```text
web/app/App.tsx          route shell, global filters, data orchestration
web/app/views/           one module per public section
web/app/charts/          reusable accessible figures
web/app/lib/data.ts      typed snapshot contract and cached loaders
web/app/lib/viz.ts       shared scales, palette roles, and formatting
web/public/data/         production-safe derived payloads
web/local-data/          localhost-only research payloads
```

Build output must not contain files from `web/local-data/`.
