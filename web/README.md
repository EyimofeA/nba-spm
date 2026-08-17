# NBA Impact web

The compact client has three views:

- Impact: SPM, normal RAPM, and decomposed AIO by side and window.
- Roles: soft offense and defense memberships. Position is excluded.
- Aging: the observed 1997--2024 annual RAPM curve.

RAPM error bars appear only for the exact 2022--24 and 2025 uncertainty runs.
SPM and AIO intervals are not estimated.

Regenerate the derived-data snapshot from the repository root:

```bash
uv run python -m nba_impact.cli build-web-snapshot
```

The export writes a small player index and 32 rating shards under
`web/public/data/`. The browser loads about 140 KB for search and one shard for
the selected player. It does not need the Python API at runtime and contains no
raw NBA event data.

Run and check the client:

```bash
cd web
npm install
npm run dev
npm test
npm run lint
```
