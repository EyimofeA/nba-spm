# CourtSignal / NBA Impact

One repository for NBA player-impact data, models, research evidence, and the
CourtSignal site.

## What is active

- `src/nba_impact/`: canonical data and model package.
- `research/`: estimands, season exposure, experiment plans, and release audit.
- `artifacts/`: immutable model and feature runs.
- `web/`: static CourtSignal client built from derived data.
- `tests/`: model, data-contract, API, and release checks.

The root `src/*.py`, `rapm/`, and `zts/` directories are legacy research
references. Do not extend them as production pipelines. See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the exact boundary.

## Models

- **RAPM** estimates retrospective possession impact from the ten players on
  court. The public reference uses a terminal lineup and no player-stat prior.
- **SPM** learns offense and defense from player-season statistics. Each target
  is a one-season RAPM rating.
- **AIO** uses SPM as the prior mean and updates it with the same season's RAPM
  possession evidence.
- **zTS** measures shooting above the expectation implied by playtype mix. It is
  an SPM feature, not an extra RAPM column.

The validated release scope and research-only timelines are documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and controlled by `research/`.

## Start here

```bash
uv sync
uv run python -m nba_impact.cli --help
```

Run the focused model checks:

```bash
uv run pytest tests/test_repository_boundaries.py tests/test_rapm.py \
  tests/test_current_single_season_rapm.py tests/test_single_season_spm.py \
  tests/test_annual_spm_priors.py tests/test_playtype_features.py
```

Run the client:

```bash
cd web
npm install
npm run dev
npm test
npm run lint
```

Regenerate the production-safe web snapshot only after changing a pinned
rating, role map, player sheet, or aging artifact:

```bash
uv run python -m nba_impact.cli build-web-snapshot
```

The client loads derived JSON only. Matchups data lives in `web/local-data/`
and is available only from the local development server.

## Operating documents

- [`AGENTS.md`](AGENTS.md): repository rules.
- [`ROADMAP.md`](ROADMAP.md): active queue.
- [`docs/README.md`](docs/README.md): documentation index.
- [`docs/modeling/PLAYBOOK.md`](docs/modeling/PLAYBOOK.md): statistical workflow.
- [`docs/product/UI_GUIDE.md`](docs/product/UI_GUIDE.md): site contract and style.
- [`RESEARCH_LOG.md`](RESEARCH_LOG.md): append-only evidence ledger.

Do not start downloads, model fits, or deployments as side effects of a smoke
test. Use the specific CLI and model specification for the requested run.
