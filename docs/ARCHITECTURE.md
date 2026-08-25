# Repository architecture

This repository contains one active product and two supporting lanes. Keep the
boundaries explicit. A research result does not become a public model because
its code and artifact exist here.

## Source-of-truth map

| Lane | Purpose | Canonical path |
| --- | --- | --- |
| Production package | Data contracts, canonical builders, model fits, API export | `src/nba_impact/` |
| Public client | Static, derived-data CourtSignal site | `web/` |
| Research control | Estimands, season exposure, preregistration, release evidence | `research/` |
| Research implementations | Challengers that use the production contracts | `src/nba_impact/models/` and `src/nba_impact/data/` |
| Legacy reference | Earlier global SPM, MySQL RAPM, and standalone zTS work | root `src/*.py`, `rapm/`, `zts/` |
| Frozen history | Superseded scripts and outputs | `archive/` and `docs/historical/` |

New production code must not import the legacy root scripts, `rapm/src`, a lab,
or a notebook. Legacy code can remain while artifact manifests and old studies
still depend on it.

## Current model contracts

### RAPM

- One row is one completed possession.
- The design contains the five offensive players, five defensive players, and
  a home indicator. Game, season, period, and possession IDs are QA fields.
- The public reference is terminal-lineup, zero-prior ridge.
- Penalties are 3000 offense, 3000 defense, and 300 home.
- Output is points per 100 possessions. Positive defense means points prevented.
- Offense plus defense equals net.

Canonical implementation:
`src/nba_impact/models/rapm.py` and
`src/nba_impact/models/current_single_season_rapm.py`.

### Annual SPM

- One training row is one player-season.
- Each target is that season's one-season, zero-prior RAPM. The target is not a
  multi-year RAPM.
- The learner sees player statistics, not possessions, minutes, games, age,
  height, position, on/off, plus-minus, or external impact metrics.
- Offense and defense are fit separately. Net is their sum.
- Historical evaluation holds out the scored season. The final mapping refits
  on the available labeled seasons.
- The validated public contract is the 2014-24 training panel with 2017-24
  outputs. The 2014-26 refresh and its 2025-26 outputs remain research results
  because their defensive confirmation was weak.

Canonical implementation:
`src/nba_impact/models/single_season_spm.py` and
`src/nba_impact/models/annual_spm_priors.py`.

### Annual AIO

- SPM supplies the prior mean in RAPM coefficient units.
- The same season's possession likelihood updates that mean in one ridge fit.
- AIO is not an arithmetic sum of two independently fit ratings.
- Roles, zTS, and box features enter only through the fitted SPM prior. They are
  not extra columns in the possession design.
- The validated public contract is 2017-24. Later full-timeline artifacts are
  research-only until a release review changes that status.

Canonical implementation:
`src/nba_impact/models/annual_aio_ratings.py` and
`src/nba_impact/models/rapm.py`.

## Product boundary

The browser loads derived JSON. It does not fit a model and it must not receive
raw NBA events. The production bundle contains ratings, player details, role
maps, and validation summaries. Matchups and shot-quality diagnostics are
localhost-only until they pass a held-out model gate.

Read `docs/product/UI_GUIDE.md` before changing the client. Read
`research/README.md`, `research/estimands.yml`, and
`research/season_exposure.yml` before changing a claim.

## Consolidation decision

The active history already contains the useful commits from the data, model,
uncertainty, and modular UI branches. Do not cherry-pick the old branch tips.
They either duplicate current work or replace the modular client with a stale
prototype.

Keep dirty sibling worktrees intact as research scratch space. Integrate only a
reviewed file or result, never a whole worktree. The isolated
`research/rapm_lab` remains
local research and must not enter production imports or public assets.

## Parked work

The following questions keep their code and evidence but are not current
integration tasks:

- start-lineup versus terminal-lineup assignment;
- fractional possession attribution and an exact constrained solver;
- another RAPM penalty search;
- alternative SPM target horizons;
- state-space promotion;
- public Matchups, projections, or win-probability pages.

## Verification

For a normal repository change, run:

```bash
uv run pytest tests/test_repository_boundaries.py tests/test_rapm.py \
  tests/test_current_single_season_rapm.py tests/test_single_season_spm.py \
  tests/test_annual_spm_priors.py tests/test_playtype_features.py
cd web
npm test
npm run lint
```

Do not start downloads or full model runs as a side effect of a repository or
UI check.
