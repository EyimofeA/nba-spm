# AGENTS.md — CourtSignal / NBA Impact

This repository contains the CourtSignal NBA impact product and its research
work. Read this file before making changes. Keep it under 250 lines.

## Start here

1. Read `ROADMAP.md` for current priorities.
2. Read `docs/ARCHITECTURE.md` for the production, research, and legacy map.
3. Read `docs/README.md` for topic-specific specifications.
4. Read `research/estimands.yml` and `research/season_exposure.yml` before
   changing a model claim or evaluation split.
5. Read `docs/product/UI_GUIDE.md` and `web/README.md` before changing the site.

The public site is `https://courtsignalnba.pages.dev`. Do not deploy unless the
user explicitly asks. Do not publish raw NBA rows.

## Working style

- Lead with the outcome and next action.
- Be concise, direct, and candid. Separate verified facts, interpretations, and
  unresolved uncertainty.
- Use short active sentences and one consistent term for each concept.
- Push back on statistically weak assumptions.
- Preserve the user's goal and unrelated dirty files.
- Ask only when a decision is materially ambiguous, risky, or needs approval.
- Use subagents only for independent work that materially improves speed or
  verification. Never exceed four concurrent tasks. Verify their output.
- Use Terra high for normal repository work. Use Sol xhigh only for a frozen
  statistical review or promotion decision.
- Python computes numerical results. A language model designs and audits them.
- Before an expensive run, estimate runtime and value. Start with the smallest
  decisive pilot. Never start a downloader or full model run during a smoke test.
- Test observable behavior. Do not claim completion from a successful build
  alone when a user-facing result also needs inspection.

Use a visualization only when it materially improves understanding. Do not add
decorative diagrams.

## Canonical lanes

| Lane | Path | Rule |
| --- | --- | --- |
| Production package | `src/nba_impact/` | New model and data code goes here. |
| Public client | `web/` | Derived static data only. |
| Research control | `research/` | Estimands, exposure, plans, and release evidence. |
| RAPM research lab | `research/rapm_lab/` | Local challengers only; data and outputs stay ignored. |
| Model artifacts | `artifacts/` | Immutable run outputs and manifests. |
| Legacy reference | root `src/*.py`, `rapm/`, `zts/` | Do not extend as production code. |
| Frozen history | `archive/`, `docs/historical/` | Read-only. |

Production code must not import root legacy scripts, `rapm/src`, `zts`, a lab,
or a notebook. Do not move `rapm/data` until absolute artifact paths and the
canonical compatibility paths are migrated.

Keep Matchups and `research/rapm_lab` local and research-only. The lab is
self-contained and can reuse canonical loaders. It must write only below its
own `data/` and `outputs/` paths. Neither surface can enter public assets,
production imports, or model claims without a held-out promotion gate.

## Current model contracts

### RAPM

- One row is one completed possession.
- Inputs are five offensive player IDs, five defensive player IDs, and a home
  indicator. IDs for game, season, period, and possession are QA fields.
- The reference model uses the terminal lineup, zero player prior, and ridge
  penalties 3000 offense, 3000 defense, and 300 home.
- Positive defense means points prevented. Offense plus defense equals net.
- RAPM is available through 2026. The public interface calls it `RAPM`.
- One-year RAPM is the retrospective season estimand. Unweighted five-year
  `3000 / 3000 / 300` RAPM is the stable multi-year reference. A tuned
  actual-age, time-decayed challenger improved 2025 but worsened reused 2026;
  do not promote it.
- A direct joint actual-clock rubber-band fit keeps possession points unchanged
  and adds eight signed-margin columns beside home. It slightly improved reused
  2026 correlation but worsened margin RMSE; keep it local and unpromoted.
- A blocked comparison of normal, smooth age-only, ten signed score-bucket, and
  joint age-plus-score controls found no promotable model. Every player-only
  adjusted rating worsened reused 2026 RMSE. Keeping known lineup age at
  prediction time changed RMSE by only -0.012 with an interval crossing zero.
  Keep these leaderboards in the local Lab.
- Teammate-event and observable shot-finish RAPMs are descriptive local Lab
  views. They are not causal effects or Synergy possession play types.

Canonical code: `src/nba_impact/models/rapm.py` and
`src/nba_impact/models/current_single_season_rapm.py`.

### Annual SPM

- One row is one player-season.
- Each label is that season's one-season zero-prior RAPM. It is not a multi-year
  RAPM label.
- The model trains across historical player-seasons. Historical scoring holds
  out the scored season; the final mapping refits on all eligible labels.
- Offense and defense are separate. Net is their sum.
- Possessions supply sample weights, not features.
- Minutes, games, age, experience, height, position, on/off, plus-minus, BPM,
  xRAPM, and external ratings are excluded as general inputs.
- The validated public contract trains on 2014-24 and publishes 2017-24. The
  2014-26 refresh and 2025-26 rows are research-only because defense was weak.

Canonical code: `src/nba_impact/models/single_season_spm.py` and
`src/nba_impact/models/annual_spm_priors.py`.

### Annual AIO

- SPM supplies the coefficient prior mean.
- The same season's possession likelihood updates that mean in one ridge fit.
- AIO is not an addition of independently fit SPM and RAPM leaderboards.
- Roles, zTS, box, tracking, playtype, and matchup features enter only through
  SPM. The RAPM possession design has no player features.
- The validated public contract is 2017-24. Full-timeline artifacts remain
  research-only until a release review promotes them.

Canonical code: `src/nba_impact/models/annual_aio_ratings.py` and
`src/nba_impact/models/rapm.py`.

## Evidence rules

- Season 2027 is reserved as untouched annual confirmation. Do not use it for
  development, debugging, feature selection, or hyperparameter selection.
- Use chronological outer folds. Tune only inside the training period.
- Baseline and challenger must score identical games and rows.
- Resample whole games when uncertainty is required. Random seeds are not
  independent evidence.
- Record code, data, configuration, row-set hashes, fixed seeds, and convergence.
- Log null and failed runs in `RESEARCH_LOG.md`.
- Do not promote a model because its leaderboard looks plausible.

The public reference remains zero-prior RAPM. Annual SPM, AIO, roles, matchup
factors, trajectories, peaks, expected-possession residuals, and WP credit are
research unless their pinned contract explicitly says otherwise.

## Parked questions

Keep their code and evidence, but do not reopen them without a new request or a
predeclared hypothesis:

- start versus terminal lineup assignment;
- fractional possession attribution;
- exact constrained RAPM solver;
- another RAPM penalty search;
- alternative SPM target horizons;
- state-space promotion;
- neural win-probability models on this Mac.
- further tuning of the failed actual-age time-decay challenger;

## Data and file hygiene

- Preserve dirty worktrees. Stage only explicit files.
- Use `rg` and `rg --files` for search.
- Use `src/nba_impact/paths.py` and configured paths. Do not add absolute machine
  paths to manifests or release bundles.
- Large raw files stay out of prompts and public bundles. Profile them with code.
- Keep immutable source provenance and source-rights notes.
- Do not weaken game, score, lineup, identity, or minute-reconciliation gates to
  increase coverage.
- Do not use fuzzy player-name matching in canonical data.
- Generated Python metadata, virtual environments, build output, downloads,
  checkpoints, and local research payloads are not source code.

## Site rules

- The site is a static derived-data client. The browser never fits a model.
- Ratings is one view. Its table is the default; its chart is optional.
- The URL hash is the route source of truth.
- Load the catalog/index once. Lazy-load season, role, player, and local research
  data only when the active view needs them.
- Every chart needs exact values through a table or equivalent accessible view.
- Use color for one purpose per chart. Keep offense, defense, and net identities
  stable. Use the diverging scale only for polarity around zero.
- Matchups data lives in `web/local-data/` and is served only by the dev server.

## Normal verification

Run focused Python tests first. Then run the client contract, build, and lint:

```bash
uv run pytest tests/test_repository_boundaries.py tests/test_rapm.py \
  tests/test_current_single_season_rapm.py tests/test_single_season_spm.py \
  tests/test_annual_spm_priors.py tests/test_playtype_features.py
cd web
npm test
npm run lint
```

Use broader tests only when the changed surface requires them. Do not trigger a
download, model fit, deployment, or expensive bootstrap from these checks.
