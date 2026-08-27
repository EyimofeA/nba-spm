# AGENTS.md — CourtSignal / NBA Impact

This repository contains the CourtSignal NBA impact product and its research
work. Read this file before making changes. Keep it under 250 lines.

## Start here

1. Read `ROADMAP.md` for current priorities.
2. Read `docs/ARCHITECTURE.md` for the production, research, and legacy map.
3. Read `docs/README.md` for topic-specific specifications.
4. Read `research/estimands.yml` and `research/season_exposure.yml` before changing a model claim or evaluation split.
5. Read `docs/product/UI_GUIDE.md` and `web/README.md` before changing the site.

The public site is `https://courtsignalnba.pages.dev`. Do not deploy unless the
user explicitly asks. Do not publish raw NBA rows.

## Working style

- Lead with the outcome and next action.
- Be concise, direct, and candid. Separate verified facts, interpretations, and unresolved uncertainty.
- Use short active sentences and one consistent term for each concept.
- Use clear subject, verb, and object constructions. Do not use cleft sentences, contrastive appositives, appended glosses, or trailing clauses.
- Assume the user may edit documents, especially Markdown files. Preserve the user's edits and inspect current contents before changing them.
- Write Markdown for an independent reader. Do not refer to conversations, tasks, threads, or unstated context that the reader cannot access.
- Push back on statistically weak assumptions.
- Preserve the user's goal and unrelated dirty files.
- Ask only when a decision is materially ambiguous, risky, or needs approval.
- Use subagents only for independent work that materially improves speed or verification. Never exceed four concurrent tasks. Verify their output.
- Use Terra high for normal repository work. Use Sol xhigh only for a frozen statistical review or promotion decision.
- Python computes numerical results. A language model designs and audits them.
- Before an expensive run, estimate runtime and value. Start with the smallest decisive pilot. Never start a downloader or full model run during a smoke test.
- Test observable behavior. Do not claim completion from a successful build alone when a user-facing result also needs inspection.

Use a visualization only when it materially improves understanding. Do not add decorative diagrams.

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

### Five-year SPM research default

- One row pools five seasons of player statistics; its label is RAPM over the
  identical five seasons.
- Complete windows run from 2018 through 2026. Historical fits train only on
  earlier window ends.
- The model retains the frozen 127 offense and 68 defense inputs and learners.
- Canonical complete feature input:
  `full_spm_features_2014_2026_v1_21885aaf37`. It contains 6,942 annual rows
  for 2014--26 and 8,620 rolling rows for window ends 2018--26.
- Hustle and matchup-assignment sources start in 2018. The coverage ledger
  marks 2014--17 unavailable. Do not fabricate observed rows for those seasons.
- This feature refresh closes the 2025--26 input gap. It does not refit or
  promote the five-year SPM or AIO.
- Coverage run `full_feature_coverage_v1_3de4ec8954` audits all 170 selected
  fields before imputation. It assigns an explicit source or opportunity cause
  to every field below 99% observed coverage. Undefined rate fields remain
  missing and receive training-fold median imputation.
- Completion run `semantically_complete_spm_features_v1_8be676bd0f` converts
  the same contract into 175 fully finite inputs. Event rates use zero, raw
  ratios use same-season empirical-Bayes estimates, and source-specific values
  use zero with an availability field. Low-sample zTS uses all observed
  playtype possessions. Rows without playtype data use season-relative TS.
  The annual and five-year panels have zero missing selected inputs.
- Comparison run `semantic_feature_completion_comparison_v1_235b4dea34` finds
  no AIO gain over the previous missing-data implementation. Removing matchup
  fields makes the standalone SPM worse. Keep matchup fields and their source
  flag. Keep BoxPIPM-style and the completed SPM as separate research AIO
  challengers.
- Final ladder `final_box_feature_ladder_v1_8bb26f12e7` adds eight frozen
  feature families to Box15 cumulatively. No step lowers equal-season
  next-season game-margin MSE after the same RAPM update. Box15 scores `207.421`
  MSE; the closest matchup step scores `207.537`. Keep Box15 as the research AIO
  prior and end the current feature search. Interpretation run
  `final_box_interpretability_v1_652799efb6` splits its dependence into
  disruption/fouls, shooting/scoring, creation/security, and rebounding. Use
  `active_2026_leaderboard.parquet`; the earlier unfiltered table includes
  zero-exposure historical players.
- Run `full_spm_history_ablation_v1_2eb5eb428c` refits the corrected panel and
  removes the 13 hustle and matchup fields that start in 2018 as one fixed
  block. The reduced SPM loses. Keep the full 127/68 contract.
- Corrected full SPM beats BoxPIPM-style standalone by `-2.750` paired MSE,
  with interval `[-4.821, -0.702]`. After the same RAPM update, full minus
  BoxPIPM-style is `+0.681`, with interval `[-0.227, +1.587]`. The final AIO
  comparison remains unresolved.
- Prior-scale audit `aio_prior_scale_audit_v1_aeca5715b3` selects each season's
  scale from earlier folds. Full SPM selects `0.75`; BoxPIPM-style selects
  `1.00`. Scaling does not improve full SPM AIO accuracy. Keep both as research
  challengers.
- The run emits current 2026 SPM and AIO ratings. It does not load or score
  2027.
- For AIO season `t`, this SPM is the prior and only season `t` possessions are
  the likelihood. It is not a five-year RAPM likelihood.
- Run `five_year_target_spm_v1_65550acb79` beat the annual-prior AIO and
  zero-prior RAPM in next-season game-margin RMSE for every 2022--26 test.
- Teammate-context correction `five_year_spm_teammate_context_v1_13d270986a`
  improves two reused next-season RAPM folds by only `.002` and `.012` RMSE.
  Keep the family for a joint refit; do not change the frozen research SPM.
- Stabilized rim assists failed as an additional offense input. Run
  `rim_assist_spm_challenger_v1_23e599d812` lost all three reused future-game
  folds and increased paired MSE. Keep rim assists in the descriptive skill
  layer. Do not add them to the impact prior.
- Treat it as the research replacement, not a public promotion, until the
  untouched 2027 confirmation.

Canonical code: `src/nba_impact/models/five_year_target_spm.py`.

- Sparse challenger `sparse_function_spm_v1_4f1ecaa353` is invalid for its
  declared feature contract: `ShootingFouls` is fouls committed, but the run
  labeled it shooting fouls drawn and put it on offense. Its numerical output
  remains reproducible but must not be used as a model-selection result.
- Hand-selected challenger `hand_selected_sparse_spm_v1_f04379a684` fixes that
  lineage and uses eight offense plus four defense functions. It lowers future
  one-year RAPM RMSE but loses correlation and both team-win folds; team-win R²
  is `.472` versus `.545`. Treat it as a research null and do not run its AIO.
- The full five-year SPM baseline also contains the old mislabeled field in its
  offense feature set. That does not erase its predictive benchmark, but its
  side interpretation is not publication-clean. Correct and refit it before
  any promotion decision.
- Full factor-target run `factor_target_full_feature_spm_v1_69496cee37` uses
  127 offense and 60 available defense inputs. Context raises reused 2026
  shooting-defense R² only to `.126`; five of six full factor heads beat their
  sparse versions. Predicted factors plus context reach `.312` normal-RAPM R²,
  within `.010` RMSE of a direct annual model plus context. Keep it diagnostic.

### Same-season feature research

- Stabilize every player-season against that season only. Do not use career,
  previous-season, future-season, or pooled-era centers to create an annual SPM
  input.
- Five-year SPM may possession-weight five already-frozen annual estimates.
  That model window is intentional history, not a stabilization prior.
- Run `five_year_spm_feature_research_v1_93c148510e` selected Basketball Index
  passing context plus RAPTOR-style defended-shot and matchup-volume context on
  2022-24 development folds. It improved AIO game RMSE through 2025 but lost by
  `0.0126` in reused 2026, so it remains a localhost-only challenger.
- Do not add opponent shooting outcome, generic hustle, shooting, screening,
  transition, or playtype families from this run. They failed the frozen gate.
- The feature builder now distinguishes raw `rim_points_saved_p100_raw` from
  the frozen EB-stabilized `rim_points_saved_p100`, and calculates exact
  SelfORB-adjusted TS from Gabriel's observed `SelfOReb` count. Both new fields
  are candidates, not selected SPM inputs. KOBE-style row-level shot context
  remains a 2014-15 historical prototype because modern nearest-defender rows
  are unavailable under the current source contract.
- Controlled run `spm_stabilization_ablation_v1_db618f06e8` compares one raw or
  stabilized value for each of 37 offense and 10 defense concepts. Stabilization
  improves standalone five-year SPM target fit and wins four of five future-game
  folds, but the one-season RAPM update weakens the difference. Raw minus
  stabilized AIO MSE is `+0.1338`, with interval `[-0.2335, +0.5050]`. Retain
  same-season stabilization in SPM research. Do not claim an AIO gain.

Canonical code: `src/nba_impact/models/five_year_spm_feature_research.py`.

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

- Validation run `impact_validation_suite_v1_4f2ad7cdd8` keeps BoxPIPM-style as the research AIO prior. Four-way run `pipm_four_way_comparison_v1_0f1473b838` also beats a third-party PIPM reference after the same RAPM update, but the attached original file is partial and 2027 remains required.
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
- further SPM target-horizon tuning after the selected five-year run;
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
