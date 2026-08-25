# NBA RAPM/SPM — Master State Document

Single entry point. Agents: read this file, then the last 3 entries of
`RESEARCH_LOG.md`, then act. Humans: read `RESEARCH_LOG.md` for the story,
`rapm/outputs/figures/` for the pictures.

Last updated: 2026-07-03 ~16:00 (Feature Foundry Phase 0 live; minutes prior still interim product).

## What this project is

Possession-level NBA RAPM (1997–2024, 6.6M possessions in MySQL `matchups`,
parquet-cached per season) plus a same-window SPM prior under construction.
Everything is judged by ONE gate (below). The user directs; agents execute;
nothing ships on vibes.

## State of the world

- Production panel: `rapm/outputs/rapm_results/final_20260703_hl250/`
  (26 rolling 3-yr windows, SEs/CIs, A/B/C tiers; hl365 twin kept beside it).
- Champion config: players + single home effect, garbage-time filtered,
  exponential recency decay half-life 250 days, lambda 3000 symmetric, zero prior.
  Gate scores: corr 0.6596 (2021-23→2024), 0.5939 (2020-22→2023).
- Best known result (interim prior product): minutes-only prior at c=2:
  0.7335 / 0.6953 (harness repro: 0.7346 / 0.6954). Saturates ~c=8.
- SPM v1.2: 0.6988/0.6619 — below minutes. SPM v2 pooled (Phase A): 0.7304/0.6983 — still below minutes.
- **Feature Foundry Phase 0 shipped:** `rapm/src/feature_eval.py` (splice harness),
  `rapm/features/` (program.md, results.tsv, prepare.py), foundry gen 0 running
  (`rapm/outputs/foundry_g0.log`). User lane: `feature_submit.py`.

## The gate (how every change is judged)

1. Two chronological folds: train 2021-23 → test all 2024 games; train
   2020-22 → test 2023. Metric: corr + RMSE of predicted vs actual game margins
   from frozen ratings. Both folds must agree for a win.
2. Anchor test (auto-reject): Jokic total > 0, Gobert total > 0, Gobert
   defensive coefficient must not flip harmful. In `experiments.anchor_check`.
3. ESS logged for any weighting change (watch for silent sample collapse).
4. Banned: possession-level RMSE as a selection metric; priors whose features
   are computed from the same possessions as the labels (on/off ratings);
   global-strength priors (must be per-player tau²-derived).
5. Every run appends to `rapm/outputs/diagnostics/experiments.csv` including
   failures. Negative results get logged, never deleted.

## Conventions (get these wrong and everything silently breaks)

- Def coefficient: POSITIVE = bad defense (points allowed). Player total =
  (off − def) × 100. The Gobert check exists because of this sign.
- All rates per-100-possessions. Ratings are per-100 (≈ points per 100).
- PLAYER_ID = NBA person id (int). Names via `rapm/data/all_names.csv`.
- Windows named by END season: "2024" window = 2022+2023+2024 regular seasons.
- Prior semantics (clean, mandatory): likelihood penalty UNTOUCHED (lambda 3000
  → zero), prior enters as SEPARATE per-player pull c·sigma²/tau²_side toward
  center; tau² measured from stage-1 OOF residuals per side; c is the only
  tuned scalar. No playoffs anywhere. Descriptive SPM sees window-T info only.

## Artifact map

- `PROJECT.md` — this file. `IDEAS.md` — ledger of every idea + status.
  `RESEARCH_LOG.md` — append-only journal (the memory).
- `rapm/README.md` — script-by-script map of `rapm/src/`.
- `rapm/src/standard_rapm.py` — the engine (design matrix, ridge, SEs).
  `experiments.py` — harness lib + overnight queue (import from it).
  `spm_minutes_prior.py`, `spm_v1.py` — prior machinery (clean semantics).
  `final_windows.py` — production runner. `aging_curve.py`, `d0/d2 decay` —
  measured curves. `fig_*.py` — regenerable figures.
- Data: `rapm/data/possession_cache/*.parquet` (delete a season to refetch);
  `rapm/data/spm_features_windows.parquet` (56 feats × 17K player-windows,
  coverage 100%, tracking tier valid 2001+).
- Runs: tmux session + `outputs/<name>_run.log` + done-flag file; watcher
  pattern in transcript. Results: `outputs/diagnostics/experiments.csv`.

## Verdict table (compressed; details in RESEARCH_LOG)

WON: recency decay (hl250 two-fold champion; buckets licensed exponential,
power law unidentifiable at 3yr); minutes-only prior (+0.07/+0.10, saturates
c≈8, use 2-4); clean-semantics prior plumbing (flipped stale-prior verdict);
aging curve (delta method; peak 25-27; aging is an OFFENSE phenomenon;
defense flat); game-margin gate itself.
LOST: rubberband (endogenous, twice); season dummies on 3yr; soft-GT weighting
(= hard drop); pooling; elastic net via SGD (optimizer verdict only);
stale priors under OLD semantics (retracted — clean semantics: mild win);
iterated/infinite prior chain (degrades, converges to raw APM — proven);
1yr window (anchor fail); LightGBM possession-level; window ensembles vs decay.
OPEN: v1.2 verdict; calibration fix; third fold; defense features; role
clustering; WP-RAPM; state-space model; era normalization for long windows.

## TODO stack (ranked; next session starts at 1)

1. [VERDICT IN, 2026-07-03 15:00] v1.2 complete: 0.6988/0.6619 — both folds
   below minutes prior (0.7335/0.6953). CONFIRMED: coach-revealed preference
   (minutes) beats the player's own box+tracking stats as a prior for exactly
   the players where priors bind. Defense SPM R² ≈ 0.006-0.035 — box stats are
   blind to defense. Logged; minutes prior is the interim prior product.
   NEXT decisive experiment: RESIDUAL SPM — fit the SPM to predict the
   residual of the minutes prior (what box stats add BEYOND minutes), use
   minutes+residual as center. If residual OOF R²≈0, box adds nothing at prior
   granularity (strong publishable negative); if >0, blend properly. This also
   creates the battleground for the rate-stabilization ablation (user skeptical
   of regressing rates; stabilized 3P% etc. should mainly help the residual SPM).
2. Calibration audit of the winning prior: regress actual margins on predicted
   (train side), check slope<1, apply slope/intercept correction, re-gate.
   Hypothesis: recovers the RMSE creep while keeping corr. ~30 min.
3. Third fold for era robustness: train 2015-17 → test 2018 (cache exists).
   Champion + winning prior only. ~15 min.
4. Ship panel v5: production rerun with winning prior via `final_windows.py`
   (extend it: SPM/minutes prior support), new dated folder, top-20 sanity,
   README + PROJECT update.
5. Walk-forward variant of the prior (train SPM on window T-1, apply to T) —
   required before any FORECASTING product; descriptive product is fine as-is.
6. Defense: add tracking defensive features (contested shots, rim protection)
   to try lifting def-SPM R² off zero; else document "defense = minutes +
   position" honestly.
7. Software debt: split `experiments.py` into `harness.py` (lib) + queue
   script; dedupe champion_fit (3 copies now); loader tolerating experiments.csv
   schema drift. One session, no behavior change, tests: reproduce champion
   gate numbers exactly.
8. git init both projects + rotate DB password out of legacy
   `rapm_with_prior.py` (STILL PENDING, flagged twice).
9. Website/data viewer: **v1 shipped** — `rapm/outputs/viewer/human.html` (FM/xRAPM-style
   panel explorer) + `agent.html` (foundry results for agents). Rebuild via
   `rapm/src/build_human_viewer.py`. Greps: `rapm/OPERATOR_GREPS.md` +
   `outputs/grep_digest.log`.
10. Speed (one session, ~5-10x experiment turnaround; do before big foundry queue):
    a. Cache built design matrices: `scipy.sparse.save_npz` + metadata keyed by
       (seasons, spec-flags, season_type). Every experiment tonight rebuilt the
       IDENTICAL matrix via Python loops (~2 min each; the actual CG solve is
       seconds). Biggest win by far.
    b. Cache pass-1 champion fits (beta, intercept, sigma2) keyed by fold+config
       — every prior script refits it from scratch.
    c. Warm-start CG with the previous beta across c/lambda grids (same dm,
       slightly different penalty → few iterations to converge).
    d. Vectorize `build_design_matrix` (numpy column-stacking instead of
       per-possession Python append loops) for the cold-cache path.
    e. Run the two gate folds as parallel processes (independent; memory is
       small — X is ~200MB). tmux two sessions or multiprocessing.
    Verify after: champion gate numbers reproduce EXACTLY (0.6596/0.5939).
11. Research backlog (ledger has full list): role-clustered priors, rate
    stabilization ablation (user skeptical — test both), WP-RAPM target,
    state-space DARKO-lite (all ingredients exist: decay + aging + SEs),
    era-normalized long-window RAPM, GBM-vs-ridge SPM comparison.

## Principal backlog (user-owned parallel tracks)

Not agent TODO until redirected. Full table in `IDEAS.md` § Principal backlog.

| Track | One-line | Blocked on |
|---|---|---|
| Win-probability RAPM | ΔWP target instead of margin | WP model + bbgm port |
| Draft modeling | pre-NBA → pro impact | draft/college data ingest |
| Player impact product | panel + prior + foundry | minutes prior ships first |
| Skill-based modeling | skills not just off/def | def-SPM + playtype/role structure |

Current agent sprint stays: foundry gen 0+ → speed caches → residual SPM verdict → ship minutes or SPM prior.

## Handoff protocol (any model, fresh context)

Read this file → last 3 RESEARCH_LOG entries → `IDEAS.md` skim → check tmux
(`tmux ls`) and `outputs/*_run.log` tails for running/finished jobs → execute
TODO stack top-down. Long jobs: tmux + caffeinate + done-flag + watcher.
Never trust a result that isn't in experiments.csv. Update this file's
"State of the world" + "Last updated" when the champion or TODO 1-4 change.

## Agent docs vs human docs (deliberate difference)

- Agents need: state, rules, exact paths, banned moves, next actions —
  front-loaded, terse, deterministic. That's THIS file.
- Humans need: narrative, why, pictures, honest uncertainty. That's
  RESEARCH_LOG.md + figures/ (+ future website; TODO 9).
- Both need the conventions block. Neither needs prose duplicated in code
  comments — code explains how, docs explain why, the log explains what happened.
