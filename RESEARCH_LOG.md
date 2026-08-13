# Research Log — NBA RAPM (New SPM project)

Append-only. Newest entries at the bottom.

## 2026-07-03 — Fixed the RAPM validation harness (it was wrong in 3 ways)

**Question:** Why did the ablation scorecard show ~0.0002 RMSE differences between specs — is the validation broken?

**What we did:**
- Audited `evaluate_rapm_models.py` + `standard_rapm.py`.
- Fix 1: garbage-time is now a per-row flag, all specs scored on the same non-garbage validation possessions (previously `no_gt` was graded on its own easier row set).
- Fix 2: lambda tuning now only sees the training side of each split (previously tuned on the full matrix including the held-out season — leakage).
- Fix 3: added game-margin metrics (RMSE + corr of predicted vs actual home-minus-away points per game); summary now ranks on margin RMSE.
- Added per-season parquet possession cache (`data/possession_cache/`): DB fetch of 2M rows went 50s → 4.7s.

**Result:** (2022–24 window, zero prior, default lambdas 3000/3000/300/100)
- Possession RMSE ~1.19 for every spec — confirmed useless as a discriminator [SOURCED: eval run final_v2].
- Game-margin metric separates specs by whole points: player+home wins (margin RMSE 13.9 CV / 14.6 chrono, corr ~0.63). Home effect worth ~0.05 pts over player-only.
- Rubberband term costs ~1.6–2.0 pts of margin RMSE and drops corr to ~0.38–0.44. Caveat: rubberband uses live score margin as a feature, so the margin metric is stacked against it — its real test is rating quality, not margin prediction.
- Season effects: no gain on a 3-year window.
- Training with garbage time included is slightly worse once scored fairly.

**Dead ends:** Possession-level RMSE as a model-selection metric — a 0/2/3-point outcome has so much irreducible noise that any sane spec ties. Do not use it to pick specs again.

**Lesson:** Validate at the level of the thing you ship. We ship player ratings and game-level impact, so the metric must aggregate to at least game margins (better: next-season retrodiction from frozen ratings). Also: any cross-model comparison is invalid unless the validation rows are identical.

**Promote?:** "Score all model variants on a common validation set" belongs in a general validation skill — second time leakage/comparability bugs have bitten this project (first: circular SPM prior).

## 2026-07-03 — Rating-level validation picks the spec; nonlinear probe clears ridge

**Question:** Which spec produces the best RATINGS (not possession fit)? Does a nonlinear model beat ridge? Is a model-based rubberband salvageable?

**What we did:**
- `validate_ratings.py`: split-half reliability (game-split refits, r across halves) + next-season retrodiction (freeze 2021–23 betas, predict all 1,201 games of 2024).
- `nonlinear_probe.py`: LightGBM on frozen lineup ratings ± live score margin, same retrodiction test.

**Result:** [SOURCED: rating_validation_ratings_v1, nonlinear_probe CSVs]
- player+home: split-half r=0.549, next-season margin corr=0.632, RMSE 14.49 (naive baseline 17.89). WINNER.
- +rubberband: split-half 0.565 but corr collapses to 0.419. Rejected.
- LightGBM with live margin: corr 0.424 — same collapse as linear rubberband. The live margin FEATURE is the problem (good teams lead more, so any model routes quality through it), not the linear functional form.
- LightGBM without margin: corr 0.613 — no nonlinear structure left over linear lineup ratings.
- Lambda sweep aborted early: λ∈[1000,8000] barely moves retrodiction. Fixed λ=3000.
- Caveat to check: probe reported def_sum feature importance = 0; verify feature construction.

**Dead ends:** (1) Rubberband/coasting adjustments via any margin-derived feature — endogeneity, not crudeness. A real fix needs an exogenous effort model, not the margin itself. (2) Tree models at the possession level on rating-derived features.

**Lesson:** When a control variable is caused by the thing you're measuring (leading ⟵ team quality), "controlling" for it removes signal, not bias. Endogeneity beats functional-form sophistication every time.

**Promote?:** Next-season retrodiction as the standard model gate for any player-value model (soccer RAPM too).

## 2026-07-03 — Overnight experiment queue: 21 variants, decisive results

**Question:** Which of the brainstormed RAPM knobs actually improve next-season retrodiction (train 2021-23, test all 1,201 games of 2024, baseline player+home ridge corr 0.6317 / RMSE 14.49)?

**What we did:** `experiments.py` — crash-safe sequential queue in tmux, one CSV row per experiment, Gobert/Jokic sign anchors, ESS logging, walk-forward priors only. `combos.py` follow-up merging winners.

**Result:** [SOURCED: outputs/diagnostics/experiments.csv]
- WON: recency decay hl=365d → corr 0.6535 (biggest single gain; ESS healthy at 493K/620K rows). hl=730 → 0.6447.
- WON: asymmetric lambdas, shrinking DEFENSE harder (off 2000 / def 4500 → 0.6347; off 1500 / def 6000 → 0.6353). The reverse direction loses (0.6216). Defense coefficients are noisier and want more shrinkage — now measured, not assumed.
- LOST: previous-window prior (2018-20 → 0.589–0.613, all below baseline). Old ratings are a *biased* shrinkage target — aging + role churn — not free information.
- LOST: infinite RAPM — walk-forward chain 2012→2015→2018→2021 scored 0.5933; degradation compounds with depth. (User's "infinite RAPM" question: answered, it degrades.)
- LOST: soft garbage-time weighting (0.6316 — dead even with the hard drop); replacement pooling at 250/500/1000 poss (~baseline); 5-year window (0.6095).
- REJECTED BY ANCHOR TEST: 1-year window (corr 0.647 but flipped a sign anchor and worst RMSE 14.64 — noisy ratings pointing the right way on average).
- INCONCLUSIVE: elastic net via SGD (0.16–0.32) — the optimizer underfit, not a verdict on sparsity.

**Dead ends:** Stale-prior lane (both variants). Do not revisit priors built from old windows; if priors return, they must be same-window information (i.e., an SPM) — upgraded that from "blocked" to "next-project candidate" in IDEAS.md.

**Lesson:** The two things that worked both amount to "trust recent, noisy-in-the-right-way data more, and shrink the noisiest block hardest." The things that failed all amount to "import information from a different distribution than the one you're predicting." Recency beats history; same-window beats stale.

**Promote?:** The tmux + crash-safe CSV + sentinel-wake pattern worked flawlessly for unattended overnight research; worth making a skill.

## 2026-07-03 — FINAL: production RAPM shipped, 26 windows, 1997–2024

**Question:** Ship the finalized regular-season RAPM with the night's winning config.

**What we did:** `final_windows.py` in tmux — 26 rolling 3-year windows (end seasons 1999–2024), config: player+home, garbage-time filtered, 365-day recency half-life, symmetric lambda=3000, zero prior, SEs/CIs + A/B/C confidence tiers. ~57 minutes total. Output: `rapm/outputs/rapm_results/final_20260703/` (per-window CSVs + `rapm_all_windows.csv`, 16,716 player-window rows + `run_meta.json`).

**Result:** Sanity checks pass everywhere [SOURCED: rapm_all_windows.csv]:
- 2022–24 top-15: Jokic clear #1 (+7.78), then George, D.White, Draymond, Embiid, Kawhi, Giannis, LeBron, Tatum — consistent with xRAPM-class lists. Low-minute Jonathan Isaac correctly flagged Tier B.
- Window ending 2010: LeBron #1 (+6.92), Wade, Dwight, Varejao (the classic RAPM darling), Nash — textbook.
- Window ending 2016: LeBron, Draymond, CP3, Curry, Kawhi.
- Combo round footnote: decay + asymmetric lambdas do NOT stack (0.6532 vs 0.6535 decay-alone); hl=250d raises corr (0.6596) but worsens RMSE — drifting toward the anchor-failing 1yr-window regime. hl=365 is the stable optimum.

**Dead ends:** none new; see previous entry.

**Lesson:** A finished model is a config + a gate + a ledger, not a bigger model. The whole night's improvement (corr 0.632 → 0.654) came from ONE knob (recency decay) surviving a fair gauntlet that killed eight fancier ideas.

**Promote?:** Next projects, in order of expected value: (1) clean same-window SPM prior (the one prior type the night did NOT kill), (2) win-probability RAPM target, (3) trio RAPM on a small window.

## 2026-07-03 — Decay function learned model-free; aging curve built; new champion hl=250

**Question:** What is the true decay shape (user predicted "between exponential and power law")? What does the aging curve look like from our own panel?

**What we did:**
- D0 `d0_bucket_decay.py`: 6 free bucket weights tuned on retrodiction (55 Nelder-Mead evals).
- D1+D2 `d2_decay_folds.py`: curve-fit families to the buckets, then confirmed all variants on TWO folds (2021-23→2024 AND 2020-22→2023).
- A1+A2 `aging_curve.py`: delta-method aging curve from the 1-yr RAPM panel (13,396 player-seasons, ages already joined), survivorship sensitivity, off/def split, level-dependence regression, year-ahead player gate.

**Result:** [SOURCED: decay_fold_confirmation.csv, aging_curve_delta.csv/png/meta]
- Learned buckets are NON-monotone [1.0, 1.18, 0.26, 0.07, 0.33, 0.49] — wiggle is single-fold overfit; on two folds the simple exponential beats them.
- Power-law fit DEGENERATED to the exponential limit (tau→huge with alpha/tau fixed): at 3-year horizons exp vs power law is empirically indistinguishable. User's fat-tail intuition untestable at this window length.
- NEW CHAMPION: exponential decay hl=250 days — wins BOTH folds (0.6596/0.5939, mean 0.627 vs hl365's 0.619, no-decay 0.592). Anchors pass (from overnight run).
- Aging curve: peak 25-27; ~-4.5 at 19; ~-4 by 36. AGING IS AN OFFENSE PHENOMENON — defense is nearly flat across career. Survivorship penalty steepens old-age decline (truth between curves). Level dependence b=-0.40 (higher-rated decline more; partly measurement-error mean reversion — needs SE-aware model to separate).
- Aging-translation gate: year-ahead player MAE improves 1.974→1.920; corr ticks down 0.522→0.511 (calibration gain, ordering wash).

**Dead ends:** Free-form bucket weights as a production choice (overfit single folds; use them only to license a parametric form). Power law at 3-yr windows (unidentifiable vs exponential).

**Lesson:** Fit free, read the shape, ship the simple form that survives multiple folds. And a non-monotone learned weight pattern is a leakage/overfit alarm, not a discovery.

**Promote?:** Two-fold confirmation should be the standard gate for any weighting/hyperparameter choice (single-fold wins are hypotheses, not results).

## 2026-07-03 — Aged prior: mechanism validated, strategy dominated

**Question:** Does translating the 2018-20 prior through the aging curve (user's idea) fix the stale-prior failure?

**What we did:** `s_aged_prior.py` — off/def coefficients of the 2018-20 fit translated by f_off/f_def over the 3-year gap (ages from the career panel), used as shrinkage target for 2021-23 with hl250 decay, strengths {1,2,4}, vs raw prior and vs zero-prior champion.

**Result:** [SOURCED: aged_prior_results.csv] Aged beats raw at every strength (0.6403 vs 0.6224 at best) — the staleness diagnosis and the aging cure are both CORRECT. But all prior variants remain below zero-prior + hl250 decay (0.6596). Recency weighting already extracts old information more cheaply than shrinking toward it.

**Dead ends:** Global-strength stale priors, even age-corrected, on top of a well-decayed likelihood.

**Lesson:** Two mechanisms can both be real and still not stack: decay and priors are substitute routes to the same old information, and the cheaper unbiased route (decay) wins. Test combinations, never assume additivity.

**Promote?:** Per-player adaptive prior strength (prior only where current data is thin) — the one untested corner of the prior lane, and the theoretically correct one (that's literally what a posterior does).

## 2026-07-03 — Production rerun with new champion config

final_v4 launched: 26 rolling 3-year windows, hl=250d decay (two-fold champion), otherwise identical to final_v3 (player+home, GT filter, lambda 3000, zero prior, SEs + tiers). Output: `outputs/rapm_results/final_20260703_hl250/`.

COMPLETED same day: all 26 windows green, 16,716 rows. Top-10 sanity passes (Jokic +6.97 #1; SEs slightly tighter than hl365 panel). README updated; hl250 panel is the production artifact.

## 2026-07-03 — Minutes-only prior: largest single win of the project; stale-prior verdict overturned

**Question:** Does ANY same-window prior help (machinery test with q=1 feature), under clean semantics?

**What we did:** `spm_minutes_prior.py` — pass-1 champion fit gives labels; OOF quadratic regression of player coefficients on log(possessions) gives prior center beta_0 and tau^2 (OOF residual variance); pass-2 refit with champion penalty UNCHANGED plus separate per-player pull c*sigma^2/tau^2 toward beta_0. Both folds. Stale 2018-20 prior re-baselined under same semantics as control.

**Result:** [SOURCED: experiments.csv mprior_* rows]
- Fold 2024: champion 0.6596 -> c=2 gives 0.7335 (+0.074). Fold 2023: champion 0.5939 -> c=2 gives 0.6953 (+0.101). Anchors pass everywhere. Largest improvement in project history (decay was +0.03).
- Prior quality is modest (OOF R^2 ~0.16, tau ~1.07/100) but applied where it matters: fringe players. Implied lam0 ~12,000 possessions of evidence.
- RMSE trade: corr rises monotonically in c but fold-2024 RMSE creeps up (14.55->14.86 at c=2); fold-2023 RMSE flat. Ordering improves, scale slightly inflates — calibration check pending. Extended grid c={4,8,16} running.
- CONTROL: stale 2018-20 prior under CLEAN semantics now BEATS champion (0.6644 vs 0.6596) — the overnight "stale priors lose" verdict was substantially a plumbing artifact (confounded lambda scaling + global strength). User's suspicion ("older experiments were done poorly") confirmed and consequential.

**Dead ends:** none new; overnight prior magnitudes formally retracted (directional aged>raw comparison stands).

**Lesson:** When an idea with sound theory loses, audit the plumbing before accepting the result. The difference between "prior loses by 0.05" and "prior wins by 0.07" was parameterization, not information.

**Promote?:** Clean prior semantics (separate per-player pull, tau^2-derived strength) is now THE pattern for all future prior work — soccer included.

## 2026-07-03 — SPM v1: a leakage class discovered; v1.1/v1.2 chase the minutes prior

**Question:** Does a 56-feature box+tracking SPM beat the minutes-only prior?

**What we did:** `build_spm_features.py` (audit: 100% coverage all 26 windows, possession-weighted; tracking tier valid from 2001, not 2014 as assumed). `spm_v1.py`: OOF weighted ridge per side → prior center + tau² → pass-2 refit, both folds.

**Result:** [SOURCED: spmv1/spmv11 logs, experiments.csv]
- v1 PARADOX: OOF R² 0.54/0.45 (3x the minutes prior) but gate WORSE than minutes (0.665/0.606 vs 0.734/0.695) and degrading in c.
- DIAGNOSIS: `OnOffRtg`/`OnDefRtg` are computed from the same possessions as the labels — the feature itself carries the label's noise. OOF-by-player cannot block this (leakage inside the feature, not the split). R² was inflated, tau² underestimated, prior over-trusted.
- v1.1 (features dropped): R² fell to 0.39 (confirming the diagnosis — the gap WAS shared noise) and the gate ROSE to 0.699/0.668. Still below minutes.
- v1.1 exposed: defensive SPM OOF R² ≈ 0.01-0.04 (box stats don't see defense) while sharing one tau² with offense — over-pulls defense; and features lacked the log-poss shape that made the minutes prior work. v1.2 (per-side tau², log-poss features) launched; verdict pending in `spmv12_run.log`.

**Dead ends:** Any feature computed from the same possessions as the label (on/off ratings, team ratings, plus-minus family). New banned class alongside rubberband.

**Lesson:** Leakage hides in features, not just in splits. "Better R², worse product" is the signature — R² measures fit to labels including their noise; the gate measures signal only. When a diagnostic improves while the product degrades, suspect contamination.

**Promote?:** Feature provenance check before any supervised fit: "could this feature share noise with the label?" — belongs in a validation skill.

## 2026-07-03 — Feature Foundry Phase 0 shipped; harness reproduces minutes; SPM v2 pooled still loses

**Question:** Can we build a Karpathy-style foundry with a fixed RAPM×SPM splice, team L2 report, and fair comparison vs the minutes prior?

**What we did:**
- `feature_eval.py`: two-pass splice (pass-1 hl250 → prior → pass-2), L1 game-margin gate, L2 team net/wins corr, folds f24/f23/vault.
- Baseline repro through new harness: minutes prior c=2 → **0.7346 / 0.6954** (matches prior runs 0.7335/0.6953).
- `spm_v2.py`: pooled ~13.7K player-windows (time leave-out), CV alpha (picked 1000), heteroskedastic τ²(log poss), residual SPM lane.
- Infra: `rapm/features/` (program.md, prepare.py, data_allowlist.yaml, results.tsv), `verify_keep.py`, `data_ingest.py`, `data_curator.py`, `feature_submit.py`, `feature_foundry.py`, `fig_foundry_progress.py`.
- Foundry gen 0 launched overnight (minutes baseline all folds).

**Result:** [SOURCED: feature_eval_baseline.log, spmv2_run.log, experiments.csv]
- Harness validated — do not trust foundry until this passed; it passed.
- SPM v2 pooled best: f24 **0.7304** (c=4), f23 **0.6983** (c=4) — still below minutes c=2 on both folds.
- Residual SPM: f24 0.7239, f23 0.6874 — no lift from box beyond minutes at prior granularity.
- Pooled OOF R² off/def: ~0.01/-0.09 (f24 pooled) — estimation fix did not unlock box signal on offense in this spec; defense still ~0.
- Team L2 (report-only): minutes c=2 f24 team_wins_corr **0.653**; pooled SPM c=2 f24 **0.652** (similar).

**Dead ends:** None new — confirms minutes prior remains interim product until foundry discovers something that clears f24+f23+vault with anchors.

## 2026-07-03 — Autoresearch loop: build.py candidates (not column subsets)

**Question:** Are we actually building new features for autoresearch, or re-filtering existing box parquet?

**What we did:**
- Honest audit: gen 2–5 and SPM v2 soup were **column ablations only** — not new research. Minutes re-runs wasted ~30+ min.
- Added `candidates/gen_NNN/build.py` pattern + `candidate_build.py` + `type: build` in foundry.
- Gen 006: 11 NEW derived features (ratios, z-scores). Gen 007: playtype PPP (sparse). Gen 008–009: tracking interactions + blend.
- `autoresearch_loop.py` + `autoresearch_proposer.py` + `scripts/run_autoresearch.sh` — pending queue → evaluate → propose → repeat.
- Stale lock cleanup in `run_lock.py` (dead pids were blocking all runs).

**Result:** Pipeline complete for v1 autoresearch. Overnight queue: gens 6–9 then proposer templates (residual SPM, etc.). Gate truth unchanged: beat frozen minutes 0.7335/0.6953 on f24+f23.

**Promote?:** Next increment is LLM proposer writing `build.py` — harness + loop are ready.

**Lesson:** The harness is the product; features are plugins. Log everything (`results.tsv` + `experiments.csv` + registry).

**Promote?:** `feature_eval.run_splice()` as the only gate entrypoint — ban ad-hoc prior scripts for comparisons.

## 2026-08-08 — Independent reset; Net Points and RL/Shapley scoped

**Question:** What should the NBA project do next after treating the existing RAPM,
SPM, validation conclusions, and frontend as stale or unverified? How should event
credit, playoffs, roles, cloud ingestion, and the 2026 Sloan RL/Shapley paper fit?

**What we did:** Read the principal's RAPM research note; independently audited the
available data and validation paths; reviewed ESPN's published Net Points explainer,
Dean Oliver's public credit-allocation discussion, the complete Sloan
DRL/temporal-difference/Shapley paper, and current Cloudflare storage/compute
pricing. Created `docs/planning/RESEARCH_BACKLOG.md` as the dependency-aware planning page.

**Result:**
- Net Points is a descriptive event-accounting product, distinct from RAPM. Our
  version must conserve team value and expose residual/unobserved credit.
- The Sloan paper is replication-worthy, not a benchmark to accept on authority.
  Its TD, replacement-coalition, neural-Shapley, synergy, and forecast claims need
  separate ablations and time-safe revalidation.
- Playoff work should first distinguish rotation tightening from per-possession
  translation and use cutoff-time projected lineups for deployable tests.
- Offense/defense components remain useful, but promotion is gated primarily on
  recombined out-of-time team/game prediction and component calibration.
- R2 is inexpensive object storage, not compute or a faster home upload. Cloud-side
  ingestion can bypass the slow residential connection; local overnight ingestion
  remains a fallback for access-sensitive sources.

**Dead ends / cautions:** Do not label passive value-function fitting as a solved
causal credit problem; Shapley allocates a model's prediction relative to a chosen
baseline. Do not make injuries, contracts, draft, or RL the current sprint before
the canonical event and identity layer exists.

**Promote?:** P0 in `docs/planning/RESEARCH_BACKLOG.md`: canonical data repair, independent simple
RAPM, new chronological folds, model registry, and resumable ingest.

## 2026-08-08 — P0 implementation: immutable ingest, quality gates, independent RAPM

**Question:** Can we begin with the data already on disk, patch the critical gaps,
and get an independent model/data spine running without inheriting the earlier
implementation's conclusions?

**What we did:** Built the clean `nba_impact` package, resumable manifest downloader,
SHA-256 sidecars, source-aware event and possession audits, whole-game structural
quarantine, DuckDB registry, independent zero-prior offense/defense RAPM, immutable
artifacts, and a one-fold regularization diagnostic. Downloaded an Apache-2.0
bootstrap of NBA Stats V3 events, PBPStats possessions, shot detail, and matchup
detail. Added tests and an operator guide in `docs/planning/NBA_IMPACT_BUILD.md`.

**Data result:**
- Current connection measured about 10.8 Mbps; the 12-file bootstrap completed
  locally with resume/checksum support.
- Complete NBA Stats V3 event partitions now exist for 2024 and 2025 regular
  seasons (1,230 games each) and playoffs (84 and 85 games). Shot and matchup
  partitions cover 2025 playoffs as well.
- Every downloaded file passes its source-level schema, identity-key, season, and
  season-type checks.
- Cross-source reconciliation finds one missing game in 2024 regular-season matchup
  detail: `0022400856`. The event and shot partitions contain all 1,230 games.
- Legacy RAPM cache quality: 2022 and 2023 pass; 2024 contains one structurally bad
  game (`0022300835`, duplicated home player ID) and is quarantined; 2025 is an
  empty cache and fails critically.

**Independent model result:** Fixed-lambda zero-prior RAPM trained on 2021–2023 and
lineup-conditioned on all valid 2024 regular-season games scored margin RMSE 14.268,
MAE 11.134, and correlation 0.409 across 1,229 games. Predicted margin SD was 6.30
versus 15.62 actual, with calibration slope 1.01. This is a reproducible baseline,
not a production rating. A six-candidate penalty diagnostic barely moved results;
best RMSE was 14.260 at offense/defense lambdas 2000/4500, while the highest
correlation was 0.412 at 1000/1000. No candidate is promoted from one fold.

**Important correction to prior evidence:** The legacy validation's reported
1,201-game 2024 sample was created by a date-based postseason cutoff that removed
late regular-season games. It also dropped estimated garbage-time possessions from
the holdout target before aggregating “game margin,” so that target was not the
actual final score margin. Its ~0.63 correlation is therefore not comparable to the
clean all-game 0.409 result and must not be used as a production gate.

**Engineering result:** Vectorizing lineup overlap checks and sparse design
construction reduced the four-season fit from roughly four minutes to about 53
seconds on this machine. Nine tests pass.

**Next:** Build the event-to-game/stint/possession silver layer with exact score and
minutes reconciliation. That bridge is required for current RAPM, honest
garbage-time experiments, win probability, Net Points, and WP-RAPM.

**Dead ends / cautions:** Player-name sanity checks are not validation. One
chronological season is a diagnostic, not hyperparameter selection. Do not mix the
source-native matchup-detail table with the legacy lineup possession table despite
their similar names.

## 2026-08-08 — Gabriel mirror refresh and four-fold RAPM evidence gate

**Question:** Can Gabriel's live data mirror patch recent aggregate coverage, and
how many model runs are enough to say a three-year RAPM idea is better?

**What we did:** Inspected `gabriel1200/site_Data` at pinned commit
`bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd` (2026-07-05). Extended the safe
downloader to validate CSV schemas and widths. Downloaded 20 pinned assets with
atomic writes, hashes, row counts, source revision, and upstream-license status.
Implemented a multi-fold three-year-to-next-year RAPM evaluator with paired
within-season game bootstraps and explicit evidence labels.

**Data result:** The refresh adds complete-ID aggregate playtype (2014–2026),
tracking (2014–2026), player shooting (2014–2026), shot zones (2001–2026), hustle
(2018–2026), passing (2014–2026), and the LEBRON identity panel (2010–2026).
The DFG master remains 91% null on player ID and is not safe as an ID-keyed
canonical table. Contract assets include 389 salary rows, 389 option rows, 328 cap
holds, 11 dead-money rows, and cap tables; they are raw inputs without stable NBA
player IDs. The upstream repository declares no license, so the manifest records
`not-declared-by-upstream` rather than assuming reuse rights.

**Model result:** Four outer folds used 3-year training windows and test seasons
2021–2024. Baseline `off=3000/def=3000` mean margin RMSE was 13.893. Asymmetric
`off=2000/def=4500` reached 13.883: a 0.08% improvement, 2/4 fold wins, and 87.35%
paired-bootstrap probability of lower loss. This does not clear the research gate.
`1000/1000` was 1.12% worse on RMSE and won 0/4 folds.

**Answer on run count:** RAPM is deterministic, so repeated seeds on the same split
add no evidence. One baseline-vs-idea screen needs at least four chronological
outer folds (eight fits total). A publishable claim should use six to eight folds
plus paired uncertainty and one untouched confirmation season. Stochastic models
add at least three seeds per fold.

**Dead ends / cautions:** Do not run Gabriel's scraper scripts unchanged in
production: many are hard-coded to one year or season type, overwrite CSVs directly,
and lack timeouts, retries, atomic writes, and schema gates. Treat the repository as
a useful pinned mirror and endpoint notebook until each source is independently
wrapped.

## 2026-08-08 — Canonical game dimension exposes season-label mismatch

**Question:** Can the new event sources share one reliable game identity for win
probability, Net Points, current RAPM, and playoff analysis?

**What we did:** Built a silver `game_dim` from NBA Stats V3 events, shot detail,
PBPStats, and matchup detail. Added explicit source-season, start-year, end-year,
season-label, phase, date provenance, home/away IDs, final scores, overtime,
source-coverage flags, content-addressed lineage, and registry integration.

**Result:** 2,629 unique games across 2024–25 and 2025–26: 2,460 regular-season and
169 playoff games. All games have dates, final scores, and two consistent teams.
There are zero duplicate game IDs, duplicate V3 action IDs, source-date conflicts,
team-code conflicts, or matchup team-ID conflicts. Shot detail supplies 2,545 games,
PBPStats supplies 1,314, and matchup detail supplies 2,544; their complementary
coverage yields a complete game dimension. Play-in games are absent from the archive
and are stated as out of scope.

**Critical semantic finding:** `nba-data-archive/season=2024` means the season
starting in 2024 (2024–25), while legacy `matchups_2024.parquet` means the season
ending in 2024 (2023–24). Any join on the integer `season` alone is off by one year.
Canonical tables must use `season_start`, `season_end`, or `season_label` explicitly.

**Next:** Within-game score-state normalization, then lineup stints and possessions
with final-score and player-minute reconciliation.

## 2026-08-08 — Reconciled event states and chronological win probability v0

**Question:** Can the current event archive support a trustworthy score-state layer
and a first leakage-safe live win-probability baseline?

**What we did:** Normalized NBA Stats V3 actions using `actionId` feed order, parsed
clocks, created regulation/overtime time state, reconstructed pre/post scores,
joined canonical games, and added terminal reconciliation. Sampled one state per 30
seconds plus exact opening and terminal states. Fit a fixed logistic state model on
2024–25 and evaluated once on untouched 2025–26 data.

**Data result:** 1,313,486 actions across 2,629 games pass all critical gates: no
duplicate event IDs, missing games, invalid clocks, time-order reversals, terminal
score mismatches, or nonzero terminal clocks. Scores reconstructed from made field
goals and made free throws match all 2,629 final scores exactly. Sparse upstream
score snapshots disagree on 53 correction rows, and `actionNumber` backtracks on
11,980 rows; therefore `actionId` is chronology and `actionNumber` is metadata.

**Model result:** Test set contains 1,315 games and about 129K uniformly sampled
states. Overall Brier is 0.1622, log loss 0.4769, AUC 0.8367, calibration slope 1.04,
and Brier skill versus the constant training home-win rate is 34.4%. Checkpoint
Brier improves from 0.247 at game start to 0.169 at halftime, 0.124 at the start of
the fourth, 0.088 with six minutes left, 0.057 with two minutes left, and 0.045 with
one minute left. Game-start AUC is correctly 0.5 because v0 has no pregame-strength
inputs.

**Interpretation:** This is a calibrated state baseline, not a finished win model.
The clean next ablations are pregame team/player strength, possession indicator,
timeouts/bonus, and availability—each added one at a time against the same test
contract. Final states are set deterministically from the completed score.

**Dead ends / cautions:** Sorting by `actionNumber` creates thousands of clock
reversals because the feed retroactively renumbers corrected actions. Forward-filling
the source score snapshots also creates false score decreases. Both approaches are
rejected.

## 2026-08-08 — Current starter seeds and five-player lineup stints

**Question:** Can 2024–25 and 2025–26 lineups be reconstructed accurately enough
to unblock current RAPM without relying on the stale legacy possession cache?

**What we did:** Added a pinned, resumable ingest for CDN NBA event partitions and
the `llimllib/nba_data` NBA/ESPN mirrors. NBA player-game positions supply starter
seeds; ESPN player boxes add independent starter checks plus Net Points and WPA
benchmarks. Reconstructed substitution stints, inferred omitted between-period
lineups in the late-playoff V3 fallback, and reconciled every player's stint seconds
against official box minutes.

**Result:** The canonical player-game table contains 69,517 rows and all 2,629
games. Every team-game has exactly five starters, team minutes reconcile, and the
2,627 ESPN-overlap games have zero starter disagreements. The lineup table contains
81,893 validated stints covering 2,622 games (99.73%). Seven games (0.27%) are
quarantined for feed/substitution or minute-reconciliation anomalies. Emitted rows
have zero duplicate stint IDs, nonpositive durations, invalid five-player sides, or
home/away player overlap. The five-second player-minute tolerance and 0.5% maximum
quarantine budget are explicit pipeline parameters.

**Source/legal note:** `cdechoch/nba-data-archive` declares Apache-2.0. The
`llimllib/nba_data` repository declares no license, so its NBA/ESPN-derived fields
are research-only and must not be redistributed until rights are clarified. ESPN
Net Points/WPA are benchmarks, not labels to copy into our own metric.

**Next:** Convert validated stints plus event states into possession outcomes, then
fit a current zero-prior RAPM baseline. Keep the seven quarantined games excluded
until a second source or a provably minute-consistent repair resolves them.

## 2026-08-08 — Ordinal possessions and current RAPM sensitivity

**Question:** Can the current two-season event archive replace the stale legacy
RAPM cache without silently misordering events or assigning same-clock actions to
the wrong lineup?

**What we did:** Built CDN possessions in physical `orderNumber` order and retained
every within-possession lineup change in a separate ordinal segment table. Used
`actionNumber` only as a guarded V3 score join when game, period, and clock agree.
Replayed substitutions in source order, keeping the prior lineup active until a
complete out/in transaction exists. Fit identical zero-prior RAPM models using the
lineup at possession start and at possession end as explicit sensitivity variants.

**Data result:** 525,279 possessions and 633,568 lineup segments cover 2,598 of
2,629 games (98.82%). The 31 excluded games are 25 CDN games after the May 9, 2026
source cutoff plus lineup-quality quarantines, with one overlap. All 2,598 included
games reconcile exactly to final scores. There are zero duplicate IDs, invalid
point outcomes, implausible game possession counts, or possession/segment point
mismatches. 76,354 possessions cross at least one lineup change, which confirms
that a single implicit clock-based lineup would be a lossy contract.

**Model result:** On 2024–25 regular season training and 2025–26 regular season
lineup-conditioned retrodiction, the start-lineup model has margin RMSE 15.798 and
correlation 0.285; terminal-lineup has RMSE 15.733 and correlation 0.294. A paired
10,000-draw game bootstrap gives a 99.73% probability that terminal-lineup squared
error is lower, with mean MSE delta -2.05 and 95% interval [-3.50, -0.58]. The
effect is real but small: about 0.4% relative RMSE, not a production verdict.

**Cautions:** This is one outer season, uses observed future lineups, and knows only
82.8% of test-season players from the training season. It is retrodiction, not a
forecast. CDN coverage stops during the 2026 playoffs, and the existing V3 fallback
must be validated prospectively before those 25 games enter possession RAPM.

**Next:** Add time-safe pregame team strength to win probability, then compare
possession-start state models. For RAPM, add a formal start-vs-terminal-vs-segment
comparison harness and more seasons before selecting a production lineup policy.

## 2026-08-08 — Time-safe pregame Elo materially improves win probability

**Question:** Does a simple pregame strength signal improve the state-only win
model without using current-game or future information?

**What we did:** Added a fixed Elo ablation (1500 initial rating, K=20, 60-point
home advantage, 25% offseason regression). Every game on the same date receives
ratings from the start of that date; results update ratings only after the entire
date is scored. Compared state-only and state-plus-Elo logistic models on identical
30-second states, training on 2024–25 and holding out all 2025–26 games. Terminal
states were excluded from the comparison, and uncertainty resampled whole games.

**Result:** Across 128,232 nonterminal test states and 1,315 games, Elo reduces
Brier from 0.16385 to 0.14987 (8.54%), log loss from 0.48175 to 0.44855, and raises
AUC from 0.83339 to 0.86294. The 5,000-draw paired game bootstrap favors Elo in
100% of draws; mean game-level Brier delta is -0.01404 with 95% interval
[-0.01949, -0.00875]. At game start, Brier improves 0.24708 to 0.21175 and AUC
improves from the expected 0.500 to 0.720. Elo's test calibration slope is 0.971.

**Interpretation:** Pregame strength is a large missing input, not a marginal
tweak. This is still one outer season and basic team Elo will lag trades, injuries,
rookies, and lineup changes. Keep it as the first promoted research challenger,
not a final production model. Older WP artifacts without a source-code hash are
superseded for provenance; new runs fingerprint both data and implementation.

**Next:** Build a possession-start WP table and add current possession as a
same-state ablation. Then replace team-only strength with frozen pregame player and
projected-lineup strength once availability/projection data exists.

## 2026-08-08 — ESPN benchmark and prior-season starter RAPM ablation

**Question:** Where does the local WP model lag ESPN, and does frozen prior-season
starter RAPM close the pregame gap without using current-game outcomes or minutes?

**What we did:** Cached ESPN play-level WP for 1,314 of 1,315 2025–26 games and
matched ESPN and local post-action states on game, period, score, and clock. The
single unavailable game has ESPN play-by-play but no WP records. For the lineup
ablation, 2024–25 games use RAPM fit only through 2023–24; 2025–26 games use RAPM
fit only through 2024–25. Official starters are treated as tipoff-time information.
Missing prior ratings are centered at zero; actual minutes and current-game results
are excluded. All comparisons resample whole games.

**External benchmark:** The validated run `wp_espn_benchmark_v1_ca79cde82d`
scores 631,380 matched nonterminal plays across 1,313 games with 99.26% raw
play-match coverage. ESPN Brier is 0.14759 versus 0.14883 for local state plus Elo.
The equal-game Brier-difference interval crosses zero. ESPN's clear advantage is at
tipoff: 0.20210 versus 0.21181, with local-minus-ESPN 95% interval
[0.00451, 0.01495]. From halftime onward the models are statistically tied at the
predeclared checkpoints.

**Starter-RAPM result:** Run `wp_lineup_ablation_v1_7570ad01c9` reduces overall
Brier from 0.14987 to 0.14922 and tipoff Brier from 0.21181 to 0.21057; tipoff AUC
rises from 0.7195 to 0.7226. Evidence is inconclusive: the tipoff paired bootstrap
favors the starter model in 70.24% of draws and its 95% delta interval
[-0.00562, 0.00319] crosses zero. It remains clearly worse than ESPN at tipoff:
local-minus-ESPN Brier delta 0.00847, 95% interval [0.00322, 0.01393]. Prior ratings
cover 91.96% of 2024–25 starters and 92.34% of 2025–26 starters on average; some
games cover only four of ten starters.

**Verdict:** Do not promote the starter-RAPM feature. The direction is plausible,
but one outer season and incomplete rookie/new-player coverage are insufficient.
The next pregame challenger should use expected rotation minutes, rookie/translation
priors, injuries/availability, and rest, while keeping ESPN as an external benchmark.

## 2026-08-08 — Rolling team context closes most of the ESPN tipoff gap

**Question:** Does a strictly time-safe rolling margin and rest signal add value
beyond Elo and frozen prior-season starter RAPM?

**What we did:** Added exponentially updated team point differential (10% update
weight, 75% offseason retention) plus rest advantage capped at seven days. Features
for every game on a date are computed before any result from that date updates the
ratings. The candidate retains Elo and prior-season starter RAPM and uses the same
2024–25 train / 2025–26 outer test split and whole-game bootstrap.

**Result:** Run `wp_pregame_ablation_v2_522e1a36f2` lowers overall Brier from
0.14922 for Elo plus starters to 0.14731. The paired mean game-level delta is
-0.00195, 98.56% of draws favor the context model, and the 95% interval
[-0.00373, -0.00020] excludes zero. At tipoff it improves Brier from 0.21057 to
0.20592 and AUC from 0.7226 to 0.7348; all 5,000 paired draws favor context over
starter RAPM. ESPN remains directionally better at 0.20210, but the paired
local-minus-ESPN interval [-0.00063, 0.00839] crosses zero.

**Verdict:** Promote rolling team context as the next research challenger, not as
production. It closes about 61% of the original Elo-to-ESPN tipoff Brier gap, but
only one outer season exists and the fixed smoothing choices have not been tested
across folds. Next acquire an earlier event season, freeze a small smoothing grid,
and require multi-season improvement before promotion.

## 2026-08-08 — Inpredictable neutral-state surface audit

**Question:** Does the local score/time probability shape agree with an independent
published calculator, and what missing state matters most?

**What we did:** Queried Inpredictable's public zero-spread calculator at eight
checkpoints, eleven margins from -15 to +15, and possession/no-possession. Removed
the local model's home intercept by averaging mirrored team orientations, then
compared it with the midpoint of Inpredictable's two possession states. This is a
reference-surface comparison, not outcome validation.

**Result:** Run `wp_inpredictable_surface_v1_56696b0386` covers 88 neutral states.
Surface correlation is 0.9983, mean absolute difference 1.64 percentage points,
RMSE difference 2.32 points, and maximum difference 5.97 points. Inpredictable's
possession input moves probability by 2.87 points on average; in a tie with ten
seconds left it moves probability by 23.4 points.

**Verdict:** The local score/time functional shape is independently plausible.
Possession/control is the highest-value missing in-game state. Build a causal
possession-start table before adding it; do not join the raw action possession tag
naïvely or interpret this surface agreement as predictive accuracy.

## 2026-08-08 — Causal possession-start WP improves late-game accuracy

**Question:** Does knowing which team controls the ball improve WP on held-out
games after score, clock, Elo, prior starter RAPM, rolling margin, and rest?

**What we did:** Rebuilt the canonical possession table with separate home and
away point deltas. A possession-start score is the cumulative sum of completed
prior possessions only; the current possession's points and every future result
are excluded. The first attempted shortcut—assigning all possession points to the
tagged offense—failed score conservation in 417 games because technical/free-throw
sequences can score for the other side. That representation was rejected. Three
logistic models were fit on identical 2024–25 starts and tested on 2025–26:
pregame context, context plus a possession sign, and possession with time-pressure
interactions. Uncertainty resampled whole games 5,000 times.

**Result:** Run `wp_possession_start_v1_9af34729ef` covers 261,222 test starts in
1,288 games. The time-interacted possession model lowers Brier from 0.146513 to
0.146325; mean game-level delta is -0.000189 with 95% interval
[-0.000214, -0.000163], and all draws favor the candidate. In 2,497 regulation
states inside two minutes with margin at most three, Brier improves from 0.17122
to 0.16719 and AUC from 0.82594 to 0.83474. Its fitted home-possession swing is
2.04 percentage points overall, 11.57 points in close last-two-minute states, and
19.67 points when tied inside ten seconds; Inpredictable's corresponding public
tied-ten-second swing is 23.4 points.

**Verdict:** Possession/control is retained as a research feature and the
time-interacted form beats a constant possession effect. It is not production yet:
this is one chronological outer fold, only CDN-covered lineup-reconciled games are
eligible, and the 2023–24 confirmation fold is not downloaded. The causal state
contract and failed score-attribution shortcut are now regression-tested.

## 2026-08-08 — Fractional and terminal lineup RAPM beat start-lineup attribution

**Provenance:** Fractional exposure was designed and implemented inside this
repository in commit `db4cb02`. It was motivated by the canonical segment table,
which preserves substitutions inside a possession. It is not attributed to a
paper or claimed as an established RAPM method. Treat it as a project-specific
sensitivity analysis unless later literature review finds a direct precedent.

**Question:** When substitutions occur inside a possession, should RAPM credit the
lineup at the start, the lineup at the terminal event, or every lineup fractionally?

**What we did:** Compared fixed zero-prior ridge RAPM models on the exact same
497,177 regular-season possessions. All use 2024–25 for training and retrodict game
margins in 2025–26 with observed lineups. Start and terminal select one lineup.
Fractional assignment keeps one outcome row per possession and gives each player
their elapsed-time share across ordinal lineup segments; possessions with zero
clock span fall back to action-count shares. Every fractional row sums to five
offensive and five defensive player exposures. Uncertainty resamples 1,228 whole
held-out games 5,000 times.

**Result:** Run `rapm_lineup_policy_v1_23149bbb29` gives held-out game-margin RMSE
15.798 for start, 15.733 for terminal, and 15.723 for fractional. Terminal minus
start squared-error delta is -2.048 with 95% interval [-3.463, -0.665]; fractional
minus start is -2.342 with interval [-3.303, -1.420]. Fractional is only
directionally better than terminal: delta -0.294, 74.5% probability better, 95%
interval [-1.082, 0.510]. Net-rating correlations with start remain 0.970 for
terminal and 0.984 for fractional, so the policy changes some rankings without
creating an unrelated metric.

**Verdict:** Reject start-lineup attribution for current RAPM. Use terminal lineup
as the simplest provisional production policy because fractional exposure has not
shown a reliable advantage over it. Keep fractional exposure as the research
challenger: it is conceptually fairer for possessions spanning substitutions, but
its sub-second timing is approximated and only one outer season exists. Neither
result is a true forecast; 82.8% player carryover and observed test lineups make it
lineup-conditioned retrodiction.

## 2026-08-08 — Nonlinear WP architecture ladder frozen before testing

**Question:** Should the next WP model be a CNN, recurrent network, transformer,
or a stronger tabular model?

**Evidence:** Published NBA work supports dynamic Bayesian smoothing and reports
small neural/recurrent mixture-density gains over logistic and random-forest
baselines, but the latter study's preprocessing data are proprietary. Generic TCN
research makes causal dilated convolutions a credible efficient sequence baseline.
TFT, NBA2Vec, and Set Transformer motivate heterogeneous temporal inputs and
permutation-invariant lineup embeddings; the Sloan distributional TD/Shapley work
is more directly relevant to the later player-credit lane than to the first WP
replacement. These sources do not establish that a transformer beats simpler NBA
WP models on our states or seasons.

**Decision:** Freeze the comparison order before the third rich season arrives:
dynamic Bayesian/GAM, tree boosting, small MLP, causal TCN, parameter-matched GRU,
then a two-layer causal transformer. A player-set encoder and recurrent mixture
density/value model follow only after the core comparison. Candidates receive
identical states, two chronological outer folds, whole-game bootstrap, calibration
gates, and five fixed seeds for optimizer noise. A seed is not an independent run.
Production remains the smallest model that clears both folds.

## 2026-08-08 — 2023–24 rich-event backfill clears the canonical gates

**Question:** Can the earlier season support an independent WP fold without
weakening lineup or score-conservation validation?

**What we did:** Downloaded the pinned ten-file 2023–24 event batch (86.96 MB)
with checksums and resumability. Added the missing CDN schema contract, supported
the older V3 schema without `shotValue`, and derived action scores from sparse
scoreboard snapshots. The historical NBA box cache contained 15,261 exact
duplicate rows and almost no 2023–24 coverage, so exact duplicates were removed
and ESPN starters/minutes were used as explicitly marked research fallbacks.
The first lineup rebuild quarantined 63 games. We downloaded official
BoxScoreTraditionalV3 JSON for the 56 affected 2023–24 games, then rebuilt every
silver table. Possession scoring was changed from broad V3 action-number
replacement to CDN `orderNumber` score deltas with V3 repair only when an
official terminal-score gap proved the CDN row wrong.

**Result:** Game dimension and event states pass for 3,941 games and 1,950,498
actions. Player-games cover all 3,941 games with no duplicate keys or missing
boxes. Lineups pass 3,931 games; ten remain transparently quarantined (0.254%,
below the 0.5% budget). Possessions pass for 3,907 games: 787,579 possessions and
946,768 lineup segments, with zero score mismatches, negative/implausible point
rows, duplicate IDs, or segment-conservation failures. Only two one-point CDN
score gaps were repaired from clock-and-period-aligned V3 free throws.

**Dead ends:** Treating `actionNumber` as a trustworthy scoring replacement.
Although 83.1% of CDN rows aligned with V3 on action number, clock, and period,
broad replacement created 37 final-score failures. `orderNumber` plus the CDN
scoreboard is the canonical path; V3 is a narrow validator/repair source.

**Verdict:** The data constraint for the second WP fold is removed. Run the
frozen 2023–24 → 2024–25 comparison next; do not tune architectures against the
existing 2025–26 test season.

## 2026-08-08 — Frozen second WP fold confirms context and possession

**Question:** Do rolling team context and causal possession control repeat on an
independent season, or were their 2025–26 gains one-fold noise?

**What we did:** Parameterized the existing frozen comparisons by explicit train
and test season without changing features, regularization, state sampling, or the
5,000-draw whole-game bootstrap. Trained on 2023–24 and tested on 2024–25, then
compared the result with the already-frozen 2024–25 → 2025–26 fold. ESPN was
skipped on the earlier fold because it is an external later-season benchmark.

**Result:** Run `wp_pregame_ablation_v2_3026c4a4b9` lowers Brier from 0.15496
with Elo plus starters to 0.15378 with rolling margin and rest. Its paired delta
is -0.00118, 95% interval [-0.00198, -0.00040]. The later fold lowers Brier from
0.14922 to 0.14731, delta -0.00195, [-0.00373, -0.00020]. Starter RAPM versus
Elo crosses zero in both folds. Run `wp_possession_start_v1_f4a1c8a2d2` lowers
2024–25 Brier from 0.15333 to 0.15314 with time-interacted control, delta
-0.000196, [-0.000230, -0.000164]. The later possession fold repeats the gain:
0.14651 to 0.14632, delta -0.000189, [-0.000214, -0.000163]. Close-last-two-minute
Brier improves in both seasons, with fitted possession swings rising above 11
percentage points.

**Null result:** Prior-season starter RAPM does not independently clear either
fold. It is still included inside the current rolling-context candidate, so that
exact combined implementation is larger than the evidence justifies.

**Verdict:** Promote rolling margin plus rest and causal possession-start control
as confirmed feature blocks. Do not freeze the exact production model until a
starter-free rolling-context ablation passes both folds. Then begin the fixed
GAM/GBM architecture ladder on identical states.

## 2026-08-08 — Starter-free WP becomes the frozen Stage 0 baseline

**Question:** Does prior-season starter RAPM add enough value to justify its
coverage gaps and dependencies once rolling team margin and rest are present?

**What we did:** Added an explicit starter-free model with the same state, Elo,
rolling margin, rest, interactions, regularization, rows, and whole-game
bootstrap as the combined model. Scored both frozen folds. Then removed player
games, lineup segments, legacy possessions, and starter ratings from the
possession-start model and repeated both possession folds.

**Result:** Pregame runs `wp_pregame_ablation_v3_30ab68d381` and
`wp_pregame_ablation_v3_cdbcea84ee` show starter-free context beating Elo on
2024–25 (0.15502 → 0.15302; delta -0.00201, 95% interval
[-0.00365, -0.00042]) and 2025–26 (0.14961 → 0.14777; delta -0.00189,
[-0.00377, -0.00008]). Against context plus starters, the smaller model wins the
first fold by 0.00077 and loses the second by 0.00053; both intervals cross zero.
Possession runs `wp_possession_start_v2_1db472e450` and
`wp_possession_start_v2_0a5d626234` retain the time-interacted control gain on
both folds, with Brier deltas -0.000177 and -0.000175 and both intervals excluding
zero. Late close-game gains remain materially larger.

**Verdict:** Freeze starter-free Elo plus rolling margin and rest as Stage 0.
Starter RAPM remains a research feature, not a production dependency. The
possession-start extension is separately validated and now uses the same smaller
context. Begin the fixed GAM/GBM comparison on identical features.

## 2026-08-08 — Spline GAM and histogram GBM fail state parity

**Question:** Can generic nonlinear shape or interactions beat the frozen
starter-free logistic WP model without adding information?

**What we did:** Run `wp_stage1_v1_7e6c77d51a` fits three models on identical 12
features, 30-second states, labels, and chronological folds: the frozen scaled
logistic control; an additive logistic model with five-knot cubic splines; and a
fixed histogram GBM with 200 trees, learning rate 0.05, at most 15 leaves and
depth six. No hyperparameter was selected on an outer test season. Uncertainty is
the same 5,000-draw whole-game bootstrap.

**Result:** Logistic Brier is 0.15302 on 2024–25 and 0.14777 on 2025–26. The
spline model is worse at 0.15560 and 0.14943; its pooled candidate-minus-logistic
delta is +0.00214, 95% interval [0.00088, 0.00340]. HistGBM is substantially
worse at 0.17030 and 0.15661; pooled delta +0.01303 [0.00970, 0.01639]. Both lose
AUC in both folds. The spline calibration slope is 0.876 on the first fold;
HistGBM slopes are 0.717 and 0.849.

**Verdict:** Reject both candidates. Their failures are not merely calibration:
ranking quality also declines. Do not tune around the result using 2024–25 or
2025–26. Run the preregistered residual MLP next; treat TCN/GRU/transformer models
as tests of causal sequence history, not automatic upgrades over tabular logistic.

## 2026-08-08 — Fixed feed-forward MLP fails across five seeds

**Question:** Does a small neural network improve the same frozen tabular WP
states where splines and trees failed?

**What we did:** PyTorch is unavailable locally, so the residual architecture was
not silently approximated. Run `wp_mlp_v1_7a7825bf09` explicitly tests a scaled
scikit-learn feed-forward MLP with two 64-unit ReLU layers, Adam, batch size 1024,
early stopping, a frozen 100-epoch cap, and seeds 7/17/29/43/71. Both outer folds
use identical rows and the five-seed probability ensemble; seeds measure optimizer
variance only.

**Result:** On 2024–25 the ensemble Brier is 0.18965 versus logistic 0.15302; on
2025–26 it is 0.17448 versus 0.14777. The pooled candidate-minus-logistic delta is
+0.03171, 95% interval [0.02619, 0.03745]. AUC falls from 0.8582 to 0.8092 and
from 0.8666 to 0.8255. Calibration slopes are 0.411 and 0.485. Nine of ten seed
fits hit 100 epochs, but every seed is materially worse.

**Verdict:** Reject this feed-forward implementation and do not tune it against
the outer seasons. This is not evidence against residual networks or sequence
history. The next meaningful experiment is a prefix-invariant completed-
possession token table, followed by parameter-matched TCN/GRU/transformer tests.

## 2026-08-08 — WP frozen after playoff and temporal audit

**Question:** Is the current model good enough to close the WP lane, including
playoffs and older-season evidence?

**What we did:** Segmented both frozen action-state and possession-start artifacts
by regular season versus playoffs and recomputed Brier, AUC, calibration slope,
and whole-game paired intervals. Also segmented the 2025–26 ESPN matched-play
benchmark and audited whether the legacy cache can support an older WP backtest.

**Result:** Regular-season action-state performance is stable: 2024–25 Brier
0.15096/AUC 0.862/slope 0.954 and 2025–26 Brier 0.14696/AUC 0.868/slope 1.047.
The earlier context-versus-Elo delta is -0.00250 with interval
[-0.00417, -0.00085]; the later interval narrowly crosses zero. Playoff samples
are only 84 and 85 games. Their Brier/slope pairs are 0.18317/0.648 and
0.15945/0.937; context-versus-Elo intervals cross zero. ESPN playoff Brier is
0.15696 versus 0.16182 for the older local matched-play benchmark, also unresolved
by whole-game bootstrap. Possession improves the 60-game later CDN playoff slice,
but the source omits the final 25 playoff games. Legacy pre-2023 possessions lack
clock/state fields and cannot evaluate this estimand.

**Verdict:** Freeze WP as good enough for regular-season platform use with an
explicit playoff calibration caveat. Do not fit a playoff-only model from this
sample, continue local neural training, or claim a pre-2023 backtest. Move active
work to RAPM and the all-in-one impact model.

## 2026-08-08 — Fractional lineup attribution wins the two-fold RAPM check

**Question:** Which lineup should receive credit when a substitution occurs
inside a possession: the start lineup, the terminal lineup, or fractional
exposure across every lineup segment?

**What we did:** Run `rapm_lineup_policy_v2_911d8bfce1` fits the same zero-prior
ridge specification for all three policies. It uses 743,946 identical regular-
season possessions and two one-season-train chronological folds: 2023–24 to
2024–25 and 2024–25 to 2025–26. It resamples whole games 5,000 times. Fractional
weights use elapsed-time share, with action-count share only when the clock does
not move. The policy does not use the possession outcome.

**Result:** On the first fold, RMSE is 15.0107 for start, 15.0541 for terminal,
and 15.0101 for fractional. Fractional versus start is a tie: squared-error
delta -0.016, 95% interval [-1.028, 1.021]. Fractional beats terminal by -1.321,
[-2.196, -0.443]. On the second fold, RMSE is 15.7975, 15.7326, and 15.7232.
Fractional beats start by -2.342, [-3.306, -1.393], and is unresolved versus
terminal at -0.294, [-1.107, 0.519]. Across 2,454 disjoint test games,
fractional beats start by -1.180, [-1.876, -0.487], and terminal by -0.807,
[-1.426, -0.208].

**Verdict:** Freeze fractional exposure as the working current-RAPM attribution
policy. Keep start and terminal as mandatory sensitivity outputs. This is a
small, project-specific engineering decision. It is not a published standard.
The current evidence has only two outer folds, and one fractional-versus-start
fold is effectively tied. Select ridge penalties next without using 2025–26 for
model selection.

## 2026-08-08 — Normal RAPM keeps the original ridge penalties

**Question:** Do new offensive, defensive, and home penalties improve normal
RAPM outside the season used to select them?

**What we did:** Run `normal_rapm_v1_85e0cc8e27` compares 20 predeclared penalty
triples on 2023–24 to 2024–25 game-margin retrodiction. It then evaluates the
selection winner once on 2025–26 after fitting 2023–24 and 2024–25. The run uses
terminal lineup assignment, 743,946 possessions, 3,681 games, and no uncertainty
estimation.

**Result:** Selection chose 4500/4500/1000 with RMSE 15.05339, only 0.00067
better than 3000/3000/300. On untouched confirmation, the original penalties win:
RMSE 15.47320 versus 15.50980, correlation 0.3344 versus 0.3230, and MAE 12.1970
versus 12.2262.

**Verdict:** Reject the selected challenger. Normal RAPM remains zero-prior ridge
with offensive/defensive/home penalties 3000/3000/300. Park fractional lineup
attribution as research-only by user direction.

## 2026-08-08 — First three-year statistical impact baseline

**Question:** Do the available advanced features and on/off add predictive signal
beyond box rates for three-year normal RAPM?

**What we did:** Run `statistical_impact_v1_6dff345dc2` joins 6,513 player-window
rows from 2016–2024 to separate offensive and positive-is-good defensive RAPM
targets. It compares ridge models using box rates, 49 allowed advanced features,
and the same advanced panel plus OnOffRtg/OnDefRtg. Age, experience, height,
position, minutes, games, and possession counts are not predictive features.
Square-root possession counts weight noisy labels. Test windows end in 2022,
2023, and 2024; a two-window purge prevents three-year target overlap.

**Result:** Advanced features improve net weighted RMSE over box rates in all
three folds by 0.0342, 0.0466, and 0.0410. Mean net RMSE is 1.3594 versus 1.4000.
Adding on/off improves every fold again by 0.2815, 0.2276, and 0.2206; mean net
RMSE is 1.1162 and correlation is 0.5629. The independent advanced model has
mean offensive/defensive correlations 0.4341/0.3092. The on/off-assisted model
has 0.5242/0.4800.

**Verdict:** Keep the advanced ridge model as the independent baseline. Keep the
on/off-assisted model as a distinct, stronger impact-assisted challenger; do not
mislabel it as an independent statistical prior. Before further model comparison,
rebuild percentages and average tracking fields with natural attempt or touch
denominators because the inherited panel minute-weights those fields.

## 2026-08-08 — Direct net target adds no value to the ridge baseline

**Question:** Should the statistical model predict net RAPM directly, or predict
offensive and defensive RAPM separately and add them?

**What we did:** Run `statistical_impact_v2_48f6ad776f` adds a direct three-year
net RAPM target to the same purged chronological folds, feature sets, reliability
weights, preprocessing, and ridge alpha search used by the component models.

**Result:** On the advanced features, direct net and summed components have the
same mean weighted RMSE to floating-point precision: 1.3593836544. Their mean
correlations are also the same: 0.4159608624. The on/off-assisted variants are
also identical: RMSE 1.1161642634 and correlation 0.5629409007. This is expected
because a ridge solution is linear in its target when the design, weights,
preprocessing, and penalty are the same. The box-only variants differ by 0.00085
RMSE because their independently selected penalties are not identical in every
fold.

**Verdict:** Use separate offensive and defensive ridge models, then add their
predictions for net. The direct target gives no practical gain and cannot explain
the offense/defense split. Reconsider a direct net target only for a nonlinear
model, a different loss, or a different feature set.

## 2026-08-08 — Natural-denominator statistical feature rebuild

**Question:** Does replacing the legacy minute-weighted feature panel with
pooled counts and natural attempt/touch denominators change the ridge baseline?

**What we did:** Run `statistical_features_v1_940f99ed54` builds 97 box,
tracking, shot-zone, creation, turnover, and rebound features for 6,689
player-windows ending 2016–2024. It excludes age, minutes, games, position, and
experience. Possessions remain reliability weights only. The builder collapses
four duplicate source rows that agree on the declared feature contract. It
rejects conflicting feature rows, duplicate output keys, and invalid bounded
ratios. Rebound-chance conversion was removed because total rebounds and
tracking chances do not have consistent coverage.

Run `statistical_impact_v2_5224a3b4a6` evaluates the rebuilt table on the same
three purged outer folds as the legacy baseline.

**Result:** Independent advanced net RMSE is 1.3578 with correlation 0.4212,
versus 1.3594 and 0.4160 on the legacy panel. It beats box-only RMSE in all three
folds by 0.0202, 0.0517, and 0.0518. The on/off-assisted model remains much
stronger at RMSE 1.1157 and correlation 0.5583, but it is not an independent
statistical prior. The clean rebuild is a correctness improvement; its aggregate
metric lift is small.

**Verdict:** Promote the rebuilt table as the feature baseline. Do not claim a
large accuracy win. Use it for the fixed ridge/elastic-net/tree comparison and
for the user's future feature challenge.

## 2026-08-09 — First statistical model-family comparison

**Question:** Do elastic net or a small nonlinear tree model beat ridge on the
validated 97-feature table?

**What we did:** Run `statistical_model_comparison_v1_dd31e7957d` compares
ridge, elastic net, and histogram gradient boosting on identical player-windows,
three-year offensive and defensive normal RAPM targets, possession reliability
weights, and 2022–2024 purged outer folds. Hyperparameters are selected only in
each fold's inner chronological validation window. The histogram models use at
most 15 leaves; this is not a neural model.

**Result:** Histogram GBM improves component-summed net RMSE in all three folds
by 0.0412, 0.0327, and 0.0263. Mean net RMSE is 1.3259 versus 1.3593 for ridge;
correlation is 0.5169 versus 0.4265. The gain is offensive: histogram offense
beats ridge in all three folds, with mean RMSE 0.9349 versus 0.9725 and
correlation 0.5448 versus 0.4699. Histogram defense loses RMSE in all three folds
(0.9083 versus 0.9028) despite higher correlation. Elastic net has mean net RMSE
1.3804 and beats ridge in only one fold.

**Verdict:** Promote histogram GBM as the offensive challenger. Keep ridge as
the defensive baseline. Reject elastic net. The mixed histogram-offense plus
ridge-defense model is the next component candidate; compare it with direct
nonlinear net prediction before any production choice. Three chronological fold
wins are initial evidence, not proof across every NBA era.

## 2026-08-09 — Direct nonlinear net loses to decomposed AIO

**Question:** Does a histogram GBM trained directly on net RAPM beat histogram
offense plus ridge defense?

**What we did:** Run `statistical_direct_net_v1_286a104216` tunes the direct
histogram model only inside each chronological training fold. It compares direct
net with saved component predictions from
`statistical_model_comparison_v1_dd31e7957d` on identical test players.

**Result:** Direct net loses weighted RMSE in all three folds. Mean RMSE is
1.3413, versus 1.3257 for histogram offense plus ridge defense and 1.3593 for
ridge components. Direct net has higher correlation than the hybrid, 0.5093
versus 0.4825, but worse magnitude calibration. The hybrid is better on the
primary loss in 2022, 2023, and 2024.

**Verdict:** Reject direct nonlinear net for this feature set. Keep the AIO
decomposed as histogram GBM offense plus ridge defense. Next, ablate feature
families before adding complexity.

## 2026-08-09 — Frozen-model feature-family optimization

**Question:** With histogram GBM offense and ridge defense frozen, which broad
feature families add chronological signal?

**What we did:** Run `statistical_feature_ablation_v1_918be14a38` removes core
box, shot profile, creation/role, turnover detail, and tracking rebound/defense
families one at a time. Model families and hyperparameters never change. The
2022–23 folds select combined removals; 2024 is scored once as the feature-level
confirmation fold.

**Result:** Offense removes the 25-feature creation/role block. Defense removes
35 shot-profile and 10 turnover-detail features. Core box and tracking
rebound/defense survive. On 2024, offensive RMSE improves 0.8846→0.8822,
defensive RMSE improves 0.9055→0.8998, and combined net RMSE improves
1.3096→1.3020. Net correlation improves 0.5155→0.5410. The resulting models
use 70 offensive and 50 defensive features and are saved in
`statistical_aio_v1_b0295558c6`.

**Verdict:** Freeze this statistical feature contract as a research challenger.
Do not continue subset search on the same three outer folds. Another search would
reuse the only tracking-era evidence and overfit research choices. The next AIO
step is cross-fitted prior generation and prior-informed RAPM, not another model
or feature sweep.

## 2026-08-09 — Basketball-domain feature engineering improves offense only

**Question:** Can stabilized rates, era context, scoring topology, creation
quality, behavioral role, and within-window temporal shape improve the frozen
statistical AIO without age, experience, height, position, minutes, games, on/off,
or plus-minus inputs?

**What we did:** Run `statistical_features_v2_6f7b3c5c57` adds 131 engineered
features to 6,689 validated player-windows. It recalculates empirical-Bayes
percentages from pooled natural denominators; adds possession-weighted
era-relative rates; separates latest-season level, linear trend, and volatility;
and adds scoring-topology, creation, role, rebound, and defensive interactions.
Missing engineered values receive a neutral within-window fill. The table has
zero duplicate keys, infinities, bounded-feature failures, or remaining missing
engineered values.

Run `statistical_feature_v2_comparison_e1cba6dd1d` holds the learners fixed:
Histogram GBM for offense (`learning_rate=0.03`, seven leaves, L2=1) and ridge
for defense (`alpha=3000`). It tests predeclared blocks on 2022 and 2023. A block
must reduce component RMSE in both folds. The selected combination is scored once
on 2024.

**Result:** Offense selects era-relative rates, latest-season levels, and
trend/volatility. Discovery offense RMSE improves 0.97270 to 0.90945 in 2022 and
0.92624 to 0.88170 in 2023. No defensive block passes both folds. On 2024,
offense RMSE improves 0.88224 to 0.84528 and correlation improves 0.56423 to
0.58908. Recombined net RMSE improves 1.30196 to 1.27987 and correlation improves
0.54104 to 0.55221. The saved challenger uses 117 offensive and 50 defensive
features.

**Null results:** Stabilized percentages, scoring topology, and creation-quality
blocks do not improve offense in both discovery folds. Era context, recent level,
temporal dynamics, direct defensive interactions, and offensive role context do
not improve defense in both folds. The available public panel still contains
little direct defensive matchup information.

**Verdict:** Keep v2 as the current research challenger. Do not call it final or
optimal: 2024 was held out from this feature selection but was inspected during
earlier model-family work, and the target RAPM run ends in 2024. Stop subset
search on 2022–24. Next, create cross-fitted priors and test whether this feature
gain improves the downstream possession model.

## 2026-08-10 — Public all-in-one methods support a factor challenger

**Question:** Should the statistical AIO estimate direct offensive and defensive
RAPM only, or also estimate interpretable team-factor effects?

**What we reviewed:** Public methodology for MAMBA and its six-factor RAPM,
Thinking Basketball's Box Creation, Offensive Load, Passer Rating, ScoreVal, and
AuPM, BPM 2.0, RAPTOR, LEBRON, EPM, DARKO, PIPM, the Dean Oliver Four Factors,
and the linked APBR RAPM implementation discussion. Exact source links and the
resulting model contract are in `docs/impact/FACTOR_DECOMPOSITION.md`.

**Verified findings:** Teemo's factor RAPM uses six targets: TS, turnovers, and
rebounds on offense and defense, then learns a linear scale back to offensive and
defensive RAPM. EPM and DARKO both support stat-specific stabilization and decay
instead of one shared rolling average. RAPTOR supports expected shot value,
assisted-shot, contested-rebound, and spacing features. BPM and LEBRON support
behavioral role interactions and role-based stabilization. Ben Taylor's work
supports creation and turnover features that distinguish responsibility from
efficiency. It does not provide a reproducible current all-in-one formula.

**Decision:** Make an eight-head eFG, turnover, rebound, and free-throw model the
primary factor challenger. Make the six-head TS version an ablation. Do not use
TS with a separate free-throw head. Keep the direct offense and defense model as
the production baseline and allow a cross-fitted residual for effects the four
factors do not assign. Current rich events are sufficient to validate the target
builder for 2023–24 through 2025–26, but historical event ingestion is required
before production promotion.

## 2026-08-10 — Correction: factors structure features, not required targets

**User clarification:** The shooting, turnover, rebound, and free-throw factors
are the user's mental decomposition of basketball value. The first AIO does not
need a separate supervised RAPM target for every factor.

**Additional source review:** The CraftedNBA glossary publishes Box Creation,
Offensive Load, Shooting Proficiency, Spacing, and an explicit Passer Rating
formula. CraftedNBA labels its Passer Rating as inspired by Ben Taylor, not as
Taylor's exact formula. It uses standardized Load, assist-to-Load,
position-standardized assist-to-Load, turnover-to-Load, creation-to-Load, and
height. The glossary does not specify the standardization population, padding,
position assignment, or final 1–10 transform. CraftedOPM, CraftedDPM, and
Portability include external impact metrics and are not valid independent prior
inputs.

**Corrected decision:** Keep direct three-year offensive and defensive RAPM as
the primary targets. Use the factors as feature families, diagnostics, and the
published explanation layer. Add versioned public formulas as candidate inputs.
Test the exact and behavioral passer ratings as separate challengers. Raw height
and listed position remain excluded as general columns; the exact composite is
blocked until canonical metadata and its standardization contract exist. Factor
RAPM remains optional research.

## 2026-08-10 — Public basketball features pass an exploratory offense test

**Question:** Do versioned Box Creation, Offensive Load, passing-ratio,
Shooting Proficiency, and Spacing features add signal to the frozen direct
offensive RAPM model?

**What we did:** Feature run `statistical_features_v2_8b2566243f` adds exact
public-formula benchmarks plus possession-shrunk model variants. The model-safe
block has eight features: Shooting Proficiency, Box Creation, Offensive Load,
assist-to-Load, turnover-to-Load, creation-to-Load, a behavioral passer score,
and a stable spacing proxy. The feature table has 6,689 player-windows, zero
duplicate keys, zero infinities, and zero missing engineered values after the
documented neutral fill. The exact Crafted passer variant is not present because
the current canonical table lacks height and position metadata.

Run `statistical_feature_v2_comparison_9b8d0555e0` holds the learner and target
fixed. It evaluates the public block separately from generic empirical-Bayes
ratios. The public block reduces offense RMSE from the 0.94912 discovery mean to
0.92292 and improves both 2022 and 2023. The final selected offense combination
also includes stabilized ratios, era-relative rates, recent level, and temporal
dynamics. No defensive block passes.

**Result:** On the reused 2024 check, offense RMSE improves from 0.87881 to
0.82701 and correlation improves from 0.56890 to 0.62133. Recombined net RMSE
improves from 1.29843 to 1.26244 and correlation improves from 0.54462 to
0.57485. These are exploratory estimates because 2024 informed earlier model
choices and the historical RAPM labels may not be the final target run.

**Verdict:** Keep the public block in the research challenger. Do not promote it
from these reused seasons or search individual public features on the same
folds. The next valid test is whether cross-fitted statistical priors improve a
downstream prior-informed RAPM on chronological future games.

## 2026-08-10 — Cross-fitted statistical priors are ready for RAPM

**Question:** Can the selected statistical AIO generate a prior for every
historical player-window without allowing a target window to train itself?

**Method:** Run `statistical_priors_v1_2c81b23662` freezes the selected 162-feature
offense histogram GBM and 50-feature defense ridge. For prediction window `T`,
training includes only RAPM target windows ending by `T-3`. The three-year train
and prediction targets therefore share no seasons. Same-window box and tracking
features are observed before prediction, so this is an end-of-window
retrodiction rather than a forecast. Models predict all feature rows; labels are
joined only afterward for evaluation.

**Quality:** The artifact contains 4,656 unique player-windows across 2019–24
and 1,270 players. It scores every eligible feature row. Minimum label coverage
is 96.21%; the remaining feature-covered players still receive priors. There are
zero duplicate keys, missing predictions, non-finite predictions, purge
violations, or offense-plus-defense identity errors. A test confirms that
changing a window's target cannot change that window's prior.

**Result:** Across six chronological folds, offense RMSE is 0.85494 with 0.60664
correlation, defense RMSE is 0.87365 with 0.29882 correlation, and net RMSE is
1.25131 with 0.51980 correlation. Defense remains the weak component. The 2019
fold has only one older target window, and the 2022–24 metrics are reused
research evidence.

**Verdict:** The priors are valid inputs for a prior-informed RAPM experiment.
They are not yet a final all-in-one. Next compare zero-prior RAPM, prior-only
ratings, and prior-informed RAPM on identical chronological possession windows.

## 2026-08-10 — Prior-informed RAPM does not clear confirmation

**Question:** Does the purged statistical AIO improve three-season normal RAPM
when used as the ridge center rather than as a standalone rating?

**Method:** Run `prior_informed_rapm_v1_122ef63045` uses matched regular-season
games and three-season training windows. It maps positive-good offense to the
offensive coefficient and positive-good defense to the negative points-allowed
coefficient, centers both blocks by training possession exposure, and compares
zero-prior, prior-only, and center scales 0.25/0.50/0.75/1.00. Scale selection
uses test seasons 2020–22. Seasons 2023–24 are a later research check. A
2,000-draw whole-game bootstrap resamples within season and weights seasons
equally.

**Result:** Scale 1.0 wins selection (13.8002 RMSE versus 13.8800 zero-prior).
On 2023–24 it records 13.5263 RMSE versus 13.5296, a 0.00327-point difference,
and wins one of two folds. The paired MSE delta is -0.195 with 95% interval
[-1.119, +0.729] and 0.6425 bootstrap probability of improvement. Prior-only is
worse at 13.7952 RMSE. The comparison covers 2,459 matched confirmation games;
minimum test lineup-slot prior coverage is 89.51%.

**Verdict:** Improvement is not demonstrated. Keep terminal-lineup zero-prior
normal RAPM with penalties 3000/3000/300 as production. Keep the statistical
model separate and labeled as research. Do not tune more scales on these reused
seasons. Revisit integration only with new seasons or one predeclared
sample-size-adaptive rule.

## 2026-08-10 — SPM benchmarked against BPM and xRAPM

**Question:** Does the current three-season SPM resemble established box and
prior-informed adjusted-impact metrics on matched historical windows?

**Method:** Run `external_impact_benchmark_v1_bab43a4087` downloads and hashes
annual 2017–24 Basketball Reference advanced tables and xrapm.com historical
tables. It resolves traded-player BPM rows to the multi-team total, converts
xRAPM defense from lower-is-better to positive-good, normalizes accents and
suffixes, and minutes-weights annual values over each `T-2:T` SPM window. The
comparison covers all 4,656 SPM rows and separately reports rows with at least
3,000 offensive and defensive possessions.

**Quality:** External source-to-source name matching is at least 99.43% in every
season. SPM-to-BPM and SPM-to-xRAPM coverage is at least 98.52% and 98.47% per
window. There are zero duplicate player-window keys. Seventeen normalized names
are ambiguous in the historical NBA name dimension and are quarantined rather
than guessed.

**Result:** Across 2,295 high-exposure player-windows, SPM net correlates 0.876
Pearson / 0.841 Spearman with BPM and 0.756 / 0.692 with xRAPM. Offensive SPM
correlates 0.831 with xRAPM offense. Defensive SPM correlates only 0.630 with
xRAPM defense, the clearest component gap. Including all tiny samples lowers net
Pearson correlation to 0.541 versus BPM and 0.610 versus xRAPM, showing that
sample/reliability handling materially changes any leaderboard comparison.

**Verdict:** The SPM has strong external face validity for established players
but remains closer to BPM than to the possession-informed xRAPM, especially on
defense. Use player-level defensive disagreements as the next feature diagnostic.
Do not train on BPM or xRAPM and do not treat either as ground truth.

## 2026-08-10 — Single-season SPM baseline and defensive disagreement report

**Question:** Can one global statistical model turn each season's own box and
tracking evidence into an annual SPM, while keeping annual xRAPM only as a
multi-window comparator?

**Method:** Target run `single_season_rapm_targets_v1_fd876680da` fits separate
zero-prior normal RAPM models for 2014–24 with penalties 3000/3000/300. Feature
run `statistical_features_v2_0e1350d95a` uses one-season pooling. It excludes
age, experience, height, position, minutes, games, on/off, plus-minus, and all
three-season temporal features. Run `single_season_spm_v1_51adc53061` keeps the
frozen offense histogram GBM and defense ridge. For each reported 2017–24
season, it trains on every other 2014–24 season. The final leaderboard refits on
all 2014–24 labels. This is descriptive leave-one-season-out evaluation, not a
forecast.

**Quality:** The panel has 5,791 unique player-seasons and the reported table has
4,341 rows. It has zero duplicate keys, missing names, non-finite predictions,
or offense-plus-defense identity errors. BPM matches 98.89% and xRAPM matches
98.99% of reported rows. The high-exposure comparison has 2,860 rows matched to
both sources. The 2024 target cache has 1,229 regular-season games, one fewer
than expected, and all legacy targets stop after 2024.

**Result:** Mean held-out weighted RMSE/correlation is 1.0060/0.6178 on offense,
1.0578/0.3091 on defense, and 1.4611/0.5314 on net. For high-exposure rows, net
Pearson correlation is 0.897 with BPM and 0.762 with xRAPM. Defensive correlation
with xRAPM is 0.590. The annual net xRAPM correlation remains between 0.725 and
0.795 in every season. Stable disagreement examples include Dillon Brooks and
Robin Lopez ranked higher by xRAPM defense, and Damian Jones and Mo Bamba ranked
higher by SPM defense. These are audit leads, not evidence that one metric is
correct.

**Verdict:** Share with caveats as the first annual SPM baseline. It is not the
final all-in-one because the target is noisy and the evaluation uses later as
well as earlier seasons. Next use annual out-of-fold SPM as the prior mean in a
one-season RAPM comparison. Estimate or tune prior precision inside training
seasons; do not restore the earlier arbitrary amplitude scale.

## 2026-08-10 — Annual zTS and playtype challenger

**Method:** `playtype_features_v1_db63ed1132` rebuilds 2014–24 zTS from the
newest local Gabriel O'Connell playtype snapshot and annual box totals. zTS is
player TS% minus expected TS% from the player's playtype mix. It also calculates
overall playtype POE, transition share, and transition POE contribution per 75
total Synergy possessions. Existing thresholds remain 250 minutes, 50 player
Synergy possessions, and 20 possessions per player-playtype row for the league
playtype TS calculation. Models and eight leave-one-season-out folds are frozen.

**Quality:** The table has 4,299 unique player-seasons. Every key matches the old
rounded `zts_results.csv`; zTS correlation exceeds 0.9999998 and maximum absolute
difference is 0.005 percentage points. Feature run
`statistical_features_v2_ab9646062b` has no duplicate keys or non-finite values.

**Result:** Baseline offense RMSE/correlation is 1.0060/0.6178. Adding zTS alone
(`single_season_spm_v1_fcdb9559f6`) improves it to 0.9972/0.6302; net improves
from 1.4611/0.5314 to 1.4553/0.5383. Adding overall POE, transition share, and
transition POE (`single_season_spm_v1_1f2125a38b`) records offense
0.9967/0.6301 and net 1.4558/0.5376. Defense is unchanged.

**Verdict:** Keep zTS in the annual offensive research model. The larger block's
tiny RMSE gain and slightly worse correlation do not justify promotion. These
seasons are now inspected, so require new data or a predeclared nested rule for
promotion. Next canonicalize DFG, rim-defense, and hustle features.

## 2026-08-10 — Canonical defensive tracking block

**Question:** Do direct shot-defense and hustle measurements repair the weak
annual defensive SPM?

**Source and quality:** The existing manifest already pinned and downloaded
Gabriel O'Connell's current `site_Data` commit
`bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd`. The aggregate rim-defense files
were missing from the manifest, so they were added and downloaded through the
resumable ingest path. Upstream declares no license; use these inputs for local
research only. Run `defensive_tracking_features_v1_9f66c664eb` contains 5,791
unique player-seasons with no non-finite values. DFG and rim source-name joins
cover 99.60% and 99.61% of rows; hustle joins cover 100%. Hustle begins in 2018,
so earlier seasons receive the global neutral median rather than a zero-valued
era marker.

**Features:** The predeclared block contains defended attempts per 100,
empirical-Bayes overall DFG differential, rim DFGA per 100, empirical-Bayes rim
DFG differential, rim points saved per 100, deflections, charges, contested
2-point and 3-point shots, and defensive loose balls recovered. Hustle counts
use defensive possessions. DFG differentials shrink toward zero with 200 total
or 100 rim attempts. Positive rim points saved means better defense.

**Result:** `single_season_spm_v1_bff6060df6` keeps the zTS offense challenger
and adds all ten fields to the frozen defensive ridge. Defense held-out RMSE
improves from 1.0578 to 0.9595 and correlation from 0.3091 to 0.4964. Net RMSE
improves from 1.4553 to 1.3859 and correlation from 0.5383 to 0.5991. Defense
and net improve on both metrics in all eight 2017–24 season folds. High-exposure
defense correlation with xRAPM rises from 0.590 to 0.701, while correlation with
BPM defense falls from 0.803 to 0.690. This is consistent with adding information
that is less box-score-like; neither comparator is ground truth.

**Verdict:** Retain the full defensive tracking block in the annual research
model. Do not search its subsets on these folds. Confirm on new seasons or use a
predeclared nested historical selection design before production promotion.

## 2026-08-10 — Gabriel assist audit and locked annual AIO integration

**Assist audit:** Gabriel O'Connell's pinned `assist_data` table defines adjusted
potential assists as potential assists plus 0.88 times free-throw assists. Its
assist efficiency is adjusted assist points divided by adjusted potential
assists. It is points per adjusted potential assist, not true shooting. The
clean builder removes four identical duplicate player-season rows, repairs three
infinite ratios, and emits 5,791 unique finite player-seasons. The two
nonduplicate candidates are free-throw assists per 100 and empirical-Bayes assist
points per adjusted potential assist.

**Assist result:** `single_season_spm_v1_d6de68348c` slightly improves the
eight-fold average, but offense and net lose both RMSE and correlation in 2023
and 2024. Do not promote the assist candidates. Contract
`configs/models/annual_spm_v1.json` freezes `single_season_spm_v1_bff6060df6`:
offense histogram GBM plus zTS, defense ridge plus all ten canonical defensive
tracking fields, and net equal to offense plus defense.

**Time-safe method:** Leave-one-season-out SPM is valid for descriptive mapping
tests but leaks later labels into a next-season experiment. Run
`annual_spm_priors_v1_1107680642` instead trains each SPM(T) mapping only on
seasons before T. It creates 3,769 priors for 2017–23 with no duplicate keys or
nonfinite values. Run `prior_informed_rapm_v1_e1239679c1` fits one-season normal
RAPM(T) with the fixed 3000/3000/300 penalties. It compares zero center, the
exact full SPM(T) center, and SPM alone on T+1 regular-season game margins. No
prior-amplitude grid is used.

**Result:** Full SPM centering wins RMSE in all four 2018–21 selection folds and
all three 2022–24 later folds. On 2022–24, mean RMSE is 13.8118 for the combined
model, 13.8892 for normal RAPM, and 14.0128 for SPM alone. Correlation is 0.3656,
0.3486, and 0.3412 respectively. Across 3,689 matched games, the equal-season
MSE delta for combined minus normal RAPM is -2.143. The paired game bootstrap
95% interval is [-3.270, -0.965].

**Verdict:** The annual SPM-centered normal RAPM is the leading research AIO.
This is not a production promotion because the later seasons already influenced
feature design. Freeze the specification. The next deliverable is its annual
offense, defense, net, SPM-center, RAPM-update, and possession-exposure table.

## 2026-08-10 — Decomposed annual AIO ratings

**Method:** Run `annual_aio_ratings_v1_23c4895f8f` fits the frozen full-SPM-center
normal RAPM separately for each 2017–24 regular season. Each row records the raw
SPM value, the possession-weighted centered SPM coefficient, zero-prior normal
RAPM, final AIO, and the RAPM update equal to final AIO minus centered SPM. It
also records offensive and defensive possession exposure and within-season net
rank.

**Quality:** The panel has 4,341 unique player-seasons and 1,270 players. All
player names and SPM priors match. The minimum lineup-slot prior coverage is
100%. Offense plus defense equals net, and SPM center plus RAPM update equals
final AIO, with maximum error below 9e-16. The 2024 input has 1,229 regular-season
games and remains one game short.

**Distribution check:** The 2024 AIO net distribution spans -6.07 to +6.74 and
has standard deviation 2.06. It is mildly right-skewed (0.59), not exactly
normal. Its unweighted player mean is -0.63 because many low-exposure players
are below average; its possession-weighted mean is effectively zero (-0.002).
The AIO is wider than zero-prior normal RAPM (standard deviation 1.66) because
the statistical center restores signal that zero-centered ridge suppresses.

**2024 high-exposure leaders:** Nikola Jokic (+6.74), Shai
Gilgeous-Alexander (+6.39), Paul George (+6.16), Joel Embiid (+5.49), OG Anunoby
(+5.16), and Jalen Brunson (+5.15), using at least 1,000 possessions on both
sides. Treat the precise order as exploratory, not ground truth.

**Verdict:** Use this versioned panel for annual leaderboards and decomposition.
Do not add uncertainty labels until they are estimated. Build three-year and
five-year peak tables next.

## 2026-08-10 — Independent three-year and five-year normal-RAPM peaks

**Contract:** `configs/models/rolling_normal_rapm_peaks_v1.json` freezes 1997–2024
regular-season windows, zero-prior ridge penalties 3000/3000/300, terminal
lineups, and a minimum 1,000 offensive and defensive possessions per included
window season. Peak selection is descriptive and winner's-curse biased. The
model removes each season's mean points per possession before fitting player
effects so changing league scoring environments do not become player credit.

**Implementation and quality:** Run `rolling_rapm_peaks_v1_584adf4f3d` fits 26
three-year and 24 five-year windows. It saves 36,530 rolling ratings and 7,866
offense/defense/net player peaks. Rating and peak keys are unique; component
identity error is below 9e-16. Annual player sheets fill historical crosswalk
gaps, including Sasha Danilović (ID 390), so every eligible peak has a name.
Legacy IDs 471 and 775 remain unresolved in four non-eligible rating rows. ID
775 appears across unrelated teams in 1997 and is likely a parser placeholder;
do not assign it a player name. Window fits checkpoint atomically after the
full archive audit.

**Leaders:** Three-year net peaks start with LeBron James 2009–11 (+9.19), Kevin
Garnett 2007–09 (+8.96), Nikola Jokić 2022–24 (+8.82), Stephen Curry 2016–18
(+8.77), and Kawhi Leonard 2020–22 (+8.32). Five-year net peaks start with Curry
2015–19 (+9.58), LeBron 2007–11 (+9.34), Garnett 2003–07 (+9.23), Chris Paul
2014–18 (+8.67), and Steve Nash 2007–11 (+8.16). Five-year offense is led by
Nash 2006–10 (+8.93); five-year defense is led by Garnett 2007–11 (+6.37).

**Validation:** For 1,223 players eligible in both window lengths, three-year
and five-year peaks correlate 0.963 for net, 0.964 for offense, and 0.950 for
defense. The independent three-year ratings correlate 0.842 with the legacy
three-year net panel across 16,716 matched rows. The difference is expected:
the legacy pipeline removes garbage time and adds score-margin and penalized
season terms, while this model uses all regular-season possessions and a direct
season scoring-level adjustment.

**Verdict:** Use this artifact for the first all-time peak product. Keep it
labeled normal RAPM, not AIO, because it has no statistical prior. Next build
the read-only query layer and player trajectory/decomposition schemas.

## 2026-08-10 — Read-only ratings API v1

- Added a pinned read-only query contract over the annual AIO and rolling normal
  RAPM peak artifacts. Artifact selection is explicit by run ID, never filesystem
  recency.
- Routes expose metadata/caveats, annual leaderboards, three- and five-year peak
  leaderboards, player search, and a player payload containing annual SPM center,
  RAPM update, AIO decomposition, rolling history, and component peaks.
- Added schema/key validation, deterministic ordering, pagination limits,
  possession filters, JSON null handling, HTTP 400/404 behavior, and permissive
  local CORS.
- Observable validation used the real pinned artifacts: the 2024 AIO net endpoint
  returned Nikola Jokic, Shai Gilgeous-Alexander, and Paul George at the top with
  a 1,000-possession-side filter; the five-year peak endpoint returned Stephen
  Curry, LeBron James, and Kevin Garnett. LeBron search and the full player payload
  also passed.
- This is a local-development server, not an internet-facing production server.
  Public hosting still needs a managed runtime, caching/compression, and restricted
  CORS.

## 2026-08-10 — First player trajectory product slice

- Replaced the discarded dashboard direction with one focused player page in
  `web/`. It consumes the pinned ratings API rather than copying model values
  into frontend code.
- The page shows annual AIO offense/defense/net, SPM-center plus RAPM-update
  decomposition, 3Y and 5Y normal-RAPM trajectories, component peak ranks,
  possession exposure, and the known research caveats.
- Search and quick-player controls use canonical NBA IDs. LeBron James is the
  default inspectable case.
- The Cloudflare-compatible production build and rendered product-shell test
  pass. This slice remains local: deploying only the page would be broken until
  the ratings API also has a managed public runtime.

## 2026-08-10 — Current possessions and frozen 2024–26 normal RAPM

- Re-audited the canonical current possession and ordinal lineup tables for the
  regular-season RAPM estimand. They contain 743,946 possessions across 3,681 of
  3,690 source games (99.76%). Nine games fail minute/substitution reconciliation
  and remain quarantined.
- Possession and segment keys are unique, player lineup slots have zero nulls,
  offense/defense team domains are valid, game scores and child-segment points
  conserve exactly, and there are no negative or greater-than-seven-point rows.
- Added current-box fallback names to the RAPM runner. This replaces stale legacy
  names when available and yields zero missing names across 802 fitted players.
- Frozen run `rapm_v0_01b5084f0a` uses terminal-lineup zero-prior ridge at
  3000/3000/300 over the 2023–24 through 2025–26 regular seasons. Its ratings are
  numerically identical to the earlier validated frozen run. The new run adds
  explicit input hashes, corrected research status, current names, and caveats.
- On 2025–26 observed-lineup retrodiction, margin RMSE is 15.473 and correlation
  is 0.334. Earlier-season player coverage is only 86.5%, so this remains a
  descriptive baseline, not a forecast.
- Start-versus-terminal net ratings correlate 0.971 with mean absolute difference
  0.358. This confirms that lineup policy matters; terminal remains the active
  simple contract by prior validation and user direction.
- The pinned API now exposes `/v1/leaderboards/current` and embeds current normal
  RAPM in each player payload. With at least 3,000 possessions per side, the top
  five are Shai Gilgeous-Alexander (+8.78), Nikola Jokić (+8.69), Kawhi Leonard
  (+6.79), Giannis Antetokounmpo (+6.70), and Victor Wembanyama (+5.90). Treat the
  order as a research result.

## 2026-08-10 — Current feature audit and untouched 2025 SPM confirmation

**Question:** Does the frozen 2014–24 annual SPM mapping retain its held-out
performance on the first new complete season?

**Data:** Rebuilt one-season features through 2025. Run
`statistical_features_v2_9e7c27e281` has 265 features, 6,360 player-seasons,
unique keys, and no non-finite or bounded-ratio failures. Playtype and defensive
tracking sources extend through 2025. The 2026 base sheet has only 81.8% of the
prior two-season median possession exposure and is marked partial.

**Method:** Applied saved offense and defense models from
`single_season_spm_v1_bff6060df6` to 2025 without refitting. Built the 2025
target from 1,226 regular-season games and 247,630 canonical possessions using
terminal-lineup normal RAPM at 3000/3000/300. All 569 target players match.

**Result:** Run `current_spm_confirmation_v1_9b4cca0b12` produces offense RMSE
1.102/correlation 0.619, defense 1.154/0.331, and net 1.610/0.500. Every RMSE is
worse than the worst 2017–24 held-out fold. Defense and net correlation are also
below the prior held-out range.

**Pipeline checks:** The rebuilt and frozen 2024 selected features and saved
predictions are numerically identical. Legacy and canonical 2024 annual RAPM
targets correlate 0.964–0.975. The target change adds about 0.03 RMSE on the same
2024 out-of-fold predictions, which does not explain the larger 2025 loss.

**Verdict:** Do not promote the frozen annual SPM or publish a 2025 AIO from it.
Do not tune it on 2025. Use 2025 for failure diagnosis, develop the next defense
model inside nested or forward-only 2014–24 validation, and reserve a new season
for confirmation. The historical-range rule was recorded after observing 2025;
it is a diagnostic comparison, not a predeclared gate.

## 2026-08-10 — No-tuning diagnosis of the 2025 defensive SPM miss

**Question:** Is the 2025 defensive regression caused by low exposure, unstable
RAPM targets, schema drift, or the new defensive tracking block?

**Method:** Run `current_spm_diagnostics_v1_59632783de` keeps the frozen model
unchanged. It stratifies errors by possession exposure, compares selected-feature
distributions with 2014–24, fits 2024 and 2025 first-half/second-half normal RAPM
diagnostics, and neutralizes saved defensive feature blocks at historical
medians. Neutralization is interpretation, not a retrained challenger.

**Result:** The 276 players above 2,000 possessions still have defense RMSE 1.232
and correlation 0.433. Their target standard deviation is 1.346 versus 0.649 for
predictions. At a 1,000-possession floor in each half, split-half defensive RAPM
correlation is nearly unchanged: 0.329 in 2024 and 0.331 in 2025. Neutralizing
all ten DFG/rim/hustle features worsens RMSE from 1.154 to 1.182 and correlation
from 0.331 to 0.230. Neutralizing DFG/rim alone is worse than neutralizing hustle
alone. The largest median shift is contested-threes per 100 at 1.32 historical
IQR, but its 2025 values remain inside the historical player range.

**Verdict:** The current evidence rejects two simple explanations: low-minute
noise alone and a harmful defensive-tracking block. The frozen model lacks enough
stable, calibrated defensive separation in 2025. Keep tracking. Design the next
defensive challenger under nested or forward-only validation and reserve a new
season for confirmation.

## 2026-08-10 — Nested forward defensive ridge selection

**Question:** Is the frozen defense model under-dispersed because its ridge
penalty is too strong?

**Contract:** `configs/models/annual_defense_ridge_nested_v1.json` freezes ridge
alphas 300, 1000, 3000, and 10000 before the run. For outer test seasons 2020–24,
each alpha is scored on the two immediately prior validation seasons, with model
training restricted to seasons before each validation year. The selected alpha
is then refit on all seasons before the outer test. Fixed alpha 3000 is the
baseline. The gate requires at least four of five RMSE wins, lower mean RMSE,
and noninferior mean correlation. No 2025 evidence enters selection.

**Result:** Run `annual_defense_ridge_nested_v1_5b06407982` selects alpha 300
once, 3000 once, and 10000 three times. It beats fixed 3000 on only one of five
outer folds. Selected-minus-fixed mean RMSE is +0.0015 and mean correlation is
-0.0004. The predeclared promotion gate fails.

**Verdict:** Keep fixed alpha 3000. Lower regularization does not fix the 2025
under-dispersion and is not supported by older forward folds. The next defensive
experiment must add stable signal rather than retune ridge strength.

## 2026-08-10 — Nested annual defense feature-block experiment

**Question:** Do interpretable defensive interactions improve the frozen annual
defense model under forward-only nested selection?

**Features:** Added overall DFG two-point-equivalent points saved per 100, rim
matchup-attempt share, and contested-three share to the defensive tracking
builder. Combined these with pre-existing stocks/foul, rebound-contest,
block-recovery, interior-role, and era-relative defensive features. On/off,
plus-minus, team ratings, games, minutes, age, experience, height, and position
remain forbidden.

**Contract:** `configs/models/annual_defense_features_nested_v1.json` freezes
five variants before the valid run: baseline, seven defensive interactions,
three matchup interactions, four era-relative rates, and all 14 additions. Ridge
alpha remains 3000. Each 2020–24 outer fold selects on the two prior validation
seasons. The promotion gate requires four of five RMSE wins, lower mean RMSE,
and noninferior mean correlation.

**Leakage correction:** Initial run `annual_defense_features_nested_v1_d913f807c5`
used a feature artifact built through 2025. Although outer rows ended in 2024,
global defensive-source fallback medians could include 2025. That run is invalid
for the stated contract. Rebuilt defensive and combined feature panels strictly
through 2024 before the final run.

**Result:** Valid run `annual_defense_features_nested_v1_22b677e1ef` selects the
combined block three times, defensive interactions once, and matchup interactions
once. It beats baseline in two of five outer folds. Selected-minus-baseline mean
RMSE is +0.00024 and mean correlation is -0.00274. The gate fails.

**Verdict:** Keep the frozen 60-feature defense model and the new derived fields
as research features only. Existing annual aggregates appear near their current
signal limit. The next defense experiment should start with new matchup-level or
spatial assignment data, not another subset search on 2014–24.

## 2026-08-10 — Licensed player-matchup archive ingest

**Question:** Is a new matchup-level defensive source available without a slow
NBA endpoint crawl?

**Audit:** The local Gabriel mirror contains 121 regular-season defensive
playtype files (`*d.csv`), and all 121 are header-only. Its consolidated tables
are player-season aggregates, not player-versus-defender matchups.

**Source:** Pinned revision `e829d467...` of
https://github.com/shufinskiy/nba_data. The repository declares Apache-2.0 and
publishes NBA Stats matchup archives from 2017–18 through 2024–25. Added manifest
`configs/ingest/shufinskiy_matchups_2017_2024.json` and `.tar.xz` member
validation to the resumable downloader.

**Result:** Downloaded and validated all eight regular-season archives: 30.38 MB
compressed and 1,769,658 game/offensive-player/defender rows. Per-season rows
range from 181,840 to 232,985. Every archive passes byte size, member name,
minimum row count, header width, and required field checks. Sidecars record the
actual SHA-256 values.

**Verdict:** This is the next defensive feature source. Build opponent-quality-
adjusted, sample-shrunk defender features before any new SPM comparison. Matchup
assignment is not optical tracking and must not be interpreted as sole causal
credit.

## 2026-08-10 — Opponent-adjusted matchup-defense feature test

**Question:** Does primary-defender assignment data add stable annual defensive
SPM signal beyond the frozen box, DFG, rim, and hustle model?

**Feature layer:** Run `matchup_defense_features_v1_86d13d7357` aggregates
1,769,658 source rows into 4,409 defender-seasons for 2018–25. The main feature
compares each scorer-defender pair with the scorer's leave-one-defender-out
points rate, centers the result within season, and shrinks it with 500 assigned
matchup possessions. Point reconstruction is exact. Exposure-weighted ID match
is at least 99.46%. The output has no duplicate or non-finite feature rows.

**Contract:** `configs/models/annual_defense_matchup_nested_v1.json` freezes the
baseline and four matchup blocks. It keeps ridge alpha 3000 and excludes raw
exposure. For outer seasons 2022–24, it selects a block using the two prior
seasons. The gate requires at least two RMSE wins, lower mean RMSE, and
noninferior mean correlation. Season 2025 is excluded.

**Result:** Run `annual_defense_features_nested_v1_eaeca704eb` selects the
event-context block in all three folds. It improves weighted RMSE in all three:
0.966 to 0.944 in 2022, 0.940 to 0.932 in 2023, and 0.968 to 0.959 in 2024.
Mean RMSE improves by 0.0131. Mean correlation falls by 0.0202. Prediction spread
moves closer to target spread, but player ordering becomes worse.

**Verdict:** The gate fails. Keep the canonical matchup data and feature panel,
but do not promote this block to the frozen SPM. The result is not evidence that
matchup assignments are useless. It shows that scorer-quality adjustment alone
does not remove shot-role, scheme, help, and lineup context. Add those controls
before one new predeclared test. Do not search more subsets on 2022–24.

## 2026-08-10 — Matchup-defense factor residuals

**Question:** Can the matchup panel express defensive mechanisms separately
instead of using one total points-allowed feature?

**Method:** Run `matchup_defense_features_v1_09829b48c8` adds six scorer-adjusted
factor residuals. For each scorer-defender pair, the expected event rate uses the
scorer's results against every other defender. Positive values mean the defender
suppressed attempts, saved shot-making points, suppressed threes, forced extra
turnovers, suppressed assists, or prevented shooting fouls. Each factor is
season-centered and empirical-Bayes shrunk.

**Result:** All six residuals conserve their season centers to numerical
precision. On same-season 2018–24 defensive RAPM, shot-making points saved has
correlation 0.407–0.457 in every season. Turnovers forced has 0.198–0.234.
Attempt and three-point-attempt suppression have small negative correlations.

**Verdict:** The factor layer is usable for explanations and future research.
The correlations are descriptive and the target seasons are already inspected.
Do not run a new subset promotion search on them. The negative volume result is
a warning that assignment, scheme, and shot-location context remain important.

## 2026-08-12 — Research control plane and rolling-peak eligibility repair

**Question:** Which recommendations from the supplied GPT Pro static diagnosis
survive contact with the live checkout, and is its peak-eligibility concern real?

**Evidence boundary:** The source diagnosis did not mount local artifacts or run
tests. Its numerical claims remain documented claims. The live checkout started
at commit `6c00463e`; its baseline suite passed 98 tests. The accepted governance
decisions are recorded in `research/estimands.yml`,
`research/season_exposure.yml`, and `research/pinned_artifact_audit.csv`.

**Verified bug:** The rolling-peak contract required at least 1,000 offensive and
defensive possessions in every constituent season. The implementation instead
required only 3,000 or 5,000 total-window possessions per side. This allowed a
player to miss or fall below the threshold in a constituent season.

**Repair and result:** The builder now counts offense and defense by player and
season and requires both per-season minima to meet the threshold. Rebuilt run
`rolling_rapm_peaks_v1_a8a612143c` fits the same 50 windows and 36,530 rating
rows. It removes 8,715 falsely eligible player-windows and produces 5,505
player/component peak rows. There are 1,088 eligible three-year players and 747
eligible five-year players. Net peaks correlate 0.956 among the 747 eligible in
both. Component identity error remains below `9e-16`; peak keys and names pass.

**Verdict:** Retire `rolling_rapm_peaks_v1_584adf4f3d` and pin the Ratings API to
the corrected run. Keep peaks research-only because selection uncertainty and
winner's-curse correction remain absent. Keep terminal-lineup, zero-prior normal
RAPM as the production reference. Preregister precision-aware prior work, but do
not tune it on the inspected 2025 failure or partial 2026 season. Reserve Season
2027 for one untouched annual confirmation.

## 2026-08-12 — Research control plane strengthened

**Question:** Can the pinned rating API expose an artifact without complete,
portable lineage?

**What we did:** Added a machine-readable pinned-artifact contract and a
`validate-research-control` command. The gate requires an estimand, evidence and
uncertainty status, season scope/completeness, relative artifact location, code
and configuration hashes, data-hash status, and a forbidden interpretation for
every API pin. It rejects incomplete hashes, machine-specific release paths,
research-only artifacts marked as production, and explicit Season 2027 use.

**Result:** The current four API pins pass the gate. This validates their
lineage records. It does not add uncertainty or upgrade their scientific status.

**Next:** Freeze and implement game-cluster uncertainty for normal RAPM before
publishing any rank precision.

## 2026-08-12 — Normal RAPM uncertainty: 2025 pilot complete

**Question:** Does the frozen terminal-lineup zero-prior normal RAPM have
reproducible game-level uncertainty on a complete annual slice?

**What we did:** Fit 2025 regular-season terminal-lineup normal RAPM with fixed
3000/3000/300 penalties. Resampled complete games with replacement within the
season for 1,000 deterministic, atomically checkpointed draws. Estimated a CR0
game-cluster ridge sandwich only as a diagnostic.

**Result:** The run covers 1,226 lineup-quality-passing games and 569 players.
It completed all 1,000 draws. Component identity holds to `4.4e-16`. For players
with at least 2,000 possessions on each side, median analytic/bootstrap 95%
interval-width ratios are 1.011 (offense), 1.011 (defense), and 1.009 (net).
The bootstrap remains the publication method. Only 521 players have complete
offense/defense/net coverage in every draw; players absent from a draw remain
missing rather than being treated as zero evidence.

**Next:** Let the 2022-24 three-year historical pilot finish. Do not publish
rank precision until synthetic coverage and the historical run also pass.

## 2026-08-12 — Normal RAPM uncertainty: historical 2022-24 pilot complete

**Result:** The three-year terminal-lineup normal RAPM run completed 1,000
season-stratified whole-game draws across 3,689 games, 728,269 possessions, and
818 players. Component identity holds to `4.4e-16`. For the 451 players with at
least 2,000 possessions on each side, median analytic/bootstrap 95% width ratios
are 1.004 (offense), 1.004 (defense), and 1.006 (net). There are 704 players with
complete joint draw coverage.

**Interpretation:** This passes the stated analytic/bootstrap-width diagnostic.
It does not establish empirical coverage or justify rank precision by itself.
Bootstrap intervals remain the publication method. The selection-aware rolling
peak run has started separately and remains research-only.

## 2026-08-12 — Error quantification audit for normal RAPM

**Question:** Does the uncertainty contract retain the RAPM covariance that is
needed for net ratings and rank claims?

**External check:** Squared Statistics' RAPM error analysis derives a ridge
posterior covariance, shows that offensive-defensive covariance is required for
a net interval, and demonstrates that a marginal player leader can be
statistically indistinguishable from many other players. It also warns against
extrapolating to unobserved lineups. Source: "Exercising Error: Quantifying
Statistical Tests Under RAPM (Part IV)," 2019-10-03,
https://squared2020.com/2019/10/03/exercising-error-quantifying-statistical-tests-under-rapm-part-iv/

**Decision:** Keep the analytic game-cluster ridge sandwich as a diagnostic,
not as the public method. Continue to use whole-game bootstrap draws as the
publication method, retain joint offense/defense draws before forming net and
ranks, and do not publish exact rank precision from marginal intervals.

**New automated check:** The uncertainty test suite now simulates correlated
game-level shocks around a known regularized, recentered RAPM estimand. It
checks directional 80% and 95% coverage for offense and joint net intervals.
This is a regression test of the resampling implementation, not a claim that
the simulator is an NBA data-generating process or a full calibration study.

## 2026-08-12 — Precision-aware prior promotion review

**Result:** `precision_aware_prior_rapm_v1_c4b0878571` is invalid for the
preregistered promotion gate. The run scored 2021-24, not the frozen
2018-21 selection and 2022-24 diagnostic schedule. The available three-year
cross-fitted prior/calibration history cannot provide the fourth model in the
omitted earlier seasons.

**Descriptive result:** It also loses on its non-preregistered 2023-24 slice:
mean RMSE improvement is -0.1425 points per game, correlation changes by
-0.0210, and candidate-minus-zero MSE has a paired 95% interval of
[+0.675, +7.502]. Zero-prior normal RAPM remains the reference. Do not retune
this candidate on these outcomes.

## 2026-08-13 — Stop all-time peak bootstrap; retain analytic RAPM covariance

**Decision:** Stop `rolling_peak_uncertainty_v1_af2fd07f284e` after 65 of the
planned 1,000 draws. Each draw refits all 50 historical three-year and five-year
windows. The local runtime cost is disproportionate to the current product
value. Completed checkpoints remain local and resumable, but are incomplete and
must not enter an artifact, API response, or research result.

**What remains useful:** For one fixed normal-RAPM fit, the analytic
game-cluster ridge sandwich supplies offense, defense, and joint-net standard
errors quickly. It already agrees closely with our whole-game bootstrap pilots
for high-exposure players: median 95% interval-width ratios are 1.009--1.011 in
2025 and 1.004--1.006 in 2022--24. This supports using analytic covariance as a
fast fixed-window diagnostic.

**Boundary:** Analytic covariance does not include the extra uncertainty from
choosing a player's maximum among many correlated windows, reapplying
eligibility, or interpreting ranks. Therefore rolling peak tables remain
descriptive and have no rank intervals. Justin Jacobs-style public peak tables
are suitable external descriptive comparators, not a replacement for missing
selection-aware uncertainty.

## 2026-08-13 — Pin role-context bronze inputs; do not promote them yet

**Question:** What small, current data addition can support future continuous
role research without reopening the frozen all-in-one feature search?

**What we did:** Audited the latest pinned `gabriel1200/site_Data` revision
(`bc583cb0188a6d5ae59d052d08ac0d6efe1b14fd`). The existing core archive and
the repository's current aggregate-source manifest already verify, so no broad
redownload was warranted. Added a separate resumable manifest for two additive
player-season-dribble-bucket inputs: shooting by dribble count (31,376 rows,
1,692 IDs) and jump-shot dribble context (28,435 rows, 1,656 IDs). Both cover
2014–26 and have unique `(PLAYER_ID, year, dribbles)` keys.

**Data-quality finding:** the jump-shot source provides populated makes and
attempts for every year but leaves supplied percentage and team display fields
blank in 2014–24. A future silver transform must derive shares from counts and
carry explicit missingness; it must not fill those fields with zero.

**Decision:** Keep both assets bronze and research-only. Do not add them to the
frozen AIO or interpret them as a rating. The next valid use is a time-safe
role table followed by a preregistered chronological role-block experiment
against the frozen baseline.

**Follow-through:** `build-role-context-features` now produces that
research-only player-season table from counts: 6,326 rows covering 1,590 IDs
in 2014–25, with no duplicate keys or infinite values. It excludes incomplete
2026 and deliberately drops source age, games, minutes, teams, and upstream
percentage display fields.

## 2026-08-13 — Current event-source coverage contract

**Finding:** NBA Stats V3 is locally complete for 2023–25: each regular season
has 1,230 games and the playoff slices have 82, 84, and 85 games respectively.
CDN NBA is complete through 2024 but has only 60 of 85 2025 playoff games;
PBPStats ends after 2024; 2024 playoff shot-detail and matchup slices are
absent.

**Decision:** V3 is the primary raw event source for 2023–25. CDN remains a
possession-tag validation source and must use `orderNumber`, not
`actionNumber`. No model may substitute any absent source slice with zero rows
or silently join a regular-only table as playoff-complete. The documented
coverage is a data-availability claim, not a licence grant.

## 2026-08-13 — Two-part shot-defense design revised; team pilot is null

**Sol checkpoint:** The proposed primary-defender, exact-zone, exact-lineup
model is not identified by the available licensed matchup aggregates. Only
about 36--37% of matched scorer-games exactly reconcile official makes and
attempts, allocated matchup attempts exceed official shots by about 9--11%, and
87% of shooter-games allocate attempts to multiple defenders. The revised
estimand is observed defensive-unit association with shot-zone mix and make
probability. Primary-defender and causal player-defense claims are forbidden.

**Data result:** `build-shot-defense-events` produced 654,376 regular-season
shots across 3,681 games in 2023--25. It aligns official shots to V3, CDN
`orderNumber`, stable event `actionId`, and exact ordinal five-player lineups.
All output gates passed: unique shot IDs, valid identities, known zones, finite
model fields, and 99.75% exact segment coverage.

**Pilot result:** `shot_defense_team_pilot_v1_a1d8880794` compared sparse L2
logistic zone and make models with and without defense-team context on a strict
chronological 2024-season split. Combined held-out log loss improved by 0.0893%
(2.04882 to 2.04699), below the frozen 0.5% practical gate. Make Brier improved
slightly. Only 41.5% of test shots used a lineup observed in training.

**Decision:** Classify the model `research_null`. Keep the validated event panel
for future exact-guarding data, but do not bootstrap this pilot, fit individual
defender rankings, merge it into the AIO, or publish a defense leaderboard. Move
the active model task to dynamic time-decayed impact trajectories.

## 2026-08-14 — Time-decay baseline passes forward proxy checks

**Question:** Can a small, no-future-leakage smoothing rule predict the next
annual normal-RAPM observation better than the latest annual value alone?

**Method:** Built `time_decayed_trajectory_v1_4706719bfb` from 2014--24 annual
zero-prior normal-RAPM targets. For each observed player-season it applies an
exponentially decayed filter using only that season and earlier observations,
separately on offense and defense; net is their exact sum. A frozen twelve-cell
grid varied annual decay (0.50, 0.65, 0.80, 0.90) and the possession exponent
(0.0, 0.5, 1.0). Selection used 2018--21 origins to predict the next annual
target; 2022--23 origins are later diagnostics. Players need 1,000 possessions
per side in the target season. One source row with zero defensive possessions is
explicitly excluded.

**Result:** Decay 0.80 and exponent 0.0 won selection. Equal-season next-year
net RMSE improves from 1.9481 to 1.7166 in selection and from 2.0285 to 1.7758
in later diagnostics. Net correlation is 0.419 and 0.435, respectively.

**Decision:** Retain this as a research trajectory baseline. It is not a full
latent-state model, has no trajectory intervals, and uses a legacy target
archive ending in 2024. Do not interpolate missing seasons, expose it in the
API, or call it current NBA strength. A later state-space model must beat this
baseline on a frozen forward test.

## 2026-08-14 — Expected-possession residual RAPM: causal state is ready

**Question:** Can normal RAPM be challenged with actual-minus-expected possession
points, without leaking the current possession or fitting away player effects?

**Data result:** `build-possession-start-context` created
`possession_start_context_ebcae214e662d404`: 787,579 canonical possessions
across 3,907 games. Start scores are reconstructed from completed prior canonical
possessions, rather than an incomplete cross-source action-number join. The
contract provides only period/time, score differential, home/away side, and
prior-possession context; it forbids player/team/lineup identity and all current
possession actions or outcomes as expected-points inputs.

**Decision:** The data contract is ready. Do not fit residual RAPM until expected
points are cross-fitted chronologically and player-neutral. The first model is a
simple Poisson or multinomial baseline; compare residual RAPM with normal RAPM
on identical games using equal-season margin RMSE and paired whole-game error.

## 2026-08-14 — Possession-start expected points is a useful null

**Method:** `expected_possession_points_v1_c9581a23b1` fits a player-neutral
Poisson expected-points model using only the frozen possession-start fields. It
uses whole-season chronological cross-fitting: train 2023 and test 2024, then
train 2023--24 and test 2025. No player, team, lineup, current-possession event,
or current-possession outcome is an input. The run yields 497,177 out-of-fold
regular-season predictions over 2,454 games.

**Result:** Context beats a training-mean constant in both folds, but only
slightly. 2024 RMSE is 1.19556 versus 1.19587 and Poisson deviance is 1.63425
versus 1.63493; 2025 RMSE is 1.19258 versus 1.19306 and deviance is 1.61978
versus 1.62078. Mean context improvement is 0.00039 RMSE and 0.00084 deviance.
Mean bias remains under 0.01 points per possession and both fits converge.

**Decision:** Do not fit residual RAPM. The contextual change is too close to a
constant subtraction to justify another RAPM evaluation. A prospective reopening
gate now requires at least 0.25% mean Poisson-deviance improvement in both
chronological folds, small bias, and player-neutral out-of-fold predictions. New
causal context—not extra model complexity—is needed before retrying.
