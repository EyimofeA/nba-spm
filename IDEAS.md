# RAPM Ideas Ledger

NOTE: `PROJECT.md` is the master state doc (agents start there). Baseline to
beat as of 2026-07-03 evening: minutes-prior c=2 → 0.7335/0.6953 on the two
folds (champion zero-prior: 0.6596/0.5939).

Statuses: `untested` / `testing` / `tested-won` / `tested-lost` / `blocked` / `not-tonight`.
Every experiment run updates its row. Scores = next-season (2024) game-margin corr/RMSE
unless stated. Baseline to beat: player+home ridge, corr 0.632 / RMSE 14.49.

## Sample construction
| idea | status | result / note |
|---|---|---|
| Hard garbage-time drop | tested-lost (current 5Y) | A stale fast-lab result called this a win. Frozen 2019--26 five-year testing dropped 9.84% of rows and worsened selection and diagnostic RMSE; keep all rows in the current reference. |
| Soft garbage down-weighting (competitiveness weight) | tested-lost | 0.6316 = baseline; hard drop already sufficient |
| Recency decay (exp. by date, hl 1yr/2yr) | tested-won | CHAMPION now hl=250d (two-fold mean 0.627); hl365 0.619; no-decay 0.592 |
| Model-free decay shape (learned buckets) | tested-lost | non-monotone wiggle = single-fold overfit; licensed the exponential, don't ship buckets |
| Power-law / hyperbolic decay | tested-lost | degenerates to exponential at 3-yr horizons; unidentifiable |
| Age normalization / equal-age RAPM | tested-won for forecast translation only | The descriptive curve peaks around 25--27 and is offense-led. In reused walk-forward tests, a one-year Gaussian age smoother lowered net weighted RMSE by 0.014/0.027/0.036 for trailing 1/3/5-season ratings. Stored ages are integers, so 0.1-year aging was not truly testable. Keep age out of retrospective RAPM and use the curve only to translate ratings forward. |
| Level-dependent aging (b=-0.40) | untested | confounded with mean reversion; needs SE-aware state-space model to separate |
| Min-stint filtering / stint weighting | not-tonight | our data is possession-level already |
| Playoff possessions at reduced weight | blocked | user: nothing involving playoffs |

## Regularization
| idea | status | result / note |
|---|---|---|
| Lambda magnitude | tested-lost | flat 1000–8000; fixed 3000 |
| Asymmetric off/def lambdas (learned, agnostic) | tested-lost (current 5Y) | An older all-history lab selected 2000/4500 on a different substrate. Grid, Sobol, GCV, bivariate, and EB five-year searches did not clear future-game gates; retain 3000/3000/300. |
| Elastic net / LASSO (sparsity) | tested-lost | SGD underfit badly (0.16–0.32); verdict is on the optimizer, not sparsity — retry with proper solver someday |
| Replacement-level pooling (<N poss share one column) | tested-lost | Old 250/500/1000-possession pooling was flat. A 2026 Dummy RAPM paper reports a small 0.30% RMSE gain when excluded low-minute players remain in stints, but our canonical design excludes no player columns, so that mechanism does not apply directly. |
| Regularization path diagnostic | untested | diagnostic, not a gate |

## Shrinkage targets / old information (separate lane — wins by construction)
| idea | status | result / note |
|---|---|---|
| Previous-window RAPM prior | tested-lost | SURPRISE: 0.589–0.613 < baseline 0.632. Stale prior = biased target (aging, role changes) |
| Age-translated previous-window prior | tested-lost | mechanism VALIDATED (beats raw prior at every strength, 0.640 vs 0.622) but still < zero-prior+decay champion 0.6596. Revisit as per-player adaptive prior for thin-data players |
| Infinite RAPM (iterated prior chain) | tested-lost | depth-3 chain 0.5933; degradation compounds with iteration. Question answered |
| SPM/box-score prior | tested-lost (vs minutes) | v1 leaked; v1.1 0.699/0.668; v1.2 FINAL 0.6988/0.6619 — flat feature set cannot beat minutes as prior center. Residual SPM is the remaining question |
| Residual SPM (predict minutes-prior residual) | untested | THE next experiment: isolates box's marginal value over coach judgment; also the fair test bed for rate stabilization |
| Own-box-stat noise sharing | untested | subtle cousin of on/off leakage: player's box rates share his lineups' possession noise. Suspected reason SPM R² flatters vs gate. Diagnose via residual SPM |
| Minutes-only prior | tested-won | 0.7335/0.6953 at c=2 — biggest win in project. Saturates c≈8; RMSE creep → calibration TODO |
| Clean prior semantics (per-player tau² pull) | tested-won | flipped the stale-prior verdict; now mandatory (see PROJECT.md conventions) |
| Feature provenance rule (no label-noise features) | tested-won | on/off ratings banned class; "R² up, gate down" = contamination signature |
| Window-length ensemble (1/3/5yr blend) | tested-lost | win1 0.647 but FAILS anchor test; win5 0.6095. Decay dominates window-length games |

## Context columns
| idea | status | result / note |
|---|---|---|
| Single home effect | tested-won | Frozen five-year diagnostics improve RMSE 15.0512→14.9358 and MAE 11.8529→11.7277 versus no home. |
| Rubberband score adjustment | tested-lost as RAPM correction | The actual-clock curve and fixed 25-possession progress proxy agree closely (eight-slope correlation 0.971). After removing only the signed-margin slope from the target, clock-adjusted and possession-adjusted RAPM improved 2026 game-margin correlation by about .009 but worsened RMSE by .018/.026; paired intervals cross zero. Keep the descriptive curve and reject both adjusted ratings. |
| Quadratic / nonlinear rubberband | tested-lost | Quadratic curvature is negligible in Q2--Q4 after lineup and season adjustment. Earlier LightGBM live-margin control also collapsed. Do not spend another full run on shape search. |
| Season scoring environment | tested-won as normalization | Current five-year RAPM removes each training season's mean PPP before fitting. Explicit season dummies are an equivalent, less compact parameterization for this purpose; the old 3-year dummy test added no value. |
| Per-team home effects | tested-lost | Official home-team map now covers 2014--26. Exposure-constrained deviations selected λ=30,000 but worsened diagnostic RMSE 14.9358→14.9433; only 15.35% of paired draws favored them. This is not an altitude/travel estimate. |
| Rest / back-to-back indicator | untested in RAPM | The schedule join is feasible and rest helped the separate WP model, but that does not prove player-rating lift. Test global signed rest categories first, with no player interactions. |
| Coach offense / defense effects | tested-lost | The 2017--26 joint player/coach fit covered all 11,969 games. Selection chose the strongest coach shrinkage tested (100,000), then coach terms worsened 2026 RMSE by .0147 and correlation by .0012. Coach, roster, and organization remain too confounded for a portable rating. |
| Referee effects | not-tonight | needs ref data |
| Clock-state fatigue proxy | tested-lost | Numerically inert in frozen five-year folds; game clock is not observed player fatigue. |
| Player fatigue / foul trouble / injuries | not-tonight | needs richer pre-outcome data and a separate causal contract |
| Luck-adjusted RAPM | untested, definition required | Cross-fit expected values for predeclared noisy events, then fit the same lineup model. Do not subtract all shotmaking variance or call one arbitrary residual target canonical luck adjustment. |

## Models & targets
| idea | status | result / note |
|---|---|---|
| Ridge (standard) | tested-won | the baseline everything must beat |
| LightGBM possession-level | tested-lost | 0.613 vs 0.632; no nonlinear signal over ratings |
| Win-probability RAPM target | built diagnostic | Cross-fitted 2025--26 WP surfaces produce a conserved leverage-credit target over 497,177 possessions. Net correlates .738 with points RAPM but annual net stability is only .125. Useful for descriptive high-leverage credit, not an all-in-one or forecast. |
| Conserved points-channel RAPM | built research pilot | The 2022--26 run splits points into one-point, two-point, and three-plus channels in one shared ridge factorization. The channels reproduce normal RAPM within `1.32e-7` points per 100. This is an exact descriptive decomposition, not a probability model. |
| Event-factor RAPM (shooting, turnovers, OREB; offense/defense) | built descriptive pilot | The 2024--26 run fits six sides: shooting eFG, turnover avoidance/forcing, and OREB conversion/prevention. Event mapping is 98.43%. These mechanism ratings use different opportunity denominators and intentionally do not sum to points RAPM. A conserved FT/2P/3P/OREB ledger remains separate work. |
| True multinomial expected-points RAPM | tested-lost | A 0/1/2/3+ softmax selected alpha=.001 on 2025. On 2026 it had margin RMSE 15.508 and correlation .333 versus 15.473/.334 for linear points RAPM; constant class rates also beat its log loss. Retain only as a descriptive outcome surface. |
| Rim FGA, 4-factor, and shot-quality RAPM | untested | Modern shot/event data are local. Historical event-grade coverage remains incomplete. |
| Residual 2/3/4/5-player interaction layers | tested-lost / null | These are additions after one-player RAPM, not standalone unit RAPM. They selected penalties on 2025 and were checked on 2026. Pair, trio, and four-player layers worsened RMSE by .035/.070/.036. The five-player layer improved RMSE by only .0039 and correlation by .0006, effectively noise. Do not call these causal chemistry. |
| Standalone pair/trio/four-man/lineup RAPM | tested-lost | Corrected estimand has no individual player columns and fits raw possession points. Five-season windows selected penalties on 2025 and checked the same 1,228 games as one-player RAPM in reused 2026. RMSE worsened by .471/.780/1.022/1.226. Pair-to-lineup test-slot coverage fell from 41.4% to 3.7% under the frozen exposure floors, so sparsity is part of the result. |
| Matchup-RAPM | not-tonight | needs who-guarded-whom ingest |
| Defender-assignment RAPM (who-guarded-whom inside the design matrix) | parked-by-principal 2026-08-22 | Data now in hand: 1.77M licensed possession-level matchup pairs 2018-25. Ceiling math says this raises defensive *measurement quality* (target reliability 0.31-0.48 today), not just prediction. Structural change, new estimand - predeclare forward gates when unparked. |
| Simple APM (OLS, no ridge) | built 2026-08-22 | Full-era baseline: `research/rapm_lab/outputs/apm_1997_2025.csv` (6.64M possessions, 5,309 players, 49s). Anchors pass: Jokic +14.7, LeBron +13.1 (132k possessions), KG/Stockton/Duncan top-era mix correct. The 2014-26 loader baseline is also saved. Its +111 fringe-player estimate shows why ridge is required. |
| RAPTOR On-Off (courtmate-chain regression) | queued | zero new data needed; tests the 538 claim that on/off chains match RAPM out-of-sample |
| SPM vs BPM head-to-head | queued | correlation documented (0.876-0.897) but no predictive study exists in-repo |
| Shot-profile RAPMs: rim-assist / midrange-freq / shot-freq / shot-efficiency | untested (principal priority) | shot-level 2017-26 local; 1997-2013 factors exist in Gabriel old_data pipeline spec |
| JE 6/8-factor recovery | queued | faithful reconstruction buildable (inputs local or one GitHub download); EXACT replication additionally requires JE's target definitions, filters, lambda/shrinkage, luck adjustment, and verification table - until then label output 'JE-style', never 'exact'. Needs event-grade PBP 97+; source = gabriel1200/merged_playbyplay old_data (GitHub); NBA stats/cdn endpoints unreachable from this network (stats hangs, cdn 403) - direct scraping parked |
| GPM / WOWYR (1957-96) | blocked | BBRef game logs scrape pending; spec = Thinking Basketball Part IV (2017-11-17) |
| Playtype RAPM | backburner (principal 2026-08-22) | possession tags not in public PBP; state-split tier derivable now; aggregate-prior tier uses owned Synergy export |
| DRIP (CraftedNBA; ODRIP/DDRIP) | verification-target | name confirmed in glossary mentions; expansion/method unverified (page 404) - external comparator only |
| State-split RAPM (halfcourt vs transition) | not-tonight | needs possession-type tags |
| Bayesian / NN RAPM | not-tonight | infeasible locally tonight |

## Validation & diagnostics
| idea | status | result / note |
|---|---|---|
| Next-season margin retrodiction | tested-won | the gate |
| Split-half reliability | tested-won | tiebreaker for finalists |
| Possession RMSE for model selection | tested-lost | banned; cannot discriminate |
| Gobert sign-anchor test | testing | auto-reject variants that flip anchors |
| ESS logging on weighted fits | testing | catch silent sample collapse |
| def_sum importance=0 in LGBM probe | tested-won | feature math verified correct on synthetic case; LGBM simply routed all splits through off_sum (correlated features), not a bug |

---

## Principal backlog (user-directed — parallel tracks, not current sprint)

These are **yours** to prioritize. Agents finish the foundry + minutes/SPM lane first unless you explicitly redirect. Log new rows here when an idea gets tested.

| track | status | note |
|---|---|---|
| **Win-probability RAPM** | built diagnostic | Conserved 2025--26 possession credit is implemented from prior-season-trained player-neutral WP surfaces. It is leverage-weighted retrospective credit and is not promoted as future strength. See Models table above. |
| **Draft modeling** | untested | Pre-NBA signal → pro impact. Needs draft + college/G-league/international data ingest; eval = year-1/2/3 RAPM or minutes-survival, not same-window descriptive gate. |
| **Player impact modeling (product)** | in-progress | Umbrella for what we're shipping: descriptive RAPM panel + SPM/minutes prior + foundry. Forecasting variant (walk-forward SPM) is TODO 5 in PROJECT.md. |
| **Skill-based modeling** | untested | Decompose impact into interpretable skills (shooting, creation, defense, rebounding) rather than off/def only — SPM features, playtypes, tracking roles. Natural fit after def-SPM audit + role clusters; may feed draft model priors. |

**Dependency sketch (not execution order):**
```
possession RAPM (champion) → WP-RAPM target (same matrix, new y)
                           → skill SPM (multi-head or multi-target)
descriptive panel → walk-forward prior → draft model (pre-NBA features → pro RAPM)
```

**Do not mix into foundry v1:** draft data, WP target, or skill heads — separate harnesses, separate gates, same logging discipline (`experiments.csv` + RESEARCH_LOG).

## External sources & verification targets (2026-08-22)

- Verify against: xrapm.com (Engelmann) | DARKO DPM leaderboard (local data/raw/darko/) | EPM + LEBRON public tables | JE 6-factor values (principal pull) | canonical RAPM Drive folder (principal export) | CraftedNBA DRIP.
- Method sources: Joe Sill RAPM (Sloan 2010) + stabilization review (godismyjudgeok.com/DStats) | Engelmann Substack build guides with SEs/CIs | thespax.com 25-year RAPM build | garbage-time identification (nbainrstats.netlify.app; CTG rules) | rd11490 NBA_Tutorials | RussDT/pbp_rapm docs (Gabriel old_data ingest spec; 18 metric surfaces incl. LA_RAPM + shot-profile factors 1997-2013).
- Data provenance: pre-2014 possession data ALREADY LOCAL at rapm/data/possession_cache/matchups_{1997..2024}.parquet (10-man lineups per possession; 2025 file empty). Event-grade 1997-2013 (shots/playtypes) via Gabriel old_data on GitHub - not yet downloaded.
- Zach Stone = Backpicks reader contributor; GPM methodology exists only in Thinking Basketball Part IV post.
- Exa search wiring pending principal API key (EXA_API_KEY in .env).
- Scrape-day-one rule (advisory 2026-08-22): when the event-grade 1997-2013
  ingest happens (Gabriel old_data or otherwise), pull official player minutes /
  team totals (BoxScoreTraditional-class or BBRef game logs) in the SAME pass.
  6/8-factor targets and lineup-stint QA gates require official minutes;
  bolting them on later forces a second full pass. Note: stats.nba.com is
  unreachable from this network - BBRef game logs are the reachable minutes
  source for that era.
