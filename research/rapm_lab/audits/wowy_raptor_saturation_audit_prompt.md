# CourtSignal RAPM saturation audit

Date: 2026-08-25

This is a read-only statistical audit. Do not edit files or run expensive
models. Inspect the named code and artifacts. Report bugs before research ideas.

## Files to inspect

- `research/rapm_lab/run_wowy_raptor_reproduction.py`
- `research/rapm_lab/run_raptor_onoff_proxy.py`
- `tests/test_wowy_raptor_reproduction.py`
- `tests/test_raptor_onoff_proxy.py`
- `research/rapm_lab/outputs/wowy_raptor_reproduction/wowy_raptor_reproduction_v1_4983f2cd47/run.json`
- `research/rapm_lab/outputs/raptor_onoff_proxy/raptor_onoff_proxy_v1_bb23b07cc8/run.json`
- `research/rapm_lab/audits/external_reproduction_audit_results.md`
- `RESEARCH_LOG.md`, especially the 2026-08-25 RAPM entries

## Frozen evidence

1. DARKO publishes player-game Final Cut WOWY histories. The runner downloads
   all 1,478 public histories and reconstructs each public season average as the
   unweighted mean of game-level offense, defense, and net. All 5,497 published
   player-seasons match. Pearson and rank correlations equal 1.000000. Maximum
   absolute errors are 7.55e-15 offense, 6.66e-15 defense, and 7.55e-15 net.
   This does not reproduce DARKO's underlying daily causal model.
2. The local FiveThirtyEight player-season RAPTOR file is semantically identical
   to the official GitHub CSV across 4,684 rows and all three on/off components.
3. FiveThirtyEight described an on/off construction using opposition-adjusted
   own on-court rating, direct courtmates' without-player ratings with negative
   influence, and second-order courtmate context with positive influence. It did
   not publish the three fitted coefficients, exact opposition adjustment, or
   exact second-order construction. The local implementation is therefore named
   a RAPTOR-on/off-inspired proxy, not an exact reproduction.
4. The proxy freezes one shared offense/defense coefficient vector on 2014-2018
   and scores 2019-2022. Courtmate context is computed within team for traded
   players, and the focal player is excluded from second-order context. For
   1,000-minute player-seasons, held-out net Pearson and rank correlations
   against the published regular-season team-stint RAPTOR on/off target are
   .9658 and .9575 across 1,036 rows. This is distinct from the player CSV in
   item 2, which combines regular season and playoffs. The fitted signs are own
   on-court +.5919, direct courtmates -.5964, and
   second-order courtmates +.2431.
5. CourtSignal normal RAPM agrees strongly with independent same-key normal RAPM
   references. Net correlations are .967 for Ryan Davis annual, .980 for exact
   three-year windows, .957 for exact five-year windows, and .897 for current
   xRAPM with a season-weight mismatch.
6. DARKO WOWY and RAPTOR on/off are different estimands, not validation targets.
   Against CourtSignal annual RAPM, pooled DARKO net correlation is .574. RAPTOR
   on/off net correlation is .917 Pearson and .912 rank after restricting to
   1,000-minute player-seasons.
7. Blocked future-game tests rejected score-state rubber-band controls, age
   controls, joint age plus score, and a time-decayed actual-age model. Luck
   adjustment improved one reused season but its paired interval crossed zero.
   The current production candidate remains zero-prior, terminal-lineup,
   possession-level, rolling five-year RAPM with 3000/3000/300 penalties.
8. Pair, trio, quartet, five-man, factor, outcome, WP-credit, coach, age-27, and
   multinomial fits exist as local research views. Most answer different
   questions or lack a clean future-data promotion gate.

## Questions

1. Find any P0 or P1 defect in the two new reproductions, joins, weighting,
   signs, held-out split, or claims. Give file and line evidence.
2. Is the .9658 held-out proxy correlation against the regular-season team
   target meaningful evidence that the public RAPTOR on/off construction was
   implemented closely enough for a reproduction check? Explain what it proves
   and what it cannot prove.
3. Do the combined external checks rule out a gross CourtSignal RAPM sign, join,
   scale, or windowing bug?
4. Are we done researching the core normal RAPM estimator for now? Answer yes or
   no. Distinguish normal RAPM work from RAPM-like decomposition and credit
   models.
5. List at most five remaining projects, ranked by expected decision value.
   For each, state the estimand, the smallest decisive test, and whether it
   belongs before or after SPM/AIO, uncertainty, and dynamic projections.
6. End with one verdict: FREEZE NORMAL RAPM, CONTINUE ONE NAMED RAPM TEST, or
   BLOCKED BY DEFECT. Do not produce a broad idea dump.
