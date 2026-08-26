# NBA Impact Roadmap

This is the one file to follow remotely. `RESEARCH_LOG.md` contains evidence and
dead ends. Updated 2026-08-25. See `docs/README.md` for the document index,
`docs/impact/ROADMAP.md` for the detailed RAPM/all-in-one plan, and
`docs/modeling/PLAYBOOK.md` for the common statistical modeling procedure.

## Current position

- Three-season events, player-games, lineups, and possessions now run from
  2023–24 through 2025–26.
- Lineups pass 3,931/3,941 games (99.75%); canonical possessions cover 3,907
  games with zero score or point-domain failures.
- ESPN WP benchmark covers 1,313 matched 2025–26 games.
- WP's score/time surface agrees closely with Inpredictable.
- Starter-free rolling context and causal possession control improve WP in both
  chronological folds; prior-season starter RAPM has not earned promotion.
- Additive splines and a bounded histogram GBM lose to logistic on both WP folds.
- The fixed five-seed 64×64 feed-forward MLP also loses badly; tabular logistic
  remains the production model.
- WP is frozen as good-enough infrastructure. Regular-season evidence is strong;
  playoff calibration remains a documented small-sample caveat.
- Normal RAPM uses the simple terminal lineup assignment. Fractional exposure is
  parked as a research sensitivity despite its small pooled gain.
- Normal RAPM keeps penalties 3000/3000/300. A 4500/4500/1000 selection winner
  lost on the untouched 2025–26 confirmation season.
- Two normal-RAPM uncertainty pilots show that analytic game-cluster intervals
  closely match whole-game bootstrap widths for high-exposure players. The
  expensive all-time peak bootstrap is stopped by user decision; peak ranks stay
  descriptive with no rank intervals.
- The first three-year statistical ridge baseline now compares box, advanced,
  and advanced-plus-on/off feature sets across three purged chronological folds.
- Frozen feature engineering improves the statistical AIO's reused 2024 net
  RMSE from 1.2984 to 1.2624. The gain is offensive; no new defensive block
  passes. Treat this as exploratory because 2024 informed earlier research.
- Diagnostic run `statistical_interpretability_v1_94d3f2c24b` confirms that
  offense relies mainly on shooting/scoring and public creation composites.
  Defense relies on disruption and rebounding, but offensive-role proxies are
  almost as important. This is a diagnostic, not causal attribution.
- Validated feature run `player_skill_features_v1_cf800d4e7e` adds 12 explicit
  shot-making, passing, screening, and hustle measurements over 5,791 player-
  seasons from 2014--24. Eleven are model candidates; absolute shot difficulty
  remains audit-only after QA found a large 2018--19 level shift. The annual
  and rolling integration tables are `statistical_features_v2_2515b57958` and
  `statistical_features_v2_d67bb64ac7`. They are inputs, not promoted models.
- Aging diagnostic `aging_balanced_validation_v1_ec5122d5a3` scores 1,768
  matched transitions in each direction. Raw forward/reverse net correlations
  are similar at 0.409/0.405. Age adjustment improves forward correlation but
  worsens RMSE; it is a diagnostic, not a replacement target.
- Behavior-role run `behavior_roles_v1_e0fb51c026` passes all frozen gates:
  93.71% coverage, 0.9845 seed stability, 61.66% out-of-sample adjacent exact-role
  persistence, and 0.9149 median adjacent-axis cosine. Six axes and seven soft
  affinities are integrated as candidates; the hard role is descriptive only.
- Separate role run `side_roles_v1_2c228f4b9e` selects six offense clusters and
  five defense clusters without impact targets. The fixed defense comparison
  rejects role inputs but selects eight scorer-adjusted matchup features on
  both 2020 and 2021. The matchup challenger wins all three 2022--24 diagnostic
  seasons, with mean RMSE change -0.0392 and correlation change +0.0601. It is
  research-only because those later seasons are inspected.
- Full annual SPM run `single_season_spm_v1_18496a1348` improves defense RMSE
  in all eight held-out seasons and net RMSE in seven of eight. The resulting
  leakage-safe centers feed `annual_aio_ratings_v1_b52b5aecd9`. The compact web
  client now exposes annual SPM, one-/three-/five-year RAPM, decomposed AIO,
  soft offense/defense roles, exact-scope RAPM intervals, and the observed aging
  curve from one sharded derived-data snapshot.
- Cross-fitted statistical priors cover every eligible 2019–24 feature row. For
  window `T`, training labels end by `T-3`. Six-fold prior-only net RMSE is
  1.2513 with 0.5198 correlation.
- The first matched prior-informed RAPM test did not demonstrate improvement.
  Full prior scale won 2020–22 selection, but beat zero-prior by only 0.0033
  margin RMSE on 2023–24, won 1/2 folds, and had a paired-game MSE interval of
  -1.12 to +0.73. Prior-only was clearly worse. Zero-prior remains production.
- Full factor-target run `factor_target_full_feature_spm_v1_69496cee37` gives
  every factor head 127 offense or 60 available defense inputs. Five of six
  heads improve over their sparse versions, but shooting-defense R-squared is
  still only `.126` with context. The factor route reaches `.312` reused 2026
  normal-RAPM R-squared, close to `.320` for direct full SPM with context.
- Five-year context run `five_year_spm_teammate_context_v1_13d270986a` adds a
  chronological residual correction to the exact frozen five-year SPM. It
  improves next-season net RAPM RMSE by `.002` and `.012` on the two reused
  folds. Retain the family for a later joint refit; do not change production or
  the frozen five-year research model from this small inspected gain.
- External benchmark `external_impact_benchmark_v1_bab43a4087` matches at least
  98.47% of SPM rows per window to minutes-weighted BPM and xRAPM. Among 2,295
  high-exposure player-windows, net SPM correlates 0.876 with BPM and 0.756 with
  xRAPM. Defense is the main external disagreement (0.630 with xRAPM).
- Annual run `single_season_spm_v1_51adc53061` uses current-season-only inputs
  and one-season normal RAPM labels from 2014–24. Its 2017–24 leave-one-season-out
  net RMSE is 1.4611 and correlation is 0.5314. For 2,860 matched player-seasons
  above 1,000 possessions per side, net correlation is 0.897 with BPM and 0.762
  with xRAPM. xRAPM remains a multi-window external comparator.
- The old 1997–2024 RAPM archive remains valuable but is stale after 2024.
- The pinned 2025 and 2026 player sheets now pass structural QA. The refreshed
  2026 sheet has 582 players and 100.5% of the median 2024--2025 possession
  exposure. Playtype and player-skill features reach 2026; DFG/rim DFG stop in
  2025 and scorer-matchup aggregates stop in 2024.
- Frozen annual SPM missed its untouched 2025 confirmation. Offense/defense/net
  RMSE is 1.102/1.154/1.610; defense correlation is 0.331. Do not promote or tune
  this run on 2025.
- Canonical annual normal-RAPM targets now cover 2023–24 through 2025–26. The
  audited 2023–24 source overlap has 0.975 net correlation with the legacy
  target and passes the frozen source-transition gate.
- The fixed 0.80 time-decay trajectory now extends through 2025–26. It still
  improves the historical annual proxy RMSE from 1.9481 to 1.7166 in selection
  and from 2.0549 to 1.8086 on the later diagnostic. It is research-only and
  has no trajectory intervals or new annual confirmation.
- The causal state-space challenger has a complete 6,942-row annual
  observation-variance panel. On exact matched player-season rows, it improves
  net RMSE versus frozen 0.80 time decay by 0.125 in selection and 0.106 in the
  later diagnostic. It remains research-only until a new untouched annual
  confirmation exists.
- Actual-clock rubber-band run `rubberband_adjustment_v1_34be1ee621` cross-fits
  lineup residuals by whole game on 743,946 2024--26 possessions. The scoring
  effect is near zero early and strongest in the final six minutes. It transfers
  to reused 2026 diagnostics. The frozen adjusted-RAPM follow-up
  `rubberband_progress_rapm_v2_b72716c2fb` compared actual clock with a
  no-lookahead, fixed 25-possession progress proxy. Their eight slopes correlate
  0.971, but both slightly worsen 2026 neutral game-margin RMSE. Keep the curve;
  reject the adjusted ratings under this gate.
- Standalone unit run `standalone_unit_rapm_v1_460b34a3b1` corrects the earlier
  terminology: pair, trio, four-man, and lineup RAPM contain no one-player
  columns. All four lose to one-player RAPM on the same reused 2026 games, with
  increasing out-of-sample sparsity as unit size grows.
- Joint score-signal run `rubberband_score_signal_v1_deac872ede` compared ten
  signed 5-point buckets, clipped linear terms, and cubic splines under one
  blocked gate. The selected bucket fit worsened reused 2026 RMSE by 0.036.
  Joint age-score run `age_score_context_v1_7e8689fee8` separates player-only
  ratings from pregame context. Player-only age, score, and joint fits worsen
  RMSE by 0.158, 0.036, and 0.101. Known lineup age changes pregame RMSE by
  -0.012 with a paired interval crossing zero. Retain normal RAPM.
- External reproduction run `external_reproduction_benchmark_v1_0a95702214`
  matches exact IDs and windows rather than fuzzy player names. Net RAPM
  correlates `.967` with Ryan Davis annual player-seasons, `.957` across exact
  five-year windows, `.980` across exact three-year windows, and `.897` with the
  current unequal-weight xRAPM three-year table. DARKO WOWY, RAPTOR on/off,
  AuPM, raw on-court margin, and a reproduced game-level PM ridge are retained
  as different-estimand diagnostics, not validation targets. All tables remain
  localhost-only in RAPM Lab.

## Active next task

Obtain a better event or lineup source for the nine Gabriel fallback games that
failed the strict repair gate. The adapter repaired `0022300535`; the 2024
regular-season input now covers 1,228 of 1,230 games. The rebuilt 2024 rating is
nearly unchanged (net correlation 0.99975; mean absolute change 0.0079 points
per 100). Do not weaken the existing QA gates to close the remaining gaps.

The precision-aware SPM-prior challenger is deferred. Its reviewed 2018--21
schedule cannot be produced by the frozen feature history, and the invalid
2021--24 run must not be reused. Keep zero-prior normal RAPM as the production
reference. Keep annual AIO, matchup factors, trajectories, and peaks research-
only. WP neural work stays paused on the Mac.

The public product is live at `https://nba-impact-lab.mofe.chatgpt.site`.
Normal RAPM covers 2017--26. SPM and AIO remain the validated 2017--24 model.
The full 2014--26 SPM refresh was flat-to-worse on the exact 2017--24 comparison,
and its 2025/2026 defensive correlations were 0.332/0.378. Keep that refresh as
a null result. Do not rerun the same annual specification without a new feature
or target-drift hypothesis.

Slow-network policy: each immutable file resumes from `.partial`, retries up to
20 times with exponential jitter, and waits up to five minutes for the next bytes.

## Ordered queue

1. **All-in-one challenger:** freeze factor groups with the selected eight
   matchup-defense fields. Use direct offense and defense RAPM targets. Do not
   include role interactions in the first challenger.
2. **Role research:** evaluate role-relative skill before any role-fit
   counterfactual. Require support/overlap checks for counterfactual roles.
3. **Dynamic impact:** retain the 2014--26 state-space filter as the leading
   research challenger and frozen 0.80 time decay as its baseline. Do not retune
   either from the reused historical result. Await an untouched annual
   confirmation before any API or public-rating promotion.
4. **Expected possession:** retain the player-neutral start-state baseline as a
   null. Reopen only when a richer causal state clears its prospective expected-
   points gate; then compare residual RAPM only on identical games.
5. **Defense:** retain the validated shot/lineup panel, but wait for exact
   guarding data before individual defender modeling.
6. **WP-RAPM / credit:** value possession-start-to-end WP change only after the WP
   and lineup assignment are validated; compare Net Points and TD/Shapley ideas.
7. **Product:** maintain the derived-data Ratings / Player / Roles / Research
   client. Keep annual tables and role maps lazy. Add new research pages only
   after their metric contract and caveat are frozen.
8. **Product research ideas:** define a roster net-rating calculator contract,
   then test 2-, 3-, 4-, and 5-player combination ratings and role-combination
   summaries. Require exposure floors and shrinkage. Do not present combination
   ratings as isolated causal effects.
9. **Later data:** injuries/availability, contracts, salaries, draft, roster stints,
   travel, and historical team schedules.
10. **WP later:** revisit only for playoff calibration or a cloud-trained causal
   sequence experiment after the impact platform is useful.

## Research rules

- New model ideas need chronological outer seasons, whole-game uncertainty, and
  identical scoring rows. Random seeds are not independent evidence.
- Log losses and null results; never silently promote a directional one-fold win.
- Production stays simple. Research may compare ridge variants, state-space/time
  decay, nonlinear priors, role splits, and RL credit assignment.
- No current-game box statistics, actual minutes, or future roster knowledge in
  pregame features.

## Naming

`New SPM` is now too narrow. Recommended eventual folder name: `NBA Impact Lab`.
Rename only after removing absolute paths from manifests and artifacts; doing it
now would break reproducibility links.

## Recent verified runs

- Game dimension: `game_dim_6e716feac7a2d6d6` (3,941 games)
- Event states: `event_states_e8a4cfbe25220240` (1,950,498 actions)
- Player-games: `player_games_03660947f91f97e3` (all 3,941 games)
- Lineups: `lineup_stints_7518759ccb7f181c` (3,931 passed games)
- Possessions: `possessions_769070fb3b70f511` (3,907 games)
- ESPN benchmark: `wp_espn_benchmark_v1_ca79cde82d`
- WP pregame Stage 0 folds: `wp_pregame_ablation_v3_30ab68d381`,
  `wp_pregame_ablation_v3_cdbcea84ee`
- Inpredictable surface: `wp_inpredictable_surface_v1_56696b0386`
- Possession-start WP folds: `wp_possession_start_v2_1db472e450`,
  `wp_possession_start_v2_0a5d626234`
- WP nonlinear parity: `wp_stage1_v1_7e6c77d51a`
- WP five-seed MLP parity: `wp_mlp_v1_7a7825bf09`
- RAPM lineup policy: `rapm_lineup_policy_v2_911d8bfce1`
- Current RAPM start lineup: `rapm_v0_d38f08740e`
- Current RAPM terminal lineup: `rapm_v0_ec1f17c82a`
- Normal RAPM tuning: `normal_rapm_v1_85e0cc8e27`
- Statistical features: `statistical_features_v1_940f99ed54`
- Statistical impact ridge: `statistical_impact_v2_5224a3b4a6`
- Statistical model families: `statistical_model_comparison_v1_dd31e7957d`
- Statistical direct net: `statistical_direct_net_v1_286a104216`
- Statistical feature ablation: `statistical_feature_ablation_v1_918be14a38`
- Optimized statistical AIO: `statistical_aio_v1_b0295558c6`
- Statistical features v2: `statistical_features_v2_8b2566243f`
- Statistical feature v2 comparison: `statistical_feature_v2_comparison_9b8d0555e0`
- Statistical AIO interpretation: `statistical_interpretability_v1_94d3f2c24b`
- Annual player-skill features: `player_skill_features_v1_cf800d4e7e`
- Skill-integrated rolling features: `statistical_features_v2_d67bb64ac7`
- Skill-integrated annual features: `statistical_features_v2_2515b57958`
- Aging-balanced annual validation: `aging_balanced_validation_v1_ec5122d5a3`
- Behavior-only roles: `behavior_roles_v1_e0fb51c026`
- Role-integrated rolling features: `statistical_features_v2_2bb78bc737`
- Role-integrated annual features: `statistical_features_v2_d8dd1d8dc2`
- Cross-fitted statistical priors: `statistical_priors_v1_2c81b23662`
- Prior-informed RAPM comparison: `prior_informed_rapm_v1_122ef63045`
- External BPM/xRAPM benchmark: `external_impact_benchmark_v1_bab43a4087`
- Annual normal RAPM targets: `single_season_rapm_targets_v1_fd876680da`
- Annual SPM and disagreements: `single_season_spm_v1_51adc53061`
- Corrected rolling normal-RAPM peaks: `rolling_rapm_peaks_v1_a8a612143c`
- Full annual SPM: `single_season_spm_v1_18496a1348`
- Historical annual AIO: `annual_aio_ratings_v1_b52b5aecd9`
- Split side roles: `side_roles_v1_2c228f4b9e`
- Forward role stabilization: `role_stabilization_v1_f5b426dd5d`
- Canonical annual target bridge: `canonical_annual_target_panel_v1_4586bd2f72`
- Canonical-label annual SPM refresh: `single_season_spm_v1_c4be58c72e`
- Full 2014--26 SPM null refresh: `single_season_spm_v1_47b3bd9b17`
- Selected five-year SPM plus one-season AIO: `five_year_target_spm_v1_65550acb79`
- Full 2014--26 base features: `statistical_features_v1_65446dd3e2`
- Full 2014--26 expanded features: `statistical_features_v2_b808fc1bf1`
- Repaired 2024--26 Normal RAPM: `current_single_season_rapm_targets_v1_b4cdb51de8`
- Actual-clock rubber-band estimate: `rubberband_adjustment_v1_34be1ee621`
- Joint score-signal comparison: `rubberband_score_signal_v1_deac872ede`
- Joint age-score comparison: `age_score_context_v1_7e8689fee8`
