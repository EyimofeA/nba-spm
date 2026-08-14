# NBA Impact Roadmap

This is the one file to follow remotely. `RESEARCH_LOG.md` contains evidence and
dead ends. Updated 2026-08-14. See `docs/README.md` for the document index,
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
- Cross-fitted statistical priors cover every eligible 2019–24 feature row. For
  window `T`, training labels end by `T-3`. Six-fold prior-only net RMSE is
  1.2513 with 0.5198 correlation.
- The first matched prior-informed RAPM test did not demonstrate improvement.
  Full prior scale won 2020–22 selection, but beat zero-prior by only 0.0033
  margin RMSE on 2023–24, won 1/2 folds, and had a paired-game MSE interval of
  -1.12 to +0.73. Prior-only was clearly worse. Zero-prior remains production.
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
- The full 2025 box, playtype, DFG, rim, and hustle panel passes structural QA.
  The 2026 player sheet is partial at 81.8% of recent possession exposure.
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

## Active next task

The side-specific precision-aware SPM-prior contract has passed Sol review with
a revision: it now uses heteroskedastic earlier-window variance calibration, not
pooled variance subtraction. Its frozen historical 2018--21 schedule is
currently blocked because the cross-fitted prior history starts in 2019. The
frozen three-season feature contract cannot produce the planned 2018--21
selection folds. Do not rerun the invalid 2021--24 experiment. The active task
is to specify a separate pre-2016 prior contract or defer this challenger. The
scientific control plane,
canonical identity/provenance spine, two normal-RAPM uncertainty pilots, Ratings
API v2, first filtered time-decay trajectory baseline, and expected-possession
data contract are implemented. The player-neutral expected-points pilot improves
cross-fitted Poisson deviance by only about 0.05%; residual RAPM is deferred
until a richer causal state passes its prospective gate. The exact
shot/ordinal-lineup defense panel is also complete, but its defense-team pilot
improved held-out combined log loss by only 0.089% against a frozen 0.5% gate.
Keep both as documented nulls; do not fit player defender rankings without exact
guarding assignments. Keep zero-prior normal RAPM as the production reference.
Keep annual AIO, matchup factors, trajectories, and peaks research-only. Peak
ranks are descriptive only; a 1,000-draw selection-aware refit was stopped
after 65 draws because its cost is not justified now. Reserve Season 2027 as the
next untouched annual confirmation. WP neural work stays paused on the Mac.

Slow-network policy: each immutable file resumes from `.partial`, retries up to
20 times with exponential jitter, and waits up to five minutes for the next bytes.

## Ordered queue

1. **All-in-one:** specify a separate pre-2016 prior contract, or defer the
   reviewed precision-aware prior challenger. The frozen three-season contract
   cannot fill its 2018--21 selection horizon. Do not tune amplitudes or reuse
   the invalid 2021--24 run.
2. **Dynamic impact:** retain the 2014--26 filtered trajectory as the simple
   baseline. The causal state-space challenger is implemented but awaits its
   resumable annual observation-variance panel. Run no score until all 2014--26
   rows are present, then compare it on the frozen historical schedule.
3. **Expected possession:** retain the player-neutral start-state baseline as a
   null. Reopen only when a richer causal state clears its prospective expected-
   points gate; then compare residual RAPM only on identical games.
4. **Defense:** retain the validated shot/lineup panel, but wait for exact
   guarding data before individual defender modeling.
5. **WP-RAPM / credit:** value possession-start-to-end WP change only after the WP
   and lineup assignment are validated; compare Net Points and TD/Shapley ideas.
6. **Product:** stable DuckDB/API contract first, then a restrained player explorer.
   Do not rebuild the deleted UI before metric contracts are frozen.
7. **Later data:** injuries/availability, contracts, salaries, draft, roster stints,
   travel, and historical team schedules.
8. **WP later:** revisit only for playoff calibration or a cloud-trained causal
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
- Cross-fitted statistical priors: `statistical_priors_v1_2c81b23662`
- Prior-informed RAPM comparison: `prior_informed_rapm_v1_122ef63045`
- External BPM/xRAPM benchmark: `external_impact_benchmark_v1_bab43a4087`
- Annual normal RAPM targets: `single_season_rapm_targets_v1_fd876680da`
- Annual SPM and disagreements: `single_season_spm_v1_51adc53061`
- Corrected rolling normal-RAPM peaks: `rolling_rapm_peaks_v1_a8a612143c`
