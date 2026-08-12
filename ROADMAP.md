# NBA Impact Roadmap

This is the one file to follow remotely. `RESEARCH_LOG.md` contains evidence and
dead ends. Updated 2026-08-12. See `docs/README.md` for the document index,
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

## Active next task

Freeze the scientific control plane before another AIO feature search. The
estimand registry, conservative season-exposure map, pinned-artifact audit, and
precision-aware prior preregistration now live in `research/`. Corrected rolling
peaks are pinned as `rolling_rapm_peaks_v1_a8a612143c`. Next, add whole-game
RAPM uncertainty and expose its status through the artifact/API contract. Keep
zero-prior normal RAPM as the production reference. Keep the annual AIO and
matchup factors research-only. Reserve Season 2027 as the next untouched annual
confirmation. WP neural work stays paused on the Mac.

Slow-network policy: each immutable file resumes from `.partial`, retries up to
20 times with exponential jitter, and waits up to five minutes for the next bytes.

## Ordered queue

1. **Evidence:** complete artifact lineage and uncertainty status for every
   pinned API rating. Keep Season 2027 untouched.
2. **RAPM:** add whole-game uncertainty to the terminal-lineup, zero-prior
   3000/3000/300 reference, then propagate it through peaks and the API.
3. **All-in-one:** implement the preregistered precision-aware prior challenger.
   Do not start another broad feature subset search on reused seasons.
4. **Dynamic impact:** create annual time-decayed/player-state trajectories and
   peak 1/3/5-year views in the style of NBA RAPM peaks.
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
