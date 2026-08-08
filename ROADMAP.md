# NBA Impact Roadmap

This is the one file to follow remotely. `RESEARCH_LOG.md` contains evidence and
dead ends. Updated 2026-08-08. See `docs/README.md` for the document index,
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
- The old 1997–2024 RAPM archive remains valuable but is stale after 2024.

## Active next task

Rebuild three-year advanced inputs with natural attempt/touch denominators, then
compare the user's feature set against the frozen box and advanced ridge
baselines. Keep on/off as a separately labeled impact-assisted model. WP neural
work is paused; do not train neural models on the local Mac.

Slow-network policy: each immutable file resumes from `.partial`, retries up to
20 times with exponential jitter, and waits up to five minutes for the next bytes.

## Ordered queue

1. **RAPM:** normal RAPM is the terminal-lineup, zero-prior ridge with penalties
   3000/3000/300. Keep fractional attribution research-only.
2. **All-in-one:** improve and compare box/tracking/playtype feature groups for
   offense and defense. Keep independent and on/off-assisted variants distinct.
3. **Dynamic impact:** create annual time-decayed/player-state trajectories and
   peak 1/3/5-year views in the style of NBA RAPM peaks.
4. **WP-RAPM / credit:** value possession-start-to-end WP change only after the WP
   and lineup assignment are validated; compare Net Points and TD/Shapley ideas.
5. **Product:** stable DuckDB/API contract first, then a restrained player explorer.
   Do not rebuild the deleted UI before metric contracts are frozen.
6. **Later data:** injuries/availability, contracts, salaries, draft, roster stints,
   travel, and historical team schedules.
7. **WP later:** revisit only for playoff calibration or a cloud-trained causal
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
- Statistical impact ridge: `statistical_impact_v2_48f6ad776f`
