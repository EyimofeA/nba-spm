# NBA Impact Roadmap

This is the one file to follow remotely. `RESEARCH_LOG.md` contains evidence and
dead ends; `WIN_PROBABILITY.md` contains the WP model card. Updated 2026-08-08.

## Current position

- Three-season events, player-games, lineups, and possessions now run from
  2023–24 through 2025–26.
- Lineups pass 3,931/3,941 games (99.75%); canonical possessions cover 3,907
  games with zero score or point-domain failures.
- ESPN WP benchmark covers 1,313 matched 2025–26 games.
- WP's score/time surface agrees closely with Inpredictable.
- Starter-free rolling context and causal possession control improve WP in both
  chronological folds; prior-season starter RAPM has not earned promotion.
- Current RAPM is a two-season baseline, not the final all-in-one.
- The old 1997–2024 RAPM archive remains valuable but is stale after 2024.

## Active next task

Run the preregistered Stage 1 nonlinear comparison on identical starter-free
states: interpretable GAM first, then a parameter-bounded histogram GBM. Preserve
the logistic model as the frozen Stage 0 control and do not tune on 2025–26.

Slow-network policy: each immutable file resumes from `.partial`, retries up to
20 times with exponential jitter, and waits up to five minutes for the next bytes.

## Ordered queue

1. **WP:** starter-free rolling context and time-interacted possession control
   pass both outer folds and are frozen Stage 0. Run the GAM/GBM → MLP → TCN →
   GRU → transformer ladder in `WP_ARCHITECTURES.md`; do not tune on 2025–26.
2. **RAPM:** terminal lineup is the simple current baseline; fractional segment
   exposure is the research challenger. Confirm both across additional seasons,
   then tune penalties with nested chronological folds.
3. **All-in-one:** build independent box/tracking/playtype priors for offense and
   defense, then stack only improvements that pass next-season prediction gates.
4. **Dynamic impact:** create annual time-decayed/player-state trajectories and
   peak 1/3/5-year views in the style of NBA RAPM peaks.
5. **WP-RAPM / credit:** value possession-start-to-end WP change only after the WP
   and lineup assignment are validated; compare Net Points and TD/Shapley ideas.
6. **Product:** stable DuckDB/API contract first, then a restrained player explorer.
   Do not rebuild the deleted UI before metric contracts are frozen.
7. **Later data:** injuries/availability, contracts, salaries, draft, roster stints,
   travel, and historical team schedules.

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
- RAPM lineup policy: `rapm_lineup_policy_v1_23149bbb29`
- Current RAPM start lineup: `rapm_v0_d38f08740e`
- Current RAPM terminal lineup: `rapm_v0_ec1f17c82a`
