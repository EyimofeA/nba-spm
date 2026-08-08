# NBA Impact Roadmap

This is the one file to follow remotely. `RESEARCH_LOG.md` contains evidence and
dead ends; `WIN_PROBABILITY.md` contains the WP model card. Updated 2026-08-08.

## Current position

- Current events, player-games, lineups, possessions, RAPM, and WP pipelines run.
- ESPN WP benchmark covers 1,313 matched 2025–26 games.
- WP's score/time surface agrees closely with Inpredictable.
- Current RAPM is a two-season baseline, not the final all-in-one.
- The old 1997–2024 RAPM archive remains valuable but is stale after 2024.

## Active next task

The pinned 2023–24 rich-event batch is ready (10 files, 86.96 MB). Launch it
when the laptop is on AC, then rebuild silver tables and run the second WP fold.
Do not tune further against 2025–26 alone.

Slow-network policy: each immutable file resumes from `.partial`, retries up to
20 times with exponential jitter, and waits up to five minutes for the next bytes.

## Ordered queue

1. **Data:** ingest 2023–24 event PBP, game dimension, player-games, and lineups;
   run the existing completeness and chronology gates.
2. **WP:** causal possession-start control passes one outer fold; confirm it and
   rolling team context on at least one additional outer season before promotion.
   Then run the frozen GAM/GBM → MLP → TCN → GRU → transformer ladder in
   `WP_ARCHITECTURES.md`; do not tune architectures on 2025–26.
3. **RAPM:** terminal lineup is the simple current baseline; fractional segment
   exposure is the research challenger. Confirm both across additional seasons,
   then tune penalties with nested chronological folds.
4. **All-in-one:** build independent box/tracking/playtype priors for offense and
   defense, then stack only improvements that pass next-season prediction gates.
5. **Dynamic impact:** create annual time-decayed/player-state trajectories and
   peak 1/3/5-year views in the style of NBA RAPM peaks.
6. **WP-RAPM / credit:** value possession-start-to-end WP change only after the WP
   and lineup assignment are validated; compare Net Points and TD/Shapley ideas.
7. **Product:** stable DuckDB/API contract first, then a restrained player explorer.
   Do not rebuild the deleted UI before metric contracts are frozen.
8. **Later data:** injuries/availability, contracts, salaries, draft, roster stints,
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

- ESPN benchmark: `wp_espn_benchmark_v1_ca79cde82d`
- WP pregame challenger: `wp_pregame_ablation_v2_522e1a36f2`
- Inpredictable surface: `wp_inpredictable_surface_v1_56696b0386`
- Possession-start WP: `wp_possession_start_v1_9af34729ef`
- RAPM lineup policy: `rapm_lineup_policy_v1_23149bbb29`
- Current RAPM start lineup: `rapm_v0_d38f08740e`
- Current RAPM terminal lineup: `rapm_v0_ec1f17c82a`
