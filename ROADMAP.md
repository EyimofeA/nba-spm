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

Acquire and canonicalize one earlier rich event season so WP and current RAPM
have another chronological outer fold. Do not tune further against 2025–26 alone.

## Ordered queue

1. **Data:** ingest 2023–24 event PBP, game dimension, player-games, and lineups;
   run the existing completeness and chronology gates.
2. **WP:** build causal possession-start states and add possession as an ablation;
   confirm rolling team context across at least two outer seasons.
3. **RAPM:** run current start/terminal/segment policies across repeated seasons;
   freeze a simple production specification and keep research variants separate.
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
- Current RAPM start lineup: `rapm_v0_d38f08740e`
- Current RAPM terminal lineup: `rapm_v0_ec1f17c82a`

