# Ratings API

This local, read-only JSON API is the data contract for future player pages. It
pins explicit model runs; it never selects an artifact because it has the newest
timestamp.

Start it from the project root:

```bash
python3 -m nba_impact.cli serve-ratings
```

Default address: `http://127.0.0.1:8765`.

## Stable v1 routes

| Route | Purpose |
|---|---|
| `GET /v1/health` | Process and contract health |
| `GET /v1/meta` | Run IDs, estimands, seasons, metrics, and caveats |
| `GET /v1/leaderboards/annual?season=2024&metric=aio_net` | Annual AIO/SPM/normal-RAPM leaderboard |
| `GET /v1/leaderboards/current?metric=net` | Current 2024–26 frozen normal-RAPM leaderboard |
| `GET /v1/leaderboards/peaks?window=5&component=net` | Three- or five-year all-time peak table |
| `GET /v1/leaderboards/matchup-defense?season=2024` | Research-only scorer-adjusted matchup-factor leaderboard |
| `GET /v1/players/search?q=lebron` | Case-insensitive player lookup |
| `GET /v1/players/2544` | Annual decomposition, rolling history, peaks, and matchup-defense factors |

Leaderboard routes accept `limit` and `offset`; annual leaderboards also accept
`minimum_possessions`. The server caps `limit` at 100. Invalid metrics,
components, and parameters return HTTP 400. Unknown players and routes return
HTTP 404.

The matchup-defense route accepts `minimum_matchup_possessions`. It exposes six
positive-good factor metrics. The default is scorer-adjusted shot-making points
saved. Every response is labeled `research_only` and includes the source caveat.

## Published runs

The pinned run IDs live in `configs/api/ratings_v1.json`:

- annual: `annual_aio_ratings_v1_b52b5aecd9`
- rolling/peaks: `rolling_rapm_peaks_v1_a8a612143c`
- current normal RAPM: `rapm_v0_01b5084f0a`
- matchup-defense factors: `matchup_defense_features_v1_09829b48c8`

These are research artifacts, not production truth. `/v1/meta` returns their
caveats so a frontend cannot silently hide that status.

## Versioned v2 uncertainty routes

`/v2` preserves the underlying v1 results inside a versioned envelope and adds
lineage: estimand, evidence status, uncertainty status, model/config hashes,
row-set hash, and an explicit forbidden interpretation.

| Route | Purpose |
|---|---|
| `GET /v2/meta` | Lineage for every pinned artifact, including scoped uncertainty pilots |
| `GET /v2/leaderboards/normal-rapm-uncertainty?scope=single_season_2025&metric=net` | 2025 normal-RAPM estimates with whole-game bootstrap intervals |
| `GET /v2/leaderboards/normal-rapm-uncertainty?scope=trailing_2022_2024&minimum_possessions=2000` | 2022–24 legacy-cache uncertainty pilot |
| `GET /v2/players/{player_id}` | Existing player record plus scope-matched uncertainty summaries |

These two pilots are deliberately separate from `/v1/leaderboards/current`:
their scope does not match the current 2024–26 run. Their intervals quantify
resampled-game variation, not latent ability, causal value, or precise rank.

## Deployment boundary

This standard-library server is for local development and contract validation.
A public deployment should put the same query layer behind a managed runtime,
add caching and compression, and restrict CORS. Do not expose the local server
directly to the internet.
