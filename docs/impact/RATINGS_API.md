# Ratings API v1

This local, read-only JSON API is the data contract for future player pages. It
pins explicit model runs; it never selects an artifact because it has the newest
timestamp.

Start it from the project root:

```bash
python3 -m nba_impact.cli serve-ratings
```

Default address: `http://127.0.0.1:8765`.

## Routes

| Route | Purpose |
|---|---|
| `GET /v1/health` | Process and contract health |
| `GET /v1/meta` | Run IDs, estimands, seasons, metrics, and caveats |
| `GET /v1/leaderboards/annual?season=2024&metric=aio_net` | Annual AIO/SPM/normal-RAPM leaderboard |
| `GET /v1/leaderboards/peaks?window=5&component=net` | Three- or five-year all-time peak table |
| `GET /v1/players/search?q=lebron` | Case-insensitive player lookup |
| `GET /v1/players/2544` | Annual decomposition, rolling history, and peaks |

Leaderboard routes accept `limit` and `offset`; annual leaderboards also accept
`minimum_possessions`. The server caps `limit` at 100. Invalid metrics,
components, and parameters return HTTP 400. Unknown players and routes return
HTTP 404.

## Published runs

The pinned run IDs live in `configs/api/ratings_v1.json`:

- annual: `annual_aio_ratings_v1_23c4895f8f`
- rolling/peaks: `rolling_rapm_peaks_v1_584adf4f3d`

Both are research artifacts, not production truth. `/v1/meta` returns their
caveats so a frontend cannot silently hide that status.

## Deployment boundary

This standard-library server is for local development and contract validation.
A public deployment should put the same query layer behind a managed runtime,
add caching and compression, and restrict CORS. Do not expose the local server
directly to the internet.
