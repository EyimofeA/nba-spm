# NBA Impact Lab web

The first product slice is one player trajectory page. It reads the pinned
ratings API and shows:

- annual AIO offense, defense, and net;
- the SPM center plus RAPM update decomposition;
- three- and five-year normal RAPM trajectories;
- the player's offense, defense, and net peak ranks;
- research status and data caveats.

Run the API from the repository root:

```bash
python3 -m nba_impact.cli serve-ratings
```

Then run the site:

```bash
cd web
npm install
npm run dev
```

The page defaults to `http://localhost:3000` and the API defaults to
`http://127.0.0.1:8765`. Set `NEXT_PUBLIC_RATINGS_API_URL` to change the API
origin.

`npm test` compiles the Cloudflare-compatible build and validates the rendered
product shell. The site is not deployed yet; public deployment requires hosting
the Python query contract or adapting it to the chosen managed data service.
