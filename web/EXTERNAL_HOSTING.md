# External hosting

## Recommended path: Cloudflare Workers

The web client is not a plain static export. The current `vinext` build emits:

- `dist/server/index.js`: the Cloudflare Worker;
- `dist/client/`: the browser assets and derived JSON snapshot;
- `dist/server/wrangler.json`: a generated local manifest.

Cloudflare Workers is therefore the smallest external-hosting change. It runs
the same Worker shape used by the existing Sites build and serves the same
derived snapshot. The checked-in `wrangler.jsonc` makes the entry point and
asset binding explicit. It contains no account ID, token, NBA raw data, or
secret.

The GitHub Actions workflow deploys only after a successful build. To enable
it, add these repository secrets in GitHub:

- `CLOUDFLARE_API_TOKEN`: a token limited to Workers deployment for this
  account;
- `CLOUDFLARE_ACCOUNT_ID`: the target Cloudflare account ID.

The workflow is manual or runs when `main` changes under `web/`. Without both
secrets it cannot deploy. This is intentional: no credential is guessed or
committed.

## Why not GitHub Pages?

GitHub Pages is a good free host for a genuine static export. This app currently
does not produce one. Its route is rendered through a Worker and the current
build has no standalone `index.html` entry. Converting it to static export
would be a separate product change and could break the current route/data
loading behavior. Do not point Pages at `dist/client` until a static-export
build and deep-link fallback test exist.

## Why not Vercel?

Vercel can host the source, but this repo is already configured for the
Cloudflare Worker runtime. Vercel would add a second deployment adapter and
provider configuration without solving a current blocker. Use it only if the
project later needs Vercel-specific server functions or the Cloudflare account
is unavailable.

## ChatGPT Sites versus an external host

ChatGPT Sites packages this same Cloudflare-compatible build, associates it
with the project ID in `.openai/hosting.json`, and manages the hosted version
and access through the Sites service. It is convenient for a private preview
and does not require the user to manage Cloudflare credentials.

An external Cloudflare deployment is owned by the user's Cloudflare account.
The user controls the domain, deployment history, account limits, secrets,
logs, and future automation. It is independent of ChatGPT Sites, but it needs
the two GitHub secrets above (or a local Cloudflare login) and a Cloudflare
account.

No raw NBA event or possession files belong in the web deployment. The app
ships only the derived JSON snapshot already under `web/public/data/`.
