# Cloudflare Pages deploy (public demo · private data)

Public demo at **https://protosleepnet-demo.pages.dev**. The static app is served
by Cloudflare Pages; `/data/*` is proxied by a Pages Function to a **private** HF
dataset using the `HF_TOKEN` secret (never exposed to the browser). Same-origin,
so no CORS; HTTP Range is preserved for lazy per-epoch loading.

## Redeploy
```
cd demo && VITE_DATA_URL=/data npx vite build --outDir space-cf/dist --emptyOutDir  # (move public/data aside first)
cd space-cf
export CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=6b0d06ff7f43b61448a59dbd45a8f8eb
npx wrangler pages deploy --branch main            # production
```
Config: `wrangler.toml` (`DATASET_REPO` var). Secret: `wrangler pages secret put HF_TOKEN --project-name protosleepnet-demo` (production env). Data repo: `4rooms/protosleepnet-demo-data` (private).
