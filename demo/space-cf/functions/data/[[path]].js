// Cloudflare Pages Function: authenticated proxy for a PRIVATE HF dataset.
// Serves same-origin at /data/* (no CORS). The read token lives in the Pages
// secret HF_TOKEN and never reaches the browser. HTTP Range is forwarded so the
// frontend keeps lazy-loading per-epoch spectrograms / IG.
//
// Env (Pages project):
//   DATASET_REPO  e.g. "4rooms/protosleepnet-demo-data"  (variable)
//   HF_TOKEN      read token scoped to that repo          (secret)
const PASS = ["content-range", "content-length", "accept-ranges", "content-type",
              "etag", "last-modified", "cache-control"];

export async function onRequestGet({ request, env }) {
  const url = new URL(request.url);
  const path = url.pathname.replace(/^\/data\//, "");        // keep %20 encoding as-is
  const base = `https://huggingface.co/datasets/${env.DATASET_REPO}/resolve/main`;
  const range = request.headers.get("Range");

  const h1 = {};
  if (env.HF_TOKEN) h1["Authorization"] = `Bearer ${env.HF_TOKEN}`;
  if (range) h1["Range"] = range;

  // Let fetch follow HF's 302 to the signed LFS/Xet CDN. fetch preserves Range
  // across the redirect and drops Authorization cross-origin (the signed url
  // needs none), so this serves both small files (200) and LFS ranges (206).
  // ("redirect: manual" would yield an unreadable opaqueredirect in Workers.)
  const resp = await fetch(`${base}/${path}`, { headers: h1, redirect: "follow" });

  const out = new Headers();
  for (const k of PASS) {
    const v = resp.headers.get(k);
    if (v) out.set(k, v);
  }
  return new Response(resp.body, { status: resp.status, headers: out });
}
