"""Public demo server for a PRIVATE data repo.

Serves the built static app at ``/`` and proxies ``/data/*`` to a private
Hugging Face dataset repo using a Space secret token. The token never reaches
the browser; HTTP Range requests are forwarded so the frontend keeps lazy-
loading per-epoch spectrograms / IG.

Env (set as Space variables/secrets):
  DATASET_REPO  e.g. "4rooms/protosleepnet-demo-data"   (variable)
  HF_TOKEN      read token scoped to that private repo   (secret)
"""
import os
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

DATASET_REPO = os.environ.get("DATASET_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# DATA_BASE lets a local/CI run point the proxy at any origin (e.g. a local
# static server) for testing; in the Space it defaults to the HF resolve URL.
BASE = (os.environ.get("DATA_BASE")
        or f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main").rstrip("/")
# headers worth passing back so Range + caching + content-type work in the browser
PASS = ("content-range", "content-length", "accept-ranges", "content-type",
        "etag", "last-modified", "cache-control")

app = FastAPI()


@app.get("/healthz")
def healthz():
    return {"ok": True, "dataset_repo_set": bool(DATASET_REPO), "token_set": bool(HF_TOKEN)}


@app.get("/data/{path:path}")
async def data(path: str, request: Request):
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    rng = request.headers.get("range")
    if rng:
        headers["Range"] = rng
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        r = await client.get(f"{BASE}/{path}", headers=headers)
    out = {k: v for k, v in r.headers.items() if k.lower() in PASS}
    return Response(content=r.content, status_code=r.status_code, headers=out)


# static app last, so /data and /healthz take precedence
app.mount("/", StaticFiles(directory="dist", html=True), name="static")
