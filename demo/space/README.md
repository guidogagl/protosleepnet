---
title: ProtoSleepNet Explainability Atlas
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ProtoSleepNet — Explainability Atlas

Interactive, prototype-based explainability demo for ProtoSleepNet: a PaCMAP
atlas of the model's 128-d epoch-embedding space with its 12 learned prototypes,
per-epoch predictions, input→prototype matching, per-epoch Integrated-Gradients
attributions, and a guided tour of the paper's explainability claims.

The demo shows only **de-identified, derived visualizations** (log-power
spectrograms, embeddings, attributions). No raw source signals are served. The
data lives in a private companion dataset repo and is streamed through an
authenticated in-Space proxy (`/data/*`), so the access token is never exposed
to the browser and HTTP Range requests still drive lazy loading.

Configuration (Space settings):
- Variable `DATASET_REPO` — the private dataset repo id.
- Secret `HF_TOKEN` — a read token scoped to that repo.
