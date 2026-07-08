---
license: mit
tags: [sleep-staging, eeg, polysomnography, ablation, physioex]
library_name: physioex
pipeline_tag: other
---

# {MODEL_TITLE}

{VARIANT_DESC}

This is a **component-ablation variant** released with the paper *"Prototype-based
interpretable sleep staging with physiologically meaningful sub-stage pattern
discovery"* (npj Digital Medicine). It is one of the four progressive
configurations (baseline → + channel dropout → + channel mixer → full
ProtoSleepNet) used in the ablation study (Supplementary §2), and ships with the
per-scenario occlusion metrics/predictions used there.

- **Backbone:** {BACKBONE}  ·  **Training cohort:** {DATASET}  ·  **Channels:** EEG, EOG, EMG (3)  ·  **Sequence length L:** {SEQLEN}
- **Code / reproduction:** https://github.com/guidogagl/protosleepnet
- **Library:** [`physioex`](https://github.com/guidogagl/physioex) v2.0.0

## Usage

```python
from physioex.models import load_from_pretrained
model = load_from_pretrained("{MODEL_ID}", repo_id="{REPO}", verbose=True)
# input  (batch, L, 3, 29, 129) STFT log-power (EEG/EOG/EMG)
# output (batch, L, 5) AASM stage logits: W=0 N1=1 N2=2 N3=3 REM=4
```

The full models are `protosleeptransformer-gagliardi` (PST) and
`protosleepnet-gagliardi` (PSN) in the same repo; the Phan originals are
`{{seqsleepnet,sleeptransformer}}-phan` in `4rooms/physioex`.

## Intended use & limitations

Research use only. **Not a medical device and not for clinical diagnosis.**
Trained on {DATASET}; performance on other populations/montages may differ.

## Citation

See `CITATION.cff` in the code repository (paper + software DOI).
