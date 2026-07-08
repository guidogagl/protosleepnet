---
license: mit
tags: [sleep-staging, eeg, polysomnography, interpretability, prototypes, physioex]
library_name: physioex
pipeline_tag: other
---

# {MODEL_TITLE}

Pretrained **ProtoSleepNet** — an interpretable, prototype-based sleep-staging
model from the paper *"Prototype-based interpretable sleep staging with
physiologically meaningful sub-stage pattern discovery"* (npj Digital Medicine).

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

## Intended use & limitations

Research use only. **Not a medical device and not for clinical diagnosis.**
Trained on specific PSG cohorts ({DATASET}); performance on other populations,
montages or hardware may differ. The prototype codebook supports interpretable,
AASM-coherent sub-stage analysis as described in the paper.

## Citation

See `CITATION.cff` in the code repository (paper + software DOI).
