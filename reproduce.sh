#!/usr/bin/env bash
# Smoke reproduction path: load the pretrained model from HuggingFace and run a
# forward pass, then (optionally) regenerate one figure from the committed
# figure-source data. Full experiment reproduction is documented in the README
# (Figure/Table -> experiment -> script map).
set -euo pipefail

echo "[1/2] Loading pretrained ProtoSleepNet (PST) from HuggingFace 4rooms/physioex ..."
python - <<'PY'
from physioex.models import load_from_pretrained
m = load_from_pretrained("protosleepnet-st-3ch-mixer", verbose=True)
import torch
x = torch.randn(2, 21, 3, 29, 129)   # (batch, L, channels, T, F)
with torch.no_grad():
    y = m(x)
print("forward OK, logits shape:", tuple(y.shape))   # expect (2, 21, 5)
PY

echo "[2/2] Regenerating a figure from committed figure-source data (if present) ..."
if [ -d data/reconstructions/M12 ]; then
  python src/protosleepnet/figure_reconstruction/spectral_signature.py data/reconstructions/M12 --figures-only || true
else
  echo "  (skip) data/reconstructions/M12 not populated — see README 'Data'."
fi
echo "Done."
