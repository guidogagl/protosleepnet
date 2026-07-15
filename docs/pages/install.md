# Installation

ProtoSleepNet requires **Python ≥ 3.10**. The model, data loaders,
preprocessing, dataset splits, seeds and metrics come from
[physioex](https://github.com/guidogagl/physioex) `== 2.0.0`, which is installed
as a dependency.

```bash
conda env create -f environment.yml && conda activate protosleepnet   # optional
pip install -e .                     # installs the `protosleepnet` package + physioex==2.0.0
```

After `pip install -e .` every script runs as a module from the repo root —
**one** convention throughout:

```bash
python -m protosleepnet.train --help
python -m protosleepnet.figure_reconstruction.spectral_signature --help
```

## Pretrained weights

Weights are on the HuggingFace Hub and load through physioex. Names follow the
physioex convention `<model>-<primary-author>`. The paper's models live in
[`4rooms/sleep-prototypes`](https://huggingface.co/4rooms/sleep-prototypes):

```python
from physioex.models import load_from_pretrained

# PST — SleepTransformer backbone (trained on SHHS, L=21)
model = load_from_pretrained("protosleeptransformer-gagliardi", repo_id="4rooms/sleep-prototypes")

# PSN — SeqSleepNet backbone (trained on MASS, L=20)
# load_from_pretrained("protosleepnet-gagliardi", repo_id="4rooms/sleep-prototypes")

# input  (batch, L, 3, 29, 129)  STFT log-power (EEG/EOG/EMG)
# output (batch, L, 5)           AASM logits (W, N1, N2, N3, REM)
```

The 3-channel baselines and ablation variants are published alongside them:
`{seqsleepnet,sleeptransformer}-gagliardi[-dropout|-mixer]`. The Phan originals
`{seqsleepnet,sleeptransformer}-phan` are in `4rooms/physioex`.

`bash reproduce.sh` runs a smoke path (load weights → forward pass → regenerate
one figure).

## Documentation dependencies

Building this documentation needs only a light, **import-free** set of tools
(no torch / physioex) — the module reference is parsed statically by
`sphinx-autoapi`, so the docs environment installs the doc tools alone (not the
package):

```bash
pip install -r docs/requirements.txt
sphinx-build -b html -W --keep-going docs docs/_build/html
```

## Configuration (environment variables)

Scripts read paths from the environment so no absolute paths are baked in:

| Variable | Meaning | Default |
|---|---|---|
| `PROTOSLEEPNET_DATA` / `EXPERIMENT_DIR` | root for predictions, embeddings, reconstruction outputs | `data` |
| `PROTOSLEEPNET_MODELS` | checkpoints / training stats | `<DATA>/models` |
| `PROTO_RECON_SRC` | reconstruction source arrays (optional; else symlink `reconstructions_bulk/`) | — |
| `PHYSIOEX_ROOT` | physioex checkout (only if running physioex from source, not pip) | — |

Cluster launchers additionally source `examples/slurm/env.sh` (copy from
`examples/slurm/env.sh.example`).
