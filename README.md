# ProtoSleepNet

Code for **"Prototype-based interpretable sleep staging with physiologically
meaningful sub-stage pattern discovery"** (ProtoSleepNet), in publication at
*npj Digital Medicine*.

ProtoSleepNet wraps a sleep-staging backbone (SeqSleepNet or SleepTransformer)
with a **Prototype Sub-Stage (PSS)** module: per-channel encoding, modality
embeddings, inverse-accuracy channel dropout, a Transformer channel mixer and a
dual-residual path, yielding contextualized epoch embeddings that are quantized
into a small, interpretable **prototype codebook**. The prototypes are shown to
carry physiologically meaningful, AASM-coherent sub-stage patterns and to
support exploratory clinical probing (Alzheimer's, Parkinson's).

![ProtoSleepNet](docs/figures/schema.png)

> The model architecture, data loaders, preprocessing, dataset splits, seeds and
> evaluation live in the [`physioex`](https://github.com/guidogagl/physioex)
> library (**v2.0.0**). This repository holds the **experiment pipeline** that
> produced every figure and table in the paper.

## Installation

```bash
conda env create -f environment.yml && conda activate protosleepnet
# or: pip install -r requirements.txt   (Python 3.12, physioex==2.0.0)
```

Scripts import `physioex` (installed) and their local siblings, so run each from
inside its directory (e.g. `cd src/protosleepnet && python train.py ...`).

## Pretrained weights

Weights are on the HuggingFace Hub under `4rooms/physioex` and load via physioex:

```python
from physioex.models import load_from_pretrained
model = load_from_pretrained("protosleepnet-st-3ch-mixer", verbose=True)  # PST
# also: "protosleepnet-seq-3ch-mixer" (PSN)
# input  (batch, L, 3, 29, 129) STFT log-power (EEG/EOG/EMG); output (batch, L, 5) AASM logits
```

`bash reproduce.sh` runs a smoke path (load weights -> forward pass -> one figure).

## Data

11 PSG datasets / 20 dataset-sources (12,317 subjects). Raw data is **not**
redistributed here — obtain it from the sources below and preprocess with
physioex; see `src/protosleepnet/*/preprocessing` usage and `examples/slurm`.

| Datasets | Access |
|---|---|
| SHHS, MrOS, MESA, WSC, HomePAP | NSRR — controlled/registered (https://sleepdata.org) |
| HMC, Sleep-EDF | PhysioNet — open |
| MASS, DCSM | open on request |
| ASD (Alzheimer), KPD (Parkinson) | KU Leuven — restricted (contact authors; ethics S61792/S70708) |

## Repository layout

```
src/protosleepnet/
  train.py, test_*.py, extract_*.py, seed_*.py   # staging + seed-stability
  ablation/  baselines/                          # ablation & residual/mixer variants
  posthoc_prototypes/  proto-reconstruction/     # VQ codebook learning + reconstruction
  fix-alzheimer/  probing/                        # OOD (AD/PD) + clinical probing (compute)
  plot/                                           # portable figure scripts (from predictions)
  figure_reconstruction/                          # prototype cards / rules / spectral signatures
  clinical_probing/                               # AD/PD feature analysis + staging probe
examples/slurm/    # cluster launchers (Leonardo/Sofia); copy env.sh.example -> env.sh
notebooks/         # global_explain.ipynb, local_explain.ipynb
data/              # small committed figure-source JSON (see below)
```

Large artifacts (predictions, embeddings, reconstruction arrays, checkpoints)
are **not** in git — they live in a backup root and are wired in via the
git-ignored `json/` and `reconstructions_bulk/` symlinks. Only ~small
figure-source JSON is committed under `data/`.

## Reproducing the paper (Figure/Table -> experiment -> code)

| Paper asset | Experiment | Code |
|---|---|---|
| Fig 2a; Supp §1 | In-domain + OOD staging, per-class, confusion, stats | `train.py`, `test_pretrained.py`, `clinical_probing/staging/extract_embeddings.py` |
| Fig 2b; Supp §3,§8 | M-sweep, residual/VQ robustness, VQ methods, randomization | `posthoc_prototypes/`, `baselines/`, `plot/residual*.py`, `plot/vq.py` |
| Supp §2 | Ablation + occlusion robustness | `ablation/`, `test_occlusion*.py`, `plot/occlusion*.py` |
| Fig 4; Supp §5,§6 | Prototype reconstruction (data/model/hybrid) + cross-dataset | `proto-reconstruction/`, `figure_reconstruction/compute_*`, `figure_reconstruction/prototype_card*.py` |
| Fig 3; Tab 1,2; Supp §4,§6 | Codebook summaries, band-ablation rules, coherence, local IG | `figure_reconstruction/{combinatorial_ablation,rule_learning,spectral_signature,relevance_signature,compute_local_explanations}.py` |
| Supp §9 | Seed stability | `seed_*.py` |
| Fig 5; Supp §7 | Clinical probing (AD/PD) | `clinical_probing/{parkinsons,alzheimers}/*`, `fix-alzheimer/` |

Manuscript figure/table *assembly* (LaTeX-coupled emitters) is kept out of this
repo, with the paper sources.

## Citation

See `CITATION.cff` (paper + software DOI). License: MIT (`LICENSE`).
