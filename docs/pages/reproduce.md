# Reproducing the paper

Every paper asset maps to an entry-point module run as `python -m
protosleepnet.<module>`. Pass `--help` for options; full experiment settings
(M value, seeds, datasets) are documented per script and mirrored in the
{doc}`/api/protosleepnet/index`.

:::{note}
Raw recordings are **not** redistributed here — obtain them from their sources
and preprocess with [physioex](https://github.com/guidogagl/physioex) (this repo
ships none of its own preprocessing). Large artifacts (predictions, embeddings,
reconstruction arrays, checkpoints) live in a backup root wired in via the
git-ignored `json/` and `reconstructions_bulk/` symlinks; only small
figure-source JSON is committed under `data/`.
:::

## Figure / table → experiment → command

| Paper asset | Experiment | Command |
|---|---|---|
| Fig 2a; Supp §1 | In-domain + OOD staging, per-class, confusion, stats | `python -m protosleepnet.train` · `python -m protosleepnet.test_pretrained` |
| Fig 2b; Supp §3,§8 | M-sweep, residual/VQ robustness, VQ methods, randomization | `python -m protosleepnet.posthoc_prototypes.learn_prototypes_vq` · `python -m protosleepnet.plot.residual` · `python -m protosleepnet.plot.vq` |
| Supp §2 | Ablation (4 component variants) + occlusion robustness | `python -m protosleepnet.ablation.train_ablation --variant {baseline,dropout,mixer,protosleepnet}` (SleepTransformer) / `...train_ablation_seqsleepnet ...` (SeqSleepNet) · `python -m protosleepnet.plot.occlusion` |
| Fig 4; Supp §5,§6 | Prototype reconstruction (data/model/hybrid) + cross-dataset | `python -m protosleepnet.proto_reconstruction.data_driven` (`.model_driven`, `.hybrid`) · `python -m protosleepnet.figure_reconstruction.compute_cross_dataset_metrics` |
| Fig 3; Tab 1,2; Supp §4,§6 | Codebook summaries, band-ablation rules, coherence, local IG | `python -m protosleepnet.figure_reconstruction.{rule_learning,spectral_signature,relevance_signature,compute_local_explanations,combinatorial_ablation}` |
| Supp §9 | Seed stability | `python -m protosleepnet.seed_stability_pipeline` |
| Fig 5; Supp §7 | Clinical probing (AD/PD), frozen-model LOSO | `python -m protosleepnet.clinical_probing.staging.extract_embeddings` · `python -m protosleepnet.probing.{parkinsons,alzheimers}.diagnosis_probe` · `python -m protosleepnet.clinical_probing.{parkinsons,alzheimers}.analyze_features` |

Manuscript figure/table *assembly* (LaTeX-coupled emitters) is kept out of this
repo, with the paper sources.

## Repository layout

```text
src/protosleepnet/            # installable package (run as python -m protosleepnet.<mod>)
  train.py  test_*.py  extract_*.py  seed_*.py   # staging training/eval + seed-stability
  build_protosleepnet.py                          # shared model factory
  ablation/  baselines/                           # ablation & residual/mixer/dropout variants
  posthoc_prototypes/  proto_reconstruction/      # VQ codebook learning + reconstruction
  figure_reconstruction/                          # prototype cards / rules / spectral & relevance signatures
  probing/  clinical_probing/                      # clinical probing (AD/PD) compute + analysis
  ood_disease/                                     # OOD disease evaluation / transport
  plot/                                            # portable figure scripts (from predictions)
  demo/                                            # the interactive-demo precompute pipeline
examples/slurm/    # cluster launchers; copy env.sh.example -> env.sh
notebooks/         # global_explain.ipynb, local_explain.ipynb
data/              # small committed figure-source JSON only
tests/             # smoke tests (import graph + optional pretrained forward pass)
```

## Smoke test

```bash
bash reproduce.sh      # load weights → forward pass → regenerate one figure

pip install -e . pytest
pytest                 # import-graph smoke test (no network)
pytest --runslow       # also pull weights from HuggingFace and run a forward pass
```

## The interactive demo

The `protosleepnet.demo` subpackage builds the static bundle behind the
{doc}`demo`: subject selection, subsetting/anonymization, the PaCMAP atlas,
per-epoch IG, and the clinical-plausibility audit. It ships **only anonymized
derived artifacts** — never the source signals. See its module reference in the
{doc}`/api/protosleepnet/index`.
