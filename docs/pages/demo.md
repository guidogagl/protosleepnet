# Live demo

The interactive explainability atlas is embedded below and also lives at
**<https://protosleepnet-demo.pages.dev>**. It is a static, GPU-free app: a
public front-end whose derived visualizations are served from a private data
store — **no source recordings are shipped**, only anonymized, non-invertible
derived artifacts (log-power spectrograms, embeddings, prototype assignments and
Integrated-Gradients maps).

```{raw} html
<iframe class="pst-demo-frame"
        src="https://protosleepnet-demo.pages.dev"
        loading="lazy"
        allowfullscreen
        title="ProtoSleepNet interactive explainability demo"></iframe>
```

:::{tip}
If the frame does not load in your browser's privacy mode, open it in a new tab:
**<https://protosleepnet-demo.pages.dev>**.
:::

## A claim-by-claim walkthrough

The demo's guided **Story mode** walks the same path. Each panel names the claim
it demonstrates (see {doc}`interpret` for the full explanation).

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Claim A — Faithful by construction
The **atlas** is a PaCMAP projection of the 128-d embedding space with the 12
prototypes placed inside it. Because the model can only stage an epoch through
its nearest prototype, this map *is* the model's reasoning surface — start here.
:::

:::{grid-item-card} Claim B — Every decision is traceable
**Click any epoch.** You get its true stage, its position in the night's
hypnogram, the model's (non-quantized) prediction, its matched prototype and L2
distance, and a **per-epoch Integrated-Gradients** heatmap over the spectrogram —
the exact time–frequency evidence for the match — with an honest **plausibility
badge**.
:::

:::{grid-item-card} Claim C — Monosemantic prototypes
**Open a prototype card.** Each of the 12 prototypes shows its label purity and
monosemanticity score: clean, single-concept, stage-specialized patterns.
:::

:::{grid-item-card} Claim D — Clinically meaningful microstructure
Each card's **rule, band-relevance, channel-relevance and spectral envelope**
follow AASM physiology — spindles for N2, delta for N3, eye movements for REM,
alpha/EMG for wake.
:::

:::{grid-item-card} Claim G — Prototype-gram
The ribbon under each recording renders the whole night as its **sequence of
activated prototypes** — a representation between the raw PSG and the coarse
hypnogram.
:::

:::{grid-item-card} Claims E & F — in the paper
**No accuracy cost** vs. the black-box backbone (E) and **cross-dataset transfer
and disease biomarkers** (F) require multiple cohorts and are covered in the
[paper](https://www.nature.com/npjdigitalmed/) — the demo does not fabricate
single-cohort visuals for them.
:::
::::

## What is (and isn't) shipped

The featured recordings are the *cleanest* subjects from each model's own
**training split**, shown as anonymized "Recording A–D" with no dataset name and
no demographics. Only derived, non-invertible artifacts are published; the atlas
coordinates come from a stratified sample of the training embeddings, and the
nearest-prototype agreement is reported honestly on-screen.
