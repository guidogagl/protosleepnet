# How to interpret ProtoSleepNet

This is the page to read to understand *what the explanations mean*. Unlike
post-hoc attribution on a black box, ProtoSleepNet's explanation and its
prediction are the **same object**: the model can only reach a stage through the
prototypes it activates.

## 1. The prototype codebook (Claim A — faithful by construction)

Every 30-second epoch is encoded into a **128-dimensional embedding** and then
matched to the **nearest of $M=12$ prototypes** by squared-L2 distance:

$$
k^\star(x) \;=\; \arg\min_{k \in \{1,\dots,12\}} \; \bigl\lVert\, \mathrm{encode}(x) - p_k \,\bigr\rVert_2^2 .
$$

The stage is decided from these prototype activations. There is no surrogate
model standing between the explanation and the decision — **inspecting which
prototypes an epoch resembles *is* inspecting the decision**. This is what
"faithful by construction" means, and it is why the atlas (a 2-D projection of
the embedding space with the 12 prototypes placed inside it) is a legitimate map
of the model's reasoning, not a decoration.

:::{admonition} Why the atlas is trustworthy
:class: note
The 2-D atlas is produced with **PaCMAP**, fit jointly on epoch embeddings *and*
the codebook. We report a **nearest-prototype agreement** — the fraction of
epochs whose nearest prototype in 2-D matches their true nearest prototype in the
full 128-D space. It is high (≈0.83–0.85 in-domain), so distances you see on the
map reflect the distances that actually drive staging.
:::

## 2. Per-epoch evidence (Claim B — traceable decisions)

For any epoch you can ask *why this prototype?*. We answer with **Integrated
Gradients** on the matching objective $-\lVert \mathrm{encode}(x)-p_{k^\star}\rVert^2$
(zero baseline), which highlights the **time–frequency regions that pull the
epoch toward its prototype**. In the demo this appears as a heatmap over the
epoch's spectrogram, per channel (EEG / EOG / EMG), next to the epoch's place in
the night's hypnogram. Each decision is therefore *traceable*: a stage, a
prototype, the evidence for the match, and the temporal context.

## 3. Clinical microstructure (Claim C & D — monosemantic, AASM-coherent)

Each prototype carries a **card**: a 4-sentence natural-language rule, a spectral
envelope, EEG band-relevance, channel-relevance, its dominant stage, and two
quality scores — **label purity** and a **monosemanticity** score. High values
mean the prototype is a clean, single-concept pattern (Claim C).

The signatures follow textbook AASM physiology (Claim D):

- <span class="stage stage-n3">N3</span> prototypes concentrate on **delta** (0.5–4 Hz) EEG.
- <span class="stage stage-n2">N2</span> prototypes concentrate on **sigma / spindles** (11–16 Hz).
- <span class="stage stage-rem">REM</span> prototypes are **EOG-driven** (eye movements) with **theta**.
- <span class="stage stage-w">Wake</span> prototypes show **alpha/beta** EEG and **EMG** muscle tone.

## 4. The plausibility badge — an honest audit

The demo shows a per-epoch **plausibility badge**: does the Integrated-Gradients
relevance actually land on the frequency band **and** channel that the matched
prototype's stage predicts? This is a genuine sanity check of the *local*
explanations, not a cherry-pick — epochs where the evidence lands off-band are
shown as such, not hidden.

The audit below aggregates it across the featured recordings (anonymized
"Recording A–D", one set per model). "band ok" / "channel ok" are the fractions
of epochs whose IG concentrates where physiology expects; "N3 pos" / "REM pos"
are the mean normalized night-positions of N3 and REM prototypes (deep sleep
should precede REM — and it does).

```{include} ../explanation_audit.md
:start-line: 7
```

:::{admonition} Reading the numbers honestly
:class: warning
Per-epoch plausibility sits around **65–77%**, not 100%. That is expected and
intentional to report: Integrated Gradients marks *relevance magnitude*, not
sign or a single mechanism, and real nights contain ambiguous, transitional
epochs. What matters for the claims is that the evidence concentrates in the
physiologically correct band/channel **far more often than chance**, and that the
night-level structure (N3 before REM) is coherent for every recording.
:::

## 5. The prototype-gram (Claim G)

Rendering a whole night as its **sequence of activated prototypes** — colored by
prototype rather than by stage — gives the *prototype-gram*, an intermediate
representation that sits between the raw PSG and the coarse 5-class hypnogram. It
exposes sub-stage structure (which *kind* of N2, which *kind* of REM) that the
hypnogram flattens away.

---

Next: see it all together on the {doc}`demo` page, or regenerate the underlying
analyses from {doc}`reproduce`.
