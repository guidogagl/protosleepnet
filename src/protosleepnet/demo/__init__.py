"""Static-bundle precompute for the ProtoSleepNet explainability demo.

The demo is a GPU-free static web app (SemanticLens-style) served from a
Hugging Face Static Space. Everything it needs is precomputed here on the
A30 and emitted as a compact bundle: a per-model UMAP of the SleepEDF
epoch-embedding space with the M=12 prototypes co-embedded, per-epoch
predictions/labels/prototype-assignments, per-prototype cards assembled from
the committed reconstruction JSON, and per-subject raw signals for the
time-series + spectrogram panels.

Entry point: ``python -m protosleepnet.demo.precompute`` (see its ``--help``).
"""
