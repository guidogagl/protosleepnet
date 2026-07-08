"""Assemble per-prototype cards for the demo from committed reconstruction JSON.

No GPU, no raw arrays: everything here is read from the committed
``data/reconstructions/M12/<model>/`` tree (rules, band-ablation, spectral
signatures, summary, local explanations, cross-dataset metrics). The UMAP
2-D coordinate of each prototype is added later by ``precompute.py`` once the
reducer is fit.

Committed dir naming differs from the HF id:
    seq -> protosleepnet-seq-3ch-mixer   (HF: protosleepnet-gagliardi)
    st  -> protosleepnet-st-3ch-mixer    (HF: protosleeptransformer-gagliardi)
"""
import json
from pathlib import Path

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


def _load(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _proto_cross_metrics(cross: dict, k: int) -> dict:
    """Pull the per-prototype slice out of cross_dataset_metrics.json."""
    if not cross:
        return {}
    out = {}
    fidelity = cross.get("fidelity", {})
    if "sleepedf" in fidelity.get("per_dataset", {}):
        out["fidelity_sleepedf"] = fidelity["per_dataset"]["sleepedf"]
    out["fidelity_indomain_mean"] = fidelity.get("indomain_mean")
    stab = cross.get("stability", {}).get("per_prototype", {})
    if str(k) in stab:
        out["stability_cv"] = stab[str(k)]
    plaus = cross.get("plausibility", {}).get("per_prototype", {})
    if str(k) in plaus:
        out["plausibility"] = plaus[str(k)]
    return out


def build_prototype_cards(committed_dir: Path, m: int = 12,
                          method: str = "data_driven") -> list:
    """Return a list of M card dicts assembled from committed JSON.

    Each card carries the human-readable rule, the band/channel relevance,
    the EEG spectral signature (+ temporal profile), and monosemanticity /
    cluster descriptors. ``xy`` is filled in by the caller after UMAP.
    """
    committed_dir = Path(committed_dir)
    mdir = committed_dir / method
    summary = _load(mdir / "summary.json") or {}
    per_proto_summary = {p["prototype"]: p for p in summary.get("per_prototype", [])}
    cross = _load(committed_dir / "cross_dataset_metrics.json") or {}

    cards = []
    for k in range(m):
        rule = _load(mdir / "rules" / f"proto_{k:03d}_rule.json") or {}
        abl = _load(mdir / "ablation" / f"proto_{k:03d}" / "metadata.json") or {}
        spec = _load(mdir / "spectral_analysis" / "statistics" / f"proto_{k:03d}.json") or {}
        local = _load(mdir / "local_explanations" / f"proto_{k:03d}" / "metadata.json") or {}
        summ = per_proto_summary.get(k, {})

        card = {
            "idx": k,
            "dominant_stage": (
                rule.get("dominant_class")
                or spec.get("dominant_stage")
                or summ.get("dominant_stage")
            ),
            "label_purity": spec.get("label_purity"),
            "monosemanticity": spec.get("monosemanticity_score"),
            "spectral_consistency": spec.get("spectral_consistency"),
            "peak_band_eeg": spec.get("peak_band_eeg"),
            "cluster_size": summ.get("cluster_size") or spec.get("cluster_size"),
            "mean_distance": summ.get("mean_distance"),
            "n_samples": summ.get("n_samples") or spec.get("n_samples"),
            # rule text (4 sentences + full)
            "rule": {
                "s1": rule.get("s1"), "s2": rule.get("s2"),
                "s3": rule.get("s3"), "s4": rule.get("s4"),
                "text": rule.get("rule_text"),
            },
            # band-ablation relevance (8 EEG bands)
            "band_names": abl.get("feature_names"),
            "band_relevance_pct": abl.get("band_relevance_pct"),
            "feature_direction": abl.get("feature_direction"),
            "channel_importance_pct": abl.get("channel_importance_pct"),
            "predicted_distribution": abl.get("predicted_distribution"),
            "predicted_purity": abl.get("predicted_purity"),
            # spectral signature (for the card's PSD / envelope plots)
            "spectral_band_names": spec.get("band_names"),
            "eeg_band_powers": spec.get("eeg_band_powers"),
            "spectral_envelope": spec.get("spectral_envelope"),      # (3, 129)
            "temporal_profile": spec.get("temporal_profile"),        # (3, 29)
            "eog_total_power": spec.get("eog_total_power"),
            "emg_tone": spec.get("emg_tone"),
            # label distribution over the assigned training epochs
            "label_distribution": _label_dist(spec, local),
            # cross-dataset generalisation
            "cross": _proto_cross_metrics(cross, k),
            "xy": None,  # filled after UMAP
        }
        cards.append(card)
    return cards


def _label_dist(spec: dict, local: dict) -> dict:
    """Best-effort label distribution over stages (from spectral stats if present)."""
    ld = spec.get("label_distribution")
    if isinstance(ld, dict):
        return ld
    return {}
