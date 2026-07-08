"""Compute cross-dataset reconstruction metrics for M=12.

Metrics (from the old article):
  1. Fidelity: mean L2 distance to codebook vector (lower = better alignment)
  2. Stability: cross-dataset variance of per-prototype mean distances
  3. Plausibility: spectral cosine similarity between cross-dataset and in-domain reconstructions

Usage:
    python compute_cross_dataset_metrics.py --backbone seq --base_dir /path/to/pretrained/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

def load_summary(path):
    with open(path) as f:
        return json.load(f)

def compute_spectral_similarity(epochs_a, epochs_b):
    """Cosine similarity between mean spectrograms of two reconstruction sets."""
    mean_a = epochs_a.mean(axis=0).flatten()
    mean_b = epochs_b.mean(axis=0).flatten()
    cos = np.dot(mean_a, mean_b) / (np.linalg.norm(mean_a) * np.linalg.norm(mean_b) + 1e-12)
    return float(cos)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["seq", "st"])
    parser.add_argument("--base_dir", required=True, type=Path)
    parser.add_argument("--m", type=int, default=12)
    args = parser.parse_args()

    if args.backbone == "seq":
        model = "protosleepnet-seq-3ch-mixer"
    else:
        model = "protosleepnet-st-3ch-mixer"

    base = args.base_dir / model
    indomain_dir = base / "reconstruction_m12" / "data_driven"
    crossds_base = base / "reconstruction_per_dataset_m12"
    M = args.m

    if not indomain_dir.exists():
        print(f"Error: {indomain_dir} not found")
        return
    if not crossds_base.exists():
        print(f"Error: {crossds_base} not found")
        return

    datasets = sorted([d.name for d in crossds_base.iterdir() if d.is_dir()])
    print(f"Backbone: {args.backbone} ({model})")
    print(f"In-domain: {indomain_dir}")
    print(f"Cross-dataset: {len(datasets)} datasets: {datasets}")
    print()

    # ── Per-dataset fidelity ──
    print("=== Fidelity (mean L2 distance to codebook) ===")
    indomain_summary = load_summary(indomain_dir / "summary.json")
    indomain_dists = [p["mean_distance"] for p in indomain_summary["per_prototype"]]
    indomain_mean = np.mean(indomain_dists)
    print(f"  In-domain: {indomain_mean:.3f} ± {np.std(indomain_dists):.3f}")

    cross_means = {}
    cross_per_proto = {k: [] for k in range(M)}
    for ds in datasets:
        summary_path = crossds_base / ds / "summary.json"
        if not summary_path.exists():
            print(f"  {ds}: MISSING summary.json")
            continue
        summary = load_summary(summary_path)
        dists = [p["mean_distance"] for p in summary["per_prototype"]]
        cross_means[ds] = np.mean(dists)
        for k, p in enumerate(summary["per_prototype"]):
            cross_per_proto[k].append(p["mean_distance"])
        print(f"  {ds}: {np.mean(dists):.3f} ± {np.std(dists):.3f}")

    all_cross = list(cross_means.values())
    print(f"\n  Cross-dataset mean: {np.mean(all_cross):.3f} ± {np.std(all_cross):.3f}")
    print(f"  In-domain mean:    {indomain_mean:.3f}")
    print(f"  Ratio (cross/in):  {np.mean(all_cross)/indomain_mean:.2f}x")

    # ── Stability (cross-dataset variance per prototype) ──
    print("\n=== Stability (per-prototype distance variance across datasets) ===")
    stabilities = []
    for k in range(M):
        vals = cross_per_proto[k]
        if len(vals) > 1:
            cv = np.std(vals) / (np.mean(vals) + 1e-12)
            stabilities.append(cv)
            stage = indomain_summary["per_prototype"][k].get("dominant_stage", "?")
            print(f"  P{k} ({stage}): mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, CV={cv:.3f}")
    print(f"\n  Mean CV across prototypes: {np.mean(stabilities):.3f} ± {np.std(stabilities):.3f}")

    # ── Plausibility (spectral cosine similarity) ──
    print("\n=== Plausibility (spectral cosine similarity vs in-domain) ===")
    plausibilities = {k: [] for k in range(M)}
    for ds in datasets:
        for k in range(M):
            indomain_epochs_path = indomain_dir / f"proto_{k:03d}" / "epochs.npy"
            cross_epochs_path = crossds_base / ds / f"proto_{k:03d}" / "epochs.npy"
            if not indomain_epochs_path.exists() or not cross_epochs_path.exists():
                continue
            indomain_epochs = np.load(indomain_epochs_path)
            cross_epochs = np.load(cross_epochs_path)
            sim = compute_spectral_similarity(indomain_epochs, cross_epochs)
            plausibilities[k].append(sim)

    for k in range(M):
        vals = plausibilities[k]
        if vals:
            stage = indomain_summary["per_prototype"][k].get("dominant_stage", "?")
            print(f"  P{k} ({stage}): cosine_sim={np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")
    all_plaus = [v for vals in plausibilities.values() for v in vals]
    print(f"\n  Overall plausibility: {np.mean(all_plaus):.4f} ± {np.std(all_plaus):.4f}")

    # ── Save results ──
    results = {
        "backbone": args.backbone,
        "model": model,
        "n_datasets": len(datasets),
        "datasets": datasets,
        "fidelity": {
            "indomain_mean": float(indomain_mean),
            "indomain_std": float(np.std(indomain_dists)),
            "cross_mean": float(np.mean(all_cross)),
            "cross_std": float(np.std(all_cross)),
            "ratio": float(np.mean(all_cross) / indomain_mean),
            "per_dataset": {ds: float(v) for ds, v in cross_means.items()},
        },
        "stability": {
            "mean_cv": float(np.mean(stabilities)),
            "std_cv": float(np.std(stabilities)),
            "per_prototype": {str(k): float(stabilities[k]) for k in range(len(stabilities))},
        },
        "plausibility": {
            "overall_mean": float(np.mean(all_plaus)),
            "overall_std": float(np.std(all_plaus)),
            "per_prototype": {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                              for k, v in plausibilities.items() if v},
        },
    }

    out_path = crossds_base / "cross_dataset_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
