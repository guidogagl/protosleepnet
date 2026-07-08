"""Local instance-to-prototype explanations via Integrated Gradients.

For each prototype, selects the single nearest epoch from data-driven
reconstruction and computes IG attributions explaining why that epoch
is assigned to the prototype (L2 objective).

Usage:
    python compute_local_explanations.py --backbone seq --m 12
    python compute_local_explanations.py --backbone st --m 12
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("PHYSIOEX_ROOT", ""))
sys.path.insert(0, os.environ.get("PROTO_RECON_SRC", ""))

from physioex.explain.posthoc.gradients import IntegratedGradients

torch.backends.cudnn.enabled = False

RECON = Path(os.environ.get("PROTOSLEEPNET_RECON_M12", "data/reconstructions/M12"))
MODELS_DIR = Path(os.environ.get("PROTOSLEEPNET_MODELS", "models"))
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

CONFIGS = {
    "seq": {"model_name": "protosleepnet-seq-3ch-mixer"},
    "st":  {"model_name": "protosleepnet-st-3ch-mixer"},
}


def make_prototype_distance_fn(model, prototype_vec, device):
    """f(x) = -||epoch_encode(x) - p_k||^2  (higher = closer to prototype)."""
    p_k = torch.from_numpy(prototype_vec).float().to(device)

    def f(x_batch):
        h = model.epoch_encode(x_batch.unsqueeze(1), quantize=False)
        h = h.squeeze(1)
        return -((h - p_k.unsqueeze(0)) ** 2).sum(dim=1)

    return f


def main():
    parser = argparse.ArgumentParser(description="Local explanations via IG (L2 objective)")
    parser.add_argument("--backbone", required=True, choices=["seq", "st"])
    parser.add_argument("--m", type=int, default=12)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=128, help="IG integration steps")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--codebook_path", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = CONFIGS[args.backbone]["model_name"]

    # Load model
    from utils import load_frozen_model, load_codebook, get_paths
    model = load_frozen_model(args.backbone, device, checkpoint_path=args.checkpoint_path)
    codebook = load_codebook(args.backbone, m=args.m, codebook_path=args.codebook_path)
    M = codebook.shape[0]
    print(f"Model loaded, codebook M={M}")

    # Load training mean for baseline context
    paths = get_paths(args.backbone, m=args.m)
    training_mean_path = paths.get("training_mean")
    if training_mean_path and training_mean_path.exists():
        training_mean = np.load(training_mean_path)
        print(f"Training mean loaded: shape={training_mean.shape}")
    else:
        training_mean = None
        print("No training mean found, will use zero baseline only")

    # Paths
    dd_dir = RECON / model_name / "data_driven"
    out_base = RECON / model_name / "local_explanations"
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"Data-driven dir: {dd_dir}")
    print(f"Output dir: {out_base}")
    print(f"IG steps: {args.steps}")

    for k in range(M):
        proto_dir = dd_dir / f"proto_{k:03d}"
        if not (proto_dir / "epochs.npy").exists():
            print(f"  P{k}: no epochs, skipping")
            continue

        epochs = np.load(proto_dir / "epochs.npy")  # (256, C, T, F)
        distances = np.load(proto_dir / "distances.npy")  # (256,)
        labels = np.load(proto_dir / "labels.npy")  # (256,)

        # Select the single nearest epoch
        best_idx = int(distances.argmin())
        best_epoch = epochs[best_idx:best_idx+1]  # (1, C, T, F)
        best_dist = float(distances[best_idx])
        best_label = int(labels[best_idx])
        valid_labels = labels[labels >= 0].astype(int)
        dominant_stage = STAGE_NAMES[int(np.bincount(valid_labels, minlength=5).argmax())] if len(valid_labels) > 0 else "?"

        print(f"\n  P{k} ({dominant_stage}): nearest epoch idx={best_idx}, "
              f"dist={best_dist:.3f}, label={STAGE_NAMES[best_label]}")

        # Compute IG
        x = torch.from_numpy(best_epoch).float().to(device)  # (1, C, T, F)
        baseline = torch.zeros_like(x)  # zero baseline

        f = make_prototype_distance_fn(model, codebook[k], device)
        ig = IntegratedGradients(f, steps=args.steps, expects_batch=True)
        attr = ig(x, baseline=baseline)  # (1, C, T, F)

        attr_np = attr.detach().cpu().numpy().squeeze(0)  # (C, T, F)

        # Completeness check
        with torch.no_grad():
            fx = f(x).item()
            fb = f(baseline).item()
        attr_sum = attr_np.sum()
        expected = fx - fb
        gap = abs(attr_sum - expected) / (abs(expected) + 1e-8)
        print(f"    IG completeness: sum={attr_sum:.4f}, expected={expected:.4f}, gap={gap:.4f}")

        # Save
        out_dir = out_base / f"proto_{k:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "local_epoch.npy", best_epoch.squeeze(0))
        np.save(out_dir / "local_attr.npy", attr_np)

        # Band-aggregated relevance
        from physioex.explain.foundational.sleep_bands import SLEEP_BANDS, bands_to_bin_ranges
        bin_ranges = bands_to_bin_ranges(SLEEP_BANDS, fs=100.0, signal_length=256)
        band_names = [b[0] for b in bin_ranges]
        band_rel = np.zeros((3, len(bin_ranges)), dtype=np.float32)  # (C, n_bands)
        for bi, (_, bstart, bend) in enumerate(bin_ranges):
            for ch in range(3):
                band_rel[ch, bi] = np.abs(attr_np[ch, :, bstart:bend]).sum()
        np.save(out_dir / "band_relevance.npy", band_rel)

        # Channel relevance
        chan_rel = np.abs(attr_np).sum(axis=(1, 2))  # (C,)
        np.save(out_dir / "channel_relevance.npy", chan_rel)

        meta = {
            "prototype_idx": k,
            "dominant_stage": dominant_stage,
            "selected_epoch_idx": best_idx,
            "selected_epoch_label": STAGE_NAMES[best_label],
            "l2_distance": best_dist,
            "ig_steps": args.steps,
            "ig_completeness_gap": float(gap),
            "band_names": band_names,
            "channel_names": ["EEG", "EOG", "EMG"],
        }
        with open(out_dir / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)

        print(f"    Chan relevance: EEG={chan_rel[0]:.2f}, EOG={chan_rel[1]:.2f}, EMG={chan_rel[2]:.2f}")

    print(f"\nDone. Output at {out_base}/")


if __name__ == "__main__":
    main()
