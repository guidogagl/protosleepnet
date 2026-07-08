"""Combinatorial feature ablation for prototype matching analysis.

Measures the marginal contribution of each EEG band to prototype matching
by adding bands back to a mean-EEG baseline (with actual EOG/EMG held constant).
Channel importance computed separately via whole-channel ablation.

All 256 subset variants are batched on GPU for maximum throughput.

Usage:
    python combinatorial_ablation.py /path/to/data_driven/ \
        --backbone seq --checkpoint_path ... --codebook_path ...
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
PHYSIOEX_ROOT = None
for c in [Path(os.environ.get("PHYSIOEX_ROOT", "")), Path(os.environ.get("PHYSIOEX_ROOT", "")),
          Path(os.environ.get("PHYSIOEX_ROOT", ""))]:
    if (c / "physioex").is_dir():
        sys.path.insert(0, str(c)); PHYSIOEX_ROOT = c; break

from physioex.explain.foundational.sleep_bands import SLEEP_BANDS, bands_to_bin_ranges
from protosleepnet.proto_reconstruction.utils import load_frozen_model, load_codebook, add_common_args, get_device, get_paths, CONFIGS

torch.backends.cudnn.enabled = False

FS, NFFT = 100.0, 256
FREQ_RES = FS / NFFT
N_FREQ = NFFT // 2 + 1
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]


# ── Band definitions ─────────────────────────────────────────────────

def get_eeg_bands():
    """Get 8 EEG bands (no mains) as (name, bstart, bend) tuples."""
    return [(n, s, e) for n, s, e in
            bands_to_bin_ranges(SLEEP_BANDS, fs=FS, signal_length=NFFT)
            if n != "mains"]


# ── Training set mean ────────────────────────────────────────────────

def load_training_mean(backbone, override_path=None):
    """Load training set mean spectrogram (C, F) float32.

    Computed offline over the full training set (not prototype epochs).
    """
    if override_path:
        p = Path(override_path)
    else:
        p = get_paths(backbone)["training_mean"]
    if not p.exists():
        raise FileNotFoundError(
            f"Training mean not found at {p}. "
            "Run compute_training_mean.py first."
        )
    tm = np.load(p).astype(np.float32)
    print(f"  Training mean loaded from {p}, shape={tm.shape}")
    return tm


# ── Channel importance (whole-channel ablation) ──────────────────────

@torch.no_grad()
def compute_channel_importance(model, codebook_k, epochs_np, training_mean_np,
                                device, batch_size=256):
    """Whole-channel ablation. training_mean_np is (C, F)."""
    p_k = torch.from_numpy(codebook_k).float().to(device)
    gm = torch.from_numpy(training_mean_np).float().to(device)  # (C, F)
    epochs_t = torch.from_numpy(epochs_np).float().to(device)
    N = epochs_t.shape[0]

    orig_dists = []
    for i in range(0, N, batch_size):
        batch = epochs_t[i:i + batch_size].unsqueeze(1)
        h = model.epoch_encode(batch, quantize=False).squeeze(1)
        orig_dists.append(((h - p_k.unsqueeze(0)) ** 2).sum(dim=1))
    orig_dist = torch.cat(orig_dists).mean().item()

    channel_importance = np.zeros(3, dtype=np.float32)
    for ch in range(3):
        abl_dists = []
        for i in range(0, N, batch_size):
            batch = epochs_t[i:i + batch_size].clone()
            batch[:, ch, :, :] = gm[ch]  # (F,) broadcasts to (T, F)
            h = model.epoch_encode(batch.unsqueeze(1), quantize=False).squeeze(1)
            abl_dists.append(((h - p_k.unsqueeze(0)) ** 2).sum(dim=1))
        channel_importance[ch] = torch.cat(abl_dists).mean().item() - orig_dist

    return channel_importance


# ── Predict classes ──────────────────────────────────────────────────

@torch.no_grad()
def predict_classes(model, epochs_np, device, batch_size=256):
    N = epochs_np.shape[0]
    all_preds = []
    epochs_t = torch.from_numpy(epochs_np).float()
    for i in range(0, N, batch_size):
        batch = epochs_t[i:i + batch_size].unsqueeze(1).to(device)
        model.epoch_encode(batch, quantize=False)
        logits = model.get_metrics()["epoch_logits"].squeeze(1)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(all_preds)
    counts = np.bincount(preds, minlength=5)
    return preds, counts


# ── EEG band marginal contribution (GPU-batched) ────────────────────

@torch.no_grad()
def compute_band_importance(model, codebook_k, epochs_np, training_mean_np,
                            eeg_bands, device):
    """Compute marginal contribution of each EEG band by adding bands back
    to a mean-EEG baseline. All 256 subsets batched on GPU per epoch.

    training_mean_np is (C, F) — the training set mean spectrogram.

    Returns:
        marginal_importance: (n_bands,) — positive = band helps matching
        eeg_contribution: float — total EEG effect (dist_base - dist_actual)
    """
    p_k = torch.from_numpy(codebook_k).float().to(device)
    gm = torch.from_numpy(training_mean_np).float().to(device)  # (C, F)
    n_bands = len(eeg_bands)
    n_subsets = 2 ** n_bands  # 256

    # Precompute band bin ranges on GPU
    band_slices = [(bstart, bend) for _, bstart, bend in eeg_bands]

    # Precompute subset masks: (n_subsets, n_bands) bool
    subset_masks = np.array([[(si >> bi) & 1 for bi in range(n_bands)]
                              for si in range(n_subsets)], dtype=bool)

    N = epochs_np.shape[0]
    # Accumulate distances per subset across epochs
    dist_sums = torch.zeros(n_subsets, device=device)
    dist_actual_sum = 0.0
    dist_base_sum = 0.0

    for ei in range(N):
        x = torch.from_numpy(epochs_np[ei]).float().to(device)  # (C, T, F)

        # Baseline: training-set mean EEG + actual EOG/EMG
        x_base = x.clone()
        x_base[0, :, :] = gm[0]  # (F,) broadcasts to (T, F)

        # Build all 256 variants: (256, C, T, F)
        variants = x_base.unsqueeze(0).expand(n_subsets, -1, -1, -1).clone()
        for bi, (bstart, bend) in enumerate(band_slices):
            # For all subsets where band bi is active, restore actual EEG bins
            mask = torch.from_numpy(subset_masks[:, bi]).to(device)  # (256,)
            variants[mask, 0, :, bstart:bend] = x[0, :, bstart:bend]

        # Forward all 256 at once: (256, 1, C, T, F)
        h_all = model.epoch_encode(variants.unsqueeze(1), quantize=False).squeeze(1)  # (256, d)
        dists = ((h_all - p_k.unsqueeze(0)) ** 2).sum(dim=1)  # (256,)
        dist_sums += dists

        # Actual and baseline distances (subset 0 = no bands = baseline,
        # subset all-ones = all bands = actual)
        dist_base_sum += dists[0].item()      # subset 0: no bands restored = baseline
        dist_actual_sum += dists[-1].item()    # subset 2^n-1: all bands restored = actual

        if (ei + 1) % 50 == 0:
            print(f"    {ei+1}/{N} epochs processed...")

    # Average across epochs
    dist_avg = (dist_sums / N).cpu().numpy()  # (n_subsets,)
    dist_base_avg = dist_base_sum / N
    dist_actual_avg = dist_actual_sum / N
    eeg_contribution = dist_base_avg - dist_actual_avg

    # Marginal importance: Shapley-like
    # importance[b] = mean over S not containing b of: dist[S] - dist[S ∪ {b}]
    importance = np.zeros(n_bands, dtype=np.float32)
    for bi in range(n_bands):
        bi_bit = 1 << bi
        marginals = []
        for si in range(n_subsets):
            if si & bi_bit:
                continue  # b already in S
            si_with_b = si | bi_bit
            marginals.append(dist_avg[si] - dist_avg[si_with_b])
        importance[bi] = np.mean(marginals)

    return importance, float(eeg_contribution)


# ── Feature direction ────────────────────────────────────────────────

def compute_feature_direction(epochs_np, training_mean_np, eeg_bands):
    """Is each band elevated or suppressed vs training set mean?

    training_mean_np is (C, F).
    """
    n_bands = len(eeg_bands)
    proto_power = np.zeros(n_bands, dtype=np.float32)
    mean_power = np.zeros(n_bands, dtype=np.float32)

    proto_lin = np.power(10.0, epochs_np / 10.0)       # (N, C, T, F)
    mean_lin = np.power(10.0, training_mean_np / 10.0)  # (C, F)

    for bi, (bname, bstart, bend) in enumerate(eeg_bands):
        proto_power[bi] = proto_lin[:, 0, :, bstart:bend].mean()
        mean_power[bi] = mean_lin[0, bstart:bend].mean()

    directions = proto_power - mean_power
    return directions, proto_power, mean_power


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combinatorial feature ablation for prototype matching"
    )
    add_common_args(parser)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--proto_idx", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--training_mean", type=str, default=None,
                        help="Override training_mean.npy path")
    args = parser.parse_args()

    device = get_device(args)
    print(f"Device: {device}")

    model = load_frozen_model(args.backbone, device,
                              checkpoint_path=args.checkpoint_path)
    codebook = load_codebook(args.backbone, m=args.m,
                             codebook_path=args.codebook_path)
    M = codebook.shape[0]
    print(f"Model loaded, codebook M={M}")

    print("Loading training set mean...")
    training_mean = load_training_mean(args.backbone, override_path=args.training_mean)

    eeg_bands = get_eeg_bands()
    band_names = [b[0] for b in eeg_bands]
    n_bands = len(band_names)
    print(f"EEG bands ({n_bands}): {band_names}")
    print(f"Subsets: 2^{n_bands} = {2**n_bands}")

    if args.proto_idx is not None:
        proto_indices = [args.proto_idx]
    else:
        proto_indices = sorted(
            int(p.name.split("_")[1])
            for p in args.data_dir.glob("proto_*")
            if (p / "epochs.npy").exists()
        )

    out_base = args.data_dir / "ablation"
    out_base.mkdir(parents=True, exist_ok=True)
    with open(out_base / "feature_names.json", "w") as f:
        json.dump(band_names, f, indent=2)
    np.save(out_base / "training_mean.npy", training_mean)

    for k in proto_indices:
        proto_dir = args.data_dir / f"proto_{k:03d}"
        if not (proto_dir / "epochs.npy").exists():
            continue

        print(f"\nPrototype {k}:")
        epochs = np.load(proto_dir / "epochs.npy")

        # Channel importance (whole-channel ablation)
        ch_imp = compute_channel_importance(
            model, codebook[k], epochs, training_mean, device, batch_size=args.batch_size
        )
        ch_total = abs(ch_imp).sum() + 1e-12
        ch_pcts = ch_imp / ch_total
        print(f"  Channels: EEG={ch_pcts[0]:.0%} EOG={ch_pcts[1]:.0%} EMG={ch_pcts[2]:.0%}")

        # Band importance (add-back combinatorial)
        importance, eeg_contrib = compute_band_importance(
            model, codebook[k], epochs, training_mean, eeg_bands, device
        )
        print(f"  EEG contribution: {eeg_contrib:.2f}")

        # Direction
        directions, proto_power, mean_power = compute_feature_direction(
            epochs, training_mean, eeg_bands
        )

        # Predict classes
        preds, pred_counts = predict_classes(model, epochs, device, batch_size=args.batch_size)
        dominant = STAGE_NAMES[pred_counts.argmax()]
        pred_purity = float(pred_counts.max() / len(preds))
        pred_dist = {STAGE_NAMES[i]: int(pred_counts[i]) for i in range(5)}
        print(f"  Predicted: {dominant} ({pred_purity:.0%})")

        # Relevance %
        band_rel_pct = {}
        for bi, bn in enumerate(band_names):
            pct = importance[bi] / (eeg_contrib + 1e-12) * 100
            band_rel_pct[bn] = float(pct)

        # Save
        out_dir = out_base / f"proto_{k:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "marginal_importance.npy", importance)
        np.save(out_dir / "channel_importance.npy", ch_imp)
        np.save(out_dir / "feature_direction.npy", directions)
        np.save(out_dir / "proto_power.npy", proto_power)
        np.save(out_dir / "mean_power.npy", mean_power)
        np.save(out_dir / "predicted_labels.npy", preds)

        meta = {
            "prototype_idx": k,
            "dominant_class": dominant,
            "predicted_purity": pred_purity,
            "predicted_distribution": pred_dist,
            "n_bands": n_bands,
            "feature_names": band_names,
            "eeg_contribution": float(eeg_contrib),
            "band_relevance_pct": band_rel_pct,
            "marginal_importance": {bn: float(importance[bi]) for bi, bn in enumerate(band_names)},
            "feature_direction": {bn: ("elevated" if directions[bi] > 0 else "suppressed")
                                  for bi, bn in enumerate(band_names)},
            "channel_importance": {"EEG": float(ch_imp[0]), "EOG": float(ch_imp[1]), "EMG": float(ch_imp[2])},
            "channel_importance_pct": {"EEG": float(ch_pcts[0]), "EOG": float(ch_pcts[1]), "EMG": float(ch_pcts[2])},
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Print
        ranked = sorted(zip(band_names, importance, directions), key=lambda x: -x[1])
        print(f"  Band relevance (% of EEG contribution):")
        for bn, imp, d in ranked[:5]:
            arrow = "↑" if d > 0 else "↓"
            pct = imp / (eeg_contrib + 1e-12) * 100
            print(f"    {bn:>12} {pct:>6.1f}%  {arrow}")

    print(f"\nDone. Output at {out_base}/")


if __name__ == "__main__":
    main()
