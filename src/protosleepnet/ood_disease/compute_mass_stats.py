"""Precompute MASS STFT statistics (per-channel, global and per-class).

Saves a .pt file with mean/std per frequency bin for EEG, EOG, EMG.
Used for domain transport of Alzheimer data to MASS distribution.

Usage:
    python compute_mass_stats.py --output mass_stft_stats.pt
"""
import argparse
import torch
import numpy as np
from physioex.data.datasets import get_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="mass_stft_stats.pt")
    parser.add_argument("--max_per_class", type=int, default=2000)
    args = parser.parse_args()

    DatasetClass = get_dataset("mass")
    ds = DatasetClass(
        channels=["EEG", "EOG", "EMG"],
        pipelines="seqsleepnet",
        sequence_length=1,
    )

    modalities = ["EEG", "EOG", "EMG"]
    # Collect all spectra per channel per class
    # Each spectrum is (29, 129) — average over time (dim 0) → (129,)
    class_spectra = {mod: {c: [] for c in range(5)} for mod in modalities}
    n_per_class = {c: 0 for c in range(5)}

    print(f"Collecting MASS spectra (max {args.max_per_class}/class)...")
    for i in range(len(ds)):
        item = ds[i]
        label = item["labels"][0].item()
        if label < 0 or label > 4:
            continue
        if n_per_class[label] >= args.max_per_class:
            if all(v >= args.max_per_class for v in n_per_class.values()):
                break
            continue

        for ch in item["channel_order"]:
            mod = ch.split("_")[0]
            sig = item["signals"][ch][0]  # (29, 129)
            # Average over time frames → (129,)
            class_spectra[mod][label].append(sig.mean(dim=0).numpy())

        n_per_class[label] += 1
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(ds)}  {n_per_class}")

    print(f"Final counts: {n_per_class}")

    stats = {"global": {}, "per_class": {}}

    for mod in modalities:
        # Global: all classes together
        all_specs = []
        for c in range(5):
            all_specs.extend(class_spectra[mod][c])
        all_specs = np.array(all_specs)  # (N, 129)

        stats["global"][mod] = {
            "mean": torch.from_numpy(all_specs.mean(axis=0).astype(np.float32)),
            "std": torch.from_numpy(all_specs.std(axis=0).astype(np.float32)),
        }

        # Per-class
        means = []
        stds = []
        for c in range(5):
            arr = np.array(class_spectra[mod][c])
            means.append(arr.mean(axis=0))
            stds.append(arr.std(axis=0))

        stats["per_class"][mod] = {
            "mean": torch.from_numpy(np.array(means, dtype=np.float32)),  # (5, 129)
            "std": torch.from_numpy(np.array(stds, dtype=np.float32)),    # (5, 129)
        }

        g = stats["global"][mod]
        print(f"\n{mod} global: mean={g['mean'].mean():.2f}, std={g['std'].mean():.2f}")
        for c in range(5):
            m = stats["per_class"][mod]["mean"][c].mean()
            s = stats["per_class"][mod]["std"][c].mean()
            print(f"  class {c}: mean={m:.2f}, std={s:.2f}")

    torch.save(stats, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
