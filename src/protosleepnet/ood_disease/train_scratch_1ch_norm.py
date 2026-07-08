"""Train SeqSleepNet 1ch from scratch on Alzheimer HC with frequency-wise StandardScaling.

Computes mean/std per frequency bin from training subjects, normalizes all data.

Usage:
    python train_scratch_1ch_norm.py --gpu_id 0 --lr 1e-3 --output_dir /path/to/output
"""
import argparse
import json
import os

import torch
import torch.utils.data

from physioex.data.datasets import get_dataset
from physioex.models.seqsleepnet import SeqSleepNet
from physioex.train.trainer import Trainer


class NormalizedDataset(torch.utils.data.Dataset):
    """Wraps a BasePhysioDataset and applies frequency-wise z-score normalization."""

    def __init__(self, base_dataset, mean, std):
        """
        Args:
            base_dataset: BasePhysioDataset instance
            mean: dict {channel_key: (F,) tensor} — per-freq-bin mean
            std: dict {channel_key: (F,) tensor} — per-freq-bin std
        """
        self.base = base_dataset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        for ch_name in item["channel_order"]:
            sig = item["signals"][ch_name]  # (L, T, F)
            m = self.mean[ch_name]  # (F,)
            s = self.std[ch_name]   # (F,)
            item["signals"][ch_name] = (sig - m) / s
        return item

    # Forward attributes needed by Trainer
    @property
    def sequence_length(self):
        return self.base.sequence_length

    @property
    def _subjects(self):
        return self.base._subjects

    @property
    def _n_epochs(self):
        return self.base._n_epochs

    @property
    def _subject_ranges(self):
        return self.base._subject_ranges

    def split(self, fold=0):
        return self.base.split(fold=fold)

    def get_splits(self, fold=0):
        return self.base.get_splits(fold=fold)

    def get_n_subjects(self):
        return self.base.get_n_subjects()

    def _build_item(self, spec, start, end):
        item = self.base._build_item(spec, start, end)
        for ch_name in item["channel_order"]:
            sig = item["signals"][ch_name]
            m = self.mean[ch_name]
            s = self.std[ch_name]
            item["signals"][ch_name] = (sig - m) / s
        return item

    def _subject_ids_to_flat_indices(self, ids):
        return self.base._subject_ids_to_flat_indices(ids)


def compute_train_stats(dataset, fold=0, max_epochs=5000):
    """Compute per-frequency-bin mean/std from training subjects."""
    train_ids, _, _ = dataset.get_splits(fold=fold)

    # Collect spectra from training subjects
    sums = {}
    sq_sums = {}
    counts = {}

    n = 0
    for spec in dataset._subjects:
        if spec.subject_id not in train_ids:
            continue
        n_ep = dataset._n_epochs[spec.subject_id]
        item = dataset._build_item(spec, 0, n_ep)

        for ch_name in item["channel_order"]:
            sig = item["signals"][ch_name]  # (N, T, F)
            # Flatten over epochs and time → (N*T, F)
            flat = sig.reshape(-1, sig.shape[-1])

            if ch_name not in sums:
                sums[ch_name] = torch.zeros(sig.shape[-1], dtype=torch.float64)
                sq_sums[ch_name] = torch.zeros(sig.shape[-1], dtype=torch.float64)
                counts[ch_name] = 0

            sums[ch_name] += flat.double().sum(dim=0)
            sq_sums[ch_name] += (flat.double() ** 2).sum(dim=0)
            counts[ch_name] += flat.shape[0]

        n += 1

    mean = {}
    std = {}
    for ch_name in sums:
        m = sums[ch_name] / counts[ch_name]
        s = ((sq_sums[ch_name] / counts[ch_name]) - m ** 2).clamp(min=1e-12).sqrt()
        mean[ch_name] = m.float()
        std[ch_name] = s.float()
        print(f"  {ch_name}: mean=[{m.min():.2f}, {m.max():.2f}], std=[{s.min():.2f}, {s.max():.2f}]")

    print(f"  Computed from {n} training subjects")
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="scratch_1ch_norm_output")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seq_len = 20

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if args.gpu_id is not None and torch.cuda.is_available()
        else torch.device("cpu")
    )

    model = SeqSleepNet(in_chan=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SeqSleepNet 1ch from scratch: {n_params:,} params")

    DatasetClass = get_dataset("alzheimers")
    hc_dataset = DatasetClass(
        channels=["EEG"],
        pipelines="seqsleepnet",
        sequence_length=seq_len,
        subset="HC",
    )
    print(f"HC dataset: {hc_dataset.get_n_subjects()} subjects, {len(hc_dataset)} sequences")

    # Compute normalization stats from training subjects
    print("\nComputing training set statistics (frequency-wise)...")
    mean, std = compute_train_stats(hc_dataset, fold=0)

    # Save stats for later use
    stats_path = os.path.join(args.output_dir, "norm_stats.pt")
    torch.save({"mean": mean, "std": std}, stats_path)
    print(f"Saved normalization stats to {stats_path}")

    # Build dataloaders from original dataset, then wrap with normalization
    from physioex.data.collate import dict_collate_fn

    train_loader, valid_loader, _ = Trainer.build_dataloaders(
        dataset=hc_dataset,
        train_batch_size=args.batch_size,
        eval_batch_size=1,
        fold=0,
    )

    # Wrap dataloaders with normalization transform
    class NormCollate:
        def __init__(self, base_collate, mean, std):
            self.base_collate = base_collate
            self.mean = mean
            self.std = std

        def __call__(self, batch):
            out = self.base_collate(batch)
            for ch_name in out["channel_order"]:
                out["signals"][ch_name] = (out["signals"][ch_name] - self.mean[ch_name]) / self.std[ch_name]
            return out

    norm_collate = NormCollate(dict_collate_fn, mean, std)
    train_loader.collate_fn = norm_collate
    valid_loader.collate_fn = norm_collate

    # Train
    print(f"\n{'='*60}")
    print(f"Training SeqSleepNet 1ch + StandardScaling (lr={args.lr})")
    print(f"{'='*60}")

    model = Trainer.train(
        model=model,
        dataset=(train_loader, valid_loader),
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=1e-5,
        train_batch_size=args.batch_size,
        fold=0,
        gpu_id=args.gpu_id,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        early_stopping_patience=args.patience,
    )

    # Save model
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Evaluate
    from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

    model = model.to(device).eval()

    for subset, subset_name in [("HC", "HC test"), ("AD", "AD")]:
        ds = DatasetClass(
            channels=["EEG"],
            pipelines="seqsleepnet",
            sequence_length=seq_len,
            subset=subset,
        )

        if subset == "HC":
            _, _, test_ids = ds.get_splits(fold=0)
            eval_ids = [sid for _, sid in test_ids]
        else:
            eval_ids = [s.subject_id for s in ds._subjects]

        all_preds = []
        all_labels = []

        for spec in ds._subjects:
            if spec.subject_id not in eval_ids:
                continue
            n_ep = ds._n_epochs[spec.subject_id]
            item = ds._build_item(spec, 0, n_ep)
            order = item["channel_order"]
            x = item["signals"][order[0]].unsqueeze(1)  # (N, 1, T, F)

            # Apply same normalization
            ch_name = order[0]
            x = (x - mean[ch_name]) / std[ch_name]

            labels = item["labels"]
            N = x.shape[0]
            L = seq_len
            votes = torch.zeros(N, 5)
            counts_v = torch.zeros(N)

            with torch.no_grad():
                for offset in range(L):
                    usable = ((N - offset) // L) * L
                    if usable == 0:
                        continue
                    chunk = x[offset:offset+usable].reshape(-1, L, *x.shape[1:]).to(device)
                    out = model(chunk).cpu().reshape(-1, 5)
                    votes[offset:offset+usable] += out
                    counts_v[offset:offset+usable] += 1

            safe = counts_v.clamp(min=1).unsqueeze(-1)
            preds = (votes / safe).argmax(dim=-1)
            valid = labels >= 0
            all_preds.append(preds[valid])
            all_labels.append(labels[valid])

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        acc = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"\n{subset_name}: ACC={acc*100:.1f}%  Kappa={kappa:.3f}  F1={f1*100:.1f}%")

        stage_names = ["Wake", "N1", "N2", "N3", "REM"]
        for c in range(5):
            mask = all_labels == c
            if mask.sum() > 0:
                cls_acc = (all_preds[mask] == c).mean()
                print(f"  {stage_names[c]}: {cls_acc*100:.1f}%")


if __name__ == "__main__":
    main()
