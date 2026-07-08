"""Train SeqSleepNet 1ch from scratch on Alzheimer HC (EEG only).

Usage:
    python train_scratch_1ch.py --gpu_id 0 --lr 1e-3 --output_dir /path/to/output
"""
import argparse
import json
import os

import torch

from physioex.data.datasets import get_dataset
from physioex.models.seqsleepnet import SeqSleepNet
from physioex.train.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="scratch_1ch_output")
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

    # Model: SeqSleepNet 1ch, random init
    model = SeqSleepNet(in_chan=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SeqSleepNet 1ch from scratch: {n_params:,} params")

    # Dataset: Alzheimer HC, EEG only
    DatasetClass = get_dataset("alzheimers")
    hc_dataset = DatasetClass(
        channels=["EEG"],
        pipelines="seqsleepnet",
        sequence_length=seq_len,
        subset="HC",
    )
    print(f"HC dataset: {hc_dataset.get_n_subjects()} subjects, {len(hc_dataset)} sequences")

    # Train
    print(f"\n{'='*60}")
    print(f"Training SeqSleepNet 1ch from scratch on HC (lr={args.lr})")
    print(f"{'='*60}")

    model = Trainer.train(
        model=model,
        dataset=hc_dataset,
        max_epochs=args.max_epochs,
        lr=args.lr,
        weight_decay=1e-5,
        train_batch_size=args.batch_size,
        fold=0,
        gpu_id=args.gpu_id,
        checkpoint_path=os.path.join(args.output_dir, "checkpoints"),
        early_stopping_patience=args.patience,
    )

    # Save
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Evaluate on HC test + AD
    from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score

    model = model.to(device).eval()

    for subset, subset_name in [("HC", "HC test"), ("AD", "AD")]:
        ds = DatasetClass(
            channels=["EEG"],
            pipelines="seqsleepnet",
            sequence_length=seq_len,
            subset=subset,
        )

        # Voting evaluate on test subjects (for HC) or all (for AD)
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
            labels = item["labels"]
            N = x.shape[0]
            L = seq_len
            votes = torch.zeros(N, 5)
            counts = torch.zeros(N)

            with torch.no_grad():
                for offset in range(L):
                    usable = ((N - offset) // L) * L
                    if usable == 0:
                        continue
                    chunk = x[offset:offset+usable].reshape(-1, L, *x.shape[1:]).to(device)
                    out = model(chunk).cpu().reshape(-1, 5)
                    votes[offset:offset+usable] += out
                    counts[offset:offset+usable] += 1

            safe = counts.clamp(min=1).unsqueeze(-1)
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

        # Per-class
        stage_names = ["Wake", "N1", "N2", "N3", "REM"]
        for c in range(5):
            mask = all_labels == c
            if mask.sum() > 0:
                cls_acc = (all_preds[mask] == c).mean()
                print(f"  {stage_names[c]}: {cls_acc*100:.1f}%")

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"lr": args.lr, "method": "scratch_1ch_seqsleepnet"}, f, indent=2)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
