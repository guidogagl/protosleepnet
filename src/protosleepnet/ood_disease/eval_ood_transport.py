"""Out-of-domain eval with statistical transport (Alzheimer → MASS).

Two modes:
  --mode global:    align overall mean/std per channel to MASS
  --mode per_class: first pass → pseudo-labels, then per-class transport

Usage:
    python eval_ood_transport.py --gpu_id 0 --mode global --stats mass_stft_stats.pt
    python eval_ood_transport.py --gpu_id 0 --mode per_class --stats mass_stft_stats.pt
"""
import argparse
import importlib
import json
import os

import torch
import torch.nn as nn
from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from sklearn.metrics import cohen_kappa_score as sk_kappa, f1_score as sk_f1


MODELS_INFO = [
    ("seqsleepnet-phan (1ch)", "hf", "seqsleepnet-phan", 1, 20),
    ("sleeptransformer-phan (1ch)", "hf", "sleeptransformer-phan", 1, 21),
    ("seqsleepnet-phan-3ch", "dir", "seqsleepnet-phan-3ch", 3, 20),
    ("sleeptransformer-phan-3ch", "dir", "sleeptransformer-phan-3ch", 3, 21),
    ("protosleepnet-seq", "dir", "protosleepnet-seq-3ch-mixer", 3, 20),
    ("protosleepnet-st", "dir", "protosleepnet-st-3ch-mixer", 3, 21),
]


def load_model_from_dir(model_dir, device):
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    module_path, class_name = config["model_class"].rsplit(":", 1)
    mod = importlib.import_module(module_path)
    ModelClass = getattr(mod, class_name)
    if "factory" in config:
        factory = getattr(ModelClass, config["factory"])
        model = factory(**config.get("factory_kwargs", {}))
    else:
        model = ModelClass(**config["model_kwargs"])
    weights_path = os.path.join(model_dir, "model.pt")
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model.to(device).eval()


def load_subject(dataset, spec, n_ch):
    """Load full-night spectrogram for a subject. Returns (x, labels)."""
    n_ep = dataset._n_epochs[spec.subject_id]
    item = dataset._build_item(spec, 0, n_ep)
    order = item["channel_order"]
    if n_ch == 3:
        x = torch.stack([item["signals"][ch] for ch in order], dim=1)  # (N, C, T, F)
    else:
        x = item["signals"][order[0]].unsqueeze(1)  # (N, 1, T, F)
    return x, item["labels"]


def transport_global(x, n_ch, src_stats, tgt_stats):
    """Apply global (channel-wise) z-score transport: x → MASS distribution.

    x: (N, C, T, F)  — per-epoch spectrograms
    Computes source mean/std from x itself, maps to target mean/std from MASS.
    """
    mods = ["EEG", "EOG", "EMG"][:n_ch]
    x_out = x.clone()
    for c, mod in enumerate(mods):
        chan = x[:, c]  # (N, T, F)
        # Source stats: from this subject's data, per freq bin
        src_mean = chan.mean(dim=(0, 1))  # (F,)
        src_std = chan.std(dim=(0, 1)).clamp(min=1e-6)
        # Target stats from MASS
        tgt_mean = tgt_stats["global"][mod]["mean"]
        tgt_std = tgt_stats["global"][mod]["std"].clamp(min=1e-6)
        x_out[:, c] = (chan - src_mean) / src_std * tgt_std + tgt_mean
    return x_out


def transport_per_class(x, pseudo_labels, n_ch, src_stats, tgt_stats):
    """Apply per-class transport using pseudo-labels.

    x: (N, C, T, F)
    pseudo_labels: (N,) predicted class for each epoch
    """
    mods = ["EEG", "EOG", "EMG"][:n_ch]
    x_out = x.clone()
    for c_idx, mod in enumerate(mods):
        chan = x[:, c_idx]  # (N, T, F)
        for cls in range(5):
            mask = pseudo_labels == cls
            if mask.sum() == 0:
                continue
            subset = chan[mask]  # (K, T, F)
            src_mean = subset.mean(dim=(0, 1))  # (F,)
            src_std = subset.std(dim=(0, 1)).clamp(min=1e-6)
            tgt_mean = tgt_stats["per_class"][mod]["mean"][cls]
            tgt_std = tgt_stats["per_class"][mod]["std"][cls].clamp(min=1e-6)
            x_out[mask, c_idx] = (subset - src_mean) / src_std * tgt_std + tgt_mean
    return x_out


def voting_predict(model, x, seq_len, device):
    """Voting prediction for a full-night recording. Returns (N,) predictions."""
    N = x.shape[0]
    L = seq_len
    n_classes = 5
    votes = torch.zeros(N, n_classes)
    counts = torch.zeros(N)

    with torch.no_grad(), torch.autocast("cuda"):
        for offset in range(L):
            usable = ((N - offset) // L) * L
            if usable == 0:
                continue
            chunk = x[offset:offset + usable].reshape(-1, L, *x.shape[1:]).to(device)
            out = model(chunk).cpu()
            out = out.reshape(-1, n_classes)
            votes[offset:offset + usable] += out
            counts[offset:offset + usable] += 1

    safe = counts.clamp(min=1).unsqueeze(-1)
    voted = votes / safe
    return voted.argmax(dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--mode", type=str, choices=["global", "per_class"], required=True)
    parser.add_argument("--stats", type=str, required=True, help="Path to mass_stft_stats.pt")
    parser.add_argument("--models_dir", type=str,
                        default=os.environ.get("PROTOSLEEPNET_MODELS", "models"))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    tgt_stats = torch.load(args.stats, map_location="cpu", weights_only=False)
    DatasetClass = get_dataset("alzheimers")
    stage_names = ["W", "N1", "N2", "N3", "R"]
    results = {}

    print(f"\nMode: {args.mode}")

    for subset in ["HC", "AD"]:
        print(f"\n{'=' * 90}")
        print(f" OOD EVAL ({args.mode} transport): Alzheimer {subset}")
        print(f"{'=' * 90}")
        print(f"  {'Model':35s}  {'ACC':>6s}  {'K':>6s}  {'F1':>6s}  | {'W':>6s}  {'N1':>6s}  {'N2':>6s}  {'N3':>6s}  {'REM':>6s}")
        print(f"  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  | {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

        for model_name, load_type, model_id, n_ch, seq_len in MODELS_INFO:
            if load_type == "hf":
                model = load_from_pretrained(model_id, device=str(device))
            else:
                model = load_model_from_dir(
                    os.path.join(args.models_dir, model_id), device
                )
            model.eval()

            channels = ["EEG", "EOG", "EMG"] if n_ch == 3 else ["EEG"]
            ds = DatasetClass(
                channels=channels,
                pipelines="seqsleepnet",
                sequence_length=seq_len,
                subset=subset,
            )

            all_preds = []
            all_labels = []

            for spec in ds._subjects:
                x, labels = load_subject(ds, spec, n_ch)

                if args.mode == "global":
                    x_t = transport_global(x, n_ch, None, tgt_stats)
                    preds = voting_predict(model, x_t, seq_len, device)
                else:
                    # First pass: get pseudo-labels from raw data
                    pseudo = voting_predict(model, x, seq_len, device)
                    # Transport per-class
                    x_t = transport_per_class(x, pseudo, n_ch, None, tgt_stats)
                    # Second pass: predict on transported data
                    preds = voting_predict(model, x_t, seq_len, device)

                valid = labels >= 0
                all_preds.append(preds[valid])
                all_labels.append(labels[valid])

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            acc = (all_preds == all_labels).float().mean().item()
            kappa = sk_kappa(all_labels.numpy(), all_preds.numpy())
            f1 = sk_f1(all_labels.numpy(), all_preds.numpy(), average="macro")

            per_class = {}
            for c in range(5):
                mask = all_labels == c
                if mask.sum() > 0:
                    per_class[c] = (all_preds[mask] == c).float().mean().item()
                else:
                    per_class[c] = 0.0

            pc_str = "  ".join(f"{per_class[c]*100:5.1f}%" for c in range(5))
            print(f"  {model_name:35s}  {acc*100:5.1f}%  {kappa:5.3f}  {f1*100:5.1f}%  | {pc_str}")

            results[f"{model_name}_{subset}"] = {
                "accuracy": acc,
                "cohen_kappa": kappa,
                "f1_score": f1,
                "per_class": {stage_names[c]: per_class[c] for c in range(5)},
            }

            del model
            torch.cuda.empty_cache()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"ood_{args.mode}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
