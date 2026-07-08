"""Out-of-domain evaluation on Parkinsons Night (HOA + PD), with/without EMG.

Usage:
    python eval_ood_parkinsons.py --gpu_id 0
    python eval_ood_parkinsons.py --gpu_id 0 --zero_emg
"""
import argparse
import importlib
import json
import os

import torch
from physioex.data.datasets import get_dataset
from physioex.models import load_from_pretrained
from sklearn.metrics import cohen_kappa_score as sk_kappa, f1_score as sk_f1


MODELS_3CH = [
    ("seqsleepnet-phan-3ch", "dir", "seqsleepnet-phan-3ch", 3, 20),
    ("sleeptransformer-phan-3ch", "dir", "sleeptransformer-phan-3ch", 3, 21),
    ("protosleepnet-seq", "dir", "protosleepnet-seq-3ch-mixer", 3, 20),
    ("protosleepnet-st", "dir", "protosleepnet-st-3ch-mixer", 3, 21),
]

MODELS_1CH = [
    ("seqsleepnet-phan (1ch)", "hf", "seqsleepnet-phan", 1, 20),
    ("sleeptransformer-phan (1ch)", "hf", "sleeptransformer-phan", 1, 21),
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


def voting_evaluate(model, dataset, seq_len, n_ch, device, zero_emg=False):
    all_preds = []
    all_labels = []

    for spec in dataset._subjects:
        n_ep = dataset._n_epochs[spec.subject_id]
        item = dataset._build_item(spec, 0, n_ep)
        order = item["channel_order"]

        if n_ch == 3:
            x = torch.stack([item["signals"][ch] for ch in order], dim=1)
            if zero_emg:
                x[:, 2] = 0.0
        else:
            x = item["signals"][order[0]].unsqueeze(1)

        labels = item["labels"]
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
                out = model(chunk).cpu().reshape(-1, n_classes)
                votes[offset:offset + usable] += out
                counts[offset:offset + usable] += 1

        safe = counts.clamp(min=1).unsqueeze(-1)
        preds = (votes / safe).argmax(dim=-1)
        valid = labels >= 0
        all_preds.append(preds[valid])
        all_labels.append(labels[valid])

    return torch.cat(all_preds), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--models_dir", type=str,
                        default=os.environ.get("PROTOSLEEPNET_MODELS", "models"))
    parser.add_argument("--zero_emg", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = (
        torch.device(f"cuda:{args.gpu_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    DatasetClass = get_dataset("parkinsons")
    models = MODELS_3CH if args.zero_emg else (MODELS_1CH + MODELS_3CH)
    results = {}
    emg_tag = " (EMG=0)" if args.zero_emg else ""

    for group in ["HOA", "PD"]:
        print(f"\n{'=' * 90}")
        print(f" OOD EVAL{emg_tag}: Parkinsons Night {group}")
        print(f"{'=' * 90}")
        print(f"  {'Model':35s}  {'ACC':>6s}  {'K':>6s}  {'F1':>6s}  | {'W':>6s}  {'N1':>6s}  {'N2':>6s}  {'N3':>6s}  {'REM':>6s}")
        print(f"  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*6}  | {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

        for model_name, load_type, model_id, n_ch, seq_len in models:
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
                recording="night",
                group=group,
            )

            preds, labels = voting_evaluate(model, ds, seq_len, n_ch, device,
                                            zero_emg=args.zero_emg)
            acc = (preds == labels).float().mean().item()
            kappa = sk_kappa(labels.numpy(), preds.numpy())
            f1 = sk_f1(labels.numpy(), preds.numpy(), average="macro")

            per_class = {}
            for c in range(5):
                mask = labels == c
                if mask.sum() > 0:
                    per_class[c] = (preds[mask] == c).float().mean().item()
                else:
                    per_class[c] = 0.0

            pc_str = "  ".join(f"{per_class[c]*100:5.1f}%" for c in range(5))
            print(f"  {model_name:35s}  {acc*100:5.1f}%  {kappa:5.3f}  {f1*100:5.1f}%  | {pc_str}")

            results[f"{model_name}_{group}"] = {
                "accuracy": acc, "cohen_kappa": kappa, "f1_score": f1,
                "per_class": {["W","N1","N2","N3","R"][c]: per_class[c] for c in range(5)},
            }
            del model
            torch.cuda.empty_cache()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = "zero_emg" if args.zero_emg else "full"
        out_path = os.path.join(args.output_dir, f"parkinsons_night_{tag}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
