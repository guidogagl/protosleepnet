"""Extract sequence embeddings and run linear staging probe.

Loads a pretrained model from a local directory (config.json + model.pt),
extracts per-epoch contextualized embeddings via model.encode() with
sliding-window voting, and runs 5-fold subject-wise linear probing.

Supports all 4 models (proto-st, proto-seq, st-phan, seq-phan) and
22 staging datasets. Handles both factory-based (ProtoSleepNet) and
model_kwargs-based (baseline) config formats.

Output per (model, dataset):
  {PHYSIOEX_CACHE_DIR}/embeddings/{model_name}/{dataset_name}/
    {subject_id}/embeddings.npy          # (n_epochs, D) sequence embeddings
    {subject_id}/labels.npy              # (n_epochs,) sleep stage labels
    linear_probe_results.json            # 5-fold CV summary (kappa/acc/mf1)
    linear_probe_predictions.json        # per-subject softmax proba + labels

Usage:
    python extract_embeddings.py \
        --model_dir /path/to/protosleepnet-st-3ch-mixer \
        --model_name protosleepnet-st-3ch-mixer \
        --gpu_id 0

    python extract_embeddings.py \
        --model_dir /path/to/sleeptransformer-phan \
        --model_name sleeptransformer-phan \
        --datasets shhs_visit1 mesa hmc \
        --gpu_id 0
"""
import argparse
import importlib
import json
from pathlib import Path

import torch

from physioex.models import extract_embeddings, linear_probe

PIPELINE = "seqsleepnet"

# ── 22 staging datasets ─────────────────────────────────────────────
# (name, dataset_class_name, kwargs)
# name: used for output directory naming (matches archive convention)
# dataset_class_name: key for get_dataset()
# kwargs: passed to DatasetClass constructor

DATASET_CONFIGS = [
    ("shhs_visit1",      "shhs",       {"visit": 1}),
    ("shhs_visit2",      "shhs",       {"visit": 2}),
    ("mesa",             "mesa",       {}),
    ("mros",             "mros",       {}),
    ("wsc_visit1",       "wsc",        {"visit": 1}),
    ("wsc_visit2",       "wsc",        {"visit": 2}),
    ("wsc_visit3",       "wsc",        {"visit": 3}),
    ("wsc_visit4",       "wsc",        {"visit": 4}),
    ("wsc_visit5",       "wsc",        {"visit": 5}),
    ("mass_cohort1",     "mass",       {"cohort": 1}),
    ("mass_cohort2",     "mass",       {"cohort": 2}),
    ("mass_cohort3",     "mass",       {"cohort": 3}),
    ("mass_cohort4",     "mass",       {"cohort": 4}),
    ("mass_cohort5",     "mass",       {"cohort": 5}),
    ("hpap_lab-full",    "hpap",       {"subset": "lab-full"}),
    ("hpap_lab-split",   "hpap",       {"subset": "lab-split"}),
    ("sleepedf",         "sleepedf",   {}),
    ("hmc",              "hmc",        {}),
    ("dcsm",             "dcsm",       {}),
    ("alzheimers",       "alzheimers", {}),
    ("parkinsons_night", "parkinsons", {"recording": "night"}),
    ("parkinsons_nap",   "parkinsons", {"recording": "nap"}),
]

DATASET_NAME_MAP = {cfg[0]: cfg for cfg in DATASET_CONFIGS}


def _resolve_class(spec: str):
    """Import a class from a 'module.path:ClassName' string."""
    module_path, class_name = spec.rsplit(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def load_local_model(model_dir: str, device: str = "cpu"):
    """Load a pretrained model from a local directory.

    Handles both model_kwargs (baselines) and factory (ProtoSleepNet) configs.

    Returns:
        (model, channels, seq_len)
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"No model.pt in {model_dir}")

    with open(config_path) as f:
        config = json.load(f)

    model_cls = _resolve_class(config["model_class"])

    if "factory" in config:
        factory_method = getattr(model_cls, config["factory"])
        model = factory_method(**config["factory_kwargs"])
    else:
        model = model_cls(**config["model_kwargs"])

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    training = config.get("training", {})
    channels = training.get("channels", ["EEG"])
    seq_len = training.get("sequence_length", 21)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__}, params={n_params:,}")
    print(f"  channels={channels}, seq_len={seq_len}")

    return model, channels, seq_len


def main():
    parser = argparse.ArgumentParser(
        description="Extract sequence embeddings and run linear staging probe"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Local model directory (config.json + model.pt)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model identifier for output naming")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names to run (default: all 22)")
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Override PHYSIOEX_DATA env var (base dir containing dataset subdirs)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing embeddings")
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, channels, seq_len = load_local_model(args.model_dir, device)

    # Set PHYSIOEX_DATA env var if --dataset_root is provided.
    # Datasets use get_data_root() + DATASET_SUBDIR to find their data,
    # so we set the env var rather than passing root= directly.
    if args.dataset_root:
        import os
        os.environ["PHYSIOEX_DATA"] = args.dataset_root

    # Filter datasets if specified
    if args.datasets:
        configs = []
        for name in args.datasets:
            if name not in DATASET_NAME_MAP:
                print(f"[WARN] Unknown dataset: {name}, skipping")
                continue
            configs.append(DATASET_NAME_MAP[name])
    else:
        configs = DATASET_CONFIGS

    print(f"\nWill process {len(configs)} datasets")

    from physioex.data.datasets import get_dataset

    for ds_name, ds_class_name, ds_kwargs in configs:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        try:
            DatasetClass = get_dataset(ds_class_name)
            dataset = DatasetClass(
                channels=channels,
                pipelines=PIPELINE,
                sequence_length=seq_len,
                **ds_kwargs,
            )
        except Exception as e:
            print(f"  [SKIP] {ds_name}: {e}")
            continue

        if dataset.get_n_subjects() == 0:
            print(f"  [SKIP] {ds_name}: no subjects")
            continue

        print(f"  Subjects: {dataset.get_n_subjects()}")

        # Extract sequence embeddings (uses model.encode() with sliding window)
        try:
            path = extract_embeddings(
                model=model,
                dataset=dataset,
                model_name=args.model_name,
                dataset_name=ds_name,
                L=seq_len,
                device=device,
                overwrite=args.overwrite,
            )
            print(f"  Embeddings: {path}")
        except Exception as e:
            print(f"  [FAIL] extract_embeddings: {e}")
            continue

        # Linear probe with per-subject predictions
        try:
            results = linear_probe(
                model_name=args.model_name,
                dataset_name=ds_name,
                device=device,
                save_predictions=True,
            )
            kappa = results["mean_std"]["kappa"]
            print(f"  Kappa: {kappa['mean']:.4f} +/- {kappa['std']:.4f}")
        except Exception as e:
            print(f"  [FAIL] linear_probe: {e}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
