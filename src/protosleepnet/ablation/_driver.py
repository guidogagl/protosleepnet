"""Shared driver for the component-ablation study (Supplementary §2).

Trains the four progressive variants between a plain multi-channel backbone and
the full ProtoSleepNet, for either backbone (SleepTransformer / SeqSleepNet):

  baseline       — per-channel backbone, concat + pool (no dropout, no mixer)
  dropout        — + random input-level channel dropout
  mixer          — + accuracy-weighted channel dropout + Transformer channel mixer
  protosleepnet  — + dual residual connections (the full PSN / PST model)

Each variant is built exactly as its standalone counterpart in ``baselines/``
(and ``train.py`` for the full model), against **physioex v2.0.0** — this driver
just provides the single ``--variant`` entry point. The older single-class
implementation (pre-``physioex`` refactor) has been retired.

Not run directly — use ``train_ablation`` (SleepTransformer) or
``train_ablation_seqsleepnet`` (SeqSleepNet).
"""
import argparse
import json
import os

import torch

from physioex.models.protosleepnet import ProtoSleepNet, ProtoSleepNetTrainer
from physioex.models.seqsleepnet import SeqSleepNet
from physioex.models.sleeptransformer import SleepTransformer
from physioex.train.trainer import Trainer

from protosleepnet.baselines.channel_dropout import ChannelDropoutWrapper
from protosleepnet.baselines.channel_mixer import MixerTrainer
from protosleepnet.baselines.train_seq_3ch import MODEL_KWARGS as SEQ_KWARGS
from protosleepnet.baselines.train_seq_3ch_mixer import build_model as build_seq_mixer
from protosleepnet.baselines.train_st_3ch import MODEL_KWARGS as ST_KWARGS
from protosleepnet.baselines.train_st_3ch_mixer import build_model as build_st_mixer
from protosleepnet.train import BACKBONE_CONFIGS, MIXER_KWARGS, build_dataset

VARIANTS = ("baseline", "dropout", "mixer", "protosleepnet")

# per-(backbone, variant) HuggingFace artifact names — physioex convention
# <model>-<primary-author>, published under 4rooms/sleep-prototypes.
HF_REPO = "4rooms/sleep-prototypes"
_BACKBONE_TAG = {"st": "sleeptransformer", "seq": "seqsleepnet"}
_PROTO_NAME = {"st": "protosleeptransformer-gagliardi", "seq": "protosleepnet-gagliardi"}
_HF_SUFFIX = {"baseline": "", "dropout": "-dropout", "mixer": "-mixer"}


def _hf_name(backbone, variant):
    if variant == "protosleepnet":
        return _PROTO_NAME[backbone]
    return f"{_BACKBONE_TAG[backbone]}-gagliardi{_HF_SUFFIX[variant]}"


def build_variant(backbone, variant, cdropout):
    """Return (model, trainer_class, config_meta) for one ablation variant."""
    plain_cls = SleepTransformer if backbone == "st" else SeqSleepNet
    plain_kwargs = ST_KWARGS if backbone == "st" else SEQ_KWARGS
    plain_path = (
        "physioex.models.sleeptransformer:SleepTransformer" if backbone == "st"
        else "physioex.models.seqsleepnet:SeqSleepNet"
    )

    if variant == "baseline":
        model = plain_cls(**plain_kwargs)
        meta = {"model_class": plain_path, "model_kwargs": plain_kwargs}
        return model, Trainer, meta

    if variant == "dropout":
        model = ChannelDropoutWrapper(plain_cls(**plain_kwargs), p=cdropout)
        meta = {
            "model_class": plain_path, "model_kwargs": plain_kwargs,
            "wrapper": "protosleepnet.baselines.channel_dropout:ChannelDropoutWrapper",
            "cdropout": cdropout,
        }
        return model, Trainer, meta

    if variant == "mixer":
        build = build_st_mixer if backbone == "st" else build_seq_mixer
        model = build(cdropout=cdropout)
        meta = {
            "model_class": "protosleepnet.baselines.channel_mixer:ChannelMixerWrapper",
            "build_fn": f"protosleepnet.baselines.train_{backbone}_3ch_mixer:build_model",
            "cdropout": cdropout,
        }
        return model, MixerTrainer, meta

    # variant == "protosleepnet" (full model, dual residual)
    cfg = BACKBONE_CONFIGS[backbone]
    factory = getattr(ProtoSleepNet, cfg["factory"])
    factory_kwargs = {"n_channels": 3, **MIXER_KWARGS, "cdropout": cdropout}
    model = factory(**factory_kwargs)
    meta = {
        "model_class": "physioex.models.protosleepnet:ProtoSleepNet",
        "factory": cfg["factory"], "factory_kwargs": factory_kwargs,
    }
    return model, ProtoSleepNetTrainer, meta


def add_args(parser):
    parser.add_argument("--variant", required=True, choices=VARIANTS + ("full",),
                        help="Ablation variant ('full' is an alias for 'protosleepnet')")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset (default: shhs for st, mass for seq)")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--cdropout", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--valid_every", type=int, default=None,
                        help="Validate every N steps (default: 100 for seq, 1000 for st)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload", action="store_true")
    return parser


def run(backbone):
    """Entry point for the two backbone-specific wrappers."""
    args = add_args(argparse.ArgumentParser(
        description=f"ProtoSleepNet component ablation ({backbone} backbone)")).parse_args()
    variant = "protosleepnet" if args.variant == "full" else args.variant

    cfg = BACKBONE_CONFIGS[backbone]
    dataset_name = args.dataset or cfg["default_dataset"]
    seq_len = cfg["seq_len"]
    channels = cfg["channels_3ch"]
    max_epochs = args.max_epochs or cfg["max_epochs"]
    valid_every = args.valid_every or (100 if backbone == "seq" else 1000)
    hf_name = _hf_name(backbone, variant)
    output_dir = args.output_dir or f"pretrained_output/{hf_name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Ablation: backbone={backbone} variant={variant} dataset={dataset_name}")

    ds_extra = cfg["dataset_kwargs"].get(dataset_name, {})
    dataset = build_dataset(dataset_name, channels, "seqsleepnet", seq_len, ds_extra)

    model, trainer_cls, meta = build_variant(backbone, variant, args.cdropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {type(model).__name__}, {n_params:,} parameters")

    n_train = len(dataset.split(fold=0)[0])
    steps_per_epoch = max(1, n_train // args.batch_size)
    valid_interval_ratio = valid_every / steps_per_epoch

    nw = args.num_workers
    model = trainer_cls.train(
        model=model, dataset=dataset, max_epochs=max_epochs, lr=args.lr,
        weight_decay=args.weight_decay, train_batch_size=args.batch_size, fold=0,
        gpu_id=args.gpu_id, checkpoint_path=os.path.join(output_dir, "checkpoints"),
        early_stopping_patience=args.early_stopping_patience,
        valid_interval_ratio=valid_interval_ratio, num_workers=nw,
        pin_memory=nw > 0, persistent_workers=nw > 0, prefetch_factor=2, seed=args.seed,
    )
    results = trainer_cls.voting_evaluate(
        model=model, dataset=dataset, L=seq_len, fold=0, gpu_id=args.gpu_id)

    torch.save(model.cpu().state_dict(), os.path.join(output_dir, "model.pt"))
    config = {
        **meta,
        "backbone": backbone, "variant": variant,
        "training": {"dataset": dataset_name, "channels": channels,
                     "sequence_length": seq_len, "max_epochs": max_epochs,
                     "lr": args.lr, "weight_decay": args.weight_decay,
                     "batch_size": args.batch_size, "seed": args.seed},
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    metrics = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in results.items()}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results: accuracy={results['accuracy']:.4f}, "
          f"f1={results['f1_score']:.4f}, kappa={results['cohen_kappa']:.4f}")

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for fname in ["model.pt", "config.json", "metrics.json"]:
            api.upload_file(
                path_or_fileobj=os.path.join(output_dir, fname),
                path_in_repo=f"{hf_name}/{fname}",
                repo_id=HF_REPO, repo_type="model")
            print(f"Uploaded {fname} to {HF_REPO}/{hf_name}/")
