"""ProtoSleepNet vs baseline VQ prototype comparison.

Generates a figure with 2 subplots (SEQ | ST) comparing how accuracy
degrades when constraining models to use M prototypes:
  - Original model (no residual) — baseline quantization loss
  - ProtoSleepNet (with residual + mixer) — improved quantization

Metrics are computed from per-subject predictions. Skips backbones
with missing data.

Output: figures/residual.pdf + figures/residual.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── Paths ────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_DIR = os.path.join(BASE, "json")
FIG_DIR = os.path.join(BASE, "plot", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

M_VALUES = [5, 8, 12, 15, 24, 32, 48, 65, 80, 100]
VQ_METHOD = "vq_kmeans"

MODELS = {
    "SeqSleepNet": {
        "SeqSleepNet-Phan (original)": "seqsleepnet-phan",
        "ProtoSleepNet (SEQ)": "proto-seq-3ch-mixer",
    },
    "SleepTransformer": {
        "SleepTransformer-Phan (original)": "sleeptransformer-phan",
        "ProtoSleepNet (ST)": "proto-st-3ch-mixer",
    },
}

# Baseline predictions (no VQ) for reference lines
BASELINE_PRED_FILES = {
    "SeqSleepNet": {
        "SeqSleepNet-Phan (original)": os.path.join(BASE, "json", "occlusion", "predictions", "seq-1ch_clean.json"),
        "ProtoSleepNet (SEQ)": os.path.join(BASE, "json", "occlusion", "predictions", "proto-seq-3ch-mixer_clean.json"),
    },
    "SleepTransformer": {
        "SleepTransformer-Phan (original)": os.path.join(BASE, "json", "occlusion", "predictions", "st-1ch_clean.json"),
        "ProtoSleepNet (ST)": os.path.join(BASE, "json", "occlusion", "predictions", "proto-st-3ch-mixer_clean.json"),
    },
}

COLORS = {
    "SeqSleepNet-Phan (original)": "#1f77b4",
    "ProtoSleepNet (SEQ)": "#d62728",
    "SleepTransformer-Phan (original)": "#1f77b4",
    "ProtoSleepNet (ST)": "#d62728",
}
MARKERS = {
    "SeqSleepNet-Phan (original)": "o",
    "ProtoSleepNet (SEQ)": "D",
    "SleepTransformer-Phan (original)": "o",
    "ProtoSleepNet (ST)": "D",
}


# ── Metrics computation ──────────────────────────────────────────────


def compute_metrics_from_predictions(filepath):
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, dict):
        key = list(data.keys())[0]
        subjects = data[key]
    else:
        subjects = data

    all_preds, all_targets = [], []
    for subj in subjects:
        proba = np.array(subj["proba"])
        labels = np.array(subj["labels"])
        preds = proba.argmax(axis=1)
        valid = labels >= 0
        all_preds.append(preds[valid])
        all_targets.append(labels[valid])

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    acc = (preds == targets).mean()
    n_classes = max(preds.max(), targets.max()) + 1
    pe = 0.0
    for c in range(n_classes):
        pe += (preds == c).mean() * (targets == c).mean()
    kappa = (acc - pe) / (1 - pe + 1e-8)

    return acc, kappa


# ── Plot ─────────────────────────────────────────────────────────────


def plot_backbone(ax, backbone_name, models_config):
    ax2 = ax.twinx()

    pred_dir = os.path.join(JSON_DIR, VQ_METHOD, "in-domain", "predictions")
    has_data = False
    all_accs, all_kappas = [], []

    for model_label, prefix in models_config.items():
        accs, kappas, valid_ms = [], [], []

        for M in M_VALUES:
            filepath = os.path.join(pred_dir, f"{prefix}_m{M}.json")
            if not os.path.exists(filepath):
                continue
            acc, kappa = compute_metrics_from_predictions(filepath)
            accs.append(acc * 100)
            kappas.append(kappa)
            valid_ms.append(M)

        if not accs:
            continue

        has_data = True
        all_accs.extend(accs)
        all_kappas.extend(kappas)
        color = COLORS[model_label]
        marker = MARKERS[model_label]

        ax.plot(valid_ms, accs, color=color, marker=marker, markersize=5,
                linewidth=2, label=model_label)
        ax2.plot(valid_ms, kappas, color=color, marker=marker, markersize=4,
                 linewidth=1.5, linestyle="--", alpha=0.6)

    if not has_data:
        return False

    # Baseline horizontal lines (no VQ)
    baselines = BASELINE_PRED_FILES.get(backbone_name, {})
    for model_label, bl_file in baselines.items():
        if os.path.exists(bl_file):
            bl_acc, bl_kappa = compute_metrics_from_predictions(bl_file)
            all_accs.append(bl_acc * 100)
            all_kappas.append(bl_kappa)
            color = COLORS[model_label]
            ax.axhline(bl_acc * 100, color=color, linestyle=":", linewidth=1.5, alpha=0.5)
            ax2.axhline(bl_kappa, color=color, linestyle=":", linewidth=1, alpha=0.3)

    ax.set_xlabel("Codebook size M", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylabel("Cohen's Kappa", fontsize=11, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.set_title(backbone_name, fontsize=13, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(M_VALUES)
    ax.set_xticklabels([str(m) for m in M_VALUES], fontsize=8)
    ax.minorticks_off()

    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 1)

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    return True


def main():
    available = []
    pred_dir = os.path.join(JSON_DIR, VQ_METHOD, "in-domain", "predictions")
    for backbone, config in MODELS.items():
        has_any = any(
            os.path.exists(os.path.join(pred_dir, f"{prefix}_m{M}.json"))
            for prefix in config.values()
            for M in M_VALUES
        )
        if has_any:
            available.append(backbone)

    if not available:
        print("No VQ data found, skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for i, backbone in enumerate(available):
        plot_backbone(axes[0, i], backbone, MODELS[backbone])

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"residual.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
