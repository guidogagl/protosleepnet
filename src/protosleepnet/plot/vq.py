"""VQ method comparison for ProtoSleepNet.

Generates a figure with 2 subplots (ProtoSleepNet SEQ | ProtoSleepNet ST)
comparing K-Means, K-Means+supervised, and Learned prototype learning
methods as a function of codebook size M.

Output: figures/vq.pdf + figures/vq.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_DIR = os.path.join(BASE, "json")
FIG_DIR = os.path.join(BASE, "plot", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

M_VALUES = [5, 8, 12, 15, 24, 32, 48, 65, 80, 100]

MODELS = {
    "ProtoSleepNet (SEQ)": {
        "prefix": "proto-seq-3ch-mixer",
        "baseline_pred": os.path.join(BASE, "json", "occlusion", "predictions", "proto-seq-3ch-mixer_clean.json"),
    },
    "ProtoSleepNet (ST)": {
        "prefix": "proto-st-3ch-mixer",
        "baseline_pred": os.path.join(BASE, "json", "occlusion", "predictions", "proto-st-3ch-mixer_clean.json"),
    },
}

VQ_METHODS = {
    "K-Means": "vq_kmeans",
    "K-Means + supervised": "vq",
    "Learned (random init)": "vq_learned",
}

COLORS = {
    "K-Means": "#1f77b4",
    "K-Means + supervised": "#2ca02c",
    "Learned (random init)": "#d62728",
}
MARKERS = {
    "K-Means": "o",
    "K-Means + supervised": "s",
    "Learned (random init)": "D",
}


def load_predictions(filepath):
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data[list(data.keys())[0]]
    return data


def compute_metrics(filepath):
    subjects = load_predictions(filepath)
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
    pe = sum((preds == c).mean() * (targets == c).mean() for c in range(n_classes))
    kappa = (acc - pe) / (1 - pe + 1e-8)

    return acc, kappa


def load_metrics_json(filepath):
    """Fallback: load from metrics JSON if predictions unavailable."""
    with open(filepath) as f:
        d = json.load(f)
    acc = d.get("accuracy", 0)
    kappa = d.get("cohen_kappa", d.get("kappa", 0))
    return acc, kappa


def plot_backbone(ax, backbone_name, config):
    ax2 = ax.twinx()
    prefix = config["prefix"]
    has_data = False

    for method_label, method_dir in VQ_METHODS.items():
        accs, kappas, valid_ms = [], [], []
        pred_dir = os.path.join(JSON_DIR, method_dir, "in-domain", "predictions")
        met_dir = os.path.join(JSON_DIR, method_dir, "in-domain", "metrics")

        for M in M_VALUES:
            pred_f = os.path.join(pred_dir, f"{prefix}_m{M}.json")
            met_f = os.path.join(met_dir, f"{prefix}_m{M}.json")

            if os.path.exists(pred_f):
                acc, kappa = compute_metrics(pred_f)
            elif os.path.exists(met_f):
                acc, kappa = load_metrics_json(met_f)
            else:
                continue

            accs.append(acc * 100)
            kappas.append(kappa)
            valid_ms.append(M)

        if not accs:
            continue

        has_data = True
        color = COLORS[method_label]
        marker = MARKERS[method_label]

        ax.plot(valid_ms, accs, color=color, marker=marker, markersize=5,
                linewidth=2, label=method_label)
        ax2.plot(valid_ms, kappas, color=color, marker=marker, markersize=4,
                 linewidth=1.5, linestyle="--", alpha=0.6)

    if not has_data:
        return False

    # Baseline no-VQ
    bl_file = config["baseline_pred"]
    if os.path.exists(bl_file):
        bl_acc, bl_kappa = compute_metrics(bl_file)
        ax.axhline(bl_acc * 100, color="black", linestyle=":", linewidth=1.5,
                    alpha=0.5, label=f"No VQ ({bl_acc*100:.1f}%)")
        ax2.axhline(bl_kappa, color="gray", linestyle=":", linewidth=1, alpha=0.3)

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
    available = [name for name, cfg in MODELS.items()
                 if any(os.path.exists(os.path.join(JSON_DIR, md, "in-domain", "predictions", f"{cfg['prefix']}_m{M}.json"))
                        or os.path.exists(os.path.join(JSON_DIR, md, "in-domain", "metrics", f"{cfg['prefix']}_m{M}.json"))
                        for md in VQ_METHODS.values() for M in M_VALUES)]

    if not available:
        print("No VQ data found, skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for i, backbone in enumerate(available):
        plot_backbone(axes[0, i], backbone, MODELS[backbone])

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"vq.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
