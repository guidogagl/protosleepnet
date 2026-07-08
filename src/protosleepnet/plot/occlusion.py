"""ProtoSleepNet vs baseline 3ch occlusion comparison.

Generates a figure with 2 subplots (SEQ | ST) comparing ProtoSleepNet
(with channel mixer + residual) against the baseline 3-channel model
under channel occlusion scenarios.

Metrics are computed from per-subject predictions. Skips backbones
with missing data.

Output: figures/occlusion.pdf + figures/occlusion.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ── Paths ────────────────────────────────────────────────────────────

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_DIR = os.path.join(BASE, "json", "occlusion", "predictions")
FIG_DIR = os.path.join(BASE, "plot", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SCENARIOS = ["clean", "random_25", "random_50", "no_eeg", "no_eog_emg"]
SCENARIO_LABELS = ["Clean", "Rand 25%", "Rand 50%", "No EEG", "No EOG+EMG"]

MODELS = {
    "SeqSleepNet": {
        "1ch (EEG)": "seq-1ch",
        "3ch": "seq-3ch",
        "ProtoSleepNet": "proto-seq-3ch-mixer",
    },
    "SleepTransformer": {
        "1ch (EEG)": "st-1ch",
        "3ch": "st-3ch",
        "ProtoSleepNet": "proto-st-3ch-mixer",
    },
}

COLORS = {
    "1ch (EEG)": "#1f77b4",
    "3ch": "#ff7f0e",
    "ProtoSleepNet": "#d62728",
}
MARKERS = {
    "1ch (EEG)": "s",
    "3ch": "o",
    "ProtoSleepNet": "D",
}


# ── Metrics computation ──────────────────────────────────────────────


def load_predictions(filepath):
    with open(filepath) as f:
        data = json.load(f)
    if isinstance(data, dict):
        key = list(data.keys())[0]
        return data[key]
    return data


def compute_metrics_from_predictions(subjects):
    all_preds = []
    all_targets = []

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

    has_data = False
    for model_label, prefix in models_config.items():
        accs = []
        kappas = []
        valid_labels = []

        for scenario, label in zip(SCENARIOS, SCENARIO_LABELS):
            if prefix.endswith("-1ch") and scenario != "clean":
                continue

            filepath = os.path.join(PRED_DIR, f"{prefix}_{scenario}.json")
            if not os.path.exists(filepath):
                continue

            subjects = load_predictions(filepath)
            acc, kappa = compute_metrics_from_predictions(subjects)
            accs.append(acc * 100)
            kappas.append(kappa)
            valid_labels.append(label)

        if not accs:
            continue

        has_data = True
        color = COLORS[model_label]
        marker = MARKERS[model_label]
        x = [SCENARIO_LABELS.index(l) for l in valid_labels]

        if len(accs) == 1:
            ax.axhline(accs[0], color=color, linestyle="--", alpha=0.7,
                        label=f"{model_label} (ACC)")
            ax2.axhline(kappas[0], color=color, linestyle=":", alpha=0.5)
        else:
            ax.plot(x, accs, color=color, marker=marker, markersize=6,
                    linewidth=2, label=f"{model_label}")
            ax2.plot(x, kappas, color=color, marker=marker, markersize=4,
                     linewidth=1.5, linestyle="--", alpha=0.6)

    if not has_data:
        return False

    ax.set_xticks(range(len(SCENARIO_LABELS)))
    ax.set_xticklabels(SCENARIO_LABELS, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylabel("Cohen's Kappa", fontsize=11, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    ax.set_title(backbone_name, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax2.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    return True


def main():
    available = []
    for backbone, config in MODELS.items():
        has_proto = any(
            os.path.exists(os.path.join(PRED_DIR, f"{prefix}_{SCENARIOS[0]}.json"))
            for label, prefix in config.items()
            if "Proto" in label
        )
        has_baseline = any(
            os.path.exists(os.path.join(PRED_DIR, f"{prefix}_{SCENARIOS[0]}.json"))
            for label, prefix in config.items()
            if "3ch" == label
        )
        if has_proto or has_baseline:
            available.append(backbone)

    if not available:
        print("No occlusion data found, skipping.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for i, backbone in enumerate(available):
        plot_backbone(axes[0, i], backbone, MODELS[backbone])

    fig.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"occlusion.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
