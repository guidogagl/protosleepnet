"""Generate plots from probing results."""
import json
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MS = [5, 8, 12, 15, 24, 32, 48, 65, 80, 100]
MIN_TRAIN_ACC = 0.6

TASKS = {
    "diagnosis": {"dir": "diagnosis", "sub": "diagnosis", "label": "Diagnosis (HOA vs PD)"},
    "rbdsq": {"dir": "rbdsq", "sub": "rbdsq", "label": "RBDSQ (RBD screening)"},
    "ahi_all": {"dir": "ahi_all", "sub": "ahi", "label": "AHI (all, <5 vs ≥5)"},
    "ahi_no_mild": {"dir": "ahi_no_mild", "sub": "ahi", "label": "AHI (no mild, <5 vs ≥15)"},
    "psqi": {"dir": "psqi", "sub": "psqi", "label": "PSQI (poor vs very poor)"},
}

LINES = [
    ("st", "all", "Proto-ST all", "#1f77b4", "-", "o"),
    ("st", "rem_only", "Proto-ST REM", "#1f77b4", "--", "s"),
    ("seq", "all", "Proto-SEQ all", "#d62728", "-", "^"),
    ("seq", "rem_only", "Proto-SEQ REM", "#d62728", "--", "D"),
]


def load_results(task_dir, task_sub):
    results = []
    pattern = os.path.join(BASE, task_dir, "*/parkinsons_night", task_sub, "summary.json")
    for sp in sorted(glob.glob(pattern)):
        config = sp.replace(os.path.join(BASE, task_dir) + "/", "").split("/")[0]
        with open(sp) as f:
            s = json.load(f)
        train_acc = s.get("train", {}).get("accuracy_mean", 0)
        if train_acc < MIN_TRAIN_ACC:
            continue

        pred_path = os.path.join(os.path.dirname(sp), "predictions.json")
        auc = None
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                preds = json.load(f)
            yt = [p["y_true"] for p in preds.values()]
            yp = [p["y_proba"] for p in preds.values()]
            if len(set(yt)) == 2:
                auc = roc_auc_score(yt, yp)

        parts = config.split("_")
        model = parts[0]
        M = int(parts[1][1:])
        results.append({"model": model, "M": M, "features": s.get("features", "?"),
                        "acc": s["test"]["accuracy"], "mf1": s["test"]["f1_macro"], "auc": auc})
    return results


def plot_acc_vs_M():
    """Accuracy vs codebook size M for all tasks."""
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.5), sharey=True)

    for idx, (task_name, task_cfg) in enumerate(TASKS.items()):
        ax = axes[idx]
        results = load_results(task_cfg["dir"], task_cfg["sub"])

        for model, feat, label, color, ls, marker in LINES:
            accs = []
            for M in MS:
                subset = [r for r in results if r["model"] == model and r["M"] == M and r["features"] == feat]
                if subset:
                    best = max(subset, key=lambda r: r["acc"])
                    accs.append(best["acc"])
                else:
                    accs.append(np.nan)
            ax.plot(MS, accs, color=color, ls=ls, marker=marker, markersize=5,
                    label=label, linewidth=1.5, alpha=0.85)

        ax.set_title(task_cfg["label"], fontsize=10)
        ax.set_xlabel("Codebook size $M$")
        ax.set_xticks(MS)
        ax.set_xticklabels([str(m) for m in MS], fontsize=7, rotation=45)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Accuracy (LOSO)")

    axes[-1].legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(os.path.join(FIG_DIR, f"acc_vs_M{ext}"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  acc_vs_M.pdf")


def plot_summary_bar():
    """Summary bar chart: best Acc/MF1/AUC per task."""
    task_names = []
    accs, mf1s, aucs = [], [], []

    for task_name, task_cfg in TASKS.items():
        results = load_results(task_cfg["dir"], task_cfg["sub"])
        if not results:
            continue
        best = max(results, key=lambda r: r["acc"])
        task_names.append(task_cfg["label"].split("(")[0].strip())
        accs.append(best["acc"])
        mf1s.append(best["mf1"])
        aucs.append(best["auc"] if best["auc"] else 0)

    x = np.arange(len(task_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, accs, width, label="Accuracy", color="#1f77b4")
    ax.bar(x, mf1s, width, label="Macro-F1", color="#ff7f0e")
    ax.bar(x + width, aucs, width, label="AUC", color="#2ca02c")

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.legend()
    ax.set_title("Clinical probing: best results per task (LOSO CV)")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(os.path.join(FIG_DIR, f"task_summary{ext}"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  task_summary.pdf")


if __name__ == "__main__":
    print("Generating plots:")
    plot_acc_vs_M()
    plot_summary_bar()
    print("Done.")
