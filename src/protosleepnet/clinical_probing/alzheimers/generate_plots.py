"""Generate plots from Alzheimer probing results."""
import json, glob, os
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
LINES = [
    ("st", "all", "Proto-ST all", "#1f77b4", "-", "o"),
    ("st", "rem_only", "Proto-ST REM", "#1f77b4", "--", "s"),
    ("seq", "all", "Proto-SEQ all", "#d62728", "-", "^"),
    ("seq", "rem_only", "Proto-SEQ REM", "#d62728", "--", "D"),
]


def load_results():
    results = []
    for sp in sorted(glob.glob(os.path.join(BASE, "diagnosis/*/alzheimers/diagnosis/summary.json"))):
        config = sp.replace(os.path.join(BASE, "diagnosis") + "/", "").split("/")[0]
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
        model, M = parts[0], int(parts[1][1:])
        results.append({"model": model, "M": M, "features": s.get("features", "?"),
                        "acc": s["test"]["accuracy"], "mf1": s["test"]["f1_macro"], "auc": auc})
    return results


def main():
    results = load_results()

    fig, ax = plt.subplots(figsize=(8, 5))
    for model, feat, label, color, ls, marker in LINES:
        accs = []
        for M in MS:
            subset = [r for r in results if r["model"] == model and r["M"] == M and r["features"] == feat]
            if subset:
                accs.append(max(subset, key=lambda r: r["acc"])["acc"])
            else:
                accs.append(np.nan)
        ax.plot(MS, accs, color=color, ls=ls, marker=marker, markersize=6,
                label=label, linewidth=1.5, alpha=0.85)

    ax.set_title("Alzheimer Diagnosis (HC vs AD) — LOSO Accuracy vs $M$", fontsize=12)
    ax.set_xlabel("Codebook size $M$")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(MS)
    ax.set_xticklabels([str(m) for m in MS], fontsize=8)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(os.path.join(FIG_DIR, f"acc_vs_M{ext}"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  acc_vs_M.pdf")


if __name__ == "__main__":
    print("Generating plots:")
    main()
    print("Done.")
