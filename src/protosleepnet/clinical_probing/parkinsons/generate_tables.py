"""Generate CSV and LaTeX tables from probing results."""
import json
import glob
import os
import csv

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

MS = [5, 8, 12, 15, 24, 32, 48, 65, 80, 100]
MODELS = ["st", "seq"]
FEATURES = ["all", "rem_only"]
MIN_TRAIN_ACC = 0.6

TASKS = {
    "diagnosis": {
        "dir": "diagnosis",
        "label": "Diagnosis (HOA vs PD)",
        "task_subdir": "diagnosis",
        "N": 86,
        "cutoff": "HOA vs PD",
    },
    "rbdsq": {
        "dir": "rbdsq",
        "label": "RBDSQ (RBD$-$ vs RBD$+$)",
        "task_subdir": "rbdsq",
        "N": 81,
        "cutoff": "$<5$ vs $\\geq 5$",
    },
    "ahi_all": {
        "dir": "ahi_all",
        "label": "AHI (no apnea vs apnea)",
        "task_subdir": "ahi",
        "N": 86,
        "cutoff": "$<5$ vs $\\geq 5$",
    },
    "ahi_no_mild": {
        "dir": "ahi_no_mild",
        "label": "AHI (no apnea vs moderate+)",
        "task_subdir": "ahi",
        "N": 52,
        "cutoff": "$<5$ vs $\\geq 15$",
    },
    "psqi": {
        "dir": "psqi",
        "label": "PSQI (poor vs very poor)",
        "task_subdir": "psqi",
        "N": 81,
        "cutoff": "$\\leq 10$ vs $>10$",
    },
}


def load_results(task_dir, task_subdir):
    """Load all summary.json + compute AUC from predictions."""
    results = []
    pattern = os.path.join(BASE, task_dir, "*/parkinsons_night", task_subdir, "summary.json")
    for sp in sorted(glob.glob(pattern)):
        config = sp.replace(os.path.join(BASE, task_dir) + "/", "").split("/")[0]
        with open(sp) as f:
            s = json.load(f)

        train_acc = s.get("train", {}).get("accuracy_mean", 0)
        if train_acc < MIN_TRAIN_ACC:
            continue

        # Compute AUC
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

        results.append({
            "model": model, "M": M,
            "features": s.get("features", "?"),
            "acc": s["test"]["accuracy"],
            "mf1": s["test"]["f1_macro"],
            "kappa": s["test"].get("kappa", 0),
            "auc": auc,
            "train_acc": train_acc,
            "config": config,
        })
    return results


def best_per_M(results, model, feat):
    """For each M, find the config with best accuracy."""
    out = {}
    for M in MS:
        subset = [r for r in results if r["model"] == model and r["M"] == M and r["features"] == feat]
        if subset:
            out[M] = max(subset, key=lambda r: r["acc"])
    return out


def generate_per_task_csv(task_name, task_cfg):
    results = load_results(task_cfg["dir"], task_cfg["task_subdir"])
    if not results:
        print(f"  {task_name}: no results")
        return

    rows = []
    header = ["M"]
    for model in MODELS:
        for feat in FEATURES:
            header.extend([f"{model}_{feat}_Acc", f"{model}_{feat}_MF1", f"{model}_{feat}_AUC"])

    for M in MS:
        row = [M]
        for model in MODELS:
            for feat in FEATURES:
                best = best_per_M(results, model, feat)
                if M in best:
                    r = best[M]
                    row.extend([
                        f"{r['acc']:.3f}",
                        f"{r['mf1']:.3f}",
                        f"{r['auc']:.3f}" if r["auc"] else "-",
                    ])
                else:
                    row.extend(["-", "-", "-"])
        rows.append(row)

    # CSV
    csv_path = os.path.join(TABLES_DIR, f"{task_name}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {csv_path}")

    # LaTeX
    tex_path = os.path.join(TABLES_DIR, f"{task_name}_results.tex")
    with open(tex_path, "w") as f:
        n_cols = 1 + len(MODELS) * len(FEATURES) * 3
        col_spec = "r" + "|rrr" * (len(MODELS) * len(FEATURES))
        f.write(f"\\begin{{table}}[h!]\n\\centering\n\\footnotesize\n")
        f.write(f"\\caption{{{task_cfg['label']} — Accuracy / MF1 / AUC vs codebook size $M$ (LOSO, $N={task_cfg['N']}$).}}\n")
        f.write(f"\\label{{tab:{task_name}}}\n")
        f.write(f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n\\toprule\n")

        # Header row 1: model groups
        f.write("& \\multicolumn{6}{c}{Proto-ST} & \\multicolumn{6}{c}{Proto-SEQ} \\\\\n")
        f.write("\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}\n")
        # Header row 2: feature groups
        f.write("$M$")
        for model in MODELS:
            for feat in FEATURES:
                feat_label = "All" if feat == "all" else "REM"
                f.write(f" & \\multicolumn{{3}}{{c}}{{{feat_label}}}")
        f.write(" \\\\\n")
        # Header row 3: metrics
        f.write("")
        for _ in range(len(MODELS) * len(FEATURES)):
            f.write(" & Acc & MF1 & AUC")
        f.write(" \\\\\n\\midrule\n")

        # Data rows
        for row in rows:
            M = row[0]
            f.write(str(M))
            for val in row[1:]:
                if val == "-":
                    f.write(" & --")
                else:
                    f.write(f" & {val}")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  {tex_path}")


def generate_summary():
    """Cross-task summary table."""
    rows = []
    for task_name, task_cfg in TASKS.items():
        results = load_results(task_cfg["dir"], task_cfg["task_subdir"])
        if not results:
            continue
        best = max(results, key=lambda r: r["acc"])
        rows.append({
            "task": task_cfg["label"],
            "N": task_cfg["N"],
            "cutoff": task_cfg["cutoff"],
            "acc": best["acc"],
            "mf1": best["mf1"],
            "auc": best["auc"],
            "config": best["config"],
        })

    # CSV
    csv_path = os.path.join(TABLES_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task", "N", "Cutoff", "Acc", "MF1", "AUC", "Config"])
        for r in rows:
            w.writerow([r["task"], r["N"], r["cutoff"],
                        f"{r['acc']:.3f}", f"{r['mf1']:.3f}",
                        f"{r['auc']:.3f}" if r["auc"] else "-",
                        r["config"]])
    print(f"  {csv_path}")

    # LaTeX
    tex_path = os.path.join(TABLES_DIR, "summary.tex")
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Summary of clinical probing tasks on Parkinson's night recordings (LOSO CV).}\n")
        f.write("\\label{tab:summary}\n")
        f.write("\\begin{tabular}{@{}lclrrrl@{}}\n\\toprule\n")
        f.write("\\textbf{Task} & $N$ & \\textbf{Cutoff} & \\textbf{Acc} & \\textbf{MF1} & \\textbf{AUC} & \\textbf{Best config} \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            auc_str = f"{r['auc']:.3f}" if r["auc"] else "--"
            config_short = r["config"].replace("_", "\\_")
            f.write(f"{r['task']} & {r['N']} & {r['cutoff']} & {r['acc']:.3f} & {r['mf1']:.3f} & {auc_str} & \\texttt{{{config_short}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  {tex_path}")


if __name__ == "__main__":
    print("Generating per-task tables:")
    for task_name, task_cfg in TASKS.items():
        generate_per_task_csv(task_name, task_cfg)

    print("\nGenerating summary table:")
    generate_summary()
    print("\nDone.")
