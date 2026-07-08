"""Generate CSV and LaTeX tables from Alzheimer probing results."""
import json, glob, os, csv
import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(BASE, "tables")
os.makedirs(TABLES_DIR, exist_ok=True)

MS = [5, 8, 12, 15, 24, 32, 48, 65, 80, 100]
MODELS = ["st", "seq"]
FEATURES = ["all", "rem_only"]
MIN_TRAIN_ACC = 0.6


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
                        "acc": s["test"]["accuracy"], "mf1": s["test"]["f1_macro"],
                        "auc": auc, "train_acc": train_acc, "config": config})
    return results


def main():
    results = load_results()
    print(f"Loaded {len(results)} valid results")

    # Per-M table
    header = ["M"]
    for model in MODELS:
        for feat in FEATURES:
            header.extend([f"{model}_{feat}_Acc", f"{model}_{feat}_MF1", f"{model}_{feat}_AUC"])
    rows = []
    for M in MS:
        row = [M]
        for model in MODELS:
            for feat in FEATURES:
                subset = [r for r in results if r["model"] == model and r["M"] == M and r["features"] == feat]
                if subset:
                    best = max(subset, key=lambda r: r["acc"])
                    row.extend([f"{best['acc']:.3f}", f"{best['mf1']:.3f}",
                                f"{best['auc']:.3f}" if best["auc"] else "-"])
                else:
                    row.extend(["-", "-", "-"])
        rows.append(row)

    csv_path = os.path.join(TABLES_DIR, "diagnosis_results.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(rows)
    print(f"  {csv_path}")

    # LaTeX
    tex_path = os.path.join(TABLES_DIR, "diagnosis_results.tex")
    with open(tex_path, "w") as f:
        col_spec = "r" + "|rrr" * (len(MODELS) * len(FEATURES))
        f.write("\\begin{table}[h!]\n\\centering\n\\footnotesize\n")
        f.write("\\caption{Alzheimer Diagnosis (HC vs AD) --- Accuracy / MF1 / AUC vs codebook size $M$ (LOSO, $N=69$).}\n")
        f.write("\\label{tab:alz_diagnosis}\n")
        f.write(f"\\begin{{tabular}}{{@{{}}{col_spec}@{{}}}}\n\\toprule\n")
        f.write("& \\multicolumn{6}{c}{Proto-ST} & \\multicolumn{6}{c}{Proto-SEQ} \\\\\n")
        f.write("\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}\n")
        f.write("$M$")
        for _ in MODELS:
            for feat in FEATURES:
                fl = "All" if feat == "all" else "REM"
                f.write(f" & \\multicolumn{{3}}{{c}}{{{fl}}}")
        f.write(" \\\\\n")
        for _ in range(len(MODELS) * len(FEATURES)):
            f.write(" & Acc & MF1 & AUC")
        f.write(" \\\\\n\\midrule\n")
        for row in rows:
            f.write(str(row[0]))
            for val in row[1:]:
                f.write(f" & {val}" if val != "-" else " & --")
            f.write(" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  {tex_path}")

    # Summary
    best = max(results, key=lambda r: r["acc"])
    summary_csv = os.path.join(TABLES_DIR, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task", "N", "Cutoff", "Acc", "MF1", "AUC", "Config"])
        w.writerow(["Diagnosis (HC vs AD)", 69, "HC vs AD",
                     f"{best['acc']:.3f}", f"{best['mf1']:.3f}",
                     f"{best['auc']:.3f}" if best["auc"] else "-", best["config"]])
    print(f"  {summary_csv}")

    summary_tex = os.path.join(TABLES_DIR, "summary.tex")
    with open(summary_tex, "w") as f:
        auc_str = f"{best['auc']:.3f}" if best["auc"] else "--"
        config_short = best["config"].replace("_", "\\_")
        f.write("\\begin{table}[h!]\n\\centering\n")
        f.write("\\caption{Summary of Alzheimer's probing (LOSO CV).}\n\\label{tab:alz_summary}\n")
        f.write("\\begin{tabular}{@{}lclrrrl@{}}\n\\toprule\n")
        f.write("\\textbf{Task} & $N$ & \\textbf{Cutoff} & \\textbf{Acc} & \\textbf{MF1} & \\textbf{AUC} & \\textbf{Best config} \\\\\n\\midrule\n")
        f.write(f"Diagnosis (HC vs AD) & 69 & HC vs AD & {best['acc']:.3f} & {best['mf1']:.3f} & {auc_str} & \\texttt{{{config_short}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  {summary_tex}")


if __name__ == "__main__":
    main()
