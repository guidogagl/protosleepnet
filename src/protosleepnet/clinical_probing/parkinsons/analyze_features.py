"""Univariate feature analysis + stage composition for winning configs.

For each task's best configuration:
1. Computes per-prototype stage composition (stacked bar + CSV + TeX)
2. Computes per-feature univariate statistics (Cohen's d, p-value)
3. Generates forest plot of top discriminative features

Runs on Sofia (needs embeddings + GT labels).
Outputs go to tables/ and figures/ subdirectories.
"""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cdist

STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4"]
FEAT_NAMES = ["prop", "bout_mean", "bout_std"]

# Winning configs
TASKS = {
    "diagnosis": {
        "model": "protosleepnet-st-3ch-mixer", "M": 65,
        "label_field": "group", "pos_value": "PD", "neg_value": "HOA",
        "group0_label": "HOA", "group1_label": "PD",
        "filter": None,
    },
    "rbdsq": {
        "model": "protosleepnet-st-3ch-mixer", "M": 100,
        "label_field": "rbdsq_total", "cutoff": 5, "cutoff_op": ">=",
        "group0_label": "RBD$-$", "group1_label": "RBD$+$",
        "filter": None,
    },
    "ahi_all": {
        "model": "protosleepnet-seq-3ch-mixer", "M": 15,
        "label_field": "ahi", "cutoff": 5, "cutoff_op": ">=",
        "group0_label": "No apnea", "group1_label": "Apnea",
        "filter": None,
    },
    "ahi_no_mild": {
        "model": "protosleepnet-st-3ch-mixer", "M": 5,
        "label_field": "ahi", "cutoff": 15, "cutoff_op": ">=",
        "group0_label": "No apnea", "group1_label": "Mod.+sev.",
        "filter": lambda meta: meta.get("ahi") is not None and not (5 <= meta["ahi"] < 15),
    },
    "psqi": {
        "model": "protosleepnet-seq-3ch-mixer", "M": 100,
        "label_field": "psqi_total_score", "cutoff": 10, "cutoff_op": ">",
        "group0_label": "Poor", "group1_label": "Very poor",
        "filter": None,
    },
}


def get_label(meta, task_cfg):
    """Return binary label (0/1) or None if subject should be skipped."""
    if task_cfg.get("filter") and not task_cfg["filter"](meta):
        return None
    field = task_cfg["label_field"]
    val = meta.get(field)
    if val is None:
        return None
    if field == "group":
        return 1 if val == task_cfg["pos_value"] else 0
    cutoff = task_cfg["cutoff"]
    op = task_cfg["cutoff_op"]
    if op == ">=":
        return 1 if val >= cutoff else 0
    elif op == ">":
        return 1 if val > cutoff else 0
    return None


def load_and_compute(emb_base, model, M, task_cfg):
    """Load subjects, assign to codebook, compute features + stage composition."""
    cb_path = os.path.join(emb_base, model, "codebooks/vq_kmeans", f"M{M}.npy")
    codebook = np.load(cb_path).astype(np.float32)
    K = len(codebook)

    subjects = []
    all_assignments = []
    all_labels_gt = []

    for ds in [f"{model}/parkinsons_night_HOA/all", f"{model}/parkinsons_night_PD/all"]:
        emb_dir = os.path.join(emb_base, ds)
        for mp in sorted(glob.glob(os.path.join(emb_dir, "*_metadata.json"))):
            sid = os.path.basename(mp).replace("_metadata.json", "")
            with open(mp) as f:
                meta = json.load(f)

            label = get_label(meta, task_cfg)
            if label is None:
                continue

            emb_path = os.path.join(emb_dir, f"{sid}_embeddings.npy")
            lbl_path = os.path.join(emb_dir, f"{sid}_labels.npy")
            if not os.path.exists(emb_path):
                continue

            emb = np.load(emb_path).astype(np.float32)
            N = len(emb)
            if N == 0:
                continue

            assignments = cdist(emb, codebook).argmin(axis=1)

            # Features
            prop = np.zeros(K, dtype=np.float32)
            bout_mean = np.zeros(K, dtype=np.float32)
            bout_std = np.zeros(K, dtype=np.float32)
            for k in range(K):
                prop[k] = (assignments == k).sum() / N
            bouts = {k: [] for k in range(K)}
            cur, blen = assignments[0], 1
            for i in range(1, N):
                if assignments[i] == cur:
                    blen += 1
                else:
                    bouts[cur].append(blen)
                    cur, blen = assignments[i], 1
            bouts[cur].append(blen)
            for k in range(K):
                if bouts[k]:
                    bout_mean[k] = np.mean(bouts[k])
                    if len(bouts[k]) > 1:
                        bout_std[k] = np.std(bouts[k])

            subjects.append({
                "sid": sid, "label": label,
                "prop": prop, "bout_mean": bout_mean, "bout_std": bout_std,
            })

            # Stage composition
            all_assignments.append(assignments)
            if os.path.exists(lbl_path):
                gt = np.load(lbl_path).astype(np.int64)
                n = min(len(assignments), len(gt))
                all_labels_gt.append(gt[:n])
                # Pad assignments to match
                all_assignments[-1] = assignments[:n]

    # Stage composition
    all_a = np.concatenate(all_assignments)
    all_l = np.concatenate(all_labels_gt) if all_labels_gt else np.array([])

    stage_comp = np.zeros((K, 5))
    proto_counts = np.zeros(K)
    for k in range(K):
        mask = all_a == k
        proto_counts[k] = mask.sum()
        if len(all_l) > 0 and mask.sum() > 0:
            sl = all_l[mask]
            valid = sl >= 0
            sl = sl[valid]
            if len(sl) > 0:
                for s in range(5):
                    stage_comp[k, s] = (sl == s).sum() / len(sl) * 100

    return subjects, K, stage_comp, proto_counts


def analyze_task(task_name, task_cfg, emb_base, out_tables, out_figures):
    """Run full analysis for one task."""
    print(f"\n{'='*60}")
    print(f"  {task_name}: {task_cfg['model']} M={task_cfg['M']}")
    print(f"{'='*60}")

    subjects, K, stage_comp, proto_counts = load_and_compute(
        emb_base, task_cfg["model"], task_cfg["M"], task_cfg)

    n0 = sum(1 for s in subjects if s["label"] == 0)
    n1 = sum(1 for s in subjects if s["label"] == 1)
    print(f"  Subjects: {len(subjects)} ({task_cfg['group0_label']}={n0}, {task_cfg['group1_label']}={n1})")

    dominant_stage = [STAGES[int(np.argmax(stage_comp[k]))] if stage_comp[k].sum() > 0 else "?"
                      for k in range(K)]

    # ── Stage composition CSV ──
    csv_path = os.path.join(out_tables, f"{task_name}_stage_composition.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Proto", "W%", "N1%", "N2%", "N3%", "REM%", "N_epochs", "Dominant"])
        for k in range(K):
            w.writerow([k] + [f"{stage_comp[k, s]:.1f}" for s in range(5)] +
                       [int(proto_counts[k]), dominant_stage[k]])
    print(f"  {csv_path}")

    # ── Stage composition LaTeX ──
    tex_path = os.path.join(out_tables, f"{task_name}_stage_composition.tex")
    with open(tex_path, "w") as f:
        f.write(f"\\begin{{table}}[h!]\n\\centering\\footnotesize\n")
        f.write(f"\\caption{{Stage composition per prototype ({task_name}, $M={task_cfg['M']}$).}}\n")
        f.write(f"\\label{{tab:{task_name}_stages}}\n")
        f.write("\\begin{tabular}{@{}rrrrrrrl@{}}\n\\toprule\n")
        f.write("Proto & W\\% & N1\\% & N2\\% & N3\\% & REM\\% & $N$ & Dom. \\\\\n\\midrule\n")
        order = np.argsort(-stage_comp[:, 4])  # sort by %REM
        for k in order:
            if proto_counts[k] == 0:
                continue
            f.write(f"{k}")
            for s in range(5):
                f.write(f" & {stage_comp[k, s]:.1f}")
            f.write(f" & {int(proto_counts[k])} & {dominant_stage[k]} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  {tex_path}")

    # ── Stage composition stacked bar plot ──
    fig, ax = plt.subplots(figsize=(max(8, K * 0.35), 4))
    order = np.argsort(-stage_comp[:, 4])
    bottom = np.zeros(K)
    for s in range(5):
        ax.bar(range(K), stage_comp[order, s], bottom=bottom,
               color=STAGE_COLORS[s], label=STAGES[s], edgecolor="white", linewidth=0.3)
        bottom += stage_comp[order, s]
    ax.set_xticks(range(K))
    ax.set_xticklabels([str(order[i]) for i in range(K)], fontsize=6)
    ax.set_xlabel("Prototype (sorted by %REM)")
    ax.set_ylabel("Stage %")
    ax.set_title(f"{task_name} — M={task_cfg['M']} stage composition")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    for ext in [".pdf", ".png"]:
        plt.savefig(os.path.join(out_figures, f"{task_name}_stage_composition{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {task_name}_stage_composition.pdf")

    # ── Feature analysis ──
    idx0 = [i for i, s in enumerate(subjects) if s["label"] == 0]
    idx1 = [i for i, s in enumerate(subjects) if s["label"] == 1]

    rows = []
    for k in range(K):
        for fn in FEAT_NAMES:
            v0 = np.array([subjects[i][fn][k] for i in idx0])
            v1 = np.array([subjects[i][fn][k] for i in idx1])
            if v0.std() == 0 and v1.std() == 0:
                continue
            _, p = stats.ttest_ind(v0, v1, equal_var=False)
            ps = np.sqrt((v0.std()**2 + v1.std()**2) / 2)
            d = (v0.mean() - v1.mean()) / ps if ps > 0 else 0
            rows.append({
                "proto": k, "feature": fn, "stage": dominant_stage[k],
                "mean0": v0.mean(), "std0": v0.std(),
                "mean1": v1.mean(), "std1": v1.std(),
                "d": d, "p": p,
            })

    rows.sort(key=lambda r: -abs(r["d"]))
    sig_rows = [r for r in rows if abs(r["d"]) >= 0.3]

    # Feature CSV
    csv_path = os.path.join(out_tables, f"{task_name}_features.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Proto", "Feature", "Stage",
                     f"{task_cfg['group0_label']} mean", f"{task_cfg['group0_label']} std",
                     f"{task_cfg['group1_label']} mean", f"{task_cfg['group1_label']} std",
                     "Cohen_d", "p_value"])
        for r in sig_rows:
            w.writerow([r["proto"], r["feature"], r["stage"],
                        f"{r['mean0']:.4f}", f"{r['std0']:.4f}",
                        f"{r['mean1']:.4f}", f"{r['std1']:.4f}",
                        f"{r['d']:+.3f}", f"{r['p']:.4f}"])
    print(f"  {csv_path} ({len(sig_rows)} features)")

    # Feature LaTeX
    tex_path = os.path.join(out_tables, f"{task_name}_features.tex")
    with open(tex_path, "w") as f:
        g0, g1 = task_cfg["group0_label"], task_cfg["group1_label"]
        f.write(f"\\begin{{table}}[h!]\n\\centering\\footnotesize\n")
        f.write(f"\\caption{{Top discriminative features ({task_name}, $|d|\\geq 0.3$).}}\n")
        f.write(f"\\label{{tab:{task_name}_features}}\n")
        f.write("\\begin{tabular}{@{}rlllrr@{}}\n\\toprule\n")
        f.write(f"Proto & Feature & Stage & {g0} & {g1} & $d$ \\\\\n\\midrule\n")
        for r in sig_rows[:20]:
            sig = "^{***}" if r["p"] < 0.001 else "^{**}" if r["p"] < 0.01 else "^{*}" if r["p"] < 0.05 else ""
            f.write(f"{r['proto']} & {r['feature']} & {r['stage']} & "
                    f"{r['mean0']:.3f}$\\pm${r['std0']:.3f} & "
                    f"{r['mean1']:.3f}$\\pm${r['std1']:.3f} & "
                    f"${r['d']:+.2f}{sig}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  {tex_path}")

    # Forest plot
    top = sig_rows[:15]
    if top:
        fig, ax = plt.subplots(figsize=(8, max(3, len(top) * 0.35)))
        y = range(len(top))
        ds = [r["d"] for r in top]
        ps = [r["p"] for r in top]
        colors = ["#d62728" if p < 0.001 else "#ff7f0e" if p < 0.01 else "#2ca02c" if p < 0.05 else "#7f7f7f"
                  for p in ps]
        labels_y = [f"P{r['proto']}_{r['feature']} ({r['stage']})" for r in top]

        ax.barh(y, ds, color=colors, height=0.6, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels_y, fontsize=8)
        ax.set_xlabel("Cohen's d")
        ax.set_title(f"{task_name} — Top discriminative features")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis="x")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(color="#d62728", label="p < 0.001"),
                           Patch(color="#ff7f0e", label="p < 0.01"),
                           Patch(color="#2ca02c", label="p < 0.05"),
                           Patch(color="#7f7f7f", label="n.s.")]
        ax.legend(handles=legend_elements, fontsize=7, loc="lower right")

        plt.tight_layout()
        for ext in [".pdf", ".png"]:
            plt.savefig(os.path.join(out_figures, f"{task_name}_features{ext}"),
                        dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {task_name}_features.pdf")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_base", required=True, help="Base embeddings directory")
    parser.add_argument("--output_dir", required=True, help="Output directory (with tables/ and figures/)")
    args = parser.parse_args()

    out_tables = os.path.join(args.output_dir, "tables")
    out_figures = os.path.join(args.output_dir, "figures")
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_figures, exist_ok=True)

    for task_name, task_cfg in TASKS.items():
        analyze_task(task_name, task_cfg, args.emb_base, out_tables, out_figures)

    print("\nAll done.")


if __name__ == "__main__":
    main()
