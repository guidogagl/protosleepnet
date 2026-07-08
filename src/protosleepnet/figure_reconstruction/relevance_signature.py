import os
"""Combined spectral signature + relevance analysis.

For each prototype, overlays the spectral content with the IG relevance
maps to answer: which spectral features actually drive prototype proximity?

Usage:
    python relevance_signature.py /path/to/reconstructions/M24/protosleepnet-seq-3ch-mixer/
    python relevance_signature.py /path/to/reconstructions/M24/protosleepnet-seq-3ch-mixer/ --method data_driven
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

PHYSIOEX_ROOT = None
for candidate in [
    Path(os.environ.get("PHYSIOEX_ROOT", "")),
    Path(os.environ.get("PHYSIOEX_ROOT", "")),
    Path(os.environ.get("PHYSIOEX_ROOT", "")),
]:
    if (candidate / "physioex").is_dir():
        PHYSIOEX_ROOT = candidate
        break
if PHYSIOEX_ROOT:
    sys.path.insert(0, str(PHYSIOEX_ROOT))

from physioex.explain.foundational.sleep_bands import SLEEP_BANDS, bands_to_bin_ranges

# ── Constants ────────────────────────────────────────────────────────

FS, NFFT = 100.0, 256
N_FREQ_BINS = NFFT // 2 + 1
N_TIME_FRAMES = 29
CHANNEL_NAMES = ["EEG", "EOG", "EMG"]
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "W": "#e6ab02", "N1": "#66a61e", "N2": "#377eb8",
    "N3": "#7570b3", "REM": "#e7298a",
}
STAGE_ORDER = STAGE_NAMES

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.linewidth": 0.5, "axes.labelsize": 6.5,
    "axes.titlesize": 7, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
    "xtick.major.width": 0.4, "ytick.major.width": 0.4,
    "legend.fontsize": 5, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})


def savefig(fig, path_stem):
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {path_stem}.{{pdf,png}}")


def db_to_linear(x): return np.power(10.0, x / 10.0)
def freq_axis(): return np.arange(N_FREQ_BINS) * (FS / NFFT)
def get_bin_ranges(): return bands_to_bin_ranges(SLEEP_BANDS, fs=FS, signal_length=NFFT)


# ── Data loading ─────────────────────────────────────────────────────

def load_prototype_data(base_dir, k):
    """Load spectral + relevance data for one prototype."""
    proto = base_dir / f"proto_{k:03d}"
    epochs = np.load(proto / "epochs.npy")  # (N, C, T, F)
    meta_spec = {}
    spec_meta_path = base_dir / "spectral_analysis" / "statistics" / f"proto_{k:03d}.json"
    if spec_meta_path.exists():
        with open(spec_meta_path) as f:
            meta_spec = json.load(f)

    rel_dir = base_dir / "relevance" / f"proto_{k:03d}"
    relevance = {}
    if rel_dir.exists():
        rel_meta_path = rel_dir / "metadata.json"
        if rel_meta_path.exists():
            with open(rel_meta_path) as f:
                relevance["meta"] = json.load(f)
        attr_mean = rel_dir / "attr_mean.npy"
        if attr_mean.exists():
            relevance["attr_mean"] = np.load(attr_mean)  # (C, T, F)
        for stage in STAGE_NAMES:
            p = rel_dir / f"attr_vs_{stage}.npy"
            if p.exists():
                relevance[f"attr_vs_{stage}"] = np.load(p)
        for fname in ["band_relevance.npy", "temporal_relevance.npy", "channel_relevance.npy"]:
            p = rel_dir / fname
            if p.exists():
                relevance[fname.replace(".npy", "")] = np.load(p)

    return {
        "epochs": epochs,
        "meta_spec": meta_spec,
        "relevance": relevance,
    }


def load_all_prototypes(base_dir):
    """Load data for all prototypes in a method directory."""
    base_dir = Path(base_dir)
    protos = {}
    for pd in sorted(base_dir.glob("proto_*")):
        if not (pd / "epochs.npy").exists():
            continue
        k = int(pd.name.split("_")[1])
        protos[k] = load_prototype_data(base_dir, k)
    return protos


# ── Analysis ─────────────────────────────────────────────────────────

def compute_combined_stats(protos, bin_ranges):
    """Compute combined spectral + relevance statistics per prototype."""
    band_names = [b[0] for b in bin_ranges]
    n_bands = len(bin_ranges)
    stats = []

    for k in sorted(protos.keys()):
        p = protos[k]
        epochs = p["epochs"]
        rel = p["relevance"]
        meta = p["meta_spec"]

        eeg_lin = db_to_linear(epochs[:, 0])  # (N, T, F)

        # Spectral: band power (EEG)
        band_power = np.zeros(n_bands)
        for bi, (_, bstart, bend) in enumerate(bin_ranges):
            band_power[bi] = eeg_lin[:, :, bstart:bend].sum(axis=-1).mean()

        # Relevance: band relevance (EEG, overall mean)
        band_rel = np.zeros(n_bands)
        if "attr_mean" in rel:
            attr_eeg = rel["attr_mean"][0]  # (T, F) EEG channel
            for bi, (_, bstart, bend) in enumerate(bin_ranges):
                band_rel[bi] = np.abs(attr_eeg[:, bstart:bend]).sum()

        # Per-contrast band relevance
        contrast_band_rel = {}
        if "meta" in rel:
            for stage in rel["meta"].get("contrast_stages", []):
                key = f"attr_vs_{stage}"
                if key in rel:
                    attr_c = rel[key][0]  # (T, F) EEG
                    cbr = np.zeros(n_bands)
                    for bi, (_, bstart, bend) in enumerate(bin_ranges):
                        cbr[bi] = np.abs(attr_c[:, bstart:bend]).sum()
                    contrast_band_rel[stage] = cbr

        # Relevance-weighted power: which bands are both powerful AND relevant
        bp_norm = band_power / (band_power.sum() + 1e-12)
        br_norm = band_rel / (band_rel.sum() + 1e-12)
        rw_power = bp_norm * br_norm  # multiplicative combination

        # Dominant stage: prefer spectral analysis GT, fallback to relevance prediction
        dominant = meta.get("dominant_stage", None)
        purity = meta.get("label_purity", 0)
        if dominant is None or dominant == "N/A":
            rel_meta = rel.get("meta", {})
            dominant = rel_meta.get("dominant_class", "N/A")
            purity = 0  # no GT purity for optimized methods

        stats.append({
            "idx": k,
            "dominant_stage": dominant,
            "label_purity": purity,
            "band_names": band_names,
            "band_power": band_power,
            "band_relevance": band_rel,
            "band_power_norm": bp_norm,
            "band_relevance_norm": br_norm,
            "relevance_weighted_power": rw_power,
            "contrast_band_rel": contrast_band_rel,
            "has_relevance": "attr_mean" in rel,
            "attr_mean": rel.get("attr_mean"),
            "channel_relevance": rel.get("channel_relevance"),
        })

    return stats


def sort_by_stage(stats):
    order = sorted(range(len(stats)), key=lambda i: (
        STAGE_ORDER.index(stats[i]["dominant_stage"]) if stats[i]["dominant_stage"] in STAGE_ORDER else 5,
        -stats[i]["label_purity"]
    ))
    return order


# ── Figures ──────────────────────────────────────────────────────────

def fig_power_vs_relevance(stats, out_dir, method_name=""):
    """Side-by-side heatmaps: spectral power vs IG relevance (EEG bands)."""
    order = sort_by_stage(stats)
    M = len(stats)
    band_names = stats[0]["band_names"]
    n_bands = len(band_names)

    fig, (ax_pow, ax_rel, ax_rw) = plt.subplots(
        1, 3, figsize=(7.2, 3.8),
        gridspec_kw={"wspace": 0.25, "width_ratios": [1, 1, 1]}
    )

    # Build matrices
    pow_mat = np.array([stats[order[i]]["band_power_norm"] for i in range(M)])
    rel_mat = np.array([stats[order[i]]["band_relevance_norm"] for i in range(M)])
    rw_mat = np.array([stats[order[i]]["relevance_weighted_power"] for i in range(M)])

    # Normalize per band for z-scoring
    def zscore(mat):
        return (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-12)

    for ax, mat, title, cmap in [
        (ax_pow, zscore(pow_mat), "Spectral power", "Blues"),
        (ax_rel, zscore(rel_mat), "IG relevance", "Oranges"),
        (ax_rw, zscore(rw_mat), "Power × Relevance", "RdPu"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest",
                       vmin=-2.5, vmax=2.5)
        ax.set_xticks(range(n_bands))
        ax.set_xticklabels([b.replace("_", "\n") for b in band_names],
                           fontsize=4, rotation=45, ha="right")
        ax.set_yticks(range(M))
        ax.set_yticklabels([f"P{stats[order[i]]['idx']}" for i in range(M)], fontsize=4)
        ax.set_title(title, fontsize=7, fontweight="bold")

        if ax == ax_pow:
            ax.set_ylabel("Prototype", fontsize=6.5)
            for i in range(M):
                s = stats[order[i]]
                ax.plot(-0.8, i, "s", color=STAGE_COLORS.get(s["dominant_stage"], "gray"),
                        markersize=3, clip_on=False)

    handles = [Line2D([0], [0], marker="s", color="w",
                      markerfacecolor=STAGE_COLORS[s], markersize=4, label=s)
               for s in STAGE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=5,
               bbox_to_anchor=(0.5, -0.02))

    if method_name:
        fig.suptitle(f"EEG band analysis — {method_name}", fontsize=8, fontweight="bold")

    savefig(fig, out_dir / "fig_power_vs_relevance")


def fig_contrastive_relevance(stats, out_dir, method_name=""):
    """For each dominant stage group, show which bands are most relevant
    vs each other stage (contrastive decomposition)."""
    order = sort_by_stage(stats)
    band_names = stats[0]["band_names"]
    n_bands = len(band_names)

    # Group prototypes by dominant stage
    stage_groups = {s: [] for s in STAGE_ORDER}
    for s in stats:
        if s["dominant_stage"] in stage_groups and s["has_relevance"]:
            stage_groups[s["dominant_stage"]].append(s)

    active_stages = [s for s in STAGE_ORDER if stage_groups[s]]
    n_stages = len(active_stages)
    if n_stages == 0:
        print("    No stages with relevance data, skipping contrastive figure")
        return

    fig, axes = plt.subplots(1, n_stages, figsize=(7.2, 3.0),
                             gridspec_kw={"wspace": 0.35})
    if n_stages == 1:
        axes = [axes]

    for si, stage in enumerate(active_stages):
        ax = axes[si]
        group = stage_groups[stage]

        # Average contrastive band relevance across prototypes in this group
        contrast_stages = [s for s in STAGE_ORDER if s != stage]
        n_contrasts = len(contrast_stages)
        avg_cbr = np.zeros((n_contrasts, n_bands))
        count = 0

        for proto in group:
            cbr = proto["contrast_band_rel"]
            if cbr:
                for ci, cs in enumerate(contrast_stages):
                    if cs in cbr:
                        avg_cbr[ci] += cbr[cs]
                count += 1

        if count > 0:
            avg_cbr /= count

        # Normalize per contrast row
        row_sums = avg_cbr.sum(axis=1, keepdims=True) + 1e-12
        avg_cbr_norm = avg_cbr / row_sums

        im = ax.imshow(avg_cbr_norm, aspect="auto", cmap="YlOrRd",
                       interpolation="nearest", vmin=0)
        ax.set_xticks(range(n_bands))
        ax.set_xticklabels([b.replace("_", "\n") for b in band_names],
                           fontsize=3.5, rotation=45, ha="right")
        ax.set_yticks(range(n_contrasts))
        ax.set_yticklabels([f"vs {s}" for s in contrast_stages], fontsize=5)
        ax.set_title(f"{stage} ({len(group)} protos)",
                     fontsize=7, fontweight="bold",
                     color=STAGE_COLORS.get(stage, "gray"))
        if si == 0:
            ax.set_ylabel("Contrast", fontsize=6.5)

    if method_name:
        fig.suptitle(f"Contrastive band relevance — {method_name}",
                     fontsize=8, fontweight="bold")

    savefig(fig, out_dir / "fig_contrastive_relevance")


def fig_channel_relevance(stats, out_dir, method_name=""):
    """Stacked bar chart: EEG vs EOG vs EMG relevance per prototype."""
    order = sort_by_stage(stats)
    M = len(stats)

    fig, ax = plt.subplots(figsize=(7.2, 2.5))

    eeg_vals, eog_vals, emg_vals = [], [], []
    for i in range(M):
        s = stats[order[i]]
        cr = s.get("channel_relevance")
        if cr is not None and len(cr) > 0:
            # Average across contrasts
            cr_mean = cr.mean(axis=0)  # (3,)
            total = cr_mean.sum() + 1e-12
            eeg_vals.append(cr_mean[0] / total)
            eog_vals.append(cr_mean[1] / total)
            emg_vals.append(cr_mean[2] / total)
        else:
            eeg_vals.append(0)
            eog_vals.append(0)
            emg_vals.append(0)

    x = np.arange(M)
    ax.bar(x, eeg_vals, color="#1f77b4", label="EEG", width=0.8, edgecolor="white", linewidth=0.3)
    ax.bar(x, eog_vals, bottom=eeg_vals, color="#ff7f0e", label="EOG", width=0.8, edgecolor="white", linewidth=0.3)
    bottoms = [e + o for e, o in zip(eeg_vals, eog_vals)]
    ax.bar(x, emg_vals, bottom=bottoms, color="#2ca02c", label="EMG", width=0.8, edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    labels = []
    for i in range(M):
        s = stats[order[i]]
        labels.append(f"P{s['idx']}\n{s['dominant_stage']}")
    ax.set_xticklabels(labels, fontsize=3.5)
    ax.set_ylabel("Fraction of total |relevance|", fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5, ncol=3, loc="upper right")
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.3)

    if method_name:
        ax.set_title(f"Channel relevance decomposition — {method_name}",
                     fontsize=7, fontweight="bold")

    savefig(fig, out_dir / "fig_channel_relevance")


def fig_relevance_atlas(protos, stats, out_dir, method_name=""):
    """6×4 grid: mean EEG spectrogram with relevance overlay."""
    order = sort_by_stage(stats)
    M = len(stats)
    freqs = freq_axis()
    bin_ranges = get_bin_ranges()
    band_freqs = [(FS / NFFT) * bstart for _, bstart, _ in bin_ranges]

    n_cols = 4
    n_rows = (M + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 1.8 * n_rows),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30})
    axes_flat = axes.flatten()

    for pi in range(len(axes_flat)):
        ax = axes_flat[pi]
        if pi >= M:
            ax.axis("off")
            continue

        s = stats[order[pi]]
        k = s["idx"]
        epochs = protos[k]["epochs"]

        # Mean EEG spectrogram (dB)
        mean_spec = epochs[:, 0].mean(axis=0)  # (T, F)

        # Show spectrogram
        ax.imshow(mean_spec.T, aspect="auto", origin="lower", cmap="viridis",
                  interpolation="nearest", extent=[0, N_TIME_FRAMES, 0, 50],
                  alpha=0.8)

        # Overlay relevance contours (if available)
        if s["attr_mean"] is not None:
            attr_eeg = s["attr_mean"][0]  # (T, F)
            # Smooth slightly for cleaner contours
            from scipy.ndimage import gaussian_filter
            attr_smooth = gaussian_filter(attr_eeg.T, sigma=0.8)
            # Overlay as contour
            T_ax = np.linspace(0, N_TIME_FRAMES, attr_eeg.shape[0])
            F_ax = np.linspace(0, 50, attr_eeg.shape[1])
            ax.contour(T_ax, F_ax, attr_smooth,
                       levels=5, colors="red", linewidths=0.4, alpha=0.7)

        stage = s["dominant_stage"]
        purity = s["label_purity"]
        ax.set_title(f"P{k} | {stage} ({purity:.0%})",
                     fontsize=4.5, color=STAGE_COLORS.get(stage, "gray"),
                     fontweight="bold")

        if pi % n_cols == 0:
            ax.set_ylabel("Hz", fontsize=4.5)
        else:
            ax.set_yticklabels([])
        if pi >= M - n_cols:
            ax.set_xlabel("Time (s)", fontsize=4.5)
        else:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=3.5)
        ax.set_ylim(0, 45)

    if method_name:
        fig.suptitle(f"Spectrogram + relevance overlay — {method_name}",
                     fontsize=8, fontweight="bold", y=1.01)

    savefig(fig, out_dir / "fig_relevance_atlas")


def fig_cross_method_comparison(all_stats, out_dir):
    """Compare band relevance profiles across data_driven, model_driven, hybrid."""
    methods = list(all_stats.keys())
    if len(methods) < 2:
        return

    band_names = all_stats[methods[0]][0]["band_names"]
    n_bands = len(band_names)

    # Get prototype indices sorted by data_driven stage
    dd_stats = all_stats.get("data_driven", all_stats[methods[0]])
    order = sort_by_stage(dd_stats)
    M = len(dd_stats)

    fig, axes = plt.subplots(1, len(methods), figsize=(7.2, 3.8),
                             gridspec_kw={"wspace": 0.15})
    if len(methods) == 1:
        axes = [axes]

    for mi, method in enumerate(methods):
        ax = axes[mi]
        stats = all_stats[method]
        # Match ordering from dd
        idx_to_pos = {s["idx"]: i for i, s in enumerate(stats)}

        rel_mat = np.zeros((M, n_bands))
        for oi, dd_i in enumerate(order):
            k = dd_stats[dd_i]["idx"]
            if k in idx_to_pos:
                s = stats[idx_to_pos[k]]
                br = s["band_relevance_norm"]
                rel_mat[oi] = br

        z = (rel_mat - rel_mat.mean(axis=0)) / (rel_mat.std(axis=0) + 1e-12)
        im = ax.imshow(z, aspect="auto", cmap="Oranges", interpolation="nearest",
                       vmin=-2, vmax=3)
        ax.set_xticks(range(n_bands))
        ax.set_xticklabels([b.replace("_", "\n") for b in band_names],
                           fontsize=3.5, rotation=45, ha="right")
        ax.set_title(method.replace("_", " ").title(), fontsize=7, fontweight="bold")

        if mi == 0:
            ax.set_yticks(range(M))
            ax.set_yticklabels([f"P{dd_stats[order[i]]['idx']}" for i in range(M)], fontsize=4)
            ax.set_ylabel("Prototype", fontsize=6.5)
            for i in range(M):
                s = dd_stats[order[i]]
                ax.plot(-0.8, i, "s",
                        color=STAGE_COLORS.get(s["dominant_stage"], "gray"),
                        markersize=3, clip_on=False)
        else:
            ax.set_yticks(range(M))
            ax.set_yticklabels([])

    handles = [Line2D([0], [0], marker="s", color="w",
                      markerfacecolor=STAGE_COLORS[s], markersize=4, label=s)
               for s in STAGE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=5,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Band relevance across reconstruction methods (EEG)",
                 fontsize=8, fontweight="bold")

    savefig(fig, out_dir / "fig_cross_method_relevance")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined spectral signature + relevance analysis"
    )
    parser.add_argument("model_dir", type=Path,
                        help="Path to reconstructions/M24/{model}/ directory")
    parser.add_argument("--method", type=str, default=None,
                        choices=["data_driven", "model_driven", "hybrid"],
                        help="Analyze one method only (default: all)")
    args = parser.parse_args()

    model_dir = args.model_dir
    bin_ranges = get_bin_ranges()

    if args.method:
        methods = [args.method]
    else:
        methods = [d.name for d in sorted(model_dir.iterdir())
                   if d.is_dir() and (d / "proto_000" / "epochs.npy").exists()]

    out_dir = model_dir / "relevance_signature_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}

    for method in methods:
        method_dir = model_dir / method
        if not method_dir.exists():
            print(f"Skipping {method}: directory not found")
            continue

        print(f"\nAnalyzing: {method}")
        protos = load_all_prototypes(method_dir)
        if not protos:
            print(f"  No prototypes found, skipping")
            continue

        stats = compute_combined_stats(protos, bin_ranges)
        all_stats[method] = stats

        method_out = out_dir / method
        method_out.mkdir(parents=True, exist_ok=True)

        print(f"  Generating figures for {method}...")
        fig_power_vs_relevance(stats, method_out, method.replace("_", " ").title())
        fig_contrastive_relevance(stats, method_out, method.replace("_", " ").title())
        fig_channel_relevance(stats, method_out, method.replace("_", " ").title())
        fig_relevance_atlas(protos, stats, method_out, method.replace("_", " ").title())

    # Cross-method comparison
    if len(all_stats) > 1:
        print("\nGenerating cross-method comparison...")
        fig_cross_method_comparison(all_stats, out_dir)

    # Save combined stats
    stats_dir = out_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    for method, stats in all_stats.items():
        summary = []
        for s in stats:
            summary.append({
                "idx": s["idx"],
                "dominant_stage": s["dominant_stage"],
                "label_purity": s["label_purity"],
                "band_power_norm": s["band_power_norm"].tolist(),
                "band_relevance_norm": s["band_relevance_norm"].tolist(),
                "relevance_weighted_power": s["relevance_weighted_power"].tolist(),
                "band_names": s["band_names"],
            })
        with open(stats_dir / f"{method}.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nDone. Output at {out_dir}/")


if __name__ == "__main__":
    main()
