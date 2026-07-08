import os
"""Spectral signature analysis of VQ prototype reconstructions.

Computes frequency-domain statistics, temporal concentration metrics,
and monosemanticity evaluation for each prototype cluster. Generates
Nature-style figures.

Usage:
    python spectral_signature.py /path/to/data_driven/
    python spectral_signature.py /path/to/data_driven/ --no-figures
    python spectral_signature.py /path/to/data_driven/ --figures-only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Import sleep band definitions from physioex
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> reconstructions/
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

from physioex.explain.foundational.sleep_bands import (
    SLEEP_BANDS, bands_to_bin_ranges,
)

# ── Constants ────────────────────────────────────────────────────────

FS = 100.0
NFFT = 256
N_FREQ_BINS = NFFT // 2 + 1  # 129
N_TIME_FRAMES = 29
CHANNEL_NAMES = ["EEG", "EOG", "EMG"]
STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]
STAGE_COLORS = {
    "W": "#e6ab02", "N1": "#66a61e", "N2": "#377eb8",
    "N3": "#7570b3", "REM": "#e7298a",
}

# ── Nature style ─────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.linewidth": 0.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "legend.fontsize": 5,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ── Band utilities ───────────────────────────────────────────────────

def get_band_bin_ranges():
    """Get AASM band bin ranges for our STFT parameters."""
    return bands_to_bin_ranges(SLEEP_BANDS, fs=FS, signal_length=NFFT)


def db_to_linear(x_db):
    """Convert dB (10*log10) to linear power."""
    return np.power(10.0, x_db / 10.0)


def compute_band_powers(spec_linear, bin_ranges):
    """Compute band powers from linear-scale spectrogram.

    Args:
        spec_linear: (..., F) last dim is frequency bins.
        bin_ranges: list of (name, start, end) from bands_to_bin_ranges.

    Returns:
        powers: (..., n_bands) summed power per band.
        names: list of band names.
    """
    powers = []
    names = []
    for name, bstart, bend in bin_ranges:
        powers.append(spec_linear[..., bstart:bend].sum(axis=-1))
        names.append(name)
    return np.stack(powers, axis=-1), names


def freq_axis_hz():
    """Return frequency axis in Hz for the 129 STFT bins."""
    return np.arange(N_FREQ_BINS) * (FS / NFFT)


# ── EMG bin range (10–50 Hz, following Ferri et al.) ─────────────────

FREQ_RES = FS / NFFT
EMG_BIN_START = round(10.0 / FREQ_RES)   # bin 26
EMG_BIN_END = round(50.0 / FREQ_RES)     # bin 128


# ── Per-prototype analysis ───────────────────────────────────────────

def analyze_prototype(proto_dir, bin_ranges):
    """Compute all statistics for one prototype.

    EEG: AASM band decomposition.
    EOG: total broadband power + temporal CV (eye movement proxy).
    EMG: 10–50 Hz power + temporal CV (muscle tone proxy, Ferri et al.).

    Returns dict with all computed metrics.
    """
    proto_dir = Path(proto_dir)
    epochs = np.load(proto_dir / "epochs.npy")    # (N, C, T, F)
    labels_path = proto_dir / "labels.npy"
    labels = np.load(labels_path) if labels_path.exists() else None
    distances = np.load(proto_dir / "distances.npy")  # (N,)
    with open(proto_dir / "metadata.json") as f:
        meta = json.load(f)

    N, C, T, F = epochs.shape
    n_bands = len(bin_ranges)
    band_names = [b[0] for b in bin_ranges]

    # Convert to linear power
    spec_lin = db_to_linear(epochs)  # (N, C, T, F)

    # ── 1a. EEG spectral signature (AASM bands) ─────────────────
    eeg_spec = spec_lin[:, 0]  # (N, T, F)
    bp_eeg, _ = compute_band_powers(eeg_spec, bin_ranges)  # (N, T, n_bands)

    # Mean/std over epochs and time → (n_bands,)
    bp_eeg_mean = bp_eeg.mean(axis=(0, 1))
    bp_eeg_std = bp_eeg.mean(axis=1).std(axis=0)

    # Band power ratios
    total_eeg = bp_eeg_mean.sum() + 1e-12
    bp_eeg_ratios = bp_eeg_mean / total_eeg

    # Full spectral envelope: mean over epochs and time → (C, F)
    spec_envelope_mean = spec_lin.mean(axis=(0, 2))  # (C, F)
    spec_envelope_std = spec_lin.mean(axis=2).std(axis=0)  # (C, F)

    # ── 1a-bis. EOG features ────────────────────────────────────
    eog_spec = spec_lin[:, 1]  # (N, T, F)
    # Total broadband power per epoch per frame
    eog_power_per_frame = eog_spec.sum(axis=-1)  # (N, T)
    eog_total_power = float(eog_power_per_frame.mean())
    # Temporal CV across frames (averaged over epochs first)
    eog_temporal = eog_power_per_frame.mean(axis=0)  # (T,)
    eog_temporal_cv = float(eog_temporal.std() / (eog_temporal.mean() + 1e-12))

    # ── 1a-ter. EMG features (10–50 Hz, Ferri et al.) ──────────
    emg_spec = spec_lin[:, 2]  # (N, T, F)
    emg_band = emg_spec[..., EMG_BIN_START:EMG_BIN_END]  # (N, T, n_emg_bins)
    emg_power_per_frame = emg_band.sum(axis=-1)  # (N, T)
    emg_tone = float(emg_power_per_frame.mean())
    # Temporal CV
    emg_temporal = emg_power_per_frame.mean(axis=0)  # (T,)
    emg_temporal_cv = float(emg_temporal.std() / (emg_temporal.mean() + 1e-12))

    # ── 1b. Temporal concentration (EEG bands) ──────────────────
    bp_eeg_temporal = bp_eeg.mean(axis=0)  # (T, n_bands) — avg over epochs

    # Temporal CV per band: (n_bands,)
    eeg_temporal_cv = bp_eeg_temporal.std(axis=0) / (bp_eeg_temporal.mean(axis=0) + 1e-12)

    # Temporal energy profile: total power per frame → (C, T)
    temporal_profile = spec_lin.mean(axis=0).sum(axis=-1)  # (C, T)

    # Band temporal entropy: (n_bands,)
    eeg_temporal_entropy = np.zeros(n_bands)
    for bi in range(n_bands):
        p = bp_eeg_temporal[:, bi]
        p = p / (p.sum() + 1e-12)
        p = p[p > 0]
        H = -np.sum(p * np.log2(p))
        eeg_temporal_entropy[bi] = H / np.log2(T)

    # ── 1c. Monosemanticity ──────────────────────────────────────
    has_labels = labels is not None
    valid_labels = labels[labels >= 0] if has_labels else np.array([])

    if len(valid_labels) > 0:
        counts = np.bincount(valid_labels, minlength=5)
        dominant_idx = counts.argmax()
        dominant_stage = STAGE_NAMES[dominant_idx]
        label_purity = counts[dominant_idx] / len(valid_labels)

        p_labels = counts / counts.sum()
        p_labels = p_labels[p_labels > 0]
        label_entropy = -np.sum(p_labels * np.log2(p_labels)) / np.log2(5)
        effective_classes = 2 ** (-np.sum(p_labels * np.log2(p_labels)))
    else:
        dominant_stage = "N/A"
        label_purity = 0.0
        label_entropy = 1.0
        effective_classes = 5.0

    # Spectral consistency: mean pairwise cosine similarity (EEG bands)
    bp_per_epoch_mean_t = bp_eeg.mean(axis=1)  # (N, n_bands)
    norms = np.linalg.norm(bp_per_epoch_mean_t, axis=1, keepdims=True) + 1e-12
    bp_normed = bp_per_epoch_mean_t / norms
    cos_sim_matrix = bp_normed @ bp_normed.T
    triu_idx = np.triu_indices(N, k=1)
    spectral_consistency = float(cos_sim_matrix[triu_idx].mean())

    # Peak EEG band
    sorted_idx = np.argsort(bp_eeg_mean)[::-1]
    peak_band_eeg = band_names[sorted_idx[0]]
    peak_dominance_eeg = float(bp_eeg_mean[sorted_idx[0]] / (bp_eeg_mean[sorted_idx[1]] + 1e-12))

    # EEG spectral entropy
    p = bp_eeg_ratios[bp_eeg_ratios > 0]
    eeg_spectral_entropy = float(-np.sum(p * np.log2(p)) / np.log2(n_bands))

    # Composite monosemanticity score
    mono_score = 0.5 * (1 - label_entropy) + 0.5 * spectral_consistency

    return {
        "idx": meta["prototype_idx"],
        "n_samples": N,
        "cluster_size": meta.get("cluster_size", N),
        # EEG spectral
        "band_names": band_names,
        "eeg_band_powers": bp_eeg_mean.tolist(),
        "eeg_band_powers_std": bp_eeg_std.tolist(),
        "eeg_band_ratios": bp_eeg_ratios.tolist(),
        "spectral_envelope": spec_envelope_mean.tolist(),
        "spectral_envelope_std": spec_envelope_std.tolist(),
        # EOG features
        "eog_total_power": eog_total_power,
        "eog_temporal_cv": eog_temporal_cv,
        # EMG features (10–50 Hz)
        "emg_tone": emg_tone,
        "emg_temporal_cv": emg_temporal_cv,
        # EEG temporal
        "eeg_temporal_cv": eeg_temporal_cv.tolist(),
        "eeg_temporal_entropy": eeg_temporal_entropy.tolist(),
        "temporal_profile": temporal_profile.tolist(),
        # Monosemanticity
        "dominant_stage": dominant_stage,
        "label_purity": float(label_purity),
        "label_entropy": float(label_entropy),
        "effective_classes": float(effective_classes),
        "spectral_consistency": float(spectral_consistency),
        "eeg_spectral_entropy": eeg_spectral_entropy,
        "peak_band_eeg": peak_band_eeg,
        "peak_dominance_eeg": peak_dominance_eeg,
        "monosemanticity_score": float(mono_score),
        # Raw arrays for figures (not serialized to JSON)
        "_bp_eeg_mean": bp_eeg_mean,
        "_bp_eeg_ratios": bp_eeg_ratios,
        "_eeg_temporal_cv": eeg_temporal_cv,
        "_spec_envelope": spec_envelope_mean,
        "_spec_envelope_std": spec_envelope_std,
        "_temporal_profile": temporal_profile,
        "_eeg_temporal_entropy": eeg_temporal_entropy,
    }


# ── Aggregation ──────────────────────────────────────────────────────

def analyze_all(data_dir):
    """Analyze all prototypes in a data_driven directory."""
    data_dir = Path(data_dir)
    bin_ranges = get_band_bin_ranges()

    proto_dirs = sorted(data_dir.glob("proto_*"))
    if not proto_dirs:
        raise FileNotFoundError(f"No proto_* dirs in {data_dir}")

    stats = []
    for pd in proto_dirs:
        if not (pd / "epochs.npy").exists():
            continue
        print(f"  Analyzing {pd.name}...")
        s = analyze_prototype(pd, bin_ranges)
        stats.append(s)

    print(f"  Analyzed {len(stats)} prototypes")
    return stats


def sort_by_stage_purity(stats):
    """Return indices sorted by dominant stage then purity (descending)."""
    def key(s):
        stage_rank = STAGE_ORDER.index(s["dominant_stage"]) if s["dominant_stage"] in STAGE_ORDER else 5
        return (stage_rank, -s["label_purity"])
    order = sorted(range(len(stats)), key=lambda i: key(stats[i]))
    return order


# ── Serialization ────────────────────────────────────────────────────

def save_statistics(stats, out_dir):
    """Save per-prototype JSON + summary + numpy arrays."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    M = len(stats)
    n_bands = len(stats[0]["band_names"])

    # Per-prototype JSON (exclude numpy arrays)
    for s in stats:
        s_json = {k: v for k, v in s.items() if not k.startswith("_")}
        with open(out_dir / f"proto_{s['idx']:03d}.json", "w") as f:
            json.dump(s_json, f, indent=2)

    # Numpy arrays
    np.save(out_dir / "eeg_band_powers.npy",
            np.array([s["_bp_eeg_mean"] for s in stats]))  # (M, n_bands)
    np.save(out_dir / "spectral_envelopes.npy",
            np.array([s["_spec_envelope"] for s in stats]))  # (M, C, 129)
    np.save(out_dir / "temporal_profiles.npy",
            np.array([s["_temporal_profile"] for s in stats]))  # (M, C, 29)
    np.save(out_dir / "monosemanticity.npy",
            np.array([s["monosemanticity_score"] for s in stats]))  # (M,)
    np.save(out_dir / "eog_features.npy",
            np.array([[s["eog_total_power"], s["eog_temporal_cv"]] for s in stats]))  # (M, 2)
    np.save(out_dir / "emg_features.npy",
            np.array([[s["emg_tone"], s["emg_temporal_cv"]] for s in stats]))  # (M, 2)

    # Summary JSON
    summary = {
        "n_prototypes": M,
        "band_names": stats[0]["band_names"],
        "prototypes": [],
    }
    for s in stats:
        summary["prototypes"].append({
            "idx": s["idx"],
            "dominant_stage": s["dominant_stage"],
            "label_purity": s["label_purity"],
            "spectral_consistency": s["spectral_consistency"],
            "monosemanticity_score": s["monosemanticity_score"],
            "peak_band_eeg": s["peak_band_eeg"],
            "cluster_size": s["cluster_size"],
            "eog_total_power": s["eog_total_power"],
            "eog_temporal_cv": s["eog_temporal_cv"],
            "emg_tone": s["emg_tone"],
            "emg_temporal_cv": s["emg_temporal_cv"],
        })
    summary["prototypes"].sort(key=lambda x: -x["monosemanticity_score"])

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved statistics to {out_dir}/")


# ── Figure helpers ───────────────────────────────────────────────────

def savefig(fig, path_stem):
    """Save figure as both PDF and PNG."""
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {path_stem}.{{pdf,png}}")


def stage_color(stage_name):
    return STAGE_COLORS.get(stage_name, "#999999")


# ── Figure 1: Spectral Signatures ────────────────────────────────────

def fig_spectral_signatures(stats, out_dir):
    """EEG band heatmap + EOG/EMG scalar feature bars."""
    out_dir = Path(out_dir)
    order = sort_by_stage_purity(stats)
    band_names = stats[0]["band_names"]
    M = len(stats)
    n_bands = len(band_names)

    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(1, 4, wspace=0.40,
                          width_ratios=[2.5, 0.8, 0.8, 0.15])

    y_labels = [f"{stats[order[i]]['idx']}" for i in range(M)]
    y_pos = np.arange(M)

    # ── Panel 1: EEG band power heatmap ──────────────────────────
    ax_eeg = fig.add_subplot(gs[0, 0])
    ratios = np.array([stats[order[i]]["_bp_eeg_ratios"] for i in range(M)])
    z = (ratios - ratios.mean(axis=0)) / (ratios.std(axis=0) + 1e-12)

    im = ax_eeg.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                       interpolation="nearest")
    ax_eeg.set_xticks(range(n_bands))
    ax_eeg.set_xticklabels([b.replace("_", "\n") for b in band_names],
                           fontsize=4, rotation=45, ha="right")
    ax_eeg.set_yticks(y_pos)
    ax_eeg.set_yticklabels(y_labels, fontsize=4)
    ax_eeg.set_ylabel("Prototype", fontsize=6.5)
    ax_eeg.set_title("EEG (AASM bands)", fontsize=7, fontweight="bold")

    # Stage color markers on left
    for i in range(M):
        s = stats[order[i]]
        ax_eeg.plot(-0.8, i, "s", color=stage_color(s["dominant_stage"]),
                    markersize=3, clip_on=False, transform=ax_eeg.transData)

    # ── Panel 2: EOG features (horizontal bars) ──────────────────
    ax_eog = fig.add_subplot(gs[0, 1])
    eog_power = np.array([stats[order[i]]["eog_total_power"] for i in range(M)])
    eog_cv = np.array([stats[order[i]]["eog_temporal_cv"] for i in range(M)])
    # Normalize for display
    eog_power_n = eog_power / (eog_power.max() + 1e-12)
    eog_cv_n = eog_cv / (eog_cv.max() + 1e-12)

    bar_h = 0.35
    colors_stage = [stage_color(stats[order[i]]["dominant_stage"]) for i in range(M)]
    ax_eog.barh(y_pos - bar_h / 2, eog_power_n, height=bar_h,
                color=colors_stage, alpha=0.7, edgecolor="k", linewidth=0.3,
                label="Total power")
    ax_eog.barh(y_pos + bar_h / 2, eog_cv_n, height=bar_h,
                color=colors_stage, alpha=0.35, edgecolor="k", linewidth=0.3,
                hatch="//", label="Temporal CV")
    ax_eog.set_yticks(y_pos)
    ax_eog.set_yticklabels([])
    ax_eog.set_xlim(0, 1.1)
    ax_eog.set_title("EOG", fontsize=7, fontweight="bold")
    ax_eog.set_xlabel("Normalized", fontsize=5)
    ax_eog.legend(fontsize=4, loc="lower right")
    ax_eog.grid(True, axis="x", alpha=0.2, linewidth=0.3)

    # ── Panel 3: EMG features (horizontal bars) ──────────────────
    ax_emg = fig.add_subplot(gs[0, 2])
    emg_tone = np.array([stats[order[i]]["emg_tone"] for i in range(M)])
    emg_cv = np.array([stats[order[i]]["emg_temporal_cv"] for i in range(M)])
    emg_tone_n = emg_tone / (emg_tone.max() + 1e-12)
    emg_cv_n = emg_cv / (emg_cv.max() + 1e-12)

    ax_emg.barh(y_pos - bar_h / 2, emg_tone_n, height=bar_h,
                color=colors_stage, alpha=0.7, edgecolor="k", linewidth=0.3,
                label="Tone (10-50 Hz)")
    ax_emg.barh(y_pos + bar_h / 2, emg_cv_n, height=bar_h,
                color=colors_stage, alpha=0.35, edgecolor="k", linewidth=0.3,
                hatch="//", label="Temporal CV")
    ax_emg.set_yticks(y_pos)
    ax_emg.set_yticklabels([])
    ax_emg.set_xlim(0, 1.1)
    ax_emg.set_title("EMG", fontsize=7, fontweight="bold")
    ax_emg.set_xlabel("Normalized", fontsize=5)
    ax_emg.legend(fontsize=4, loc="lower right")
    ax_emg.grid(True, axis="x", alpha=0.2, linewidth=0.3)

    # ── Colorbar for EEG heatmap ─────────────────────────────────
    ax_cb = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im, cax=ax_cb)
    cbar.set_label("Z-scored\nband ratio", fontsize=5)
    cbar.ax.tick_params(labelsize=4.5)

    # Stage legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=STAGE_COLORS[s],
                      markersize=4, label=s) for s in STAGE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=5,
               bbox_to_anchor=(0.35, -0.02))

    savefig(fig, out_dir / "fig1_spectral_signatures")


# ── Figure 2: Temporal Concentration ─────────────────────────────────

def fig_temporal_concentration(stats, out_dir):
    """(a) temporal CV heatmap, (b) energy profiles for representative prototypes."""
    out_dir = Path(out_dir)
    order = sort_by_stage_purity(stats)
    band_names = stats[0]["band_names"]
    M = len(stats)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7.2, 4.0),
                                      gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.45})

    # Panel (a): Temporal CV heatmap (EEG)
    cv_matrix = np.array([stats[order[i]]["_eeg_temporal_cv"] for i in range(M)])
    im = ax_a.imshow(cv_matrix, aspect="auto", cmap="YlOrRd", vmin=0,
                     interpolation="nearest")
    ax_a.set_xticks(range(len(band_names)))
    ax_a.set_xticklabels([b.replace("_", "\n") for b in band_names],
                         fontsize=4, rotation=45, ha="right")
    ax_a.set_yticks(range(M))
    ax_a.set_yticklabels([f"{stats[order[i]]['idx']}" for i in range(M)], fontsize=4)
    ax_a.set_ylabel("Prototype", fontsize=6.5)
    ax_a.set_title("(a)  Temporal coefficient of variation (EEG)", fontsize=7, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_a, shrink=0.7, pad=0.02)
    cbar.set_label("CV", fontsize=5)
    cbar.ax.tick_params(labelsize=4.5)

    # Panel (b): Temporal energy profiles for representatives
    time_axis = np.arange(N_TIME_FRAMES)
    for stage in STAGE_ORDER:
        # Find highest-purity prototype for this stage
        candidates = [s for s in stats if s["dominant_stage"] == stage]
        if not candidates:
            continue
        best = max(candidates, key=lambda s: s["label_purity"])
        profile = best["_temporal_profile"][0]  # EEG
        profile_norm = profile / (profile.max() + 1e-12)
        ax_b.plot(time_axis, profile_norm, color=stage_color(stage),
                  linewidth=1.0, label=f"P{best['idx']} ({stage})")

    ax_b.set_xlabel("Time (s)", fontsize=6.5)
    ax_b.set_ylabel("Normalized power", fontsize=6.5)
    ax_b.set_title("(b)  Temporal energy profiles (EEG, best per stage)", fontsize=7, fontweight="bold")
    ax_b.legend(fontsize=5, ncol=5, loc="upper right")
    ax_b.set_xlim(0, N_TIME_FRAMES - 1)
    ax_b.grid(True, alpha=0.2, linewidth=0.3)

    savefig(fig, out_dir / "fig2_temporal_concentration")


# ── Figure 3: Monosemanticity ────────────────────────────────────────

def fig_monosemanticity(stats, out_dir):
    """(a) scatter, (b) sorted bars, (c) label distributions for extremes."""
    out_dir = Path(out_dir)
    M = len(stats)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.2, 2.8),
                                             gridspec_kw={"wspace": 0.40, "width_ratios": [1, 1, 1]})

    # Panel (a): Scatter — label purity vs spectral consistency
    for s in stats:
        ax_a.scatter(s["label_purity"], s["spectral_consistency"],
                     c=stage_color(s["dominant_stage"]),
                     s=max(8, s["cluster_size"] / 500),
                     edgecolors="k", linewidth=0.3, zorder=3)
        ax_a.annotate(f"{s['idx']}", (s["label_purity"], s["spectral_consistency"]),
                      fontsize=3.5, ha="center", va="bottom", xytext=(0, 2),
                      textcoords="offset points")

    ax_a.plot([0, 1], [0, 1], "k--", linewidth=0.4, alpha=0.3)
    ax_a.set_xlabel("Label purity", fontsize=6.5)
    ax_a.set_ylabel("Spectral consistency", fontsize=6.5)
    ax_a.set_title("(a)", fontsize=7, fontweight="bold")
    ax_a.set_xlim(0, 1.05)
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(True, alpha=0.2, linewidth=0.3)

    # Panel (b): Sorted bar chart of monosemanticity score
    sorted_stats = sorted(stats, key=lambda s: -s["monosemanticity_score"])
    x_pos = np.arange(M)
    colors = [stage_color(s["dominant_stage"]) for s in sorted_stats]
    scores = [s["monosemanticity_score"] for s in sorted_stats]
    ax_b.bar(x_pos, scores, color=colors, edgecolor="k", linewidth=0.3, width=0.8)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels([f"{s['idx']}" for s in sorted_stats], fontsize=3.5, rotation=90)
    ax_b.set_ylabel("Monosemanticity", fontsize=6.5)
    ax_b.set_xlabel("Prototype", fontsize=6.5)
    ax_b.set_title("(b)", fontsize=7, fontweight="bold")
    ax_b.axhline(0.7, color="k", linestyle="--", linewidth=0.4, alpha=0.5)
    ax_b.set_ylim(0, 1)
    ax_b.grid(True, axis="y", alpha=0.2, linewidth=0.3)

    # Panel (c): Stacked bars for top-3 and bottom-3
    top3 = sorted_stats[:3]
    bot3 = sorted_stats[-3:]
    selected = top3 + bot3
    y_pos = np.arange(len(selected))
    y_labels = []
    left = np.zeros(len(selected))

    for si, stage in enumerate(STAGE_ORDER):
        widths = []
        for s in selected:
            label_dist = s.get("_label_counts", None)
            if label_dist is None:
                # Reconstruct from metadata
                # We don't have raw counts in stats, use ratios
                pass
            widths.append(0)  # placeholder

        # Need actual label distributions — reload from metadata
    # Simpler approach: use label_purity and dominant_stage
    for i, s in enumerate(selected):
        # Load metadata for label distribution
        pass

    # Actually, let's compute label distributions during analysis
    # For now, show purity as a single bar with annotation
    for i, s in enumerate(selected):
        purity = s["label_purity"]
        ax_c.barh(i, purity, color=stage_color(s["dominant_stage"]),
                  edgecolor="k", linewidth=0.3, height=0.7)
        ax_c.barh(i, 1 - purity, left=purity, color="#dddddd",
                  edgecolor="k", linewidth=0.3, height=0.7)
        label = "Top" if i < 3 else "Bot"
        y_labels.append(f"P{s['idx']} (M={s['monosemanticity_score']:.2f})")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(y_labels, fontsize=4.5)
    ax_c.set_xlabel("Label purity", fontsize=6.5)
    ax_c.set_title("(c)  Top / Bottom 3", fontsize=7, fontweight="bold")
    ax_c.set_xlim(0, 1)
    ax_c.invert_yaxis()

    # Stage legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=STAGE_COLORS[s],
                      markersize=4, label=s) for s in STAGE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=5,
               bbox_to_anchor=(0.5, -0.04))

    savefig(fig, out_dir / "fig3_monosemanticity")


# ── Figure 4: Spectral Envelopes by Stage ────────────────────────────

def fig_spectral_envelopes(stats, out_dir):
    """Full 129-bin spectra grouped by dominant stage, EEG channel."""
    out_dir = Path(out_dir)
    freqs = freq_axis_hz()
    band_ranges = get_band_bin_ranges()

    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.5),
                             gridspec_kw={"wspace": 0.3}, sharey=True)

    for si, (stage, ax) in enumerate(zip(STAGE_ORDER, axes)):
        protos = [s for s in stats if s["dominant_stage"] == stage]
        if not protos:
            ax.set_title(stage, fontsize=7, fontweight="bold")
            ax.text(0.5, 0.5, "No protos", ha="center", va="center",
                    transform=ax.transAxes, fontsize=5)
            continue

        for s in protos:
            env = np.array(s["_spec_envelope"][0])  # EEG
            env_std = np.array(s["_spec_envelope_std"][0])
            ax.plot(freqs, env, linewidth=0.7, alpha=0.7,
                    label=f"P{s['idx']}")
            ax.fill_between(freqs, env - env_std, env + env_std,
                            alpha=0.1)

        # Band boundaries
        for name, bstart, bend in band_ranges:
            f_hz = bstart * (FS / NFFT)
            ax.axvline(f_hz, color="gray", linewidth=0.3, linestyle=":", alpha=0.4)

        ax.set_title(stage, fontsize=7, fontweight="bold",
                     color=stage_color(stage))
        ax.set_xlabel("Hz", fontsize=5.5)
        ax.set_xlim(0, 45)
        ax.grid(True, alpha=0.15, linewidth=0.3)

        if len(protos) <= 6:
            ax.legend(fontsize=3.5, loc="upper right")

        if si == 0:
            ax.set_ylabel("Power (linear)", fontsize=6.5)

    savefig(fig, out_dir / "fig4_spectral_envelopes")


# ── Figure 5: Prototype Atlas ────────────────────────────────────────

def fig_prototype_atlas(stats, data_dir, out_dir):
    """6x4 grid of mean EEG spectrograms."""
    out_dir = Path(out_dir)
    data_dir = Path(data_dir)
    order = sort_by_stage_purity(stats)
    freqs = freq_axis_hz()
    M = len(stats)

    n_cols = 4
    n_rows = (M + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 1.8 * n_rows),
                             gridspec_kw={"hspace": 0.5, "wspace": 0.25})
    axes = axes.flatten()

    # Band boundaries for horizontal lines
    band_ranges = get_band_bin_ranges()
    band_freqs = [(FS / NFFT) * bstart for _, bstart, _ in band_ranges]

    for panel_i in range(len(axes)):
        ax = axes[panel_i]
        if panel_i >= M:
            ax.axis("off")
            continue

        s = stats[order[panel_i]]
        k = s["idx"]

        # Load epochs and compute mean spectrogram (EEG)
        epochs = np.load(data_dir / f"proto_{k:03d}" / "epochs.npy")
        mean_spec = epochs[:, 0].mean(axis=0)  # (T, F) in dB

        im = ax.imshow(mean_spec.T, aspect="auto", origin="lower",
                       cmap="viridis", interpolation="nearest",
                       extent=[0, N_TIME_FRAMES, 0, 50])

        # Band boundary lines
        for bf in band_freqs:
            ax.axhline(bf, color="white", linewidth=0.2, linestyle=":", alpha=0.5)

        stage = s["dominant_stage"]
        purity = s["label_purity"]
        mono = s["monosemanticity_score"]
        ax.set_title(f"P{k} | {stage} ({purity:.0%}) | M={mono:.2f}",
                     fontsize=4.5, color=stage_color(stage), fontweight="bold")

        if panel_i % n_cols == 0:
            ax.set_ylabel("Hz", fontsize=4.5)
        else:
            ax.set_yticklabels([])

        if panel_i >= M - n_cols:
            ax.set_xlabel("Time (s)", fontsize=4.5)
        else:
            ax.set_xticklabels([])

        ax.tick_params(labelsize=3.5)
        ax.set_ylim(0, 45)

    savefig(fig, out_dir / "fig5_prototype_atlas")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spectral signature analysis of VQ prototype reconstructions"
    )
    parser.add_argument("data_dir", type=Path,
                        help="Path to data_driven/ output directory")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--figures-only", action="store_true",
                        help="Skip statistics, only generate figures")
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: {args.data_dir} is not a directory")
        return

    out_dir = args.data_dir / "spectral_analysis"
    stats_dir = out_dir / "statistics"
    fig_dir = out_dir / "figures"

    print(f"Analyzing: {args.data_dir}")
    stats = analyze_all(args.data_dir)

    if not args.figures_only:
        save_statistics(stats, stats_dir)

    if not args.no_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)
        print("Generating figures...")
        fig_spectral_signatures(stats, fig_dir)
        fig_temporal_concentration(stats, fig_dir)
        fig_monosemanticity(stats, fig_dir)
        fig_spectral_envelopes(stats, fig_dir)
        fig_prototype_atlas(stats, args.data_dir, fig_dir)

    print(f"\nDone. Output at {out_dir}/")


if __name__ == "__main__":
    main()
