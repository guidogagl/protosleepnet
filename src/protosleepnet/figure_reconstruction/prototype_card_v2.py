import os
"""
Prototype card v2 — Nature-quality, 1×4 horizontal layout.

Width: 2/3 textwidth (~4in for single-column article).
Uses LaTeX rendering for Greek letters and math.

Usage:
    python prototype_card_v2.py /path/to/method_dir/ --proto_idx 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── PhysioEx import ──────────────────────────────────────────────────
for c in [Path(os.environ.get("PHYSIOEX_ROOT", "")), Path(os.environ.get("PHYSIOEX_ROOT", ""))]:
    if (c / "physioex").is_dir():
        sys.path.insert(0, str(c)); break

try:
    from physioex.explain.foundational.sleep_bands import SLEEP_BANDS, bands_to_bin_ranges
except ImportError:
    # Inline definitions if torch is not available
    SLEEP_BANDS = {
        "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 11),
        "sigma_low": (11, 13), "sigma_high": (13, 16),
        "beta_low": (16, 20), "beta_high": (20, 30),
        "gamma": (30, 45), "mains": (49, 51),
    }
    def bands_to_bin_ranges(bands, fs=100.0, signal_length=256):
        freq_res = fs / signal_length
        return [(name, round(lo / freq_res), round(hi / freq_res))
                for name, (lo, hi) in bands.items()]

FS, NFFT = 100.0, 256
FREQ_RES = FS / NFFT
N_FREQ = NFFT // 2 + 1

# ── Style: match LaTeX article body (11pt) ───────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.size": 8,
    "axes.linewidth": 0.4,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.3,
    "ytick.major.width": 0.3,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

# ── Band definitions ─────────────────────────────────────────────────
BAND_ORDER = ["delta", "theta", "alpha", "sigma_low", "sigma_high",
              "beta_low", "beta_high", "gamma"]
BAND_LATEX = {
    "delta": r"$\delta$", "theta": r"$\theta$", "alpha": r"$\alpha$",
    "sigma_low": r"$\sigma_l$", "sigma_high": r"$\sigma_h$",
    "beta_low": r"$\beta_l$", "beta_high": r"$\beta_h$", "gamma": r"$\gamma$",
}

EOG_BANDS = [
    ("SEM",      0,                    round(1.0 / FREQ_RES)),
    ("REM",      round(1.0 / FREQ_RES), round(5.0 / FREQ_RES)),
    ("Blink",    round(5.0 / FREQ_RES), round(10.0 / FREQ_RES)),
    ("Residual", round(10.0 / FREQ_RES), N_FREQ),
]
EMG_START = round(10.0 / FREQ_RES)
EMG_END = round(50.0 / FREQ_RES)

STAGE_COLORS = {"W": "#C0A030", "N1": "#5B9E3A", "N2": "#3070B0",
                "N3": "#6050A0", "REM": "#C03070", "N/A": "#888888"}
CH_COLORS = {"EEG": "#3070B0", "EOG": "#D07020", "EMG": "#308040"}

# ── Helpers ──────────────────────────────────────────────────────────
def freq_hz():
    return np.arange(N_FREQ) * FREQ_RES

def compute_event_duration(epochs, training_mean, bin_ranges):
    N, C, T, F = epochs.shape
    n_bands = len(bin_ranges)
    eeg_dur = np.zeros((N, n_bands))
    for bi, (_, bs, be) in enumerate(bin_ranges):
        bp = epochs[:, 0, :, bs:be].mean(axis=-1)
        if training_mean.ndim == 2:
            th = training_mean[0, bs:be].mean()
        else:
            th = training_mean[0, :, bs:be].mean()
        eeg_dur[:, bi] = (bp > th).mean(axis=1)

    eog_dur = {}
    for label, bs, be in EOG_BANDS:
        ep = epochs[:, 1, :, bs:be].mean(axis=-1)
        if training_mean.ndim == 2:
            th = training_mean[1, bs:be].mean()
        else:
            th = training_mean[1, :, bs:be].mean()
        eog_dur[label] = (ep > th).mean(axis=1)
    return eeg_dur, eog_dur


def generate_card(k, method_dir, out_dir):
    method_dir = Path(method_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────
    epochs = np.load(method_dir / f"proto_{k:03d}" / "epochs.npy")
    spec = json.load(open(method_dir / "spectral_analysis" / "statistics" / f"proto_{k:03d}.json"))
    abl_dir = method_dir / "ablation" / f"proto_{k:03d}"
    abl_meta = json.load(open(abl_dir / "metadata.json"))
    abl_imp = np.load(abl_dir / "marginal_importance.npy")
    abl_dirs = np.load(abl_dir / "feature_direction.npy")
    proto_power = np.load(abl_dir / "proto_power.npy")
    mean_power = np.load(abl_dir / "mean_power.npy")
    ch_imp = np.load(abl_dir / "channel_importance.npy")

    tm_path = method_dir / "ablation" / "training_mean.npy"
    gm_path = method_dir / "ablation" / "global_mean.npy"
    if tm_path.exists():
        global_mean = np.load(tm_path)
    elif gm_path.exists():
        global_mean = np.load(gm_path)
    else:
        global_mean = np.zeros_like(epochs[0])

    feature_names = abl_meta.get("feature_names", BAND_ORDER)
    bin_ranges = bands_to_bin_ranges(SLEEP_BANDS, fs=FS, signal_length=NFFT)
    bin_ranges_nm = [(n, s, e) for n, s, e in bin_ranges if n != "mains"]

    dominant = abl_meta.get("dominant_class", "N/A")
    purity = abl_meta.get("predicted_purity", 0)
    sc = spec.get("spectral_consistency", 0)
    stage_color = STAGE_COLORS.get(dominant, "#888")

    # Channel percentages
    ch_total = abs(ch_imp).sum() + 1e-12
    ch_pct = {c: abs(ch_imp[i]) / ch_total * 100 for i, c in enumerate(["EEG", "EOG", "EMG"])}

    # EEG band mapping
    eeg_map = {}
    for fi, fn in enumerate(feature_names):
        band = fn.replace("EEG:", "")
        if band in BAND_ORDER:
            eeg_map[band] = fi

    # Sensitivity
    all_pp = []
    for pd in sorted(method_dir.glob("proto_*")):
        pp = method_dir / "ablation" / pd.name / "proto_power.npy"
        if pp.exists():
            all_pp.append(np.load(pp))
    median_power = np.median(np.array(all_pp), axis=0) if all_pp else mean_power

    DELTA_DB_THRESHOLD = 1.0
    sensitivities = []
    sens_colors = []
    for band in BAND_ORDER:
        if band not in eeg_map:
            sensitivities.append(0)
            sens_colors.append("#cccccc")
            continue
        fi = eeg_map[band]
        p_db = 10 * np.log10(proto_power[fi] + 1e-12)
        m_db = 10 * np.log10(median_power[fi] + 1e-12)
        delta_db = abs(p_db - m_db)
        if delta_db < DELTA_DB_THRESHOLD:
            sens = 0.0
        else:
            sens = abl_imp[fi] / delta_db
        sensitivities.append(sens)
        sens_colors.append("#C04040" if proto_power[fi] > median_power[fi] else "#4060A0")

    # Event duration
    eeg_dur, eog_dur = compute_event_duration(epochs, global_mean, bin_ranges_nm)

    # ── Figure: 1 header row + 1×4 panels ───────────────────────
    fig = plt.figure(figsize=(6.2, 2.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[0.12, 1.0],
                          width_ratios=[1.0, 1.0, 1.15, 1.45],
                          hspace=0.45, wspace=0.55,
                          left=0.07, right=0.97, top=0.86, bottom=0.24)

    # ── Header: channel relevance bar (full width) ──────────────
    ax_h = fig.add_subplot(gs[0, :])
    left = 0
    for ch in ["EEG", "EOG", "EMG"]:
        frac = ch_pct[ch] / 100
        ax_h.barh(0, frac, left=left, color=CH_COLORS[ch], height=0.6,
                  edgecolor="white", linewidth=0.3)
        if frac > 0.06:
            ax_h.text(left + frac/2, 0,
                      f"\\textbf{{{ch}}} {ch_pct[ch]:.0f}\\%",
                      ha="center", va="center", fontsize=6, color="white")
        left += frac
    ax_h.set_xlim(0, 1)
    ax_h.set_yticks([])
    ax_h.set_xticks([])
    for sp in ax_h.spines.values():
        sp.set_visible(False)

    # Title above
    fig.text(0.5, 0.96,
             f"\\textbf{{Prototype {k}}}  $|$  "
             f"\\textbf{{{dominant}}} ({purity*100:.0f}\\%)  $|$  "
             f"SC$=$\\,{sc:.2f}",
             fontsize=9, ha="center", color=stage_color)

    # ── Panel 1: EEG Spectrum ───────────────────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    freqs = freq_hz()
    eeg_db = epochs[:, 0].mean(axis=(0, 1))
    gm_db = global_mean[0] if global_mean.ndim == 2 else global_mean[0].mean(axis=0)
    ax1.plot(freqs, eeg_db, color=CH_COLORS["EEG"], linewidth=0.8, label="Proto")
    ax1.plot(freqs, gm_db, color=CH_COLORS["EEG"], linewidth=0.5,
             linestyle="--", alpha=0.4, label="Mean")
    ax1.set_xlim(0, 45)
    ax1.set_xlabel("Hz")
    ax1.set_ylabel("dB")
    ax1.set_title("EEG spectrum")
    ax1.legend(fontsize=5.5, loc="upper right")
    # Band boundaries
    for bname, bs, be in bin_ranges_nm:
        f_s = bs * FREQ_RES
        ax1.axvline(f_s, color="#cccccc", linewidth=0.3, zorder=0)
    ax1.grid(True, alpha=0.08, linewidth=0.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ── Panel 2: EOG + EMG ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 1])
    proto_spec = epochs.mean(axis=(0, 2))
    gm_spec = global_mean if global_mean.ndim == 2 else global_mean.mean(axis=1)

    items = []
    for label, bs, be in EOG_BANDS:
        p = proto_spec[1, bs:be].mean()
        m = gm_spec[1, bs:be].mean()
        items.append((label, p, m, CH_COLORS["EOG"]))
    p_emg = proto_spec[2, EMG_START:EMG_END].mean()
    m_emg = gm_spec[2, EMG_START:EMG_END].mean()
    items.append(("EMG", p_emg, m_emg, CH_COLORS["EMG"]))

    x2 = np.arange(len(items))
    w = 0.35
    ax2.bar(x2 - w/2, [i[1] for i in items], w, color=[i[3] for i in items],
            alpha=0.8, edgecolor="white", linewidth=0.2, label="Proto")
    ax2.bar(x2 + w/2, [i[2] for i in items], w, color=[i[3] for i in items],
            alpha=0.25, edgecolor="#999", linewidth=0.3, label="Mean")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([i[0] for i in items], fontsize=5.5, rotation=35, ha="right")
    ax2.set_title("EOG \\& EMG")
    ax2.set_ylabel("dB")
    ax2.legend(fontsize=5, loc="upper right")
    ax2.grid(True, axis="y", alpha=0.08, linewidth=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── Panel 3: Band Sensitivity ───────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    x3 = np.arange(len(BAND_ORDER))
    ax3.bar(x3, sensitivities, color=sens_colors, width=0.7,
            edgecolor="white", linewidth=0.2)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([BAND_LATEX[b] for b in BAND_ORDER], fontsize=6,
                         rotation=45, ha="right")
    ax3.set_title("Band sensitivity")
    ax3.set_ylabel("imp / $|\\Delta$dB$|$")
    ax3.grid(True, axis="y", alpha=0.08, linewidth=0.2)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ── Panel 4: Event Duration ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 3])
    # EEG bands + EOG bands
    dur_labels = [BAND_LATEX[b] for b in BAND_ORDER] + list(eog_dur.keys())
    dur_medians = []
    dur_q25 = []
    dur_q75 = []
    for bi in range(len(BAND_ORDER)):
        vals = eeg_dur[:, bi] * 100
        dur_medians.append(np.median(vals))
        dur_q25.append(np.percentile(vals, 25))
        dur_q75.append(np.percentile(vals, 75))
    eog_colors_list = []
    for label in eog_dur:
        vals = eog_dur[label] * 100
        dur_medians.append(np.median(vals))
        dur_q25.append(np.percentile(vals, 25))
        dur_q75.append(np.percentile(vals, 75))

    x4 = np.arange(len(dur_labels))
    colors4 = ["#888888"] * len(BAND_ORDER) + [CH_COLORS["EOG"]] * len(eog_dur)
    yerr_low = [m - q for m, q in zip(dur_medians, dur_q25)]
    yerr_high = [q - m for m, q in zip(dur_medians, dur_q75)]
    ax4.bar(x4, dur_medians, color=colors4, width=0.7, alpha=0.7,
            edgecolor="white", linewidth=0.2)
    ax4.errorbar(x4, dur_medians, yerr=[yerr_low, yerr_high],
                 fmt="none", ecolor="#555555", elinewidth=0.4, capsize=1.5, capthick=0.3)
    ax4.axhline(50, color="#cccccc", linewidth=0.4, linestyle="--")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(dur_labels, fontsize=5, rotation=55, ha="right")
    ax4.set_title("Event duration")
    ax4.set_ylabel("\\% frames $>$ mean")
    ax4.set_ylim(0, 100)
    ax4.grid(True, axis="y", alpha=0.08, linewidth=0.2)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # ── Save ────────────────────────────────────────────────────
    fig.savefig(out_dir / f"proto_{k:03d}_card.pdf")
    fig.savefig(out_dir / f"proto_{k:03d}_card.png")
    print(f"Saved: {out_dir / f'proto_{k:03d}_card.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method_dir", help="Path to method directory (e.g., hybrid/)")
    parser.add_argument("--proto_idx", type=int, default=10, help="Prototype index")
    parser.add_argument("--out_dir", default=None, help="Output directory (default: method_dir/cards_v2)")
    args = parser.parse_args()
    out = Path(args.out_dir) if args.out_dir else Path(args.method_dir) / "cards_v2"
    generate_card(args.proto_idx, args.method_dir, out)
