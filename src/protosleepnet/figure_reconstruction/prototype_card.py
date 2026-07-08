import os
"""Prototype card figure — compact Nature single-column summary.

Blocks: (1) Channel relevance, (2) EEG spectral signature,
(2b) EOG bands + EMG tone vs global mean,
(3) EEG band ablation importance [vertical], (4) Event duration % [vertical].

Usage:
    python prototype_card.py /path/to/method_dir/
    python prototype_card.py /path/to/method_dir/ --proto_idx 12
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PHYSIOEX_ROOT = None
for c in [Path(os.environ.get("PHYSIOEX_ROOT", "")), Path(os.environ.get("PHYSIOEX_ROOT", "")),
          Path(os.environ.get("PHYSIOEX_ROOT", ""))]:
    if (c / "physioex").is_dir():
        sys.path.insert(0, str(c)); break

from physioex.explain.foundational.sleep_bands import SLEEP_BANDS, bands_to_bin_ranges

FS, NFFT = 100.0, 256
FREQ_RES = FS / NFFT
N_FREQ, N_TIME = NFFT // 2 + 1, 29
STAGE_COLORS = {"W": "#e6ab02", "N1": "#66a61e", "N2": "#377eb8",
                "N3": "#7570b3", "REM": "#e7298a", "N/A": "#999999"}
CH_COLORS = {"EEG": "#1f77b4", "EOG": "#ff7f0e", "EMG": "#2ca02c"}

# EEG band frequency order: δ θ α σ_low σ_high β_low β_high γ
BAND_FREQ_ORDER = ["delta", "theta", "alpha", "sigma_low", "sigma_high",
                   "beta_low", "beta_high", "gamma"]
BAND_GREEK = {"delta": "δ", "theta": "θ", "alpha": "α",
              "sigma_low": "σ_l", "sigma_high": "σ_h",
              "beta_low": "β_l", "beta_high": "β_h", "gamma": "γ"}

# EOG bands — eye movement phenomena (AASM; Agarwal 2005; Pittman 2016)
EOG_BANDS = [
    ("SEM\n(0–1 Hz)",      0, round(1.0 / FREQ_RES)),                        # Slow Eye Movements (AASM: 0.1–1 Hz; De Carli 2000)
    ("REM\n(1–5 Hz)",      round(1.0 / FREQ_RES), round(5.0 / FREQ_RES)),    # Rapid Eye Movements (Agarwal 2005; Pittman 2016)
    ("Blink\n(5–10 Hz)",   round(5.0 / FREQ_RES), round(10.0 / FREQ_RES)),   # Blink harmonics (broadband transient)
    ("Residual\n(>10 Hz)", round(10.0 / FREQ_RES), N_FREQ),                   # EMG artifact, EEG crosstalk
]
EMG_BIN_START = round(10.0 / FREQ_RES)
EMG_BIN_END = round(50.0 / FREQ_RES)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 5.5, "axes.linewidth": 0.4, "axes.labelsize": 5.5,
    "axes.titlesize": 6, "xtick.labelsize": 4.5, "ytick.labelsize": 4.5,
    "xtick.major.width": 0.3, "ytick.major.width": 0.3,
    "xtick.major.size": 1.5, "ytick.major.size": 1.5,
    "legend.fontsize": 4, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def db_to_lin(x): return np.power(10.0, x / 10.0)
def freq_hz(): return np.arange(N_FREQ) * FREQ_RES
def greek(band): return BAND_GREEK.get(band, band)


# ── Event duration ───────────────────────────────────────────────────

def compute_event_duration(epochs_np, training_mean_np, bin_ranges_no_mains):
    """Compute % of frames where band power (dB) exceeds training mean (dB).

    All comparisons in dB space to avoid Jensen's inequality bias.
    training_mean_np can be (C, F) or (C, T, F).
    """
    N, C, T, F = epochs_np.shape

    band_names = [b[0] for b in bin_ranges_no_mains]
    n_bands = len(bin_ranges_no_mains)

    eeg_duration = np.zeros((N, n_bands), dtype=np.float32)
    for bi, (bname, bstart, bend) in enumerate(bin_ranges_no_mains):
        bp_db = epochs_np[:, 0, :, bstart:bend].mean(axis=-1)  # (N, T) mean dB across band
        if training_mean_np.ndim == 2:  # (C, F)
            thresh_db = training_mean_np[0, bstart:bend].mean()
        else:  # (C, T, F)
            thresh_db = training_mean_np[0, :, bstart:bend].mean(axis=-1).mean()
        eeg_duration[:, bi] = (bp_db > thresh_db).mean(axis=1)

    # EOG bands duration
    eog_durations = {}
    for label, bstart, bend in EOG_BANDS:
        eog_db = epochs_np[:, 1, :, bstart:bend].mean(axis=-1)  # (N, T)
        if training_mean_np.ndim == 2:
            eog_thresh_db = training_mean_np[1, bstart:bend].mean()
        else:
            eog_thresh_db = training_mean_np[1, :, bstart:bend].mean(axis=-1).mean()
        eog_durations[label] = (eog_db > eog_thresh_db).mean(axis=1)  # (N,)

    return eeg_duration, eog_durations, band_names


# ── Rule generation ──────────────────────────────────────────────────

def generate_rule_text(k, spec, abl_meta, abl_imp, abl_dir, proto_power,
                       mean_power, eeg_duration, eog_durations,
                       eog_pct_rank, emg_pct_rank, feature_names,
                       bin_ranges_no_mains, eeg_sorted_indices, **kwargs):
    dominant = abl_meta.get("dominant_class", spec.get("dominant_stage", "N/A"))
    purity = spec.get("label_purity", 0)

    # Channel importance from whole-channel ablation (passed via kwargs or computed)
    ch_pcts = kwargs.get("ch_pcts", {"EEG": 33, "EOG": 33, "EMG": 33})
    ch_sorted = sorted(ch_pcts.items(), key=lambda x: -x[1])

    stmts = []
    stmts.append(f"Prototype {k} represents {dominant} with {purity:.0%} stage purity.")
    stmts.append(f"The model identifies this prototype primarily through {ch_sorted[0][0]} "
                 f"({ch_sorted[0][1]:.0f}%, vs {ch_sorted[1][0]} {ch_sorted[1][1]:.0f}%, "
                 f"{ch_sorted[2][0]} {ch_sorted[2][1]:.0f}%).")

    # Top EEG bands by importance
    eeg_features = [(i, n) for i, n in enumerate(feature_names)
                     if n.startswith("EEG:") or n in BAND_FREQ_ORDER]
    eeg_ranked = sorted(eeg_features, key=lambda x: -abs(abl_imp[x[0]]))
    for fi, fname in eeg_ranked[:3]:
        band = fname.replace("EEG:", "")
        pp, mp = proto_power[fi], mean_power[fi]
        diff_db = 10 * np.log10(pp + 1e-12) - 10 * np.log10(mp + 1e-12)
        direction = "elevated" if abl_dir[fi] > 0 else "suppressed"
        stmts.append(f"{greek(band)} is {direction} "
                     f"(mean {10*np.log10(pp+1e-12):.1f} dB, "
                     f"{'+' if diff_db > 0 else ''}{diff_db:.1f} dB vs mean).")

    delta_suff = abl_meta.get("delta_sufficient", False)
    frac = abl_meta.get("delta_still_assigned_frac", 0)
    stmts.append(f"δ {'is' if delta_suff else 'is not'} sufficient alone ({frac:.0%} retention).")

    ranked_all = sorted(enumerate(feature_names), key=lambda x: -abs(abl_imp[x[0]]))
    parts = [f"{fn} ({abs(abl_imp[fi]):.1f}, {'↑' if abl_dir[fi] > 0 else '↓'})" for fi, fn in ranked_all[:3]]
    stmts.append(f"Key features: {', '.join(parts)}.")

    band_names_br = [b[0] for b in bin_ranges_no_mains]
    for fi, fname in eeg_ranked[:2]:
        band = fname.replace("EEG:", "")
        if band in band_names_br:
            bi = band_names_br.index(band)
            med_pct = np.median(eeg_duration[:, bi]) * 100
            stmts.append(f"{greek(band)} active >{' '}global mean for {med_pct:.0f}% of each epoch.")

    # EOG bands
    for label, dur_arr in eog_durations.items():
        med_pct = np.median(dur_arr) * 100
        short = label.split("\n")[0]
        stmts.append(f"{short} active for {med_pct:.0f}% of each epoch.")

    eog_power = spec.get("eog_total_power", 0)
    emg_tone = spec.get("emg_tone", 0)
    eog_label = "active" if eog_pct_rank > 75 else ("quiescent" if eog_pct_rank < 25 else "moderate")
    emg_label = "tonic" if emg_pct_rank > 75 else ("atonic" if emg_pct_rank < 25 else "intermediate")
    stmts.append(f"EOG: {eog_label} ({eog_power:.0f}, {eog_pct_rank:.0f}th pct). "
                 f"EMG: {emg_label} ({emg_tone:.0f}, {emg_pct_rank:.0f}th pct).")

    return "\n".join(stmts)


# ── Card generation ──────────────────────────────────────────────────

def generate_card(k, method_dir, out_dir, all_specs=None):
    method_dir = Path(method_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = np.load(method_dir / f"proto_{k:03d}" / "epochs.npy")
    spec_path = method_dir / "spectral_analysis" / "statistics" / f"proto_{k:03d}.json"
    spec = json.load(open(spec_path)) if spec_path.exists() else {}

    abl_dir_path = method_dir / "ablation" / f"proto_{k:03d}"
    abl_meta = json.load(open(abl_dir_path / "metadata.json")) if (abl_dir_path / "metadata.json").exists() else {}
    abl_imp = np.load(abl_dir_path / "marginal_importance.npy") if (abl_dir_path / "marginal_importance.npy").exists() else np.zeros(10)
    abl_directions = np.load(abl_dir_path / "feature_direction.npy") if (abl_dir_path / "feature_direction.npy").exists() else np.zeros(10)
    proto_power = np.load(abl_dir_path / "proto_power.npy") if (abl_dir_path / "proto_power.npy").exists() else np.zeros(10)
    mean_power_arr = np.load(abl_dir_path / "mean_power.npy") if (abl_dir_path / "mean_power.npy").exists() else np.zeros(10)
    # Prefer training_mean.npy (C, F); fall back to global_mean.npy (C, T, F)
    tm_path = method_dir / "ablation" / "training_mean.npy"
    gm_path = method_dir / "ablation" / "global_mean.npy"
    if tm_path.exists():
        global_mean = np.load(tm_path)  # (C, F)
    elif gm_path.exists():
        global_mean = np.load(gm_path)  # (C, T, F) legacy
    else:
        global_mean = np.zeros_like(epochs[0])

    feature_names = abl_meta.get("feature_names", [f"f{i}" for i in range(len(abl_imp))])
    bin_ranges = bands_to_bin_ranges(SLEEP_BANDS, fs=FS, signal_length=NFFT)
    bin_ranges_no_mains = [(n, s, e) for n, s, e in bin_ranges if n != "mains"]

    dominant = abl_meta.get("dominant_class", spec.get("dominant_stage", "N/A"))
    purity = abl_meta.get("predicted_purity", spec.get("label_purity", 0))
    spec_cons = spec.get("spectral_consistency", 0)
    stage_color = STAGE_COLORS.get(dominant, "#999")

    # Build EEG feature mapping sorted by frequency
    # Support both old format ("EEG:delta") and new format ("delta")
    eeg_feat_map = {}  # band_name -> (feature_index, importance, direction)
    for fi, fn in enumerate(feature_names):
        band = fn.replace("EEG:", "") if fn.startswith("EEG:") else fn
        if band in BAND_FREQ_ORDER:
            eeg_feat_map[band] = (fi, abl_imp[fi], abl_directions[fi] if fi < len(abl_directions) else 0)

    # Sorted by frequency order
    eeg_sorted = [(b, eeg_feat_map[b]) for b in BAND_FREQ_ORDER if b in eeg_feat_map]
    eeg_sorted_labels = [greek(b) for b, _ in eeg_sorted]
    eeg_sorted_imps = [v[1] for _, v in eeg_sorted]
    eeg_sorted_dirs = [v[2] for _, v in eeg_sorted]
    eeg_sorted_fi = [v[0] for _, v in eeg_sorted]

    # Event duration
    eeg_duration, eog_durations, band_names_dur = compute_event_duration(
        epochs, global_mean, bin_ranges_no_mains
    )

    # Percentile ranks
    eog_pct, emg_pct = 50, 50
    if all_specs:
        eog_vals = sorted([s.get("eog_total_power", 0) for s in all_specs])
        emg_vals = sorted([s.get("emg_tone", 0) for s in all_specs])
        eog_pct = np.searchsorted(eog_vals, spec.get("eog_total_power", 0)) / max(len(eog_vals), 1) * 100
        emg_pct = np.searchsorted(emg_vals, spec.get("emg_tone", 0)) / max(len(emg_vals), 1) * 100

    # ── Figure ───────────────────────────────────────────────────
    # Rows: (1) channel bar, (2)+(2b) spectra, (3)+(4) side by side
    fig = plt.figure(figsize=(3.5, 3.8))
    gs = fig.add_gridspec(3, 2,
                          height_ratios=[0.2, 0.9, 0.9],
                          hspace=0.65, wspace=0.35,
                          left=0.12, right=0.95, top=0.91, bottom=0.08)

    # Header
    fig.text(0.5, 0.97,
             f"Prototype {k}  |  {dominant} ({purity:.0%})  |  SC={spec_cons:.2f}",
             fontsize=7.5, fontweight="bold", ha="center", color=stage_color)

    # ── (1) Channel Relevance — full width (from whole-channel ablation) ──
    ax1 = fig.add_subplot(gs[0, :])
    ch_imp_path = method_dir / "ablation" / f"proto_{k:03d}" / "channel_importance.npy"
    if ch_imp_path.exists():
        ch_imp = np.abs(np.load(ch_imp_path))  # (3,) EEG, EOG, EMG
        ch_total = ch_imp.sum() + 1e-12
        ch_fracs = [ch_imp[0] / ch_total, ch_imp[1] / ch_total, ch_imp[2] / ch_total]
    else:
        # Fallback: sum band importances (old method)
        eeg_imp_sum = sum(abs(abl_imp[i]) for i, n in enumerate(feature_names) if n.startswith("EEG:") or n in BAND_FREQ_ORDER)
        eog_imp_val = sum(abs(abl_imp[i]) for i, n in enumerate(feature_names) if n == "EOG:all")
        emg_imp_val = sum(abs(abl_imp[i]) for i, n in enumerate(feature_names) if n == "EMG:tone")
        ch_total = eeg_imp_sum + eog_imp_val + emg_imp_val + 1e-12
        ch_fracs = [eeg_imp_sum / ch_total, eog_imp_val / ch_total, emg_imp_val / ch_total]
    left = 0
    for frac, ch in zip(ch_fracs, ["EEG", "EOG", "EMG"]):
        ax1.barh(0, frac, left=left, color=CH_COLORS[ch], height=0.6,
                 edgecolor="white", linewidth=0.3)
        if frac > 0.08:
            ax1.text(left + frac / 2, 0, f"{ch}\n{frac:.0%}",
                     ha="center", va="center", fontsize=4, fontweight="bold", color="white")
        left += frac
    ax1.set_xlim(0, 1); ax1.set_yticks([])
    ax1.set_title("(1) Channel relevance", fontsize=6, fontweight="bold")
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(bottom=False, labelbottom=False)

    # ── (2) EEG Spectral Signature — left column ────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    freqs = freq_hz()
    env_db = epochs[:, 0].mean(axis=(0, 1))  # EEG mean dB
    ax2.plot(freqs, env_db, color=CH_COLORS["EEG"], linewidth=0.8, label="Proto")
    # Training mean: (C, F) or legacy (C, T, F)
    gm_db = global_mean[0] if global_mean.ndim == 2 else global_mean[0].mean(axis=0)
    ax2.plot(freqs, gm_db, color=CH_COLORS["EEG"], linewidth=0.5,
             linestyle="--", alpha=0.5, label="Train mean")
    ax2.set_xlim(0, 45); ax2.set_ylim(-25, None)
    # Band boundaries with labels at bottom
    ybot = ax2.get_ylim()[0]
    for bname, bstart, bend in bin_ranges_no_mains:
        f_start = bstart * FREQ_RES
        f_end = min(bend * FREQ_RES, 45)
        ax2.axvline(f_start, color="gray", linewidth=0.3, linestyle=":", alpha=0.3)
        f_center = (f_start + f_end) / 2
        ax2.text(f_center, ybot + 1, greek(bname), fontsize=3,
                 ha="center", va="bottom", color="gray", alpha=0.7)
    ax2.axvline(bin_ranges_no_mains[-1][2] * FREQ_RES, color="gray",
                linewidth=0.3, linestyle=":", alpha=0.3)
    ax2.set_xlabel("Hz", fontsize=4.5); ax2.set_ylabel("dB", fontsize=4.5)
    ax2.set_title("(2) EEG spectrum", fontsize=5.5, fontweight="bold")
    ax2.legend(fontsize=3.5, loc="upper right")
    ax2.grid(True, alpha=0.1, linewidth=0.2)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ── (2b) EOG bands + EMG in dB — right column, shared y ──────
    ax2b = fig.add_subplot(gs[1, 1], sharey=ax2)
    items = []

    # Compute mean dB per band for proto and training mean
    proto_mean_spec = epochs.mean(axis=(0, 2))  # (C, F) mean over N and T, in dB
    # Training mean: (C, F) or legacy (C, T, F)
    gm_mean_spec = global_mean if global_mean.ndim == 2 else global_mean.mean(axis=1)

    for label, bstart, bend in EOG_BANDS:
        short = label.split("\n")[0]
        p_db = proto_mean_spec[1, bstart:bend].mean()
        m_db = gm_mean_spec[1, bstart:bend].mean()
        items.append((short, p_db, m_db, CH_COLORS["EOG"]))

    # EMG tone in dB
    p_emg_db = proto_mean_spec[2, EMG_BIN_START:EMG_BIN_END].mean()
    m_emg_db = gm_mean_spec[2, EMG_BIN_START:EMG_BIN_END].mean()
    items.append(("EMG\ntone", p_emg_db, m_emg_db, CH_COLORS["EMG"]))

    x_2b = np.arange(len(items))
    width = 0.35
    proto_vals = [it[1] for it in items]
    mean_vals = [it[2] for it in items]
    bar_colors = [it[3] for it in items]

    ax2b.bar(x_2b - width/2, proto_vals, width, color=bar_colors, alpha=0.8,
             edgecolor="white", linewidth=0.2, label="Proto")
    ax2b.bar(x_2b + width/2, mean_vals, width, color=bar_colors, alpha=0.3,
             edgecolor="gray", linewidth=0.3, label="Train mean")
    ax2b.set_xticks(x_2b)
    ax2b.set_xticklabels([it[0] for it in items], fontsize=3.5)
    ax2b.set_title("(2b) EOG + EMG", fontsize=5.5, fontweight="bold")
    ax2b.legend(fontsize=3, loc="upper right")
    ax2b.grid(True, axis="y", alpha=0.1, linewidth=0.2)
    ax2b.spines["top"].set_visible(False)
    ax2b.spines["right"].set_visible(False)
    ax2b.spines["left"].set_visible(False)
    ax2b.tick_params(left=False, labelleft=False)

    # ── (3) EEG Band Relevance — left column ────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    n_eeg = len(eeg_sorted)
    x3 = np.arange(n_eeg)
    # Color by power vs median across prototypes: red = above median, blue = below
    # Compute median proto_power across all prototypes in this method
    all_proto_powers = []
    for pd in sorted(method_dir.glob("proto_*")):
        pp_path = method_dir / "ablation" / pd.name / "proto_power.npy"
        if pp_path.exists():
            all_proto_powers.append(np.load(pp_path))
    if all_proto_powers:
        median_power = np.median(np.array(all_proto_powers), axis=0)
    else:
        median_power = mean_power_arr  # fallback

    colors3 = []
    for band, (fi, imp, d) in eeg_sorted:
        colors3.append("#d62728" if proto_power[fi] > median_power[fi] else "#1f77b4")
    # Sensitivity = importance / |ΔdB| — how much the model cares per unit of power change
    # Threshold: |ΔdB| < 1.0 dB is noise (below median of distribution across all prototypes)
    DELTA_DB_THRESHOLD = 1.0
    eeg_sorted_sensitivity = []
    for band, (fi, imp, d) in eeg_sorted:
        p_db = 10 * np.log10(proto_power[fi] + 1e-12)
        m_db = 10 * np.log10(median_power[fi] + 1e-12)
        delta_db = abs(p_db - m_db)
        if delta_db < DELTA_DB_THRESHOLD:
            sens = 0.0  # not significantly different from median
        else:
            sens = imp / delta_db
        eeg_sorted_sensitivity.append(sens)

    ax3.bar(x3, eeg_sorted_sensitivity, color=colors3, width=0.7, edgecolor="white", linewidth=0.2)

    delta_suff = abl_meta.get("delta_sufficient", False)
    for xi, (band, _) in enumerate(eeg_sorted):
        if band == "delta" and delta_suff:
            y_star = eeg_sorted_sensitivity[xi] + max(max(eeg_sorted_sensitivity), 0) * 0.05
            ax3.text(xi, y_star, "★", fontsize=6, ha="center", color="#d4af37")

    ax3.set_xticks(x3)
    ax3.set_xticklabels(eeg_sorted_labels, fontsize=5)
    ax3.set_ylabel("Sensitivity", fontsize=4.5)
    ax3.set_title("(3) Band sensitivity\n(imp/|ΔdB|, red=↑med blue=↓med)", fontsize=4.5, fontweight="bold")
    ax3.grid(True, axis="y", alpha=0.15, linewidth=0.2)
    ax3.axhline(0, color="gray", linewidth=0.3)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ── (4) Event Duration — right column ────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    # EEG bands in freq order + EOG bands
    dur_items = []
    for band, _ in eeg_sorted:
        if band in band_names_dur:
            bi = band_names_dur.index(band)
            dur = eeg_duration[:, bi]
            dur_items.append((greek(band), np.median(dur)*100,
                              np.percentile(dur, 25)*100, np.percentile(dur, 75)*100,
                              "#555555"))

    for label, dur_arr in eog_durations.items():
        short = label.split("\n")[0]
        dur_items.append((short, np.median(dur_arr)*100,
                          np.percentile(dur_arr, 25)*100, np.percentile(dur_arr, 75)*100,
                          CH_COLORS["EOG"]))

    x4 = np.arange(len(dur_items))
    meds4 = [d[1] for d in dur_items]
    err_lo = [d[1] - d[2] for d in dur_items]
    err_hi = [d[3] - d[1] for d in dur_items]
    col4 = [d[4] for d in dur_items]

    ax4.bar(x4, meds4, color=col4, width=0.7, edgecolor="white", linewidth=0.2, alpha=0.8)
    ax4.errorbar(x4, meds4, yerr=[err_lo, err_hi],
                 fmt="none", ecolor="gray", elinewidth=0.3, capsize=1.5, capthick=0.3)
    ax4.set_xticks(x4)
    ax4.set_xticklabels([d[0] for d in dur_items], fontsize=4, rotation=45, ha="right")
    ax4.set_ylabel("% epoch active", fontsize=4.5)
    ax4.set_ylim(0, 105)
    ax4.axhline(50, color="gray", linewidth=0.3, linestyle="--", alpha=0.4)
    ax4.set_title("(4) Event duration\n(% frames > mean)", fontsize=5, fontweight="bold")
    ax4.grid(True, axis="y", alpha=0.15, linewidth=0.2)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    stem = out_dir / f"proto_{k:03d}_card"
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)
    print(f"  Saved {stem}.{{pdf,png}}")


def main():
    parser = argparse.ArgumentParser(description="Generate prototype card figures")
    parser.add_argument("method_dir", type=Path)
    parser.add_argument("--proto_idx", type=int, default=None)
    args = parser.parse_args()

    method_dir = args.method_dir
    out_dir = method_dir / "cards"

    spec_dir = method_dir / "spectral_analysis" / "statistics"
    all_specs = []
    if spec_dir.exists():
        for f in sorted(spec_dir.glob("proto_*.json")):
            with open(f) as fh:
                all_specs.append(json.load(fh))

    if args.proto_idx is not None:
        proto_indices = [args.proto_idx]
    else:
        proto_indices = sorted(
            int(p.name.split("_")[1])
            for p in method_dir.glob("proto_*")
            if (p / "epochs.npy").exists()
        )

    print(f"Generating {len(proto_indices)} cards from {method_dir.name}...")
    for k in proto_indices:
        generate_card(k, method_dir, out_dir, all_specs=all_specs)
    print(f"Done. Cards at {out_dir}/")


if __name__ == "__main__":
    main()
