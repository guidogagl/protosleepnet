import os
"""Compact 4-sentence rule learning for prototype matching.

S1: Identity + channel dominance
S2: Spectral signature + direction + delta sufficiency (ranked by sensitivity)
S3: Temporal event duration + EOG/EMG peripheral (with context caveat)
S4: Physiological coherence flag (including top-band check)

Usage:
    python rule_learning.py /path/to/method_dir/
    python rule_learning.py /path/to/method_dir/ --proto_idx 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PHYSIOEX_ROOT = None
for c in [Path(os.environ.get("PHYSIOEX_ROOT", "")), Path(os.environ.get("PHYSIOEX_ROOT", "")),
          Path(os.environ.get("PHYSIOEX_ROOT", ""))]:
    if (c / "physioex").is_dir():
        sys.path.insert(0, str(c)); break

from physioex.explain.foundational.sleep_bands import SLEEP_BANDS, bands_to_bin_ranges

FS, NFFT = 100.0, 256
FREQ_RES = FS / NFFT
N_FREQ = NFFT // 2 + 1
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

BAND_GREEK = {"delta": "δ", "theta": "θ", "alpha": "α",
              "sigma_low": "σ_l", "sigma_high": "σ_h",
              "beta_low": "β_l", "beta_high": "β_h", "gamma": "γ"}

EOG_BANDS = [
    ("SEM",      0, round(1.0 / FREQ_RES)),
    ("REM",      round(1.0 / FREQ_RES), round(5.0 / FREQ_RES)),
    ("Blink",    round(5.0 / FREQ_RES), round(10.0 / FREQ_RES)),
    ("Residual", round(10.0 / FREQ_RES), N_FREQ),
]
EMG_BIN_START = round(10.0 / FREQ_RES)
EMG_BIN_END = round(50.0 / FREQ_RES)

def greek(band): return BAND_GREEK.get(band, band)


# ── Fix 3: EOG expected activity per stage ───────────────────────────

EOG_EXPECTED = {
    "W":   {"SEM": False, "REM": False, "Blink": True,  "Residual": True},
    "N1":  {"SEM": True,  "REM": False, "Blink": False, "Residual": False},
    "N2":  {"SEM": False, "REM": False, "Blink": False, "Residual": False},
    "N3":  {"SEM": False, "REM": False, "Blink": False, "Residual": False},
    "REM": {"SEM": False, "REM": True,  "Blink": False, "Residual": False},
}

# Fix 6: preferred EOG band to report per stage
PREFERRED_EOG_BAND = {
    "W":  "Blink",
    "N1": "SEM",
    "N2": "SEM",
    "N3": "SEM",
    "REM": "REM",
}

# Fix 4: expected top sensitivity bands per stage
EXPECTED_TOP_BANDS = {
    "W":   {"alpha", "beta_low", "beta_high", "gamma"},
    "N1":  {"theta", "alpha", "delta"},
    "N2":  {"sigma_low", "sigma_high", "delta", "theta"},
    "N3":  {"delta"},
    "REM": {"theta", "alpha", "sigma_high"},
}


# ── Physiological checklist (Fix 1: fractions, Fix 4+5: top band) ────

PHYSIOLOGY_CHECKLIST = {
    "W": [
        ("α elevated+relevant", lambda d: d["direction"].get("alpha") == "elevated"
                                          and d["band_rel"].get("alpha", 0) > 10),
        ("EMG tonic",           lambda d: d["emg_pct_rank"] > 60),
        ("EOG active",          lambda d: d["eog_pct_rank"] > 60),
    ],
    "N1": [
        ("θ elevated",          lambda d: d["direction"].get("theta") == "elevated"),
        ("α attenuated",        lambda d: d["direction"].get("alpha") == "suppressed"
                                           or d["band_rel"].get("alpha", 0) < 5),
    ],
    "N2": [
        ("σ elevated",          lambda d: d["direction"].get("sigma_high") == "elevated"
                                           or d["direction"].get("sigma_low") == "elevated"),
        ("σ relevant",          lambda d: d["band_rel"].get("sigma_high", 0) > 15
                                           or d["band_rel"].get("sigma_low", 0) > 15),
        ("EOG quiescent",       lambda d: d["eog_pct_rank"] < 40),
    ],
    "N3": [
        ("δ elevated",          lambda d: d["direction"].get("delta") == "elevated"),
        ("δ dominant",          lambda d: d["band_rel"].get("delta", 0) > 30),
        ("EMG atonic",          lambda d: d["emg_pct_rank"] < 40),
    ],
    "REM": [
        ("EOG dominant",        lambda d: d["ch_pcts"].get("EOG", 0) > 0.30),
        ("EMG atonic",          lambda d: d["emg_pct_rank"] < 40),
        ("θ elevated",          lambda d: d["direction"].get("theta") == "elevated"),
    ],
}


# ── Event duration computation ───────────────────────────────────────

def compute_event_durations(epochs_np, training_mean_np):
    """Compute % of frames where band power exceeds training mean (dB)."""
    N, C, T, F = epochs_np.shape
    bin_ranges = [(n, s, e) for n, s, e in
                  bands_to_bin_ranges(SLEEP_BANDS, fs=FS, signal_length=NFFT)
                  if n != "mains"]

    if training_mean_np.ndim == 2:
        tm = training_mean_np
    else:
        tm = training_mean_np.mean(axis=1)

    eeg_dur = {}
    for bname, bstart, bend in bin_ranges:
        bp_db = epochs_np[:, 0, :, bstart:bend].mean(axis=-1)
        thresh = tm[0, bstart:bend].mean()
        eeg_dur[bname] = float(np.median((bp_db > thresh).mean(axis=1)) * 100)

    eog_dur = {}
    for label, bstart, bend in EOG_BANDS:
        eog_db = epochs_np[:, 1, :, bstart:bend].mean(axis=-1)
        thresh = tm[1, bstart:bend].mean()
        eog_dur[label] = float(np.median((eog_db > thresh).mean(axis=1)) * 100)

    return eeg_dur, eog_dur


# ── Sensitivity (Fix 2: soft floor) ─────────────────────────────────

SENSITIVITY_SOFT_FLOOR = 3.0  # dB — minimum meaningful spectral difference

def compute_band_sensitivity(abl_meta, proto_power, mean_power, band_names):
    """Sensitivity = importance / max(|ΔdB|, SOFT_FLOOR).

    Soft floor avoids discontinuity and explosion for small ΔdB values.
    """
    importance = abl_meta.get("marginal_importance", {})
    sensitivity = {}
    for bi, bn in enumerate(band_names):
        if bi >= len(proto_power):
            continue
        imp = importance.get(bn, 0)
        p_db = 10 * np.log10(proto_power[bi] + 1e-12)
        m_db = 10 * np.log10(mean_power[bi] + 1e-12)
        delta_db = abs(p_db - m_db)
        sensitivity[bn] = imp / max(delta_db, SENSITIVITY_SOFT_FLOOR)
    return sensitivity


# ── Sentence formatters ──────────────────────────────────────────────

def format_s1(k, abl_meta):
    """S1: Identity + channel."""
    stage = abl_meta.get("dominant_class", "?")
    purity = abl_meta.get("predicted_purity", 0)
    ch = abl_meta.get("channel_importance_pct", {})
    ch_sorted = sorted(ch.items(), key=lambda x: -abs(x[1]))
    if len(ch_sorted) < 3:
        ch_sorted = [("EEG", 0.33), ("EOG", 0.33), ("EMG", 0.33)]
    return (f"Prototype {k} encodes {stage} ({purity:.0%} pure) primarily through "
            f"{ch_sorted[0][0]} ({abs(ch_sorted[0][1]):.0%}), with "
            f"{ch_sorted[1][0]} ({abs(ch_sorted[1][1]):.0%}) and "
            f"{ch_sorted[2][0]} ({abs(ch_sorted[2][1]):.0%}) contribution.")


def format_s2(abl_meta, proto_power, mean_power, band_names, sensitivity):
    """S2: Spectral signature + direction, ranked by sensitivity.
    Fix 7: surfaces negative sensitivity as 'absence of'."""
    direction = abl_meta.get("feature_direction", {})

    # Top 2 by absolute sensitivity
    ranked = sorted(sensitivity.items(), key=lambda x: -abs(x[1]))[:2]

    parts = []
    for bn, sens in ranked:
        bi = band_names.index(bn) if bn in band_names else -1
        if bi >= 0 and bi < len(proto_power):
            pp, mp = proto_power[bi], mean_power[bi]
            diff = 10 * np.log10(pp + 1e-12) - 10 * np.log10(mp + 1e-12)
            # Fix 7: negative sensitivity = absence helps matching
            if sens < 0:
                parts.append(f"absence of {greek(bn)} ({diff:+.1f} dB, sensitivity={abs(sens):.1f})")
            else:
                d = direction.get(bn, "?")
                parts.append(f"{d} {greek(bn)} ({diff:+.1f} dB, sensitivity={abs(sens):.1f})")
        else:
            parts.append(f"{greek(bn)} (sensitivity={abs(sens):.1f})")

    sig = " and ".join(parts)

    # Delta sufficiency
    delta_suff = abl_meta.get("delta_sufficient", None)
    delta_frac = abl_meta.get("delta_still_assigned_frac", None)
    delta_clause = ""
    if delta_suff is not None:
        word = "sufficient" if delta_suff else "insufficient"
        delta_clause = f", with δ {word} alone ({delta_frac:.0%} retention)" if delta_frac is not None else ""

    return f"Its EEG signature shows {sig}{delta_clause}."


def format_s3(abl_meta, eeg_dur, eog_dur, eog_pct_rank, emg_pct_rank,
              sensitivity):
    """S3: Temporal event duration + peripheral.
    Fix 3: EOG caveat for unexpected context.
    Fix 6: preferred EOG band per stage."""
    stage = abl_meta.get("dominant_class", "?")

    # Top EEG band by sensitivity
    top_band = max(sensitivity, key=lambda b: abs(sensitivity[b])) if sensitivity else "delta"
    dur = eeg_dur.get(top_band, 50)
    temporal = "sustained" if dur > 70 else ("transient" if dur < 40 else "intermittent")

    eog_label = "active" if eog_pct_rank > 75 else ("quiescent" if eog_pct_rank < 25 else "moderate")
    emg_label = "tonic" if emg_pct_rank > 75 else ("atonic" if emg_pct_rank < 25 else "intermediate")

    # Fix 6: show preferred EOG band for this stage
    preferred = PREFERRED_EOG_BAND.get(stage, "SEM")
    eog_pct_val = eog_dur.get(preferred, 0)

    # Fix 3: caveat if preferred band has high activity in unexpected context
    eog_caveat = ""
    expected = EOG_EXPECTED.get(stage, {})
    if not expected.get(preferred, False) and eog_pct_val > 50:
        eog_caveat = " (likely EEG crosstalk)"

    return (f"The {greek(top_band)} activity is present in {dur:.0f}% of each epoch ({temporal}), "
            f"with {eog_label} eye movements ({preferred} {eog_pct_val:.0f}% active{eog_caveat}) "
            f"and {emg_label} muscle tone.")


def format_s4(abl_meta, eog_pct_rank, emg_pct_rank, sensitivity):
    """S4: Physiological coherence flag.
    Fix 1: correct fraction thresholds.
    Fix 4+5: includes top sensitivity band check."""
    stage = abl_meta.get("dominant_class", "?")
    ch_pcts = abl_meta.get("channel_importance_pct", {})
    band_rel = abl_meta.get("band_relevance_pct", {})
    direction = abl_meta.get("feature_direction", {})

    # Fix 5: include top sensitivity band
    top_sens_band = max(sensitivity, key=lambda b: abs(sensitivity[b])) if sensitivity else ""

    data = {
        "direction": direction,
        "band_rel": band_rel,
        "ch_pcts": ch_pcts,
        "eog_pct_rank": eog_pct_rank,
        "emg_pct_rank": emg_pct_rank,
        "top_sens_band": top_sens_band,
    }

    checklist = PHYSIOLOGY_CHECKLIST.get(stage, [])
    if not checklist:
        return f"No physiological checklist available for {stage}."

    passed_names = []
    failed_names = []
    for name, check in checklist:
        try:
            if check(data):
                passed_names.append(name)
            else:
                failed_names.append(name)
        except (KeyError, TypeError):
            failed_names.append(f"{name} (no data)")

    total = len(checklist)
    n_pass = len(passed_names)

    if n_pass == total:
        coherence = "coherent with"
        detail = ", ".join(passed_names) + " — all confirmed"
    elif n_pass >= total / 2:
        coherence = "partially coherent with"
        detail = "confirmed: " + ", ".join(passed_names) + "; missing: " + ", ".join(failed_names)
    else:
        coherence = "conflicting with"
        detail = "missing: " + ", ".join(failed_names)

    # Informational note: is the top sensitivity band physiologically expected?
    top_band = data.get("top_sens_band", "")
    expected_bands = EXPECTED_TOP_BANDS.get(stage, set())
    if top_band and top_band not in expected_bands:
        note = f" Note: model primarily uses {greek(top_band)}, atypical for {stage}."
    else:
        note = ""

    return f"This is {coherence} {stage} physiology: {detail}.{note}"


# ── Main API ─────────────────────────────────────────────────────────

def generate_compact_rule(k, abl_meta, spectral_stats, epochs_np,
                          training_mean_np, eog_pct_rank, emg_pct_rank):
    """Generate the 4-sentence rule for one prototype."""
    band_names = abl_meta.get("feature_names", [])
    proto_power = np.load(
        Path(abl_meta["_abl_dir"]) / "proto_power.npy"
    ) if "_abl_dir" in abl_meta else np.zeros(8)
    mean_power = np.load(
        Path(abl_meta["_abl_dir"]) / "mean_power.npy"
    ) if "_abl_dir" in abl_meta else np.zeros(8)

    eeg_dur, eog_dur = compute_event_durations(epochs_np, training_mean_np)
    sensitivity = compute_band_sensitivity(abl_meta, proto_power, mean_power, band_names)

    s1 = format_s1(k, abl_meta)
    s2 = format_s2(abl_meta, proto_power, mean_power, band_names, sensitivity)
    s3 = format_s3(abl_meta, eeg_dur, eog_dur, eog_pct_rank, emg_pct_rank,
                   sensitivity)
    s4 = format_s4(abl_meta, eog_pct_rank, emg_pct_rank, sensitivity)

    return f"{s1}\n{s2}\n{s3}\n{s4}"


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate compact 4-sentence rules")
    parser.add_argument("method_dir", type=Path)
    parser.add_argument("--proto_idx", type=int, default=None)
    args = parser.parse_args()

    method_dir = args.method_dir
    out_dir = method_dir / "rules"
    out_dir.mkdir(parents=True, exist_ok=True)

    tm_path = method_dir / "ablation" / "training_mean.npy"
    if not tm_path.exists():
        print(f"Error: {tm_path} not found")
        return
    training_mean = np.load(tm_path)

    spec_dir = method_dir / "spectral_analysis" / "statistics"
    all_specs = []
    if spec_dir.exists():
        for f in sorted(spec_dir.glob("proto_*.json")):
            with open(f) as fh:
                all_specs.append(json.load(fh))

    eog_vals = sorted([s.get("eog_total_power", 0) for s in all_specs]) if all_specs else [0]
    emg_vals = sorted([s.get("emg_tone", 0) for s in all_specs]) if all_specs else [0]

    if args.proto_idx is not None:
        proto_indices = [args.proto_idx]
    else:
        proto_indices = sorted(
            int(p.name.split("_")[1])
            for p in method_dir.glob("proto_*")
            if (p / "epochs.npy").exists()
        )

    all_rules = []

    for k in proto_indices:
        epochs = np.load(method_dir / f"proto_{k:03d}" / "epochs.npy")

        abl_path = method_dir / "ablation" / f"proto_{k:03d}" / "metadata.json"
        if not abl_path.exists():
            print(f"  P{k}: no ablation data, skipping")
            continue
        with open(abl_path) as f:
            abl_meta = json.load(f)
        abl_meta["_abl_dir"] = str(method_dir / "ablation" / f"proto_{k:03d}")

        spec_path = spec_dir / f"proto_{k:03d}.json" if spec_dir.exists() else None
        spec = {}
        if spec_path and spec_path.exists():
            with open(spec_path) as f:
                spec = json.load(f)

        eog_pct = np.searchsorted(eog_vals, spec.get("eog_total_power", 0)) / max(len(eog_vals), 1) * 100
        emg_pct = np.searchsorted(emg_vals, spec.get("emg_tone", 0)) / max(len(emg_vals), 1) * 100

        rule = generate_compact_rule(k, abl_meta, spec, epochs,
                                     training_mean, eog_pct, emg_pct)

        with open(out_dir / f"proto_{k:03d}_rule.txt", "w") as f:
            f.write(rule)
        rule_json = {
            "prototype_idx": k,
            "dominant_class": abl_meta.get("dominant_class"),
            "rule_text": rule,
            "s1": rule.split("\n")[0],
            "s2": rule.split("\n")[1],
            "s3": rule.split("\n")[2],
            "s4": rule.split("\n")[3],
        }
        with open(out_dir / f"proto_{k:03d}_rule.json", "w") as f:
            json.dump(rule_json, f, indent=2)
        all_rules.append(rule_json)

        print(f"  P{k} ({abl_meta.get('dominant_class', '?')}): {rule.split(chr(10))[-1][:70]}...")

    with open(out_dir / "all_rules.json", "w") as f:
        json.dump(all_rules, f, indent=2)

    with open(out_dir / "all_rules.md", "w") as f:
        f.write(f"# Prototype Rules — {method_dir.name}\n\n")
        for r in sorted(all_rules, key=lambda x: x["prototype_idx"]):
            f.write(f"### Prototype {r['prototype_idx']} — {r['dominant_class']}\n\n")
            f.write(f"> {r['rule_text'].replace(chr(10), chr(10) + '> ')}\n\n")

    print(f"\nSaved {len(all_rules)} rules to {out_dir}/")


if __name__ == "__main__":
    main()
