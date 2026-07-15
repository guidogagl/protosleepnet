"""Clinical-plausibility audit of the per-epoch input->prototype explanations.

For every epoch of the featured recordings we ask a concrete question: does the
per-epoch Integrated-Gradients relevance land where the matched prototype's
sleep stage says it should? IG gives relevance *magnitude* (not direction), so
we test two things a clinician would check:

  1. EEG band  — the top EEG relevance band is one of the stage's expected
     sensitivity bands (``EXPECTED_TOP_BANDS``), e.g. N2->sigma (spindles),
     N3->delta, REM/N1->theta, W->alpha/beta.
  2. Channel   — relevance sits on the physiologically expected channel:
     REM is EOG-driven (eye movements), Wake shows EMG/EOG activity, and
     N1/N2/N3 are EEG-dominant.

CPU-only: reads the curated bundle (per-epoch IG already precomputed). Emits a
small per-recording ``plausibility/<id>.json`` (lazy-loaded by the demo for the
epoch-detail badge) and a human-readable ``docs/explanation_audit.md``.

Band definitions mirror ``physioex/explain/foundational/sleep_bands.py`` and the
stage priors mirror ``figure_reconstruction/rule_learning.py`` (inlined here so
the audit runs without importing physioex).

Usage:
    python -m protosleepnet.demo.evaluate_explanations \
        --bundle /.../bundle_curated --docs_out /.../protosleepnet/docs/explanation_audit.md
"""
import argparse
import json
from pathlib import Path

import numpy as np

MODEL_HF = {"seq": "protosleepnet-gagliardi", "st": "protosleeptransformer-gagliardi"}
CHANNELS = ["EEG", "EOG", "EMG"]
FS, NFFT, N_FREQ, N_TIME = 100.0, 256, 129, 29

# AASM bands kept over 0..129 rfft bins (mirror sleep_bands.SLEEP_BANDS).
# The 50 Hz "mains" powerline band is intentionally excluded — it is a recording
# artifact, not physiology, and would otherwise mask the informative bands.
_BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 11.0),
          ("sigma_low", 11.0, 13.0), ("sigma_high", 13.0, 16.0),
          ("beta_low", 16.0, 20.0), ("beta_high", 20.0, 30.0),
          ("gamma", 30.0, 45.0)]
_RES = FS / NFFT
BANDS = [(n, max(0, round(lo / _RES)), min(N_FREQ, round(hi / _RES))) for n, lo, hi in _BANDS]
BAND_NAMES = [b[0] for b in BANDS]

# expected top sensitivity bands per stage (rule_learning.EXPECTED_TOP_BANDS)
EXPECTED_TOP_BANDS = {
    "W": {"alpha", "beta_low", "beta_high", "gamma"},
    "N1": {"theta", "alpha", "delta"},
    "N2": {"sigma_low", "sigma_high", "delta", "theta"},
    "N3": {"delta"},
    "REM": {"theta", "alpha", "sigma_high"},
}


def _half_to_f32(path, per_epoch_floats):
    """Read a float16 (.ig.bin) file -> float32 array (n, 3, T, F)."""
    raw = np.fromfile(str(path), dtype="<f2")
    n = raw.size // per_epoch_floats
    return raw.reshape(n, 3, N_TIME, N_FREQ).astype(np.float32)


def _channel_score(stage, eeg, eog, emg):
    """Stage-appropriate channel expectation -> (ok, score in [0,1])."""
    if stage == "REM":
        return eog >= 0.25, min(1.0, eog / 0.30)
    if stage == "W":
        return (emg >= 0.20 or eog >= 0.25), min(1.0, max(emg, eog) / 0.25)
    # N1 / N2 / N3 -> EEG must dominate
    return eeg >= max(eog, emg), eeg


def eval_epoch(attr, stage):
    """attr = |IG| (3, T, F). Returns a per-epoch plausibility record."""
    ch = attr.sum(axis=(1, 2))                       # EEG, EOG, EMG totals
    ch_frac = ch / (ch.sum() + 1e-9)
    eeg_tf = attr[0]                                 # (T, F)
    # relevance DENSITY (mean per bin) so wide bands (gamma) aren't favoured
    # over narrow ones (delta, sigma) purely by bin count.
    band_rel = np.array([eeg_tf[:, a:b].mean() for _, a, b in BANDS])
    band_frac = band_rel / (band_rel.sum() + 1e-9)
    top = int(band_frac.argmax())
    expected = EXPECTED_TOP_BANDS.get(stage, set())
    exp_idx = [i for i, n in enumerate(BAND_NAMES) if n in expected]
    band_frac_exp = float(band_frac[exp_idx].sum()) if exp_idx else 0.0
    band_ok = BAND_NAMES[top] in expected or band_frac_exp >= 0.5

    ch_ok, ch_score = _channel_score(stage, *ch_frac)
    score = 0.5 * band_frac_exp + 0.5 * ch_score
    return {
        "st": stage, "tb": BAND_NAMES[top], "ex": sorted(expected),
        "ok": int(bool(band_ok and ch_ok)), "sc": round(float(score), 3),
        "ch": {c: round(float(f), 3) for c, f in zip(CHANNELS, ch_frac)},
        "bok": int(bool(band_ok)), "cok": int(bool(ch_ok)),
    }


def audit_model(bundle: Path, hf: str):
    mdir = bundle / hf
    proto = np.fromfile(str(mdir / "proto.u8"), dtype=np.uint8).astype(np.int64)
    subj = np.fromfile(str(mdir / "subj.u16"), dtype="<u2").astype(np.int64)
    epoch = np.fromfile(str(mdir / "epoch.u16"), dtype="<u2").astype(np.int64)
    label = np.fromfile(str(mdir / "label.u8"), dtype=np.uint8).astype(np.int64)
    cards = json.loads((mdir / "prototypes.json").read_text())
    dom = {c["idx"]: c["dominant_stage"] for c in cards}
    subjects = json.loads((mdir / "subjects.json").read_text())["subjects"]

    per_epoch_floats = 3 * N_TIME * N_FREQ
    (mdir / "plausibility").mkdir(exist_ok=True)
    summaries = []
    for s in subjects:
        sid, j = s["id"], s["idx"]
        rows = np.where(subj == j)[0]
        rows = rows[np.argsort(epoch[rows])]
        protos_j = proto[rows]
        ig = _half_to_f32(mdir / "ig" / f"{sid}.ig.bin", per_epoch_floats)
        assert ig.shape[0] == len(rows), f"{hf}/{sid}: ig {ig.shape[0]} vs rows {len(rows)}"

        recs = [eval_epoch(np.abs(ig[e]), dom[int(protos_j[e])]) for e in range(len(rows))]
        (mdir / "plausibility" / f"{sid}.json").write_text(json.dumps(recs))

        oks = np.array([r["ok"] for r in recs])
        # night coherence: mean normalized position of each stage's matched prototypes
        pos = epoch[rows][np.argsort(epoch[rows])].astype(float)
        pos = (pos - pos.min()) / (pos.ptp() + 1e-9)
        stage_pos = {}
        for st in ("N3", "REM"):
            m = np.array([r["st"] == st for r in recs])
            if m.any():
                stage_pos[st] = float(pos[m].mean())
        summaries.append({
            "id": sid, "n": len(recs), "frac_plausible": round(float(oks.mean()), 3),
            "frac_band_ok": round(float(np.mean([r["bok"] for r in recs])), 3),
            "frac_ch_ok": round(float(np.mean([r["cok"] for r in recs])), 3),
            "stage_pos": stage_pos,
        })
    return summaries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--docs_out", required=True, type=Path)
    args = ap.parse_args()

    lines = ["# Local explanation audit (per-epoch IG vs. sleep physiology)", "",
             "For each featured recording we test whether the per-epoch Integrated-Gradients",
             "relevance concentrates on the frequency band and channel that the matched",
             "prototype's stage predicts (spindles for N2, delta for N3, EOG for REM, ...).",
             "This is an honest sanity check of the *local* explanations, not a cherry-pick.",
             ""]
    for backbone, hf in MODEL_HF.items():
        summ = audit_model(args.bundle, hf)
        lines.append(f"## {hf} ({backbone})")
        lines.append("")
        lines.append("| recording | epochs | plausible | band ok | channel ok | N3 pos | REM pos |")
        lines.append("|---|---|---|---|---|---|---|")
        for s in summ:
            n3 = f"{s['stage_pos'].get('N3', float('nan')):.2f}" if "N3" in s["stage_pos"] else "-"
            rem = f"{s['stage_pos'].get('REM', float('nan')):.2f}" if "REM" in s["stage_pos"] else "-"
            lines.append(f"| {s['id']} | {s['n']} | {s['frac_plausible']:.0%} | "
                         f"{s['frac_band_ok']:.0%} | {s['frac_ch_ok']:.0%} | {n3} | {rem} |")
        # coherence note
        for s in summ:
            sp = s["stage_pos"]
            if "N3" in sp and "REM" in sp:
                order = "N3 precedes REM (expected)" if sp["N3"] < sp["REM"] else "REM precedes N3 (atypical)"
                lines.append(f"- {s['id']}: {order} — N3 mean position {sp['N3']:.2f}, REM {sp['REM']:.2f}.")
        lines.append("")
        print(f"[{hf}] " + "; ".join(f"{s['id']}={s['frac_plausible']:.0%}" for s in summ))

    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text("\n".join(lines))
    print(f"[audit] wrote {args.docs_out}")


if __name__ == "__main__":
    main()
