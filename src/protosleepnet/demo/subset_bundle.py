"""Build the curated, anonymized demo bundle from the full 153-subject bundle.

CPU-only, no GPU: the full bundle already holds every array + per-subject raw +
per-epoch IG for both backbones. We keep only the chosen recordings' epochs
(the PaCMAP coordinates stay the WHOLE-DATASET fit, just row-filtered), copy
only their raw / per-epoch-IG files, and rewrite every identifier to a neutral
label ("Recording A".."Recording D") so nothing reveals the source dataset.

Whole-dataset properties (``nn_proto_agreement`` faithfulness, the prototype
cards' spectral signatures/purity) are preserved; ``n_epochs``/``accuracy`` are
recomputed over the featured recordings.

Usage:
    python -m protosleepnet.demo.subset_bundle \
        --full_bundle /.../demo/bundle --out /.../demo/bundle_curated \
        --ids SC4012E0,SC4022E0,...            # from select_subjects.py
"""
import argparse
import json
import shutil
import string
from pathlib import Path

import numpy as np

MODEL_HF = {"seq": "protosleepnet-gagliardi", "st": "protosleeptransformer-gagliardi"}
MASK_U8 = 255
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

# per-model global arrays: filename -> (dtype, ncols)
ARRAYS = {
    "xy.f32": ("<f4", 2), "label.u8": ("u1", 1), "pred.u8": ("u1", 1),
    "proto.u8": ("u1", 1), "proba.u8": ("u1", 5), "dist.f32": ("<f4", 1),
    "subj.u16": ("<u2", 1), "epoch.u16": ("<u2", 1),
}
# prototype-level files copied verbatim (whole-dataset properties)
PROTO_FILES = ["reconstructions.f16", "recon_timeseries.f16", "ig_attr.f16", "ig_epoch.f16"]


def _read(path, dtype, ncols):
    a = np.fromfile(str(path), dtype=np.dtype(dtype))
    return a.reshape(-1, ncols) if ncols > 1 else a


def anon_labels(n):
    return [f"Recording {string.ascii_uppercase[i]}" for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ids", required=True, help="comma-separated source ids to feature")
    args = ap.parse_args()

    full, out = args.full_bundle, args.out
    chosen = [s.strip() for s in args.ids.split(",") if s.strip()]
    labels = anon_labels(len(chosen))
    anon = dict(zip(chosen, labels))              # source id -> neutral label
    print(f"[subset] featuring {len(chosen)} recordings -> {labels}")

    # canonical signals order -> global subject index (what subj.u16 references)
    signals = json.loads((full / "signals" / "subjects.json").read_text())
    gpos = {s["id"]: i for i, s in enumerate(signals)}
    missing = [c for c in chosen if c not in gpos]
    if missing:
        raise SystemExit(f"[subset] ids not in bundle: {missing}")
    chosen_pos = [gpos[c] for c in chosen]
    # global position -> new 0..k-1 index (dense remap for subj.u16)
    remap = np.full(len(signals), -1, dtype=np.int64)
    for new_i, gp in enumerate(chosen_pos):
        remap[gp] = new_i

    (out / "signals" / "subjects").mkdir(parents=True, exist_ok=True)

    # ── shared signals: subjects.json + raw waveforms (renamed) ──
    sig_meta = []
    for c in chosen:
        src = next(s for s in signals if s["id"] == c)
        sig_meta.append({"id": anon[c], "n_epochs": src["n_epochs"], "labels": src["labels"]})
        shutil.copyfile(full / "signals" / "subjects" / f"{c}.raw.bin",
                        out / "signals" / "subjects" / f"{anon[c]}.raw.bin")
    (out / "signals" / "subjects.json").write_text(json.dumps(sig_meta))

    # STFT reference: keep the numeric fixtures, scrub subject ids
    ref = json.loads((full / "signals" / "stft_reference.json").read_text())
    for i, s in enumerate(ref.get("samples", [])):
        s["subject"] = f"ref{i}"
    (out / "signals" / "stft_reference.json").write_text(json.dumps(ref))

    manifest = json.loads((full / "manifest.json").read_text())
    manifest["dataset"] = ""                       # scrub dataset name
    manifest["n_subjects"] = len(chosen)
    manifest["featured"] = labels
    manifest["projection_note"] = "PaCMAP fit on the whole cohort; only featured recordings are shown"

    # ── per-model subset ──
    for backbone, hf in MODEL_HF.items():
        mdir = full / hf
        odir = out / hf
        (odir / "ig").mkdir(parents=True, exist_ok=True)

        arrs = {name: _read(mdir / name, dt, nc) for name, (dt, nc) in ARRAYS.items()}
        subj = arrs["subj.u16"].astype(np.int64)
        keep_idx = np.where(np.isin(subj, chosen_pos))[0]
        # The full arrays are grouped by subject in FULL canonical order; reorder
        # the kept rows into the chosen (--ids) order so they line up with the
        # subjects.json we emit and the app's contiguous per-subject ranges. A
        # stable sort on the remapped index preserves within-subject epoch order.
        new_subj_all = remap[subj[keep_idx]]
        keep_idx = keep_idx[np.argsort(new_subj_all, kind="stable")]
        print(f"[{hf}] keep {keep_idx.size}/{subj.size} epochs")

        new_subj = remap[subj[keep_idx]].astype(np.uint16)
        sub = {name: a[keep_idx] for name, a in arrs.items()}
        sub["subj.u16"] = new_subj

        sub["xy.f32"].astype("<f4").tofile(str(odir / "xy.f32"))
        sub["dist.f32"].astype("<f4").tofile(str(odir / "dist.f32"))
        for name in ("label.u8", "pred.u8", "proto.u8", "proba.u8"):
            np.ascontiguousarray(sub[name].astype(np.uint8)).tofile(str(odir / name))
        for name in ("subj.u16", "epoch.u16"):
            np.ascontiguousarray(sub[name].astype("<u2")).tofile(str(odir / name))

        # prototypes.json — rename dataset-leaking keys, keep whole-cohort values
        protos = json.loads((mdir / "prototypes.json").read_text())
        for p in protos:
            if "sleepedf_label_distribution" in p:
                p["cohort_label_distribution"] = p.pop("sleepedf_label_distribution")
            if "sleepedf_cluster_size" in p:
                p["cohort_cluster_size"] = p.pop("sleepedf_cluster_size")
            # nested cross-dataset metrics carry dataset names in their keys
            if isinstance(p.get("cross"), dict):
                p["cross"] = {k.replace("sleepedf", "cohort"): v for k, v in p["cross"].items()}
        blob = json.dumps(protos)
        assert "sleepedf" not in blob.lower(), f"{hf}: prototypes still leak dataset name"
        (odir / "prototypes.json").write_text(blob)

        for f in PROTO_FILES:
            src = mdir / f
            if src.exists():
                shutil.copyfile(src, odir / f)

        # per-model subjects.json — anonymize + reindex, recompute accuracy
        smeta = json.loads((mdir / "subjects.json").read_text())["subjects"]
        by_id = {s["id"]: s for s in smeta}
        new_subjects = []
        n_ep = 0
        correct = total = 0
        for new_i, c in enumerate(chosen):
            s = by_id[c]
            new_subjects.append({"id": anon[c], "idx": new_i,
                                 "n_epochs": s["n_epochs"], "pred": s["pred"]})
            n_ep += s["n_epochs"]
            lab = np.asarray(next(x for x in signals if x["id"] == c)["labels"], dtype=np.int64)
            pred = np.asarray(s["pred"], dtype=np.int64)
            m = min(lab.size, pred.size)
            v = lab[:m] != -1
            correct += int((pred[:m][v] == lab[:m][v]).sum()); total += int(v.sum())
            # copy per-epoch IG
            shutil.copyfile(mdir / "ig" / f"{c}.ig.bin", odir / "ig" / f"{anon[c]}.ig.bin")
        (odir / "subjects.json").write_text(json.dumps({"subjects": new_subjects}))

        acc = correct / total if total else 0.0
        manifest["models"][hf].update(n_epochs=n_ep, n_subjects=len(chosen), accuracy=acc)
        print(f"[{hf}] featured accuracy = {acc:.4f} over {total} scored epochs")

    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"[subset] wrote curated bundle to {out}")


if __name__ == "__main__":
    main()
