"""In-domain demo bundle: feature each model's own TRAINING subjects.

psn (SeqSleepNet backbone) is showcased on its MASS train split; pst
(SleepTransformer backbone) on its SHHS train split. One backbone+dataset per
invocation (run twice into the same --out_dir).

Compliance: the source datasets (MASS, SHHS) are not openly redistributable, so
this bundle ships ONLY anonymized derived artifacts — embeddings/atlas coords,
predictions, prototype matches, per-epoch IG, and the model-input LOG-POWER
SPECTROGRAM (phase-less, non-invertible). It never ships the raw waveform, and
nothing names the source dataset.

Pipeline (per run):
  1. select  — score a seeded sample of train subjects (voting accuracy + a real
     sleep-architecture gate), pick the 4 cleanest to feature.
  2. atlas   — embed a stratified ~120k-epoch sample of the train split (incl.
     the 4 featured), fit PaCMAP on union(sample, codebook); keep the featured
     coords + the nearest-prototype faithfulness on the sample.
  3. featured — per featured subject: non-quantized voting prediction, L2 proto
     assignment + distance, per-epoch Integrated Gradients, and the input
     spectrogram (shipped, replaces raw).

Usage:
  python -m protosleepnet.demo.precompute_indomain --out_dir .../bundle_indomain \
     --backbone seq --dataset mass --gpu_id 0 \
     --checkpoint .../protosleepnet-seq-3ch-mixer/model.pt \
     --codebook  .../protosleepnet-seq-3ch-mixer/vq_kmeans/12/codebook.npy \
     --recon_bundle .../demo/bundle    # existing bundle to copy proto-level recon f16 from
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from physioex.data.collate import stack_channels, dict_collate_fn
from physioex.data.datasets import get_dataset

from protosleepnet.proto_reconstruction.utils import (
    load_frozen_model, load_codebook, compute_l2_sq_distances_np,
)
from protosleepnet.demo import pack
from protosleepnet.demo.build_cards import build_prototype_cards
from protosleepnet.demo.precompute import (
    MODELS, STAGE_NAMES, CHANNELS, IGNORE, MASK_U8,
    trim_window, predict_subject, embed_subject, epoch_ig_subject,
)

RECON_FILES = ["reconstructions.f16", "recon_timeseries.f16", "ig_attr.f16", "ig_epoch.f16"]


def anon_label(i):
    return f"Recording {chr(ord('A') + i)}"


def _loader(ds, flats):
    return DataLoader(Subset(ds, list(flats)), batch_size=1, shuffle=False,
                      collate_fn=dict_collate_fn)


def _subject(batch):
    """(sid, inputs (1,n,C,T,F) float tensor, labels (n,) int)."""
    sid = batch["subject"][0]["id"]
    x = stack_channels(batch)                       # (1, n, C, T, F)
    y = batch["labels"].reshape(-1).cpu().numpy().astype(np.int64)
    return str(sid), x, y


def clean_score(y, proba):
    """Return (accuracy, passes_gate, fracs) for one subject."""
    scored = y != IGNORE
    n = int(scored.sum())
    if n < 200:
        return 0.0, False, {}
    pred = proba.argmax(1)
    acc = float((pred[scored] == y[scored]).mean())
    f = {s: float((y[scored] == c).sum()) / n for c, s in enumerate(STAGE_NAMES)}
    present = len({int(c) for c in y[scored]}) == 5
    gate = (present and f["W"] <= 0.55 and f["N2"] >= 0.15
            and f["N3"] >= 0.03 and f["REM"] >= 0.06 and f["N1"] <= 0.25)
    return acc, gate, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--backbone", required=True, choices=["seq", "st"])
    ap.add_argument("--dataset", required=True, choices=["mass", "shhs"])
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("--recon_bundle", type=Path, required=True,
                    help="existing bundle dir to copy proto-level recon f16 from")
    ap.add_argument("--committed_root", type=Path,
                    default=Path("data/reconstructions/M12"))
    ap.add_argument("--cohort", type=int, default=1)   # MASS
    ap.add_argument("--visit", type=int, default=1)    # SHHS
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_featured", type=int, default=4)
    ap.add_argument("--n_candidates", type=int, default=80)
    ap.add_argument("--atlas_epochs", type=int, default=120000)
    ap.add_argument("--ig_steps", type=int, default=64)
    ap.add_argument("--ig_group", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.backends.cudnn.enabled = False  # LSTM backward in eval (IG)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    cfg = MODELS[args.backbone]
    hf, committed, L = cfg["hf"], cfg["committed"], cfg["seq_len"]
    print(f"[{hf}] backbone={args.backbone} dataset={args.dataset} L={L} device={device}")

    model = load_frozen_model(args.backbone, device, checkpoint_path=args.checkpoint)
    codebook = load_codebook(args.backbone, codebook_path=args.codebook)
    model.set_codebook(codebook)
    M = codebook.shape[0]
    assert M == 12, f"expected M=12, got {M}"

    dkw = {"cohort": args.cohort} if args.dataset == "mass" else {"visit": args.visit}
    DS = get_dataset(args.dataset)
    ds = DS(channels=CHANNELS, pipelines="seqsleepnet", sequence_length=0, **dkw)
    train_flat, _, _ = ds.split(fold=args.fold)
    train_flat = list(train_flat)
    print(f"[{hf}] {len(train_flat)} train subjects")

    # ── 1. select the 4 cleanest featured subjects ──
    cand = train_flat[:] ; rng.shuffle(cand) ; cand = cand[:args.n_candidates]
    scored = []
    for flat, batch in zip(cand, _loader(ds, cand)):
        _sid, x, y = _subject(batch)
        proba = predict_subject(model, x, L, device, quantize=False)
        s, e = trim_window(y)
        acc, gate, fr = clean_score(y[s:e], proba[s:e])
        scored.append(dict(flat=flat, sid=_sid, acc=acc, gate=gate, fr=fr))
    good = [r for r in scored if r["gate"]]
    good.sort(key=lambda r: r["acc"], reverse=True)
    featured = good[:args.n_featured]
    if len(featured) < args.n_featured:  # relax if too few pass
        rest = [r for r in scored if not r["gate"]]
        rest.sort(key=lambda r: r["acc"], reverse=True)
        featured += rest[: args.n_featured - len(featured)]
    print(f"[{hf}] {len(good)}/{len(scored)} candidates pass gate; featured:")
    for i, r in enumerate(featured):
        print(f"   {anon_label(i)} <- {r['sid']} acc={r['acc']:.3f} "
              f"W={r['fr'].get('W',0):.2f} N2={r['fr'].get('N2',0):.2f} "
              f"N3={r['fr'].get('N3',0):.2f} REM={r['fr'].get('REM',0):.2f}")
    featured_flats = [r["flat"] for r in featured]

    # ── 2. atlas: stratified embedding sample + PaCMAP fit ──
    per_subj_cap = max(1, args.atlas_epochs // max(1, min(len(train_flat), 200)))
    atlas_pool = train_flat[:] ; rng.shuffle(atlas_pool)
    H_list, LBL_list = [], []
    feat_cache = {}  # flat -> dict(sid,x,y,h)
    total = 0
    # always include the featured subjects fully (remember their slices)
    feat_slices = {}
    for flat in featured_flats:
        sid, x, y = _subject(next(iter(_loader(ds, [flat]))))
        h = embed_subject(model, x, device)
        s, e = trim_window(y)
        feat_cache[flat] = dict(sid=sid, x=x, y=y, h=h, s=s, e=e)
        feat_slices[flat] = (total, total + (e - s))
        H_list.append(h[s:e]); LBL_list.append(y[s:e]); total += (e - s)
    for flat in atlas_pool:
        if total >= args.atlas_epochs:
            break
        if flat in feat_cache:
            continue
        sid, x, y = _subject(next(iter(_loader(ds, [flat]))))
        h = embed_subject(model, x, device)
        s, e = trim_window(y)
        h, y = h[s:e], y[s:e]
        take = min(len(h), per_subj_cap, args.atlas_epochs - total)
        if take <= 0:
            continue
        sel = np.sort(np.asarray(rng.sample(range(len(h)), take)))
        H_list.append(h[sel]); LBL_list.append(y[sel]); total += take
    H = np.concatenate(H_list, 0).astype(np.float32)
    LBL_sample = np.concatenate(LBL_list, 0)
    print(f"[{hf}] atlas sample = {H.shape[0]} epochs; fitting PaCMAP ...")

    from pacmap import PaCMAP
    reducer = PaCMAP(n_components=2, random_state=args.seed)
    emb = np.asarray(reducer.fit_transform(np.vstack([H, codebook]).astype(np.float32),
                                           init="pca"), dtype=np.float32)
    XY_all, PROTO_XY = emb[:len(H)], emb[len(H):]
    lo, hi = XY_all.min(0), XY_all.max(0)
    span = np.maximum(hi - lo, 1e-6)
    norm = lambda a: ((a - lo) / span * 2 - 1).astype(np.float32)
    XY_all, PROTO_XY = norm(XY_all), norm(PROTO_XY)

    # sample proto assignment (for faithfulness + cohort stats)
    PROTO_sample = compute_l2_sq_distances_np(H, codebook).argmin(1)
    agree = float((compute_l2_sq_distances_np(XY_all, PROTO_XY).argmin(1) == PROTO_sample).mean())
    print(f"[{hf}] PaCMAP nearest-prototype agreement = {agree:.3f}")

    # ── 3. featured arrays + spectrograms + IG ──
    mdir = args.out_dir / hf
    (mdir / "spec").mkdir(parents=True, exist_ok=True)
    (mdir / "ig").mkdir(parents=True, exist_ok=True)
    xs = dict(xy=[], label=[], pred=[], proba=[], proto=[], dist=[], subj=[], epoch=[])
    subjects_meta = []
    for i, flat in enumerate(featured_flats):
        c = feat_cache[flat]
        sid, x, y, s, e = c["sid"], c["x"], c["y"], c["s"], c["e"]
        label = anon_label(i)
        h = c["h"][s:e]
        proba = predict_subject(model, x, L, device, quantize=False)[s:e]
        proto = compute_l2_sq_distances_np(h, codebook).argmin(1)
        dist = np.sqrt(np.clip(
            compute_l2_sq_distances_np(h, codebook)[np.arange(len(h)), proto], 0, None))
        xy = XY_all[slice(*feat_slices[flat])]
        n = len(h)
        assert n == (e - s) == len(xy), f"{label}: length mismatch"

        # input spectrogram (n, C, T, F) — shipped (derived, phase-less)
        spec = x.squeeze(0).cpu().numpy()[s:e].astype(np.float32)   # (n, C, T, F)
        pack.write_f16(mdir / "spec" / f"{label}.spec.bin", spec)
        # per-epoch IG toward each epoch's matched prototype
        ig = epoch_ig_subject(model, x[:, s:e], codebook[proto], device,
                              steps=args.ig_steps, group=args.ig_group)
        pack.write_f16(mdir / "ig" / f"{label}.ig.bin", ig.astype(np.float32))

        lbl = np.where(y[s:e] == IGNORE, MASK_U8, y[s:e])
        xs["xy"].append(xy); xs["label"].append(lbl); xs["pred"].append(proba.argmax(1))
        xs["proba"].append(proba); xs["proto"].append(proto); xs["dist"].append(dist)
        xs["subj"].append(np.full(n, i)); xs["epoch"].append(np.arange(n))
        subjects_meta.append({"id": label, "idx": i, "n_epochs": int(n),
                              "pred": proba.argmax(1).astype(int).tolist(),
                              "labels": y[s:e].astype(int).tolist()})
        print(f"[{hf}] {label}: {n} epochs, proto {proto.min()}..{proto.max()}")

    pack.write_f32(mdir / "xy.f32", np.concatenate(xs["xy"]))
    pack.write_u8(mdir / "label.u8", np.concatenate(xs["label"]))
    pack.write_u8(mdir / "pred.u8", np.concatenate(xs["pred"]))
    pack.write_u8(mdir / "proba.u8", pack.quantize_proba_u8(np.concatenate(xs["proba"])))
    pack.write_u8(mdir / "proto.u8", np.concatenate(xs["proto"]))
    pack.write_f32(mdir / "dist.f32", np.concatenate(xs["dist"]).astype(np.float32))
    pack.write_u16(mdir / "subj.u16", np.concatenate(xs["subj"]))
    pack.write_u16(mdir / "epoch.u16", np.concatenate(xs["epoch"]))
    (mdir / "subjects.json").write_text(json.dumps({"subjects": subjects_meta}))

    # ── prototype cards: base (committed) + xy + cohort stats, scrub dataset names ──
    cards = build_prototype_cards(args.committed_root / committed, m=M)
    for k in range(M):
        cards[k]["xy"] = [float(PROTO_XY[k, 0]), float(PROTO_XY[k, 1])]
        sel = LBL_sample[(PROTO_sample == k) & (LBL_sample != IGNORE)]
        cards[k]["cohort_label_distribution"] = {STAGE_NAMES[c]: int((sel == c).sum()) for c in range(5)}
        cards[k]["cohort_cluster_size"] = int((PROTO_sample == k).sum())
        if isinstance(cards[k].get("cross"), dict):
            cards[k]["cross"] = {kk.replace("sleepedf", "cohort"): vv for kk, vv in cards[k]["cross"].items()}
    blob = json.dumps(cards)
    assert "sleepedf" not in blob.lower(), "prototype cards still leak dataset name"
    (mdir / "prototypes.json").write_text(blob)

    # ── copy proto-level reconstruction f16 (dataset-independent) ──
    for f in RECON_FILES:
        src = args.recon_bundle / hf / f
        if src.exists():
            shutil.copyfile(src, mdir / f)

    # ── merge manifest ──
    mpath = args.out_dir / "manifest.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {
        "dataset": "", "stages": STAGE_NAMES, "channels": CHANNELS, "fs": 100,
        "epoch_sec": 30, "mask_u8": MASK_U8,
        "spec": {"dtype": "float16", "shape": [3, 29, 129], "layout": "<model>/spec/<id>.spec.bin (n,3,29,129)"},
        "stft": {"nperseg": 200, "noverlap": 100, "nfft": 256, "window": "hamming",
                 "n_time": 29, "n_freq": 129, "log_scale": "10*log10(|X|^2)"},
        "proto": {"m": 12, "match": "argmin squared-L2"}, "projection": "pacmap",
        "models": {}, "per_model_subjects": True,
    }
    manifest.setdefault("models", {})
    acc_valid = np.concatenate(xs["label"]) != MASK_U8
    acc = float((np.concatenate(xs["pred"])[acc_valid] == np.concatenate(xs["label"])[acc_valid]).mean())
    manifest["models"][hf] = {
        "backbone": args.backbone, "seq_len": L, "projection": "pacmap",
        "n_epochs": int(sum(m["n_epochs"] for m in subjects_meta)),
        "n_subjects": len(subjects_meta), "accuracy": acc,
        "nn_proto_agreement": agree, "atlas_sample_epochs": int(H.shape[0]),
    }
    mpath.write_text(json.dumps(manifest))
    print(f"[{hf}] done. featured acc={acc:.4f} agreement={agree:.3f}")


if __name__ == "__main__":
    main()
