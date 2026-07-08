"""Precompute the static bundle for the ProtoSleepNet explainability demo.

Runs on the A30 (needs GPU + SleepEDF + the released checkpoints/codebooks).
Emits a GPU-free bundle the static web app consumes. SleepEDF only.

For each proto model (SeqSleepNet / SleepTransformer backbone) it computes,
over **all** SleepEDF subjects:
  - per-epoch embedding h = epoch_encode(x, quantize=False)      (matching space)
  - per-epoch NON-quantized prediction via sliding-window voting  (displayed)
  - nearest prototype (squared-L2 to the codebook) + distance     (the match)
  - a UMAP of all epoch embeddings, with the 12 prototypes co-embedded
  - the 12 prototype cards (assembled from committed reconstruction JSON)
Shared across models (same SleepEDF signals): per-subject raw waveform
(physioex ``raw`` pipeline, float16) + a small STFT reference fixture for
validating the in-browser spectrogram.

Usage (per backbone paths resolved on the A30):
    python -m protosleepnet.demo.precompute \
        --out_dir /path/to/bundle --gpu_id 0 \
        --checkpoint-seq .../protosleepnet-seq-3ch-mixer/model.pt \
        --codebook-seq  .../codebook_m12.npy \
        --checkpoint-st  .../protosleepnet-st-3ch-mixer/model.pt \
        --codebook-st   .../codebook_m12.npy
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from physioex.data.collate import stack_channels

from protosleepnet.proto_reconstruction.utils import (
    load_frozen_model, load_codebook, compute_l2_sq_distances_np,
    build_full_loader, get_paths,
)
from protosleepnet.demo import pack
from protosleepnet.demo.build_cards import build_prototype_cards

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
CHANNELS = ["EEG", "EOG", "EMG"]
DATASET = "sleepedf"
IGNORE = -1
MASK_U8 = 255  # uint8 sentinel for unlabeled epochs

# HF id / committed-dir / seq-len per backbone
MODELS = {
    "seq": dict(hf="protosleepnet-gagliardi",
                committed="protosleepnet-seq-3ch-mixer", seq_len=20),
    "st":  dict(hf="protosleeptransformer-gagliardi",
                committed="protosleepnet-st-3ch-mixer", seq_len=21),
}


# SleepEDF recordings are full ~24 h ambulatory cassettes; physioex's
# trim_excess_wake keeps only 30 min of wake around the sleep period and marks
# the rest as -1. We run the model on the full recording (correct voting
# context, matching the benchmark) but store only this trimmed window so the
# atlas/hypnogram/raw aren't dominated by day-time wake.
def trim_window(labels):
    idx = np.where(np.asarray(labels) != -1)[0]
    if len(idx) == 0:
        return 0, len(labels)
    return int(idx[0]), int(idx[-1]) + 1


# ── prediction: sliding-window voting, non-quantized ─────────────────────
@torch.no_grad()
def predict_subject(model, inputs, L, device, quantize=False):
    """Per-epoch softmax over a full night via overlapping-window voting.

    inputs: (1, night, C, T, F). Returns (night, n_classes) numpy.
    """
    inputs = inputs.to(device)
    night = inputs.shape[1]
    if night < L:
        y = model(inputs, quantize=quantize)
        return F.softmax(y.squeeze(0), dim=-1).cpu().numpy()

    probe = model(inputs[:, :L], quantize=quantize)
    n_classes = probe.shape[-1]
    votes = torch.zeros(1, night, n_classes, device=device, dtype=probe.dtype)
    counts = torch.zeros(1, night, device=device, dtype=torch.float32)
    for offset in range(L):
        x = inputs[:, offset:]
        usable = x.shape[1] - (x.shape[1] % L)
        if usable == 0:
            continue
        x = x[:, :usable]
        nw = usable // L
        x = x.reshape(nw, L, *x.shape[2:])
        y = model(x, quantize=quantize).reshape(1, nw * L, n_classes)
        votes[:, offset:offset + usable] += y
        counts[:, offset:offset + usable] += 1
    logits = votes / counts.clamp(min=1).unsqueeze(-1)
    return F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()


@torch.no_grad()
def embed_subject(model, inputs, device, batch_size=256):
    """Per-epoch embedding h (quantize=False). inputs: (1, night, C, T, F)."""
    inputs = inputs.to(device)
    N = inputs.shape[1]
    out = []
    for i in range(0, N, batch_size):
        chunk = inputs[:, i:i + batch_size]
        h = model.epoch_encode(chunk, quantize=False)  # (1, chunk, d)
        out.append(h.squeeze(0).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ── shared signals: per-subject raw waveform + STFT reference ────────────
def compute_signals(out_dir: Path, max_subjects=None, n_ref_epochs=6):
    """Write per-subject raw waveforms and a small STFT-reference fixture.

    Returns the canonical subject order [(id, n_epochs, labels), ...].
    """
    from physioex.data.steps import XSleepNetSpectrogram

    sig_dir = out_dir / "signals" / "subjects"
    sig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[signals] building raw loader for {DATASET} ...")
    _, raw_loader = build_full_loader(DATASET, channels=CHANNELS, pipeline="raw")
    print(f"[signals] {len(raw_loader)} subjects")

    stft = XSleepNetSpectrogram(nperseg=200, noverlap=100, nfft=256, window="hamming")
    compiled = stft.compile(fs_in=100.0)

    order = []
    ref_samples = []
    for si, batch in enumerate(tqdm(raw_loader, desc="raw signals")):
        if max_subjects is not None and si >= max_subjects:
            break
        sid = batch["subject"][0]["id"]
        x = stack_channels(batch).squeeze(0).cpu().numpy()  # (N, C, S)
        assert x.ndim == 3, f"unexpected raw shape {x.shape} for {sid}"
        y = batch["labels"].squeeze(0).cpu().numpy().astype(np.int64)  # (N,)
        s, e = trim_window(y)  # drop excess pre/post-sleep wake
        x, y = x[s:e], y[s:e]
        n = x.shape[0]

        pack.write_f16(sig_dir / f"{sid}.raw.bin", x)
        order.append((sid, int(n), y.tolist()))

        # collect a few reference epochs (spanning stages) for JS validation
        if len(ref_samples) < n_ref_epochs and si < 3:
            for ei in range(0, n, max(1, n // 3)):
                if len(ref_samples) >= n_ref_epochs:
                    break
                spec = np.stack([compiled.apply(x[ei, c]) for c in range(x.shape[1])])
                ref_samples.append({
                    "subject": sid, "epoch": int(ei),
                    "label": int(y[ei]),
                    "raw": np.round(x[ei], 4).tolist(),          # (C, S)
                    "spec": np.round(spec, 4).tolist(),           # (C, T, F)
                })

    with open(out_dir / "signals" / "subjects.json", "w") as f:
        json.dump([{"id": sid, "n_epochs": n, "labels": lbl}
                   for sid, n, lbl in order], f)
    with open(out_dir / "signals" / "stft_reference.json", "w") as f:
        json.dump({
            "fs": 100, "nperseg": 200, "noverlap": 100, "nfft": 256,
            "window": "hamming", "n_time": 29, "n_freq": 129,
            "channels": CHANNELS, "samples": ref_samples,
        }, f)
    print(f"[signals] wrote {len(order)} subjects, {len(ref_samples)} ref epochs")
    return order


# ── per-model pass ───────────────────────────────────────────────────────
def compute_model(backbone, checkpoint, codebook_path, committed_root, out_dir,
                  subject_order, device, umap_kwargs, max_subjects=None):
    cfg = MODELS[backbone]
    hf, committed, L = cfg["hf"], cfg["committed"], cfg["seq_len"]
    print(f"\n[{hf}] backbone={backbone} L={L}")

    model = load_frozen_model(backbone, device, checkpoint_path=checkpoint)
    codebook = load_codebook(backbone, codebook_path=codebook_path)
    model.set_codebook(codebook)
    M = codebook.shape[0]
    assert M == 12, f"expected M=12, got {M}"
    print(f"[{hf}] model + codebook (M={M}, d={codebook.shape[1]}) loaded")

    _, loader = build_full_loader(DATASET, channels=CHANNELS, pipeline="seqsleepnet")
    order_index = {sid: i for i, (sid, _, _) in enumerate(subject_order)}

    per_subject = {}  # sid -> dict(h, proba, labels)
    for si, batch in enumerate(tqdm(loader, desc=f"{hf} forward")):
        if max_subjects is not None and si >= max_subjects:
            break
        sid = batch["subject"][0]["id"]
        inputs = stack_channels(batch)                 # (1, night, C, T, F)
        labels = batch["labels"].reshape(-1).cpu().numpy().astype(np.int64)
        h = embed_subject(model, inputs, device)       # (N, d)
        proba = predict_subject(model, inputs, L, device, quantize=False)  # (N, 5)
        s, e = trim_window(labels)  # keep only the trimmed sleep window
        per_subject[sid] = dict(h=h[s:e], proba=proba[s:e], labels=labels[s:e])

    # assemble global arrays in canonical subject order
    xs_h, xs_lbl, xs_proba, xs_subj, xs_epoch = [], [], [], [], []
    model_subjects = []
    for sid, _n_sig, _labels in subject_order:
        if sid not in per_subject:
            continue
        rec = per_subject[sid]
        n = len(rec["h"])
        sidx = order_index[sid]
        xs_h.append(rec["h"])
        xs_lbl.append(rec["labels"])
        xs_proba.append(rec["proba"])
        xs_subj.append(np.full(n, sidx, dtype=np.int64))
        xs_epoch.append(np.arange(n, dtype=np.int64))
        model_subjects.append({
            "id": sid, "idx": sidx, "n_epochs": int(n),
            "pred": rec["proba"].argmax(1).astype(int).tolist(),
        })

    H = np.concatenate(xs_h, 0)
    LBL = np.concatenate(xs_lbl, 0)
    PROBA = np.concatenate(xs_proba, 0)
    SUBJ = np.concatenate(xs_subj, 0)
    EPOCH = np.concatenate(xs_epoch, 0)
    print(f"[{hf}] {H.shape[0]} epochs, fitting UMAP ...")

    # prototype assignment (squared-L2 argmin), distance = sqrt
    dist_sq = compute_l2_sq_distances_np(H, codebook)   # (N, M)
    PROTO = dist_sq.argmin(1).astype(np.int64)
    DIST = np.sqrt(np.clip(dist_sq[np.arange(len(H)), PROTO], 0, None)).astype(np.float32)

    # UMAP on epoch embeddings; co-embed prototypes with transform()
    from umap import UMAP
    reducer = UMAP(**umap_kwargs)
    XY = reducer.fit_transform(H).astype(np.float32)          # (N, 2)
    PROTO_XY = reducer.transform(codebook).astype(np.float32)  # (M, 2)

    # normalize coords to a stable [-1, 1] box (nice for the WebGL scatter)
    lo, hi = XY.min(0), XY.max(0)
    span = np.maximum(hi - lo, 1e-6)
    def _norm(a):
        return ((a - lo) / span * 2 - 1).astype(np.float32)
    XY, PROTO_XY = _norm(XY), _norm(PROTO_XY)

    # ── emit per-model arrays ──
    mdir = out_dir / hf
    mdir.mkdir(parents=True, exist_ok=True)
    pack.write_f32(mdir / "xy.f32", XY)
    pack.write_u8(mdir / "label.u8", np.where(LBL == IGNORE, MASK_U8, LBL))
    pack.write_u8(mdir / "pred.u8", PROBA.argmax(1))
    pack.write_u8(mdir / "proba.u8", pack.quantize_proba_u8(PROBA))
    pack.write_u8(mdir / "proto.u8", PROTO)
    pack.write_f32(mdir / "dist.f32", DIST)
    pack.write_u16(mdir / "subj.u16", SUBJ)
    pack.write_u16(mdir / "epoch.u16", EPOCH)

    # ── prototype cards + xy ──
    cards = build_prototype_cards(Path(committed_root) / committed, m=M)
    for k in range(M):
        cards[k]["xy"] = [float(PROTO_XY[k, 0]), float(PROTO_XY[k, 1])]
        # observed SleepEDF label distribution for this prototype
        sel = LBL[(PROTO == k) & (LBL != IGNORE)]
        cards[k]["sleepedf_label_distribution"] = {
            STAGE_NAMES[c]: int((sel == c).sum()) for c in range(5)
        }
        cards[k]["sleepedf_cluster_size"] = int((PROTO == k).sum())
    with open(mdir / "prototypes.json", "w") as f:
        json.dump(cards, f)

    # ── per-model metrics + subject predictions ──
    valid = LBL != IGNORE
    acc = float((PROBA.argmax(1)[valid] == LBL[valid]).mean())
    with open(mdir / "subjects.json", "w") as f:
        json.dump({"subjects": model_subjects}, f)

    print(f"[{hf}] done. epochs={H.shape[0]} acc={acc:.4f} "
          f"proto range={PROTO.min()}..{PROTO.max()}")
    return {"n_epochs": int(H.shape[0]), "n_subjects": len(model_subjects),
            "accuracy": acc, "seq_len": L, "backbone": backbone}


def main():
    ap = argparse.ArgumentParser(description="Precompute the demo static bundle")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--backbones", nargs="+", default=["seq", "st"],
                    choices=["seq", "st"])
    ap.add_argument("--committed_root", default=None,
                    help="Dir with <committed_model>/ reconstruction JSON "
                         "(default: repo data/reconstructions/M12)")
    for b in ("seq", "st"):
        ap.add_argument(f"--checkpoint-{b}", dest=f"ckpt_{b}", default=None)
        ap.add_argument(f"--codebook-{b}", dest=f"cb_{b}", default=None)
    ap.add_argument("--umap_neighbors", type=int, default=25)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_subjects", type=int, default=None,
                    help="Debug: limit number of subjects")
    ap.add_argument("--skip_signals", action="store_true",
                    help="Reuse an existing signals/ pass")
    args = ap.parse_args()

    device = (torch.device(f"cuda:{args.gpu_id}")
              if args.gpu_id is not None and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    committed_root = Path(
        args.committed_root
        or (Path(__file__).resolve().parents[3] / "data" / "reconstructions" / "M12")
    )
    print(f"Committed reconstruction root: {committed_root}")

    # 1) shared signals (raw waveforms + STFT reference)
    if args.skip_signals and (out_dir / "signals" / "subjects.json").exists():
        with open(out_dir / "signals" / "subjects.json") as f:
            subject_order = [(s["id"], s["n_epochs"], s["labels"]) for s in json.load(f)]
        print(f"[signals] reused {len(subject_order)} subjects")
    else:
        subject_order = compute_signals(out_dir, max_subjects=args.max_subjects)

    umap_kwargs = dict(n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
                       n_components=2, metric="euclidean", random_state=args.seed)

    # 2) per-model passes
    model_meta = {}
    for b in args.backbones:
        ckpt = getattr(args, f"ckpt_{b}") or str(get_paths(b, m=12)["checkpoint"])
        cb = getattr(args, f"cb_{b}")  # required (M=12 codebook path unknown by default)
        if cb is None:
            raise SystemExit(f"--codebook-{b} is required (M=12 codebook path)")
        meta = compute_model(
            b, ckpt, cb, committed_root, out_dir, subject_order, device,
            umap_kwargs, max_subjects=args.max_subjects,
        )
        model_meta[MODELS[b]["hf"]] = meta

    # 3) manifest
    manifest = {
        "dataset": DATASET,
        "stages": STAGE_NAMES,
        "channels": CHANNELS,
        "fs": 100, "epoch_sec": 30, "mask_u8": MASK_U8,
        "raw": {"dtype": "float16", "n_samples": 3000, "layout": "(n, C, S)"},
        "stft": {"nperseg": 200, "noverlap": 100, "nfft": 256,
                 "window": "hamming", "n_time": 29, "n_freq": 129,
                 "log_scale": "10*log10(|X|^2)", "scipy_spectrogram": True},
        "proto": {"m": 12, "match": "argmin squared-L2"},
        "n_subjects": len(subject_order),
        "models": model_meta,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nBundle written to {out_dir}")


if __name__ == "__main__":
    main()
