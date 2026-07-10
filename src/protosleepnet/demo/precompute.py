"""Precompute the static bundle for the ProtoSleepNet explainability demo.

Runs on the A30 (needs GPU + SleepEDF + the released checkpoints/codebooks).
Emits a GPU-free bundle the static web app consumes. SleepEDF only.

For each proto model (SeqSleepNet / SleepTransformer backbone) it computes,
over **all** SleepEDF subjects:
  - per-epoch embedding h = epoch_encode(x, quantize=False)      (matching space)
  - per-epoch NON-quantized prediction via sliding-window voting  (displayed)
  - nearest prototype (squared-L2 to the codebook) + distance     (the match)
  - a PaCMAP of all epoch embeddings, with the 12 prototypes co-embedded in the fit
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


def epoch_ig_subject(model, inputs, targets, device, steps=64, group=8):
    """Per-EPOCH Integrated-Gradients attribution toward each epoch's own
    matched prototype: IG of f(x) = -||epoch_encode(x) - p_matched||^2, zero
    baseline. inputs: (1, n, C, T, F) spectrogram; targets: (n, d) codebook
    rows for the matched prototype. Returns (n, C, T, F) float32.

    Gradient flows to the INPUT only (model params stay frozen). cudnn must be
    off for LSTM backward in eval (set by the caller).
    """
    x_all = inputs.squeeze(0).to(device)                 # (n, C, T, F)
    n, C, T, Fd = x_all.shape
    tgt = torch.as_tensor(np.asarray(targets), device=device, dtype=torch.float32)  # (n, d)
    alphas = torch.linspace(0, 1, steps, device=device).view(1, steps, 1, 1, 1)
    out = np.zeros((n, C, T, Fd), dtype=np.float32)
    for i in range(0, n, group):
        xe = x_all[i:i + group]                          # (g, C, T, F)
        g = xe.shape[0]
        X = (xe.unsqueeze(1) * alphas).reshape(g * steps, 1, C, T, Fd)
        X = X.clone().detach().requires_grad_(True)      # baseline 0 → path = alpha·x
        h = model.epoch_encode(X, quantize=False).squeeze(1)          # (g*steps, d)
        te = tgt[i:i + g].repeat_interleave(steps, dim=0)             # (g*steps, d)
        f = -((h - te) ** 2).sum(dim=1).sum()
        grads, = torch.autograd.grad(f, X)               # (g*steps, 1, C, T, F)
        avg = grads.squeeze(1).reshape(g, steps, C, T, Fd).mean(dim=1)  # (g, C, T, F)
        out[i:i + g] = (xe * avg).detach().cpu().numpy()  # IG = (x-0)·mean_α ∂f/∂x
    return out


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
                  subject_order, device, seed, compare_umap, umap_kwargs,
                  max_subjects=None):
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
    print(f"[{hf}] {H.shape[0]} epochs, projecting with PaCMAP ...")

    # prototype assignment (squared-L2 argmin), distance = sqrt
    dist_sq = compute_l2_sq_distances_np(H, codebook)   # (N, M)
    PROTO = dist_sq.argmin(1).astype(np.int64)
    DIST = np.sqrt(np.clip(dist_sq[np.arange(len(H)), PROTO], 0, None)).astype(np.float32)

    # PaCMAP preserves BOTH local and global structure better than UMAP, and we
    # co-embed the 12 prototypes IN the fit (union) rather than transform()-after,
    # so their coords reflect the same optimization as the epochs.
    from pacmap import PaCMAP
    reducer = PaCMAP(n_components=2, random_state=seed)
    union = np.vstack([H, codebook]).astype(np.float32)
    emb = np.asarray(reducer.fit_transform(union, init="pca"), dtype=np.float32)
    XY, PROTO_XY = emb[:len(H)], emb[len(H):]

    # faithfulness: fraction of epochs whose 2-D-nearest prototype == the true
    # 128-D-L2-nearest prototype (the assignment that actually drives matching).
    def _agreement(xy, pxy):
        return float((compute_l2_sq_distances_np(xy, pxy).argmin(1) == PROTO).mean())
    agree = _agreement(XY, PROTO_XY)
    metrics = {"projection": "pacmap", "nn_proto_agreement": agree}
    print(f"[{hf}] PaCMAP nearest-prototype agreement = {agree:.3f}")
    if compare_umap:
        from umap import UMAP
        u = UMAP(**umap_kwargs)
        uxy = u.fit_transform(H).astype(np.float32)
        upxy = u.transform(codebook).astype(np.float32)
        metrics["umap_nn_proto_agreement"] = _agreement(uxy, upxy)
        print(f"[{hf}] UMAP  nearest-prototype agreement = {metrics['umap_nn_proto_agreement']:.3f}")

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
            "accuracy": acc, "seq_len": L, "backbone": backbone, **metrics}


def _griffin_lim(db_TF, n_iter=200, momentum=0.99):
    """Estimate a 30 s waveform from ONE coherent log-power spectrogram
    (T=29, F=129 dB) via fast Griffin-Lim (Perraudin momentum). Phase is not
    stored, so it is recovered iteratively; run on a single medoid spectrogram
    (never the cross-epoch average, which has no coherent phase to recover).

    Inversion of XSleepNet's one-sided PSD-dB back to an STFT magnitude: only
    the RELATIVE per-bin magnitude matters for GL (istft/stft are inverse in
    scipy's convention and the loop re-imposes |STFT|=mag each step), so we undo
    the one-sided ×2 on interior bins (the sole relative distortion) and map
    dB->amplitude with 10^(dB/20). Geometry: hamming/200, hop 100, nfft 256,
    100 Hz -> 200 + 28*100 = 3000 samples. Amplitude is arbitrary (auto-scaled).
    """
    from scipy.signal import stft, istft
    kw = dict(fs=100, window="hamming", nperseg=200, noverlap=100, nfft=256)
    mag = np.power(10.0, np.asarray(db_TF).T / 20.0)  # (F=129, T=29) amplitude
    mag[1:-1, :] /= np.sqrt(2.0)                       # undo one-sided doubling
    rng = np.random.default_rng(0)
    ph = np.exp(2j * np.pi * rng.random(mag.shape))
    prev = mag * ph
    for _ in range(n_iter):
        _, x = istft(prev, input_onesided=True, boundary=False, **kw)
        _, _, X = stft(x, boundary=None, padded=False, **kw)
        T = mag.shape[1]
        X = X[:, :T] if X.shape[1] >= T else np.pad(X, ((0, 0), (0, T - X.shape[1])))
        ph = X / np.maximum(np.abs(X), 1e-8)
        cur = mag * ph
        S = cur + momentum * (cur - prev)   # fast GL momentum
        prev = cur
        _, x = istft(S, input_onesided=True, boundary=False, **kw)
        _, _, X = stft(x, boundary=None, padded=False, **kw)
        X = X[:, :T] if X.shape[1] >= T else np.pad(X, ((0, 0), (0, T - X.shape[1])))
        prev = mag * (X / np.maximum(np.abs(X), 1e-8))
    _, x = istft(prev, input_onesided=True, boundary=False, **kw)
    x = np.asarray(x, dtype=np.float32)
    return (x[:3000] if len(x) >= 3000 else np.pad(x, (0, 3000 - len(x)))).astype(np.float32)


def _medoid(arr):
    """Return the single sample (C,T,F) closest to the per-cell median — a
    coherent representative spectrogram (unlike the average of all N)."""
    med = np.median(arr, axis=0)
    d = np.abs(arr - med).reshape(len(arr), -1).sum(1)
    return arr[int(d.argmin())]


def write_reconstructions(backbone, recon_root, method, out_dir):
    """Per-prototype, ship into <model>/:
      reconstructions.f16 (12,3,29,129) — MEDIAN over the optimized samples
        (sharper/robust than mean) — the heatmap.
      recon_timeseries.f16 (12,3,3000) — fast Griffin-Lim of the MEDOID sample
        (a single coherent spectrogram; phase-estimated, display-only).
      ig_attr.f16 / ig_epoch.f16 (12,3,29,129) — the committed Integrated-
        Gradients attribution + its representative epoch (why the epoch matches).
    All reuse the M12 reconstruction pipeline output (same vq_kmeans/12 codebook
    as the atlas, so indices align).
    """
    cfg = MODELS[backbone]
    hf, committed = cfg["hf"], cfg["committed"]
    mdir = Path(recon_root) / committed / method
    med_specs, wave_src = [], []
    for k in range(12):
        arr = np.load(mdir / f"proto_{k:03d}" / "epochs.npy")  # (N, 3, 29, 129) dB
        med_specs.append(np.median(arr, axis=0))               # (3, 29, 129)
        wave_src.append(_medoid(arr))                          # (3, 29, 129)
    R = np.stack(med_specs).astype(np.float32)
    (out_dir / hf).mkdir(parents=True, exist_ok=True)
    pack.write_f16(out_dir / hf / "reconstructions.f16", R)

    ts = np.zeros((12, 3, 3000), dtype=np.float32)
    for k in range(12):
        for c in range(3):
            ts[k, c] = _griffin_lim(wave_src[k][c])
    pack.write_f16(out_dir / hf / "recon_timeseries.f16", ts)

    # Integrated-Gradients attribution (already computed; no GPU here)
    le = Path(recon_root) / committed / "local_explanations"
    ig_attr = np.zeros((12, 3, 29, 129), dtype=np.float32)
    ig_epoch = np.zeros_like(ig_attr)
    have_ig = True
    for k in range(12):
        ap = le / f"proto_{k:03d}" / "local_attr.npy"
        ep = le / f"proto_{k:03d}" / "local_epoch.npy"
        if ap.exists() and ep.exists():
            ig_attr[k] = np.load(ap).reshape(3, 29, 129)
            ig_epoch[k] = np.load(ep).reshape(3, 29, 129)
        else:
            have_ig = False
    if have_ig:
        pack.write_f16(out_dir / hf / "ig_attr.f16", ig_attr)
        pack.write_f16(out_dir / hf / "ig_epoch.f16", ig_epoch)
    print(f"[{hf}] recon median {R.shape} + GL medoid {ts.shape}"
          f"{' + IG' if have_ig else ' (no IG arrays)'} ({method})")


def write_epoch_ig(backbone, checkpoint, codebook_path, out_dir, device,
                   steps, group, max_subjects=None):
    """Backfill per-EPOCH IG attribution into <model>/ig/<sid>.ig.bin (f16,
    (n,3,29,129), trimmed window) for lazy Range-fetch. Target per epoch = its
    matched prototype (argmin L2), matching the atlas assignment."""
    cfg = MODELS[backbone]
    hf = cfg["hf"]
    model = load_frozen_model(backbone, device, checkpoint_path=checkpoint)
    codebook = load_codebook(backbone, codebook_path=codebook_path)
    model.set_codebook(codebook)
    _, loader = build_full_loader(DATASET, channels=CHANNELS, pipeline="seqsleepnet")
    ig_dir = out_dir / hf / "ig"
    ig_dir.mkdir(parents=True, exist_ok=True)
    for si, batch in enumerate(tqdm(loader, desc=f"{hf} IG")):
        if max_subjects is not None and si >= max_subjects:
            break
        sid = batch["subject"][0]["id"]
        inputs = stack_channels(batch)                        # (1, n, C, T, F)
        labels = batch["labels"].reshape(-1).cpu().numpy().astype(np.int64)
        s, e = trim_window(labels)
        inputs_w = inputs[:, s:e]
        h = embed_subject(model, inputs_w, device)            # (n, d)
        proto = compute_l2_sq_distances_np(h, codebook).argmin(1)
        targets = codebook[proto]                             # (n, d)
        ig = epoch_ig_subject(model, inputs_w, targets, device, steps=steps, group=group)
        pack.write_f16(ig_dir / f"{sid}.ig.bin", ig)
    print(f"[{hf}] wrote per-epoch IG -> {ig_dir}")


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
    ap.add_argument("--compare_umap", action="store_true",
                    help="Also fit UMAP and report its nearest-proto agreement (dev comparison)")
    ap.add_argument("--recon_root", default=None,
                    help="Root of the M12 reconstruction outputs "
                         "(<recon_root>/<committed_model>/<method>/proto_XXX/epochs.npy)")
    ap.add_argument("--recon_method", default="hybrid",
                    choices=["hybrid", "data_driven", "model_driven"])
    ap.add_argument("--reconstructions_only", action="store_true",
                    help="Backfill only <model>/reconstructions.f16 into an existing bundle")
    ap.add_argument("--epoch_ig_only", action="store_true",
                    help="Backfill per-epoch IG (<model>/ig/<sid>.ig.bin) into an existing bundle")
    ap.add_argument("--ig_steps", type=int, default=64)
    ap.add_argument("--ig_group", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.reconstructions_only:
        if not args.recon_root:
            raise SystemExit("--recon_root is required with --reconstructions_only")
        for b in args.backbones:
            write_reconstructions(b, args.recon_root, args.recon_method, out_dir)
        mpath = out_dir / "manifest.json"
        if mpath.exists():
            manifest = json.load(open(mpath))
            manifest["reconstruction"] = {"method": args.recon_method,
                                          "aggregate": "median", "waveform": "griffin-lim (medoid)",
                                          "shape": [3, 29, 129], "scale": "dB"}
            manifest["ig"] = {"shape": [3, 29, 129], "source": "local_explanations (IG, zero baseline)"}
            json.dump(manifest, open(mpath, "w"), indent=2)
        print("Reconstructions backfilled.")
        return

    device = (torch.device(f"cuda:{args.gpu_id}")
              if args.gpu_id is not None and torch.cuda.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.epoch_ig_only:
        os.environ.setdefault("PHYSIOEX_DATA", os.environ.get("PHYSIOEX_DATA", ""))
        torch.backends.cudnn.enabled = False  # LSTM backward in eval
        for b in args.backbones:
            ckpt = getattr(args, f"ckpt_{b}") or str(get_paths(b, m=12)["checkpoint"])
            cb = getattr(args, f"cb_{b}")
            if cb is None:
                raise SystemExit(f"--codebook-{b} is required")
            write_epoch_ig(b, ckpt, cb, out_dir, device,
                           args.ig_steps, args.ig_group, max_subjects=args.max_subjects)
        mpath = out_dir / "manifest.json"
        if mpath.exists():
            manifest = json.load(open(mpath))
            manifest["ig_epoch"] = {"shape": [3, 29, 129], "dtype": "float16",
                                    "layout": "<model>/ig/<sid>.ig.bin (n,3,29,129)",
                                    "steps": args.ig_steps}
            json.dump(manifest, open(mpath, "w"), indent=2)
        print("Per-epoch IG backfilled.")
        return

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
            args.seed, args.compare_umap, umap_kwargs, max_subjects=args.max_subjects,
        )
        model_meta[MODELS[b]["hf"]] = meta
        if args.recon_root:
            write_reconstructions(b, args.recon_root, args.recon_method, out_dir)

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
        "projection": "pacmap",
        "n_subjects": len(subject_order),
        "models": model_meta,
    }
    if args.recon_root:
        manifest["reconstruction"] = {"method": args.recon_method,
                                      "aggregate": "median", "waveform": "griffin-lim (medoid)",
                                      "shape": [3, 29, 129], "scale": "dB"}
        manifest["ig"] = {"shape": [3, 29, 129], "source": "local_explanations (IG, zero baseline)"}
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nBundle written to {out_dir}")


if __name__ == "__main__":
    main()
