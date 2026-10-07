"""Random-codebook null for the AASM-inspired coherence checklist.

The paper rates the 24 codebook entries of PSN and PST with a stage-specific checklist and gets
10 Coherent / 10 Partially coherent / 4 Non-canonical. This script answers the question the paper
cannot answer without it: how many Coherent verdicts does the same checklist give to a codebook
that has no learned structure?

A null codebook is made of M training epochs drawn at random: each entry is the embedding of a real
epoch. Because every entry already has an input, no reconstruction is optimised. As in the
data-driven reconstruction, an entry is represented by the (up to) `top_k` training epochs that are
assigned to it and closest to it in embedding space. Everything after that is the published
pipeline, reusing its functions: whole-channel ablation, 2^8 band add-back, feature direction,
predicted dominant stage, EOG power and EMG tone, ranks within the codebook, verdict.

Draws are made in two ways:
  uniform     M epochs drawn uniformly (as in the randomization test of Supplementary Note 8);
  stratified  the same number of epochs per predicted stage as the real codebook has prototypes of
              that stage, because the real prototypes are not distributed like the stage prevalence.

The real codebook is run through the same code on the same pool of epochs (`--real`), so that the
comparison is like with like.

Nothing here writes to an existing output. Results go to `--out_dir`, one JSON per codebook, and
the run resumes where it stopped.

Usage (on the hub, environment `physioex`):
    PYTHONPATH=<worktree>/src python -m protosleepnet.coherence_null \
        --backbone seq --m 12 --codebook_path <real codebook .npy> \
        --out_dir <dir> --n_codebooks 50 --variant stratified \
        --stage_alloc W:2,N2:6,N3:2,REM:2 --real
    # timing and equivalence pilot, one entry only:
    ... --pilot --equivalence_epochs <archived data_driven/proto_000/epochs.npy> --proto_idx 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from physioex.data.collate import stack_channels

from protosleepnet.proto_reconstruction.utils import (
    STAGE_NAMES, add_common_args, build_train_loader, compute_l2_sq_distances_np, get_device,
    get_paths, load_codebook, load_frozen_model,
)
from protosleepnet.figure_reconstruction.combinatorial_ablation import (
    compute_band_importance, compute_channel_importance, compute_feature_direction, get_eeg_bands,
    load_training_mean, predict_classes,
)
from protosleepnet.figure_reconstruction import coherence_checklist as CK

FS, NFFT = 100.0, 256
EMG_BIN_START = round(10.0 / (FS / NFFT))   # 26, as spectral_signature.py
EMG_BIN_END = round(50.0 / (FS / NFFT))     # 128


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── pool of training epochs ──────────────────────────────────────────

@torch.no_grad()
def embed_and_predict(model, x: np.ndarray, device, batch_size: int = 256):
    """Epoch embeddings and predicted stage of every epoch of x (n, C, T, F)."""
    zs, ps = [], []
    xt = torch.from_numpy(x)
    for i in range(0, len(xt), batch_size):
        b = xt[i:i + batch_size].unsqueeze(1).to(device)
        h = model.epoch_encode(b, quantize=False).squeeze(1)
        logits = model.get_metrics()["epoch_logits"].squeeze(1)
        zs.append(h.cpu().numpy())
        ps.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(zs).astype(np.float32), np.concatenate(ps).astype(np.int64)


def build_pool(backbone: str, model, device, n_subjects: int | None, seed: int):
    """Raw epochs, embeddings and predicted stages of a random set of training subjects.

    Returns raw (list of arrays per subject), Z (N, d), pred (N,), index (N, 2) as
    (subject position, epoch index), and the subject ids.
    """
    from torch.utils.data import DataLoader, Subset

    _, loader = build_train_loader(backbone)
    n_total = len(loader)
    rng = np.random.RandomState(seed)
    if n_subjects and n_subjects < n_total:
        pos = sorted(rng.choice(n_total, n_subjects, replace=False).tolist())
        loader = DataLoader(Subset(loader.dataset, pos), batch_size=1, shuffle=False,
                            collate_fn=loader.collate_fn, num_workers=0)
    log(f"pool: {len(loader)} of {n_total} training subjects")

    raw, ids, Zs, Ps = [], [], [], []
    t0 = time.time()
    for n, batch in enumerate(loader):
        sid = str(batch["subject"][0]["id"])
        x = stack_channels(batch).squeeze(0).cpu().numpy().astype(np.float32)   # (n, C, T, F)
        Z, P = embed_and_predict(model, x, device)
        raw.append(x); ids.append(sid); Zs.append(Z); Ps.append(P)
        if (n + 1) % 25 == 0:
            log(f"  {n + 1} subjects, {sum(len(r) for r in raw)} epochs, "
                f"{sum(r.nbytes for r in raw) / 1e9:.1f} GB, {time.time() - t0:.0f} s")
    Z = np.concatenate(Zs); pred = np.concatenate(Ps)
    index = np.concatenate([np.stack([np.full(len(r), i), np.arange(len(r))], axis=1)
                            for i, r in enumerate(raw)])
    log(f"pool ready: {len(Z)} epochs, {sum(r.nbytes for r in raw) / 1e9:.1f} GB, "
        f"predicted stage shares " + ", ".join(
            f"{STAGE_NAMES[c]} {np.mean(pred == c):.2f}" for c in range(5)))
    return raw, Z, pred, index, ids


def neighbours(Z: np.ndarray, codebook: np.ndarray, top_k: int):
    """For each entry, the flat indices of the top_k closest epochs among those assigned to it
    (as `build_prototype_index`), and the cluster sizes."""
    dist = compute_l2_sq_distances_np(Z, codebook)
    assign = dist.argmin(axis=1)
    out, sizes = [], []
    for k in range(codebook.shape[0]):
        idx = np.where(assign == k)[0]
        sizes.append(int(len(idx)))
        order = dist[idx, k].argsort()[:top_k]
        out.append((idx[order], np.sqrt(np.maximum(dist[idx[order], k], 0.0))))
    return out, np.array(sizes)


def gather(raw, index, flat_idx) -> np.ndarray:
    return np.stack([raw[index[i, 0]][index[i, 1]] for i in flat_idx])


# ── one entry ────────────────────────────────────────────────────────

def analyse_entry(model, vec: np.ndarray, epochs: np.ndarray, tm: np.ndarray, bands, device,
                  timing: dict | None = None) -> dict:
    """The published analysis of one prototype, from the epochs that represent it."""
    t = time.time()
    names = [b[0] for b in bands]
    ch_imp = compute_channel_importance(model, vec, epochs, tm, device)
    t_ch = time.time() - t; t = time.time()
    imp, eeg_contrib = compute_band_importance(model, vec, epochs, tm, bands, device)
    t_band = time.time() - t; t = time.time()
    directions, proto_power, mean_power = compute_feature_direction(epochs, tm, bands)
    preds, counts = predict_classes(model, epochs, device)
    lin = np.power(10.0, epochs / 10.0)
    eog_power = float(lin[:, 1].sum(axis=-1).mean())
    emg_tone = float(lin[:, 2][..., EMG_BIN_START:EMG_BIN_END].sum(axis=-1).mean())
    ch_pcts = ch_imp / (np.abs(ch_imp).sum() + 1e-12)
    if timing is not None:
        timing.update(channel=t_ch, band=t_band, rest=time.time() - t)
    return {
        "stage": STAGE_NAMES[int(counts.argmax())],
        "purity": float(counts.max() / len(preds)),
        "predicted_distribution": {STAGE_NAMES[i]: int(counts[i]) for i in range(5)},
        "n_epochs": int(len(epochs)),
        "eeg_contribution": float(eeg_contrib),
        "band_rel": {n: float(imp[i] / (eeg_contrib + 1e-12) * 100) for i, n in enumerate(names)},
        "direction": {n: ("elevated" if directions[i] > 0 else "suppressed") for i, n in enumerate(names)},
        "db": {n: float(10 * np.log10(proto_power[i] + 1e-12) - 10 * np.log10(mean_power[i] + 1e-12))
               for i, n in enumerate(names)},
        "ch_pcts": {"EEG": float(ch_pcts[0]), "EOG": float(ch_pcts[1]), "EMG": float(ch_pcts[2])},
        "eog_power": eog_power,
        "emg_tone": emg_tone,
    }


def judge(records: list[dict]) -> dict:
    """Ranks within the codebook, then the verdict of every entry."""
    eog = [r["eog_power"] for r in records]; emg = [r["emg_tone"] for r in records]
    for r in records:
        r["eog_pct"] = CK.rank(r["eog_power"], eog)
        r["emg_pct"] = CK.rank(r["emg_tone"], emg)
        r["verdict"] = CK.verdict(r)
        r["criteria"] = CK.criteria(r)
    cnt = {v: sum(1 for r in records if r["verdict"] == v) for v in "CPX"}
    comp = {s: sum(1 for r in records if r["stage"] == s) for s in CK.STAGES}
    return {"counts": cnt, "stage_composition": comp}


def run_codebook(model, codebook, raw, Z, index, pred, tm, bands, device, top_k, tag) -> dict:
    nb, sizes = neighbours(Z, codebook, top_k)
    records = []
    for k in range(codebook.shape[0]):
        flat, dist = nb[k]
        if len(flat) == 0:
            records.append({"empty": True, "cluster_size": 0, "stage": "?", "eog_power": 0.0,
                            "emg_tone": 0.0})
            continue
        t = time.time()
        rec = analyse_entry(model, codebook[k], gather(raw, index, flat), tm, bands, device)
        rec["cluster_size"] = int(sizes[k]); rec["mean_distance"] = float(dist.mean())
        records.append(rec)
        log(f"  {tag} entry {k:2d}: stage {rec['stage']:>3s} purity {rec['purity']:.2f} "
            f"n={rec['n_epochs']} ({time.time() - t:.1f} s)")
    valid = [r for r in records if not r.get("empty")]
    res = judge(valid)
    res["n_empty"] = len(records) - len(valid)
    res["records"] = records
    return res


# ── draws ────────────────────────────────────────────────────────────

def draw_entries(rng, pred, m, variant, alloc):
    if variant == "uniform":
        return rng.choice(len(pred), m, replace=False)
    out = []
    for s, n in alloc.items():
        pool = np.where(pred == STAGE_NAMES.index(s))[0]
        out.extend(rng.choice(pool, n, replace=False).tolist())
    return np.array(out)


def parse_alloc(s: str) -> dict:
    return {p.split(":")[0]: int(p.split(":")[1]) for p in s.split(",") if p}


def atomic_write(path: Path, data) -> None:
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


# ── main ─────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--n_codebooks", type=int, default=50)
    ap.add_argument("--variant", choices=["uniform", "stratified"], default="stratified")
    ap.add_argument("--stage_alloc", default="W:2,N2:6,N3:2,REM:2",
                    help="entries per predicted stage for the stratified draw (real codebook's allocation)")
    ap.add_argument("--pool_subjects", type=int, default=None, help="random subset of training subjects")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--real", action="store_true", help="also run the real codebook on the same pool")
    ap.add_argument("--training_mean", default=None)
    ap.add_argument("--pilot", action="store_true", help="one codebook, one entry, with timings")
    ap.add_argument("--equivalence_epochs", default=None,
                    help="archived data_driven epochs.npy of one real prototype, to check the pipeline")
    ap.add_argument("--proto_idx", type=int, default=0)
    args = ap.parse_args()

    if not args.output_dir:
        raise SystemExit("--output_dir is required")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = get_device(args)
    log(f"device {device}; out {out}")
    model = load_frozen_model(args.backbone, device, checkpoint_path=args.checkpoint_path)
    real = load_codebook(args.backbone, m=args.m, codebook_path=args.codebook_path)
    tm = load_training_mean(args.backbone, override_path=args.training_mean)
    bands = get_eeg_bands()
    names = [b[0] for b in bands]
    if set(names) != set(CK.BANDS):
        raise SystemExit(f"band names differ from the checklist: {names}")

    # equivalence with an archived prototype (one prototype, GPU seconds)
    if args.equivalence_epochs:
        ep = np.load(args.equivalence_epochs)
        tim: dict = {}
        r = analyse_entry(model, real[args.proto_idx], ep, tm, bands, device, tim)
        r["timing_s"] = tim
        atomic_write(out / f"equivalence_proto{args.proto_idx:03d}.json", r)
        log(f"equivalence entry {args.proto_idx}: stage {r['stage']} purity {r['purity']:.2f}; "
            f"timing {tim}")
        for n in names:
            log(f"   {n:>10s} rel {r['band_rel'][n]:7.2f}%  dB {r['db'][n]:+6.2f}  {r['direction'][n]}")
        log(f"   channels {r['ch_pcts']}  EOG power {r['eog_power']:.3f}  EMG tone {r['emg_tone']:.3f}")
        if not args.pilot:
            return 0

    raw, Z, pred, index, ids = build_pool(args.backbone, model, device, args.pool_subjects, args.seed)
    alloc = parse_alloc(args.stage_alloc)

    if args.pilot:
        rng = np.random.RandomState(args.seed)
        sel = draw_entries(rng, pred, args.m, args.variant, alloc)
        nb, sizes = neighbours(Z, Z[sel], args.top_k)
        tim = {}
        t = time.time()
        rec = analyse_entry(model, Z[sel][0], gather(raw, index, nb[0][0]), tm, bands, device, tim)
        log(f"pilot, one entry of a null codebook: {time.time() - t:.1f} s total, {tim}; "
            f"stage {rec['stage']}, cluster {sizes[0]}")
        return 0

    if args.real:
        p = out / "real_pool.json"
        if not p.exists():
            log("real codebook on the same pool")
            res = run_codebook(model, real, raw, Z, index, pred, tm, bands, device, args.top_k, "real")
            res["pool_subjects"] = ids
            atomic_write(p, res)
            log(f"real: {res['counts']}  stages {res['stage_composition']}")

    rng = np.random.RandomState(args.seed)
    for k in range(args.n_codebooks):
        sel = draw_entries(rng, pred, args.m, args.variant, alloc)   # advanced for every k
        p = out / f"{args.variant}_{k:03d}.json"
        if p.exists():
            continue
        log(f"codebook {k + 1}/{args.n_codebooks} ({args.variant})")
        res = run_codebook(model, Z[sel], raw, Z, index, pred, tm, bands, device, args.top_k, f"{k:03d}")
        res.update(variant=args.variant, k=k, seed=args.seed, selected=[int(i) for i in sel])
        atomic_write(p, res)
        log(f"  -> {res['counts']}  stages {res['stage_composition']}")
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
