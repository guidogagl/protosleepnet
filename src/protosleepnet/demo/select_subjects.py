"""Rank SleepEDF recordings by how cleanly BOTH proto models stage them, and
print the top-N "particularly simple" recordings to feature in the demo.

CPU-only: reads the already-emitted static bundle (no model, no GPU). Uses
``signals/subjects.json`` (true ``labels``, trimmed sleep window) and each
``<hf>/subjects.json`` (``pred``) — the per-epoch predictions the demo shows.

A recording is a good "featured" candidate when it is EASY (both backbones
stage it accurately) and CLEAN (all five stages present, little unscored, a
consolidated night). We rank by ``min(acc_seq, acc_st)`` among recordings that
pass the cleanliness gate, so the featured nights read clearly under either
model toggle.

Usage:
    python -m protosleepnet.demo.select_subjects --bundle /path/to/bundle -n 4
"""
import argparse
import json
from pathlib import Path

import numpy as np

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
IGNORE = -1
# hf id per backbone (must match precompute.MODELS)
MODEL_HF = {"seq": "protosleepnet-gagliardi", "st": "protosleeptransformer-gagliardi"}


def _load_preds(bundle: Path):
    """sid -> {backbone -> pred list} from each <hf>/subjects.json."""
    preds = {}
    for backbone, hf in MODEL_HF.items():
        meta = json.loads((bundle / hf / "subjects.json").read_text())
        for s in meta["subjects"]:
            preds.setdefault(s["id"], {})[backbone] = np.asarray(s["pred"], dtype=np.int64)
    return preds


def score_subjects(bundle: Path):
    """Return a list of per-subject dicts with accuracy + cleanliness features."""
    signals = json.loads((bundle / "signals" / "subjects.json").read_text())
    preds = _load_preds(bundle)

    rows = []
    for s in signals:
        sid = s["id"]
        labels = np.asarray(s["labels"], dtype=np.int64)
        scored = labels != IGNORE
        n = int(labels.size)
        n_scored = int(scored.sum())
        if n_scored == 0 or sid not in preds:
            continue

        accs = {}
        for backbone, pred in preds[sid].items():
            m = min(pred.size, labels.size)
            valid = scored[:m]
            accs[backbone] = float((pred[:m][valid] == labels[:m][valid]).mean()) if valid.any() else 0.0
        if set(accs) != set(MODEL_HF):  # need both models
            continue

        present = sorted({int(c) for c in labels[scored]})
        counts = {STAGE_NAMES[c]: int((labels[scored] == c).sum()) for c in range(5)}
        transitions = int((np.diff(labels[scored]) != 0).sum())
        rows.append({
            "id": sid,
            "n_epochs": n,
            "n_scored": n_scored,
            "unscored_frac": round(1 - n_scored / n, 4),
            "acc_seq": round(accs["seq"], 4),
            "acc_st": round(accs["st"], 4),
            "min_acc": round(min(accs.values()), 4),
            "stages_present": len(present),
            "n1_frac": round(counts["N1"] / n_scored, 4),
            "transition_rate": round(transitions / max(n_scored - 1, 1), 4),
            "stage_counts": counts,
        })
    return rows


def select(rows, n=4, min_epochs=200, max_unscored=0.35, max_n1=0.25):
    """Cleanliness gate, then rank the survivors by ``min_acc`` (desc)."""
    clean = [
        r for r in rows
        if r["stages_present"] == 5
        and r["n_scored"] >= min_epochs
        and r["unscored_frac"] <= max_unscored
        and r["n1_frac"] <= max_n1
    ]
    clean.sort(key=lambda r: (r["min_acc"], min(r["acc_seq"], r["acc_st"])), reverse=True)
    return clean[:n], clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("-n", "--num", type=int, default=4)
    ap.add_argument("--json_out", type=Path, default=None,
                    help="optional: write the chosen ids + rationale as JSON")
    args = ap.parse_args()

    rows = score_subjects(args.bundle)
    chosen, clean = select(rows, n=args.num)

    print(f"\n{len(rows)} recordings scored; {len(clean)} pass the cleanliness gate.\n")
    hdr = f"{'id':<12}{'min_acc':>8}{'acc_seq':>8}{'acc_st':>8}{'stages':>7}{'n1%':>7}{'unscored%':>10}{'epochs':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in chosen:
        print(f"{r['id']:<12}{r['min_acc']:>8.3f}{r['acc_seq']:>8.3f}{r['acc_st']:>8.3f}"
              f"{r['stages_present']:>7}{r['n1_frac']*100:>6.1f}%{r['unscored_frac']*100:>9.1f}%{r['n_epochs']:>7}")

    print("\nCHOSEN_IDS=" + ",".join(r["id"] for r in chosen))
    for r in chosen:
        print(f"  {r['id']}: stage counts {r['stage_counts']}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"chosen": [r["id"] for r in chosen], "rows": chosen}, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
