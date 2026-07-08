#!/usr/bin/env python3
"""Publish ProtoSleepNet pretrained weights to the HuggingFace Hub.

Uploads, per model, into a `<name>/` subfolder of the target repo:
  model.pt, config.json, training_mean.npy, training_class_means.npy, README.md

Default target is the physioex model hub (public). Use --repo to stage to a
private repo (e.g. 4rooms/protosleepnet) and move it later.

    python scripts/upload_weights.py --models-root /path/to/outputs/protosleepnet
    python scripts/upload_weights.py --repo 4rooms/protosleepnet --private
"""
import argparse, json, tempfile, pathlib, shutil
from huggingface_hub import HfApi

MODELS = {
    "protosleepnet-st-3ch-mixer":  dict(title="ProtoSleepTransformer (PST)", backbone="SleepTransformer", dataset="SHHS", seqlen=21),
    "protosleepnet-seq-3ch-mixer": dict(title="ProtoSeqSleepNet (PSN)",      backbone="SeqSleepNet",      dataset="MASS", seqlen=20),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="4rooms/physioex")
    ap.add_argument("--models-root", required=True, help="dir containing models/<name>/ and pretrained/<name>/")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = pathlib.Path(a.models_root)
    card = (pathlib.Path(__file__).parent / "model_card.md").read_text()
    api = HfApi()
    if not a.dry_run:
        api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    for name, meta in MODELS.items():
        src_model = root / "models" / name
        src_stats = root / "pretrained" / name
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            shutil.copy(src_model / "model.pt", td / "model.pt")
            shutil.copy(src_model / "config.json", td / "config.json")
            for s in ("training_mean.npy", "training_class_means.npy"):
                if (src_stats / s).exists():
                    shutil.copy(src_stats / s, td / s)
            readme = (card.replace("{MODEL_TITLE}", meta["title"]).replace("{MODEL_ID}", name)
                          .replace("{BACKBONE}", meta["backbone"]).replace("{DATASET}", meta["dataset"])
                          .replace("{SEQLEN}", str(meta["seqlen"])))
            (td / "README.md").write_text(readme)
            print(f"[{'DRY' if a.dry_run else 'UP'}] {a.repo}/{name}:", sorted(p.name for p in td.iterdir()))
            if not a.dry_run:
                api.upload_folder(folder_path=str(td), path_in_repo=name, repo_id=a.repo, repo_type="model",
                                  commit_message=f"Add {name} (ProtoSleepNet)")
    print("done.")

if __name__ == "__main__":
    main()
