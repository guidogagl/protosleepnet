#!/usr/bin/env python3
"""Add/refresh model cards (README.md) for the 3-channel baseline + ablation
variants already published in 4rooms/sleep-prototypes.

Only uploads README.md into each existing subfolder (weights/config untouched).

    python scripts/upload_variant_cards.py            # upload
    python scripts/upload_variant_cards.py --dry-run
"""
import argparse, tempfile, pathlib
from huggingface_hub import HfApi

REPO = "4rooms/sleep-prototypes"

# hf_name -> (title, backbone, dataset, seqlen, variant description)
VARIANTS = {
    "sleeptransformer-gagliardi":          ("SleepTransformer 3ch (baseline)", "SleepTransformer", "SHHS", 21, "Per-channel SleepTransformer over 3 channels with concatenation + pooling — the ablation baseline (no channel dropout, no mixer)."),
    "sleeptransformer-gagliardi-dropout":  ("SleepTransformer 3ch + channel dropout", "SleepTransformer", "SHHS", 21, "Baseline + random input-level channel dropout."),
    "sleeptransformer-gagliardi-mixer":    ("SleepTransformer 3ch + channel mixer", "SleepTransformer", "SHHS", 21, "Baseline + accuracy-weighted channel dropout and a Transformer channel mixer (no residual)."),
    "seqsleepnet-gagliardi":               ("SeqSleepNet 3ch (baseline)", "SeqSleepNet", "MASS", 20, "Per-channel SeqSleepNet over 3 channels with concatenation + pooling — the ablation baseline (no channel dropout, no mixer)."),
    "seqsleepnet-gagliardi-dropout":       ("SeqSleepNet 3ch + channel dropout", "SeqSleepNet", "MASS", 20, "Baseline + random input-level channel dropout."),
    "seqsleepnet-gagliardi-mixer":         ("SeqSleepNet 3ch + channel mixer", "SeqSleepNet", "MASS", 20, "Baseline + accuracy-weighted channel dropout and a Transformer channel mixer (no residual)."),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tmpl = (pathlib.Path(__file__).parent / "model_card_variant.md").read_text()
    api = HfApi()
    for name, (title, backbone, dataset, seqlen, desc) in VARIANTS.items():
        card = (tmpl.replace("{MODEL_TITLE}", title).replace("{MODEL_ID}", name)
                    .replace("{BACKBONE}", backbone).replace("{DATASET}", dataset)
                    .replace("{SEQLEN}", str(seqlen)).replace("{VARIANT_DESC}", desc)
                    .replace("{REPO}", REPO))
        print(f"[{'DRY' if a.dry_run else 'UP'}] {REPO}/{name}/README.md ({len(card)} chars)")
        if not a.dry_run:
            with tempfile.TemporaryDirectory() as td:
                fp = pathlib.Path(td) / "README.md"
                fp.write_text(card)
                api.upload_file(path_or_fileobj=str(fp), path_in_repo=f"{name}/README.md",
                                repo_id=REPO, repo_type="model",
                                commit_message=f"Add model card for {name}")
    print("done.")


if __name__ == "__main__":
    main()
