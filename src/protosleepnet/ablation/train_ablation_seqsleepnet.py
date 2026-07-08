"""Component ablation — SeqSleepNet backbone (trained on MASS), Supplementary §2.

Trains one of the four progressive variants (baseline / dropout / mixer /
protosleepnet) via the shared driver, against physioex v2.0.0.

Usage:
    python -m protosleepnet.ablation.train_ablation_seqsleepnet --variant baseline --gpu_id 0
    python -m protosleepnet.ablation.train_ablation_seqsleepnet --variant dropout  --gpu_id 0
    python -m protosleepnet.ablation.train_ablation_seqsleepnet --variant mixer     --gpu_id 0
    python -m protosleepnet.ablation.train_ablation_seqsleepnet --variant protosleepnet --gpu_id 0
"""
from protosleepnet.ablation._driver import run

if __name__ == "__main__":
    run("seq")
