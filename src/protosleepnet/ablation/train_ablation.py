"""Component ablation — SleepTransformer backbone (trained on SHHS), Supplementary §2.

Trains one of the four progressive variants (baseline / dropout / mixer /
protosleepnet) via the shared driver, against physioex v2.0.0.

Usage:
    python -m protosleepnet.ablation.train_ablation --variant baseline --gpu_id 0
    python -m protosleepnet.ablation.train_ablation --variant dropout  --gpu_id 0
    python -m protosleepnet.ablation.train_ablation --variant mixer     --gpu_id 0
    python -m protosleepnet.ablation.train_ablation --variant protosleepnet --gpu_id 0
    python -m protosleepnet.ablation.train_ablation --variant baseline --dataset sleepedf --max_epochs 2
"""
from protosleepnet.ablation._driver import run

if __name__ == "__main__":
    run("st")
