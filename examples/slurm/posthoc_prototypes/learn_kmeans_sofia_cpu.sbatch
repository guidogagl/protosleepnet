#!/bin/bash
#SBATCH --partition=zen5_dense
#SBATCH --cpus-per-task=8
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --time=00:30:00
#SBATCH --job-name=vq-km
#SBATCH --output=${PSN_LOGS}/%x_%j.out

source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1

cd ${PHYSIOEX_REPO}

MODEL_NAME="${MODEL_NAME:-seqsleepnet-phan}"
M="${M:-48}"
EMB_DIR="${PSN_WORK}/posthoc_embeddings/${MODEL_NAME}"

echo "K-Means codebook: model=$MODEL_NAME, M=$M"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq_kmeans.py \
    --emb_dir $EMB_DIR \
    --n_prototypes $M \
    --output_dir $EMB_DIR

echo "Done."
