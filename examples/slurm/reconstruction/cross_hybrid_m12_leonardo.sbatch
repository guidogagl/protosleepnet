#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=${PSN_LOGS}/cross_hybrid_m12_%x_%j.out
#SBATCH --job-name=cross_hyb_m12

# Usage: sbatch --export=BACKBONE=seq,DATASET=hmc cross_hybrid_m12_leonardo.sbatch

source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

PRETRAINED=${PSN_PRETRAINED}
CODE=${PHYSIOEX_REPO}/examples/pretrained/protosleepnet-gagliardi/proto-reconstruction

if [ "$BACKBONE" = "seq" ]; then
    MODEL_NAME="protosleepnet-seq-3ch-mixer"
    CKPT="${PRETRAINED}/${MODEL_NAME}/model.pt"
    CODEBOOK="${PRETRAINED}/${MODEL_NAME}/embeddings/codebook_vq_kmeans_m12.npy"
elif [ "$BACKBONE" = "st" ]; then
    MODEL_NAME="protosleepnet-st-3ch-mixer"
    CKPT="${PRETRAINED}/${MODEL_NAME}/model.pt"
    CODEBOOK="${PRETRAINED}/${MODEL_NAME}/posthoc_embeddings/${MODEL_NAME}/codebook_vq_kmeans_m12.npy"
else
    echo "ERROR: BACKBONE must be 'seq' or 'st'"; exit 1
fi

DD_DIR="${PRETRAINED}/${MODEL_NAME}/reconstruction_per_dataset_m12/${DATASET}"
OUTPUT="${PRETRAINED}/${MODEL_NAME}/hybrid_per_dataset_m12/${DATASET}"

echo "=== Cross-dataset HYBRID M=12 ==="
echo "Backbone: $BACKBONE | Dataset: $DATASET"
echo "Data-driven dir: $DD_DIR"
echo "Output: $OUTPUT"

python ${CODE}/hybrid.py \
    --backbone $BACKBONE \
    --m 12 \
    --checkpoint_path $CKPT \
    --codebook_path $CODEBOOK \
    --data_driven_dir $DD_DIR \
    --output_dir $OUTPUT \
    --n_steps 500 \
    --force

echo "=== Done (exit $?) ==="
