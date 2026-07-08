#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=${PSN_LOGS}/cross_md_m12_%x_%j.out
#SBATCH --job-name=cross_md_m12

# Usage: sbatch --export=BACKBONE=seq,SEED=123 cross_model_driven_m12_leonardo.sbatch

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

OUTPUT="${PRETRAINED}/${MODEL_NAME}/model_driven_seeds_m12/seed_${SEED}"

echo "=== Model-driven M=12 seed=${SEED} ==="
echo "Backbone: $BACKBONE | Seed: $SEED"
echo "Output: $OUTPUT"

python ${CODE}/model_driven.py \
    --backbone $BACKBONE \
    --m 12 \
    --checkpoint_path $CKPT \
    --codebook_path $CODEBOOK \
    --output_dir $OUTPUT \
    --seed $SEED \
    --n_steps 1000 \
    --force

echo "=== Done (exit $?) ==="
