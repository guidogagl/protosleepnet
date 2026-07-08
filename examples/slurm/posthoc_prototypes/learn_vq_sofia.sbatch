#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --time=02:00:00
#SBATCH --job-name=vq-learn
#SBATCH --output=${PSN_LOGS}/%x_%j.out

source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

cd ${PHYSIOEX_REPO}

MODEL_NAME="${MODEL_NAME:-seqsleepnet-phan}"
M="${M:-48}"
METHOD="${METHOD:-vq}"
EMB_DIR="${PSN_WORK}/posthoc_embeddings/${MODEL_NAME}"

echo "VQ learning: model=$MODEL_NAME, M=$M, method=$METHOD"

if [ "$METHOD" = "kmeans" ]; then
    python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq_kmeans.py \
        --emb_dir $EMB_DIR \
        --n_prototypes $M \
        --output_dir $EMB_DIR
elif [ "$METHOD" = "vq_learned" ]; then
    python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq.py \
        --model_name $MODEL_NAME \
        --emb_dir $EMB_DIR \
        --n_prototypes $M \
        --no_kmeans_init \
        --n_epochs 50 \
        --patience 5 \
        --lr 1e-4 \
        --batch_size 32 \
        --gpu_id 0 \
        --output_dir $EMB_DIR
elif [ "$METHOD" = "vq" ]; then
    python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq.py \
        --model_name $MODEL_NAME \
        --emb_dir $EMB_DIR \
        --n_prototypes $M \
        --n_epochs 50 \
        --patience 5 \
        --lr 1e-4 \
        --batch_size 32 \
        --gpu_id 0 \
        --output_dir $EMB_DIR
fi

echo "Done."
