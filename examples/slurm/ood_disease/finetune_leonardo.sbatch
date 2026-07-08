#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=finetune-alz
#SBATCH --output=${PSN_LOGS}/%x_%j.out

# Usage: sbatch --export=BACKBONE=seq finetune_leonardo.sbatch
#        sbatch --export=BACKBONE=st,LR=1e-3 finetune_leonardo.sbatch

BACKBONE=${BACKBONE:-seq}
LR=${LR:-1e-5}
FROM_SCRATCH=${FROM_SCRATCH:-}

source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=${PHYSIOEX_CACHE_DIR}

MODELS_DIR=${PSN_PRETRAINED}
if [ -n "$FROM_SCRATCH" ]; then
    OUTDIR=${MODELS_DIR}/finetune-alzheimer-${BACKBONE}-lr${LR}-scratch
else
    OUTDIR=${MODELS_DIR}/finetune-alzheimer-${BACKBONE}-lr${LR}
fi
mkdir -p "$OUTDIR"

cd ${PHYSIOEX_REPO}

EXTRA_ARGS=""
if [ -n "$FROM_SCRATCH" ]; then
    EXTRA_ARGS="--from_scratch"
else
    if [ "$BACKBONE" = "seq" ]; then
        CKPT=${MODELS_DIR}/protosleepnet-seq-3ch-mixer/model.pt
    else
        CKPT=${MODELS_DIR}/protosleepnet-st-3ch-mixer/model.pt
    fi
    EXTRA_ARGS="--checkpoint $CKPT"
fi

python -m protosleepnet.ood_disease.finetune_alzheimer \
    --backbone "$BACKBONE" \
    $EXTRA_ARGS \
    --gpu_id 0 \
    --output_dir "$OUTDIR" \
    --lr $LR \
    --max_epochs 50 \
    --patience 10 \
    --batch_size 32
