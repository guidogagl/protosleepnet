#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=01:00:00
#SBATCH --job-name=eval-transport
#SBATCH --output=${PSN_LOGS}/%x_%j.out

# Usage: sbatch --export=MODE=global eval_transport_leonardo.sbatch
#        sbatch --export=MODE=per_class eval_transport_leonardo.sbatch

MODE=${MODE:-global}

source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=${PHYSIOEX_CACHE_DIR}

cd ${PHYSIOEX_REPO}

python -m protosleepnet.ood_disease.eval_ood_transport \
    --gpu_id 0 \
    --mode "$MODE" \
    --stats ${PROTOSLEEPNET_MODELS}/ood_disease/mass_stft_stats.pt \
    --models_dir ${PSN_PRETRAINED} \
    --output_dir ${PSN_PRETRAINED}/fix-alzheimer
