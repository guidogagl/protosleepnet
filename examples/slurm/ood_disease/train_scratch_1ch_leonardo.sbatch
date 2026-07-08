#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=scratch-1ch-alz
#SBATCH --output=${PSN_LOGS}/%x_%j.out

source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=${PHYSIOEX_CACHE_DIR}

cd ${PHYSIOEX_REPO}

python -m protosleepnet.ood_disease.train_scratch_1ch \
    --gpu_id 0 \
    --lr 1e-3 \
    --max_epochs 50 \
    --patience 10 \
    --batch_size 32 \
    --output_dir ${PSN_PRETRAINED}/scratch-1ch-seqsleepnet-alz
