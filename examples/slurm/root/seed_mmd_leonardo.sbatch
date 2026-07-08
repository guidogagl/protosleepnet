#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=04:00:00
#SBATCH --job-name=seed-mmd
#SBATCH --output=${PSN_LOGS}/%x_%A_%a.out
#SBATCH --array=0-1

source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1

cd ${PHYSIOEX_REPO}

SCRIPT=examples/pretrained/protosleepnet-gagliardi/seed_mmd_analysis.py

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "=== PSN (SEQ/MASS) ==="
    python $SCRIPT \
        --results_dir ${PSN_RESULTS}/seed_stability/seq \
        --backbone_name PSN \
        --seeds 123 456 789 1024 2048 3141
else
    echo "=== PST (ST/SHHS) ==="
    python $SCRIPT \
        --results_dir ${PSN_RESULTS}/seed_stability/st \
        --backbone_name PST \
        --seeds 123 456 789 1024 2048 3141
fi
