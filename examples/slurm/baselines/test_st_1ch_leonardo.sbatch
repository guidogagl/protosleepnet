#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=02:00:00
#SBATCH --job-name=eval-st-1ch
#SBATCH --output=${PSN_LOGS}/%x_%j.out

module load profile/deeplrn
module load cineca-ai/4.0.0
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PHYSIOEX_CACHE_DIR=${PHYSIOEX_CACHE_DIR}

OUTDIR=${PSN_PRETRAINED}/sleeptransformer-phan-1ch-eval
mkdir -p "$OUTDIR"

cd ${PHYSIOEX_REPO}

python examples/pretrained/protosleepnet-gagliardi/baselines/test_st_1ch.py \
    --gpu_id 0 \
    --output_dir "$OUTDIR"
