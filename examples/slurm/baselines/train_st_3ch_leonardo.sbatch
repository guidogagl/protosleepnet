#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=10:00:00
#SBATCH --job-name=st-phan-3ch
#SBATCH --output=${PSN_LOGS}/%x_%j.out

module load profile/deeplrn
module load cineca-ai/4.0.0
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy shhs_visit1 cache to /dev/shm
echo "Copying shhs_visit1 cache to /dev/shm..."
CACHE_SRC=${PHYSIOEX_CACHE_DIR}
mkdir -p /dev/shm/physioex_cache/v1
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

OUTDIR=${PSN_PRETRAINED}/sleeptransformer-phan-3ch
mkdir -p "$OUTDIR"

cd ${PHYSIOEX_REPO}

python examples/pretrained/protosleepnet-gagliardi/baselines/train_st_3ch.py \
    --gpu_id 0 \
    --output_dir "$OUTDIR" \
    --num_workers 14
