#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --job-name=smoke-ablation
#SBATCH --output=${PSN_LOGS}/%x_%j.out

module load profile/deeplrn
module load cineca-ai/4.0.0
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy ONLY shhs_visit1 cache to /dev/shm (124 GB)
echo "Copying shhs_visit1 cache to /dev/shm..."
CACHE_SRC=${PHYSIOEX_CACHE_DIR}
mkdir -p /dev/shm/physioex_cache/v1
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

cd ${PHYSIOEX_REPO}

VARIANT="${VARIANT:-baseline}"
echo "Running smoke test for variant: $VARIANT"

if [ "$VARIANT" = "prosleepnet" ]; then
    python -m protosleepnet.train \
        --datasets shhs \
        --max_epochs 1 \
        --gpu_id 0 \
        --num_workers 14 \
        --output_dir ${PSN_PRETRAINED}/smoke/protosleepnet-gagliardi
else
    python -m protosleepnet.ablation.train_ablation \
        --variant "$VARIANT" \
        --datasets shhs \
        --max_epochs 1 \
        --gpu_id 0 \
        --num_workers 14 \
        --output_dir "${PSN_PRETRAINED}/smoke/$VARIANT"
fi
