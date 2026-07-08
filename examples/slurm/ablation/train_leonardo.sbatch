#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=24:00:00
#SBATCH --job-name=train-ablation
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
echo "Training variant: $VARIANT"

if [ "$VARIANT" = "prosleepnet" ]; then
    python examples/pretrained/protosleepnet-gagliardi/train.py \
        --datasets shhs \
        --gpu_id 0 \
        --num_workers 14 \
        --output_dir ${PSN_PRETRAINED}/protosleepnet-gagliardi \
        --upload
else
    python examples/pretrained/protosleepnet-gagliardi/ablation/train_ablation.py \
        --variant "$VARIANT" \
        --datasets shhs \
        --gpu_id 0 \
        --num_workers 14 \
        --output_dir "${PSN_PRETRAINED}/$VARIANT" \
        --upload
fi
