#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --job-name=smoke-prosleepnet
#SBATCH --output=${PSN_LOGS}/%x_%j.out

# Use torch2x venv (torch 2.x with SimVQ/einx support)
module load python/3.11.7
module load cuda/12.1
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Verify torch version and CUDA
python -c "import torch; print(f'torch {torch.__version__}, CUDA avail: {torch.cuda.is_available()}')"

# Copy ONLY shhs_visit1 cache to /dev/shm (124 GB)
echo "Copying shhs_visit1 cache to /dev/shm..."
CACHE_SRC=${PHYSIOEX_CACHE_DIR}
mkdir -p /dev/shm/physioex_cache/v1
cp -r $CACHE_SRC/v1/shhs_visit1 /dev/shm/physioex_cache/v1/
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache
echo "Cache copied: $(du -sh /dev/shm/physioex_cache | cut -f1)"

cd ${PHYSIOEX_REPO}

python -m protosleepnet.train \
    --datasets shhs \
    --max_epochs 1 \
    --gpu_id 0 \
    --num_workers 14 \
    --output_dir ${PSN_PRETRAINED}/smoke/protosleepnet-st-3ch-mixer
