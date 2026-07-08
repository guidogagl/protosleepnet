#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=00:25:00
#SBATCH --job-name=vq-test
#SBATCH --output=${PSN_LOGS}/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

# Copy SHHS cache to /dev/shm (test set only ~25GB)
CACHE_SRC=${PHYSIOEX_CACHE_DIR}
export PHYSIOEX_CACHE_DIR=$CACHE_SRC

cd ${PHYSIOEX_REPO}

MODEL="${MODEL:-st-baseline}"
M="${M:-48}"

PRETRAINED=${PSN_PRETRAINED}
EMB_DIR=$PRETRAINED/posthoc_embeddings/$MODEL
CODEBOOK=$EMB_DIR/codebook_vq_m${M}.npy
OUTPUT_DIR=$EMB_DIR

echo "VQ test: model=$MODEL, M=$M"
echo "Codebook: $CODEBOOK"

python -m protosleepnet.posthoc_prototypes.test_posthoc_prototypes \
    --model_dir $PRETRAINED/$MODEL \
    --method vq \
    --codebook_path $CODEBOOK \
    --output_dir $OUTPUT_DIR \
    --gpu_id 0

echo "Done."
