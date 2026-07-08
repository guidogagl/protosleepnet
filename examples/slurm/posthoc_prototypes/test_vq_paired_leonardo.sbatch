#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=00:30:00
#SBATCH --job-name=vq-paired
#SBATCH --output=${PSN_LOGS}/%x_%j.out

module load python/3.11.7
module load cuda/12.1
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
export PHYSIOEX_CACHE_DIR=${PHYSIOEX_CACHE_DIR}

cd ${PHYSIOEX_REPO}

MODEL="${MODEL:-st-baseline}"
M="${M:-48}"

PRETRAINED=${PSN_PRETRAINED}
EMB_DIR=$PRETRAINED/posthoc_embeddings/$MODEL
CODEBOOK=$EMB_DIR/codebook_vq_m${M}.npy

echo "Paired test: model=$MODEL, M=$M"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/test_vq_paired.py \
    --model_dir $PRETRAINED/$MODEL \
    --codebook_path $CODEBOOK \
    --output_dir $EMB_DIR \
    --gpu_id 0
