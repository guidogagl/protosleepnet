#!/bin/bash
#SBATCH --partition=lrd_all_serial
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=vq-kmeans
#SBATCH --output=${PSN_LOGS}/%x_%j.out

module load python/3.11.7
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1

cd ${PHYSIOEX_REPO}

MODEL="${MODEL:-st-baseline}"
M="${M:-48}"

PRETRAINED=${PSN_PRETRAINED}
EMB_DIR=$PRETRAINED/posthoc_embeddings/$MODEL

echo "VQ K-Means: model=$MODEL, M=$M"

python examples/pretrained/protosleepnet-gagliardi/posthoc_prototypes/learn_prototypes_vq_kmeans.py \
    --emb_dir $EMB_DIR \
    --n_prototypes $M \
    --output_dir $EMB_DIR
