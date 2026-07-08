#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=${PSN_LOGS}/cross_m12_%x_%j.out
#SBATCH --job-name=cross_m12

# Usage: sbatch --export=BACKBONE=seq,DATASET=hmc cross_dataset_m12_leonardo.sbatch

# Use torch2x venv (PyTorch 2.4+cu121)
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PHYSIOEX_DATA=${PHYSIOEX_DATA}

# Copy dataset cache to /dev/shm for fast I/O
CACHE_SRC=${PHYSIOEX_CACHE_DIR}
mkdir -p /dev/shm/physioex_cache/v1
echo "Copying ${DATASET} cache to /dev/shm..."
# Map DATASET name to cache directory name
case "$DATASET" in
    mass_c[1-5]) CACHE_NAME="mass_ss0${DATASET: -1}" ;;
    shhs_v1)     CACHE_NAME="shhs_visit1" ;;
    shhs_v2)     CACHE_NAME="shhs_visit2" ;;
    wsc_v[1-5])  CACHE_NAME="wsc_visit${DATASET: -1}" ;;
    hpap_full)   CACHE_NAME="hpap" ;;
    hpap_split)  CACHE_NAME="hpap" ;;
    park_night)  CACHE_NAME="parkinsons_night" ;;
    park_nap)    CACHE_NAME="parkinsons_nap" ;;
    alzheimers)  CACHE_NAME="alzheimers" ;;
    *)           CACHE_NAME="$DATASET" ;;
esac
if [ -d "$CACHE_SRC/v1/$CACHE_NAME" ]; then
    cp -r "$CACHE_SRC/v1/$CACHE_NAME" /dev/shm/physioex_cache/v1/
    echo "Copied $CACHE_NAME ($(du -sh /dev/shm/physioex_cache/v1/$CACHE_NAME | cut -f1))"
else
    echo "WARNING: cache $CACHE_SRC/v1/$CACHE_NAME not found, using slow path"
fi
export PHYSIOEX_CACHE_DIR=/dev/shm/physioex_cache

# Paths
PRETRAINED=${PSN_PRETRAINED}
CODE=${PHYSIOEX_REPO}/examples/pretrained/protosleepnet-gagliardi/proto-reconstruction

# Resolve backbone-specific paths
if [ "$BACKBONE" = "seq" ]; then
    MODEL_NAME="protosleepnet-seq-3ch-mixer"
    CKPT="${PRETRAINED}/${MODEL_NAME}/model.pt"
    CODEBOOK="${PRETRAINED}/${MODEL_NAME}/embeddings/codebook_vq_kmeans_m12.npy"
elif [ "$BACKBONE" = "st" ]; then
    MODEL_NAME="protosleepnet-st-3ch-mixer"
    CKPT="${PRETRAINED}/${MODEL_NAME}/model.pt"
    CODEBOOK="${PRETRAINED}/${MODEL_NAME}/posthoc_embeddings/${MODEL_NAME}/codebook_vq_kmeans_m12.npy"
else
    echo "ERROR: BACKBONE must be 'seq' or 'st', got '$BACKBONE'"
    exit 1
fi

OUTPUT="${PRETRAINED}/${MODEL_NAME}/reconstruction_per_dataset_m12/${DATASET}"

echo "=== Cross-dataset M=12 reconstruction ==="
echo "Backbone: $BACKBONE ($MODEL_NAME)"
echo "Dataset:  $DATASET"
echo "Output:   $OUTPUT"
echo "Codebook: $CODEBOOK"
echo "Checkpoint: $CKPT"
echo "Python: $(which python)"

# Parse dataset-specific args
EXTRA_ARGS=""
case "$DATASET" in
    mass_c[1-5]) EXTRA_ARGS="--dataset mass --cohort ${DATASET: -1}" ;;
    shhs_v[12])  EXTRA_ARGS="--dataset shhs --visit ${DATASET: -1}" ;;
    wsc_v[1-4])  EXTRA_ARGS="--dataset wsc --visit ${DATASET: -1}" ;;
    hpap_full)   EXTRA_ARGS="--dataset hpap --subset lab-full" ;;
    hpap_split)  EXTRA_ARGS="--dataset hpap --subset lab-split" ;;
    park_night)  EXTRA_ARGS="--dataset parkinsons --recording night" ;;
    park_nap)    EXTRA_ARGS="--dataset parkinsons --recording nap" ;;
    alzheimers)  EXTRA_ARGS="--dataset alzheimers" ;;
    *)           EXTRA_ARGS="--dataset $DATASET" ;;
esac

echo "Extra args: $EXTRA_ARGS"

python ${CODE}/data_driven.py \
    --backbone $BACKBONE \
    --m 12 \
    --top_k 256 \
    --checkpoint_path $CKPT \
    --codebook_path $CODEBOOK \
    --output_dir $OUTPUT \
    --force \
    $EXTRA_ARGS

echo "=== Done (exit $?) ==="
