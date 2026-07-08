#!/bin/bash
#SBATCH --partition=zen4_h200
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --time=02:00:00
#SBATCH --output=${PSN_LOGS}/cross_m12_%x_%j.out
#SBATCH --job-name=cross_m12

# Usage: sbatch --export=BACKBONE=seq,DATASET=hmc cross_dataset_m12_sofia.sbatch

module swap cluster/sofia/zen4_h200 2>/dev/null
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0
module load matplotlib/3.10.3-gfbf-2025a scikit-learn/1.7.0-gfbf-2025a
module load h5py/3.14.0-foss-2025a PyYAML/6.0.2-GCCcore-14.2.0
module load tqdm/4.67.1-GCCcore-14.2.0 Pillow/11.3.0-GCCcore-14.2.0
source ${PSN_VENV_ACTIVATE}

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PHYSIOEX_DATA=${PSN_WORK}/raw_data
export PHYSIOEX_CACHE_DIR=${PHYSIOEX_CACHE_DIR}

# Paths
CODE=protosleepnet.proto_reconstruction
PRETRAINED=${PSN_WORK}/pretrained

# Resolve backbone-specific paths
if [ "$BACKBONE" = "seq" ]; then
    MODEL_NAME="protosleepnet-seq-3ch-mixer"
elif [ "$BACKBONE" = "st" ]; then
    MODEL_NAME="protosleepnet-st-3ch-mixer"
else
    echo "ERROR: BACKBONE must be 'seq' or 'st', got '$BACKBONE'"
    exit 1
fi

CKPT="${PRETRAINED}/${MODEL_NAME}/model.pt"
CODEBOOK="$WORK/posthoc_embeddings/${MODEL_NAME}/codebooks/vq_kmeans/M12.npy"
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

python -m ${CODE}.data_driven \
    --backbone $BACKBONE \
    --m 12 \
    --top_k 256 \
    --checkpoint_path $CKPT \
    --codebook_path $CODEBOOK \
    --output_dir $OUTPUT \
    --force \
    $EXTRA_ARGS

echo "=== Done (exit $?) ==="
