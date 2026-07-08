#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=${PSN_LOGS}/%x_%j.out

# --mem and --time are set by run_all.sh via sbatch flags
# MODEL_NAME, DATASET_NAME, CACHE_NAME, USE_SHM passed via --export

source ${PSN_VENV_ACTIVATE}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PHYSIOEX_DATA=${PHYSIOEX_DATA}

FAST_CACHE=${PHYSIOEX_CACHE_DIR}
MODEL_DIR=${PSN_PRETRAINED}/${MODEL_NAME}
SCRIPT=${PHYSIOEX_REPO}/probing/staging/extract_embeddings.py

echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME (cache: $CACHE_NAME)"
echo "USE_SHM: $USE_SHM"

if [ "$USE_SHM" = "1" ] && [ -d "$FAST_CACHE/v1/$CACHE_NAME" ]; then
    SHM_CACHE=/dev/shm/physioex_cache
    mkdir -p $SHM_CACHE/v1
    echo "Copying cache $CACHE_NAME to /dev/shm..."
    cp -r "$FAST_CACHE/v1/$CACHE_NAME" "$SHM_CACHE/v1/"
    echo "Cache copy done: $(du -sh $SHM_CACHE/v1/$CACHE_NAME | cut -f1)"
    export PHYSIOEX_CACHE_DIR=$SHM_CACHE
else
    echo "Using FAST cache directly"
    export PHYSIOEX_CACHE_DIR=$FAST_CACHE
fi

python $SCRIPT \
    --model_dir $MODEL_DIR \
    --model_name $MODEL_NAME \
    --datasets $DATASET_NAME \
    --dataset_root $PHYSIOEX_DATA \
    --gpu_id 0

# Copy embeddings back to FAST if we used /dev/shm
if [ "$USE_SHM" = "1" ] && [ -d "$SHM_CACHE/embeddings/$MODEL_NAME/$DATASET_NAME" ]; then
    echo "Copying embeddings back to FAST..."
    mkdir -p "$FAST_CACHE/embeddings/$MODEL_NAME"
    cp -r "$SHM_CACHE/embeddings/$MODEL_NAME/$DATASET_NAME" "$FAST_CACHE/embeddings/$MODEL_NAME/"
    rm -rf $SHM_CACHE
fi

echo "Done."
