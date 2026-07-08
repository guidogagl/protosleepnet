#!/bin/bash
# Re-submit only the timed-out jobs.
# Embeddings already exist on FAST — only linear_probe needs to run.
# 1h wall time is plenty for probe-only.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS=(protosleepnet-st-3ch-mixer protosleepnet-seq-3ch-mixer sleeptransformer-phan seqsleepnet-phan)

# dataset_name|cache_name — all get 32G, 1h, no shm (embeddings already on FAST)
MISSING=(
    "wsc_visit1|wsc_visit1"
    "wsc_visit2|wsc_visit2"
    "wsc_visit3|wsc_visit3"
    "wsc_visit5|wsc_visit5"
    "dcsm|dcsm"
)

# These only failed for proto-st and proto-seq
MISSING_PROTO_ONLY=(
    "sleepedf|sleepedf"
)

# This only failed for proto-st
MISSING_PROTO_ST_ONLY=(
    "hpap_lab-full|hpap"
)

count=0

# All 4 models × 5 datasets
for model in "${MODELS[@]}"; do
    for entry in "${MISSING[@]}"; do
        IFS='|' read -r ds_name cache_name <<< "$entry"
        sbatch --job-name="stg_${model}_${ds_name}" \
               --mem=32G \
               --time=01:00:00 \
               --export=ALL,MODEL_NAME="${model}",DATASET_NAME="${ds_name}",CACHE_NAME="${cache_name}",USE_SHM=0 \
               "${SCRIPT_DIR}/run_leonardo.sbatch"
        count=$((count + 1))
    done
done

# Proto models only × sleepedf
for model in protosleepnet-st-3ch-mixer protosleepnet-seq-3ch-mixer; do
    IFS='|' read -r ds_name cache_name <<< "${MISSING_PROTO_ONLY[0]}"
    sbatch --job-name="stg_${model}_${ds_name}" \
           --mem=32G \
           --time=01:00:00 \
           --export=ALL,MODEL_NAME="${model}",DATASET_NAME="${ds_name}",CACHE_NAME="${cache_name}",USE_SHM=0 \
           "${SCRIPT_DIR}/run_leonardo.sbatch"
    count=$((count + 1))
done

# Proto-ST only × hpap_lab-full
IFS='|' read -r ds_name cache_name <<< "${MISSING_PROTO_ST_ONLY[0]}"
sbatch --job-name="stg_protosleepnet-st-3ch-mixer_${ds_name}" \
       --mem=32G \
       --time=01:00:00 \
       --export=ALL,MODEL_NAME=protosleepnet-st-3ch-mixer,DATASET_NAME="${ds_name}",CACHE_NAME="${cache_name}",USE_SHM=0 \
       "${SCRIPT_DIR}/run_leonardo.sbatch"
count=$((count + 1))

echo "Submitted ${count} jobs"
