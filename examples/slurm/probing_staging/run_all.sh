#!/bin/bash
# Submit 88 jobs: 4 models × 22 datasets
# Each job gets appropriate --mem, --time, and USE_SHM based on dataset size

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODELS=(
    protosleepnet-st-3ch-mixer
    protosleepnet-seq-3ch-mixer
    sleeptransformer-phan
    seqsleepnet-phan
)

# dataset_name|cache_name|mem|time|use_shm
# Tier 1: Small (≤5GB) — shm, 32G, 1h
# Tier 2: Medium (5-15GB) — shm, 64G, 2h
# Tier 3: Large (15-25GB) — shm, 128G, 4h
# Tier 4: Very large (>50GB) — no shm, 32G, 2 days
DATASETS=(
    # Tier 1
    "wsc_visit5|wsc_visit5|32G|01:00:00|1"
    "parkinsons_nap|parkinsons_nap|32G|01:00:00|1"
    "mass_cohort2|mass_ss02|32G|01:00:00|1"
    "mass_cohort5|mass_ss05|32G|01:00:00|1"
    "mass_cohort1|mass_ss01|32G|01:00:00|1"
    "mass_cohort4|mass_ss04|32G|01:00:00|1"
    "mass_cohort3|mass_ss03|32G|01:00:00|1"
    "parkinsons_night|parkinsons_night|32G|01:00:00|1"
    "wsc_visit4|wsc_visit4|32G|01:00:00|1"
    "hmc|hmc|32G|01:00:00|1"
    "alzheimers|alzheimers|32G|01:00:00|1"
    "hpap_lab-full|hpap|32G|01:00:00|1"
    "hpap_lab-split|hpap|32G|01:00:00|1"
    # Tier 2
    "sleepedf|sleepedf|64G|02:00:00|1"
    "wsc_visit3|wsc_visit3|64G|02:00:00|1"
    "dcsm|dcsm|64G|02:00:00|1"
    # Tier 3
    "wsc_visit2|wsc_visit2|128G|04:00:00|1"
    "wsc_visit1|wsc_visit1|128G|04:00:00|1"
    # Tier 4
    "mesa|mesa|32G|2-00:00:00|0"
    "mros|mros|32G|2-00:00:00|0"
    "shhs_visit2|shhs_visit2|32G|2-00:00:00|0"
    "shhs_visit1|shhs_visit1|32G|3-00:00:00|0"
)

count=0
for model in "${MODELS[@]}"; do
    model_dir="${PSN_PRETRAINED}/${model}"
    if [ ! -f "${model_dir}/config.json" ]; then
        echo "[SKIP] ${model}: no config.json at ${model_dir}"
        continue
    fi

    for entry in "${DATASETS[@]}"; do
        IFS='|' read -r ds_name cache_name mem walltime use_shm <<< "$entry"

        sbatch --job-name="stg_${model}_${ds_name}" \
               --mem="${mem}" \
               --time="${walltime}" \
               --export=ALL,MODEL_NAME="${model}",DATASET_NAME="${ds_name}",CACHE_NAME="${cache_name}",USE_SHM="${use_shm}" \
               "${SCRIPT_DIR}/run_leonardo.sbatch"
        count=$((count + 1))
    done
done

echo "Submitted ${count} jobs"
