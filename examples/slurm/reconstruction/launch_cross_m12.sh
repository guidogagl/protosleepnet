#!/bin/bash
# Launch all cross-dataset M=12 reconstruction jobs on Leonardo
# Usage: bash launch_cross_m12.sh [--dry-run]

SBATCH_SCRIPT="examples/slurm/reconstruction/cross_dataset_m12_leonardo.sbatch"

DRY_RUN=false
[ "$1" = "--dry-run" ] && DRY_RUN=true

# SEQ backbone (trained on MASS) — cross-dataset on everything except MASS
SEQ_DATASETS="hmc shhs_v1 shhs_v2 mesa sleepedf wsc_v1 wsc_v2 wsc_v3 wsc_v4 dcsm mros park_night park_nap alzheimers hpap_full hpap_split"

# ST backbone (trained on SHHS v1) — cross-dataset on everything except SHHS v1
ST_DATASETS="mass_c1 mass_c2 mass_c3 mass_c4 mass_c5 hmc shhs_v2 mesa sleepedf wsc_v1 wsc_v2 wsc_v3 wsc_v4 dcsm mros park_night park_nap alzheimers hpap_full hpap_split"

COUNT=0

for DS in $SEQ_DATASETS; do
    CMD="sbatch --export=BACKBONE=seq,DATASET=$DS --job-name=cross_m12_seq_${DS} $SBATCH_SCRIPT"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY] $CMD"
    else
        echo "Submitting: seq / $DS"
        $CMD
    fi
    COUNT=$((COUNT + 1))
done

for DS in $ST_DATASETS; do
    CMD="sbatch --export=BACKBONE=st,DATASET=$DS --job-name=cross_m12_st_${DS} $SBATCH_SCRIPT"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY] $CMD"
    else
        echo "Submitting: st / $DS"
        $CMD
    fi
    COUNT=$((COUNT + 1))
done

echo ""
echo "Total jobs: $COUNT"
echo "SEQ datasets: $(echo $SEQ_DATASETS | wc -w)"
echo "ST datasets: $(echo $ST_DATASETS | wc -w)"
