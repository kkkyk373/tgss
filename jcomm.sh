#!/bin/bash
#SBATCH --job-name=jcomm
#SBATCH --partition=cluster_long
#SBATCH --cpus-per-task=52
#SBATCH --time=4-04:00:00

METHOD="gbrt"
PREFIX="jcomm"
DATE=$(date +%Y%m%d_%H%M)
PARTITION_NAME=$SLURM_JOB_PARTITION

mkdir -p outs errs

export OUT_FILE="outs/${PREFIX}_${METHOD}_${PARTITION_NAME}_${DATE}.txt"
export ERR_FILE="errs/${PREFIX}_${METHOD}_${PARTITION_NAME}_${DATE}.txt"

exec > "$OUT_FILE" 2> "$ERR_FILE"

echo "Start time: $(date)"
SECONDS=0

# 実行
python "run_$METHOD.py"
EXIT_CODE=$?

DURATION=$SECONDS
END_TIME=$(date)

if [ $EXIT_CODE -eq 0 ]; then
    echo "End time: $END_TIME"
    echo "Elapsed time: $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"
else
    echo "Script failed with exit code $EXIT_CODE at $END_TIME" >&2
fi
