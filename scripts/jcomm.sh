#!/bin/bash
#SBATCH --job-name=jcomm
#SBATCH --partition=cluster_short
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00

METHOD="rf"
PREFIX="example"
DATE=$(date +%Y%m%d_%H%M)
PARTITION_NAME=$SLURM_JOB_PARTITION

mkdir -p logs results

export OUT_FILE="logs/${PREFIX}_${METHOD}_${PARTITION_NAME}_${DATE}.txt"
export ERR_FILE="logs/${PREFIX}_${METHOD}_${PARTITION_NAME}_${DATE}.txt"

exec > "$OUT_FILE" 2> "$ERR_FILE"

echo "Start time: $(date)"
SECONDS=0

# 実行
PYTHONPATH=. python "src/examples/run_$METHOD.py"
EXIT_CODE=$?

DURATION=$SECONDS
END_TIME=$(date)

if [ $EXIT_CODE -eq 0 ]; then
    echo "End time: $END_TIME"
    echo "Elapsed time: $((DURATION / 3600))h $(((DURATION % 3600) / 60))m $((DURATION % 60))s"
else
    echo "Script failed with exit code $EXIT_CODE at $END_TIME" >&2
fi