#!/bin/bash
#SBATCH --job-name=fgw_selective
#SBATCH --partition=cluster_low
#SBATCH --cpus-per-task=8
#SBATCH --array=0-29  # 10 seeds × 3 conditions = 30 jobs
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# === 対応するSEEDと条件をマッピング ===
seeds=(0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9)
conds=("topk" "bottomk" "random" "topk" "bottomk" "random" "topk" "bottomk" "random" \
       "topk" "bottomk" "random" "topk" "bottomk" "random" "topk" "bottomk" "random" \
       "topk" "bottomk" "random" "topk" "bottomk" "random" "topk" "bottomk" "random" \
       "topk" "bottomk" "random")

SEED=${seeds[$SLURM_ARRAY_TASK_ID]}
COND=${conds[$SLURM_ARRAY_TASK_ID]}

# === 固定設定 ===
MODEL="svr"
ALPHA=0
TOP_K=100
MAX_SAMPLES=5000
DATE=$(date +%Y%m%d_%H%M)

# === パス設定 ===
DATA_DIR="/work/hideki-h/jcomm/ComOD-dataset/data"
FGW_DIR="/work/hideki-h/jcomm/ComOD-dataset/outputs"
TARGETS_PATH="source_target_lists/targets_seed${SEED}.txt"
SOURCES_PATH="source_target_lists/sources_seed${SEED}.txt"
SCRIPT="src/experiments/run_selective_${MODEL}.py"
RESULTS_DIR="results"
LOG_DIR="logs"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
source /work/hideki-h/jcomm/env/bin/activate
cd /work/hideki-h/jcomm

# === 出力ファイル名 ===
OUTFILE="${RESULTS_DIR}/topk${TOP_K}_alpha${ALPHA}_selective_seed${SEED}_${COND}_${MODEL}_max${MAX_SAMPLES}_${DATE}.txt"

# === 実行 ===
PYTHONPATH=. python -u "$SCRIPT" \
    --data_dir "$DATA_DIR" \
    --fgw_dir "$FGW_DIR" \
    --targets_path "$TARGETS_PATH" \
    --sources_path "$SOURCES_PATH" \
    --condition "$COND" \
    --alpha "$ALPHA" \
    --top_k "$TOP_K" \
    --bottom_k "$TOP_K" \
    --max_samples "$MAX_SAMPLES" \
    > "$OUTFILE"
