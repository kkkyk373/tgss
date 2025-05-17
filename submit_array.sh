#!/bin/bash
#SBATCH --job-name=fgw_selective
#SBATCH --partition=cluster_short
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a_%x_${condition}.out
#SBATCH --error=logs/%x_%A_%a_%x_${condition}.err


# === 条件の配列 ===
conditions=("topk" "bottomk" "random")
condition="${conditions[$SLURM_ARRAY_TASK_ID]}"

# === 固定設定 ===
TARGET_AREA="55001"
ALPHA=50
TOP_K=100
MAX_SAMPLES=50000
DATE=$(date +%Y%m%d_%H%M)

# === パス設定 ===
DATA_DIR="/work/hideki-h/jcomm/ComOD-dataset/data"
FGW_DIR="/work/hideki-h/jcomm/ComOD-dataset/outputs"
RESULTS_DIR="results"
SCRIPT="src/experiments/run_selective_rf.py"

# === 環境・作業準備 ===
source /work/hideki-h/jcomm/env/bin/activate
cd /work/hideki-h/jcomm
mkdir -p logs "$RESULTS_DIR"

# === 出力ファイルパス ===
OUTPUT_PATH="${RESULTS_DIR}/${TARGET_AREA}_rf_${condition}_${DATE}.json"

# === 実行 ===
PYTHONPATH=. python "$SCRIPT" \
  --data_dir "$DATA_DIR" \
  --fgw_dir "$FGW_DIR" \
  --target_area "$TARGET_AREA" \
  --condition "$condition" \
  --alpha "$ALPHA" \
  --top_k "$TOP_K" \
  --max_samples "$MAX_SAMPLES" \
  > "$OUTPUT_PATH"