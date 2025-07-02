#!/bin/bash
#SBATCH --job-name=dgm_selective_dry-run # ジョブ名を変更して単一実行であることを明示
#SBATCH --partition=gpu_long
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ===【フェーズ1】検証用パラメータを手動で設定 ===
# ここで条件を変えながら、いくつかのパターンを試す
ALPHA=0
COND="topk"
SEED=0

# === 基本設定 ===
DATE=$(date +%Y%m%d_%H%M)
MODEL_NAME="dgm"
TOP_K=100
MAX_SAMPLES=50000
EPOCHS=20

# === パス設定 ===
DATA_DIR="/work/hideki-h/jcomm/ComOD-dataset/data"
FGW_DIR="/work/hideki-h/jcomm/ComOD-dataset/outputs"
TARGETS_PATH="source_target_lists/targets_seed${SEED}.txt"
SOURCES_PATH="source_target_lists/sources_seed${SEED}.txt"
SCRIPT_PATH="src/experiments/run_selective_dgm.py"
RESULTS_DIR="results"

# === 実行準備 ===
source /work/hideki-h/jcomm/env/bin/activate
cd /work/hideki-h/jcomm

# === 出力ファイル名 ===
# テスト実行なので、ファイル名に "dry-run" を含めて区別しやすくする
OUTFILE="${RESULTS_DIR}/dry-run_${MODEL_NAME}_alpha${ALPHA}_seed${SEED}_${COND}_${DATE}.txt"

# === 実行 ===
echo "Starting single dry-run job for SEED=${SEED}, COND=${COND}, ALPHA=${ALPHA}"
PYTHONPATH=. python -u "$SCRIPT_PATH" \
    --data_dir "$DATA_DIR" \
    --fgw_dir "$FGW_DIR" \
    --targets_path "$TARGETS_PATH" \
    --sources_path "$SOURCES_PATH" \
    --condition "$COND" \
    --alpha "$ALPHA" \
    --top_k "$TOP_K" \
    --bottom_k "$TOP_K" \
    --max_samples "$MAX_SAMPLES" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch_size 32 \
    --lr 0.001 \
    > "$OUTFILE"

echo "Job finished."
