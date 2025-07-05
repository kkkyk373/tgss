#!/bin/bash
#SBATCH --job-name=dgm_exp
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-17

# --- 実験パラメータの定義 ---
CONDITIONS=("topk" "random")
ALPHAS=(0 50 100)
SEEDS=(0 1 2)

# --- SLURMのタスクIDから各パラメータを計算 ---
NUM_CONDITIONS=${#CONDITIONS[@]}
NUM_ALPHAS=${#ALPHAS[@]}
NUM_SEEDS=${#SEEDS[@]}

SEED_I=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
TMP_I=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
ALPHA_I=$((TMP_I % NUM_ALPHAS))
COND_I=$((TMP_I / NUM_ALPHAS))

PARAM_COND=${CONDITIONS[$COND_I]}
PARAM_ALPHA=${ALPHAS[$ALPHA_I]}
PARAM_SEED=${SEEDS[$SEED_I]}

# --- ログファイルのディレクトリを作成 ---
LOG_DIR="logs/dgm_array"
mkdir -p ${LOG_DIR}
export OUT_FILE="${LOG_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
export ERR_FILE="${LOG_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

exec > "$OUT_FILE" 2> "$ERR_FILE"

# --- 環境設定と実行 ---
echo "--- DGM Experiment ---"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Timestamp: $(date)"
echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA}, seed=${PARAM_SEED}"
echo "----------------------"

# 仮想環境のアクティベート
source /work/hideki-h/jcomm/env/bin/activate

# PYTHONPATHにカレントディレクトリを追加して実行
PYTHONPATH=. python src/experiments/run_selective_dgm.py \
    --data_dir "/work/hideki-h/jcomm/ComOD-dataset/data" \
    --fgw_dir "/work/hideki-h/jcomm/ComOD-dataset/outputs" \
    --targets_path "source_target_lists/targets_seed${PARAM_SEED}.txt" \
    --sources_path "source_target_lists/sources_seed${PARAM_SEED}.txt" \
    --results_dir "results" \
    --model_output_dir "outputs" \
    --condition "${PARAM_COND}" \
    --alpha ${PARAM_ALPHA} \
    --seed ${PARAM_SEED} \
    --epochs 20 \
    --max_samples 50000 \
    --lr 0.001 \
    --batch_size 32

echo "--- Job Finished ---"
echo "Timestamp: $(date)"