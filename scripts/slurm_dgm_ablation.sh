#!/bin/bash
#SBATCH --job-name=dgm_tune_samples
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
# Total jobs: 2(cond) * 5(ms) = 10
#SBATCH --array=0-9 

# --- ステップ1: max_samples のチューニング ---
CONDITIONS=("topk" "random")
MAX_SAMPLES=(1000 5000 10000 20000 50000)

# --- 他のパラメータは固定 ---
# --- In future step 2: epochs, step3: batch_size
PARAM_BATCH_SIZE=32
PARAM_EPOCHS=20
PARAM_SEED=0
PARAM_ALPHA=100

# --- パラメータ組み合わせを事前に定義 ---
PARAMS=()
for cond in "${CONDITIONS[@]}"; do
    for ms in "${MAX_SAMPLES[@]}"; do
        PARAMS+=("${cond} ${PARAM_ALPHA} ${PARAM_SEED} ${PARAM_BATCH_SIZE} ${ms} ${PARAM_EPOCHS}")
    done
done

# --- SlurmタスクIDからパラメータを取得 ---
CURRENT_PARAMS=(${PARAMS[$SLURM_ARRAY_TASK_ID]})
PARAM_COND=${CURRENT_PARAMS[0]}
PARAM_ALPHA=${CURRENT_PARAMS[1]}
PARAM_SEED=${CURRENT_PARAMS[2]}
PARAM_BATCH_SIZE=${CURRENT_PARAMS[3]}
PARAM_MAX_SAMPLES=${CURRENT_PARAMS[4]}
PARAM_EPOCHS=${CURRENT_PARAMS[5]}

# --- ログ設定 ---
LOG_DIR="logs/dgm_tuning/step1_samples/cond_${PARAM_COND}/ms_${PARAM_MAX_SAMPLES}"
mkdir -p ${LOG_DIR}
export OUT_FILE="${LOG_DIR}/${SLURM_JOB_ID}.out"
export ERR_FILE="${LOG_DIR}/${SLURM_JOB_ID}.err"
exec > "$OUT_FILE" 2> "$ERR_FILE"

# --- 環境設定と実行 ---
echo "--- DGM Tuning Step 1: max_samples ---"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Timestamp: $(date)"
echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA}, seed=${PARAM_SEED}, batch_size=${PARAM_BATCH_SIZE}, max_samples=${PARAM_MAX_SAMPLES}, epochs=${PARAM_EPOCHS}"
echo "----------------------"

# 仮想環境のアクティベート
source /work/hideki-h/jcomm/env/bin/activate

# Pythonスクリプトの実行
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
    --epochs ${PARAM_EPOCHS} \
    --max_samples ${PARAM_MAX_SAMPLES} \
    --lr 0.001 \
    --batch_size ${PARAM_BATCH_SIZE}

echo "--- Job Finished ---"
echo "Timestamp: $(date)"