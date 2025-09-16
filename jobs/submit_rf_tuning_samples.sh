#!/bin/bash
#SBATCH --job-name=rf_tune_samples
#SBATCH --partition=cluster_low
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/rf_tuning/step1_samples/%x_%a.out
#SBATCH --error=logs/rf_tuning/step1_samples/%x_%a.err

# --- ステップ1: max_samples のチューニング ---
CONDITIONS=("topk" "random")
MAX_SAMPLES=(1000 5000 10000 20000 50000)

# --- 他のパラメータは固定 ---
PARAM_TOP_K=100
PARAM_SEED=0
PARAM_ALPHA=100

# --- パラメータ組み合わせを事前に定義 ---
PARAMS=()
for cond in "${CONDITIONS[@]}"; do
    for ms in "${MAX_SAMPLES[@]}"; do
        PARAMS+=("${cond} ${PARAM_ALPHA} ${PARAM_SEED} ${PARAM_TOP_K} ${ms}")
    done
done

# --- SlurmタスクIDからパラメータを取得 ---
CURRENT_PARAMS=(${PARAMS[$SLURM_ARRAY_TASK_ID]})
PARAM_COND=${CURRENT_PARAMS[0]}
PARAM_ALPHA=${CURRENT_PARAMS[1]}
PARAM_SEED=${CURRENT_PARAMS[2]}
PARAM_TOP_K=${CURRENT_PARAMS[3]}
PARAM_MAX_SAMPLES=${CURRENT_PARAMS[4]}

# --- 環境設定と実行 ---
echo "--- RF Tuning Step 1: max_samples ---"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Timestamp: $(date)"
echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA}, seed=${PARAM_SEED}, top_k=${PARAM_TOP_K}, max_samples=${PARAM_MAX_SAMPLES}"
echo "----------------------"

source /work/hideki-h/jcomm/env/bin/activate

PYTHONPATH=. python src/experiments/run_selective_rf_json.py \
    --data_dir "/work/hideki-h/jcomm/ComOD-dataset/data" \
    --fgw_dir "/work/hideki-h/jcomm/ComOD-dataset/outputs" \
    --targets_path "source_target_lists/targets_seed${PARAM_SEED}.txt" \
    --sources_path "source_target_lists/sources_seed${PARAM_SEED}.txt" \
    --results_dir "results" \
    --model_output_dir "outputs" \
    --condition "${PARAM_COND}" \
    --alpha ${PARAM_ALPHA} \
    --seed ${PARAM_SEED} \
    --top_k ${PARAM_TOP_K} \
    --bottom_k ${PARAM_TOP_K} \
    --max_samples ${PARAM_MAX_SAMPLES}

echo "--- Job Finished ---"
echo "Timestamp: $(date)"