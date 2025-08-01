#!/bin/bash
#SBATCH --job-name=svr_all_exp
#SBATCH --partition=cluster_long 
#SBATCH --cpus-per-task=8      # マルチコアの恩恵を受けるため、コア数は維持
#SBATCH --mem=40G              # データ量に応じてメモリを確保
#SBATCH --time=40:00:00      # 実行時間（必要に応じて調整）
#SBATCH --array=0-9            # seed 0-9 に対応するため、10個のジョブ (0から9まで)

# --- パラメータ定義 ---
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# --- パラメータ組み合わせを事前に定義 (all条件のみ) ---
PARAMS=()
for seed in "${SEEDS[@]}"; do
    # condition="all", alpha=0 (all条件ではalpha値は使われないダミー), seed
    PARAMS+=("all 0 ${seed}")
done

# --- SlurmタスクIDからパラメータを取得 ---
# SLURM_ARRAY_TASK_ID は 0 から始まる
read PARAM_COND PARAM_ALPHA PARAM_SEED <<< "${PARAMS[$SLURM_ARRAY_TASK_ID]}"

# --- ログ設定 ---
LOG_DIR="logs/svr_all_exp/${PARAM_COND}" # alphaは不要なためディレクトリ構造を簡略化
mkdir -p ${LOG_DIR}
export OUT_FILE="${LOG_DIR}/${SLURM_JOB_ID}_seed${PARAM_SEED}.out"
export ERR_FILE="${LOG_DIR}/${SLUM_JOB_ID}_seed${PARAM_SEED}.err"
exec > "$OUT_FILE" 2> "$ERR_FILE"

# --- 環境設定と実行 ---
echo "--- SVR 'all' condition Experiment ---"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Timestamp: $(date)"
echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA} (dummy), seed=${PARAM_SEED}"
echo "----------------------"

# 仮想環境のアクティベート
source /work/hideki-h/jcomm/env/bin/activate

# Pythonスクリプトの実行
PYTHONPATH=. python src/experiments/run_selective_svr.py \
    --data_dir "/work/hideki-h/jcomm/ComOD-dataset/data" \
    --fgw_dir "/work/hideki-h/jcomm/ComOD-dataset/outputs" \
    --targets_path "source_target_lists/targets_seed${PARAM_SEED}.txt" \
    --sources_path "source_target_lists/sources_seed${PARAM_SEED}.txt" \
    --results_dir "results" \
    --model_output_dir "outputs" \
    --condition "${PARAM_COND}" \
    --alpha ${PARAM_ALPHA} \
    --seed ${PARAM_SEED} \
    --max_samples 50000

echo "--- Job Finished ---"
echo "Timestamp: $(date)"