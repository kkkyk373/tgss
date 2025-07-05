#!/bin/bash
#SBATCH --job-name=dgm_add_exp
#SBATCH --partition=gpu_long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --array=0-11      # 9 (bottomk) + 3 (all) = 12 jobs

# --- 実験パラメータの定義 ---
ALPHAS=(0 50 100)
SEEDS=(0 1 2)

# --- SLURMのタスクIDから各パラメータを計算 ---
TASK_ID=$SLURM_ARRAY_TASK_ID

# 最初の9ジョブ (ID 0-8) は 'bottomk'
if [ $TASK_ID -lt 9 ]; then
    PARAM_COND="bottomk"
    
    NUM_SEEDS=3
    
    SEED_I=$((TASK_ID % NUM_SEEDS))
    ALPHA_I=$((TASK_ID / NUM_SEEDS))

    PARAM_ALPHA=${ALPHAS[$ALPHA_I]}
    PARAM_SEED=${SEEDS[$SEED_I]}
else
    # 残りの3ジョブ (ID 9-11) は 'all'
    PARAM_COND="all"
    PARAM_ALPHA="0" # 'all'ではalphaは使われないが、引数として渡すため便宜的に設定

    SEED_I=$(( (TASK_ID - 9) % 3 ))
    PARAM_SEED=${SEEDS[$SEED_I]}
fi

# --- ログファイルのディレクトリを作成 ---
LOG_DIR="logs/dgm_array_additional" # ログのディレクトリを分けて管理
mkdir -p ${LOG_DIR}
export OUT_FILE="${LOG_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
export ERR_FILE="${LOG_DIR}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

exec > "$OUT_FILE" 2> "$ERR_FILE"

# --- 環境設定と実行 ---
echo "--- DGM Additional Experiment ---"
echo "Job ID: ${SLURM_JOB_ID}, Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Timestamp: $(date)"
echo "Parameters: condition=${PARAM_COND}, alpha=${PARAM_ALPHA}, seed=${PARAM_SEED}"
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
    --epochs 20 \
    --max_samples 50000 \
    --lr 0.001 \
    --batch_size 32

echo "--- Job Finished ---"
echo "Timestamp: $(date)"