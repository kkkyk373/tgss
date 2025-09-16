#!/bin/bash

# === ジョブ設定 ===
JOB_NAME="grav18_select"
LOG_BASE="logs/${JOB_NAME}"
mkdir -p "${LOG_BASE}"

# --- User-configurable paths ---
DATA_DIR="/home/hayashi/projects/tgss/syn-dataset/18domains"
FGW_DIR="/home/hayashi/projects/tgss/syn-dataset/outputs/18domains"
RESULTS_DIR="/home/hayashi/projects/tgss/results/synthetic"
VENV_ACTIVATE="/home/hayashi/projects/tgss/env/bin/activate"

# --- Experiment grid ---
CONDITIONS=("topk" "random" "bottomk" "all")
ALPHAS=(0.0 0.25 0.5 0.75 1.0)  # used for all but 'all'
TOPK=50
BOTTOMK=50
SEEDS=(0 1 2 3 4 5 6 7 8 9)

# === Slurm の array job を再現する ===
PARAMS=()
for seed in "${SEEDS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for cond in "topk" "random" "bottomk"; do
      PARAMS+=("${cond} ${alpha} ${seed}")
    done
  done
done
for seed in "${SEEDS[@]}"; do
  PARAMS+=("all 0.0 ${seed}")
done

# --- 引数で Task ID を指定 (例: ./bash_grav18_select.sh 0) ---
if [ -z "$1" ]; then
  echo "Usage: $0 <TASK_ID>"
  exit 1
fi
TASK_ID=$1

read PARAM_COND PARAM_ALPHA PARAM_SEED <<< "${PARAMS[$TASK_ID]}"

# === ログ出力 ===
ALPHA_STR=$(printf '%02d' $(echo "${PARAM_ALPHA} * 100" | bc | awk '{print int($1 + 0.5)}'))
LOG_DIR="${LOG_BASE}/${PARAM_COND}/alpha${ALPHA_STR}"
mkdir -p "${LOG_DIR}"

JOB_ID=$$
exec > "${LOG_DIR}/${JOB_ID}_seed${PARAM_SEED}.out" 2> "${LOG_DIR}/${JOB_ID}_seed${PARAM_SEED}.err"

echo "--- GravityPower 18domains Selective ---"
echo "Task: ${TASK_ID}"
echo "cond=${PARAM_COND} alpha=${PARAM_ALPHA} seed=${PARAM_SEED}"

# === 仮想環境有効化 ===
source "${VENV_ACTIVATE}"

# === 実行 ===
PYTHONPATH=. python src/experiments/run_selective_gravity_18domains.py \
  --data_dir "${DATA_DIR}" \
  --fgw_dir "${FGW_DIR}" \
  --targets_path "syn_source_target_lists/targets_seed${PARAM_SEED}.txt" \
  --sources_path "syn_source_target_lists/sources_seed${PARAM_SEED}.txt" \
  --results_dir "${RESULTS_DIR}" \
  --condition "${PARAM_COND}" \
  --top_k ${TOPK} \
  --bottom_k ${BOTTOMK} \
  --alpha ${PARAM_ALPHA} \
  --max_samples 20000 \
  --epochs 500 \
  --batch_size 64 \
  --lr 1e-4 \
  --seed ${PARAM_SEED}

echo "--- Done ---"
