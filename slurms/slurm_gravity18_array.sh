#!/bin/bash
#SBATCH --job-name=grav18_select
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-9  # seeds 0-9

# --- User-configurable paths ---
DATA_DIR="/work/hideki-h/jcomm/syn-dataset/18domains"
FGW_DIR="/work/hideki-h/jcomm/syn-dataset/outputs/18domains"
RESULTS_DIR="/work/hideki-h/jcomm/results/synthetic"
VENV_ACTIVATE="/work/hideki-h/jcomm/env/bin/activate"

# --- Experiment grid ---
CONDITIONS=("topk" "random" "bottomk" "all")
ALPHAS=(0.0 0.25 0.5 0.75 1.0)  # used for all but 'all'
TOPK=50
BOTTOMK=50

SEEDS=(0 1 2 3 4 5 6 7 8 9)

# Build PARAM list: (cond, alpha, seed)
PARAMS=()
for seed in "${SEEDS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    for cond in "topk" "random" "bottomk"; do
      PARAMS+=("${cond} ${alpha} ${seed}")
    done
  done
done
# 'all' (alpha unused -> pass 0.0)
for seed in "${SEEDS[@]}"; do
  PARAMS+=("all 0.0 ${seed}")
done

read PARAM_COND PARAM_ALPHA PARAM_SEED <<< "${PARAMS[$SLURM_ARRAY_TASK_ID]}"

# Logs
LOG_DIR="logs/gravity18/${PARAM_COND}/alpha$(printf '%02d' $(echo "${PARAM_ALPHA} * 100" | bc | awk '{print int($1 + 0.5)}'))"
mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/${SLURM_JOB_ID}_seed${PARAM_SEED}.out" 2> "${LOG_DIR}/${SLURM_JOB_ID}_seed${PARAM_SEED}.err"

echo "--- GravityPower 18domains Selective ---"
echo "Job: ${SLURM_JOB_ID} Task: ${SLURM_ARRAY_TASK_ID}"
echo "cond=${PARAM_COND} alpha=${PARAM_ALPHA} seed=${PARAM_SEED}"

source "${VENV_ACTIVATE}"

PYTHONPATH=. python src/experiments/run_selective_gravity_18domains.py \
  --data_dir "${DATA_DIR}" \
  --fgw_dir "${FGW_DIR}" \
  --targets_path "source_target_lists/targets_seed${PARAM_SEED}.txt" \
  --sources_path "source_target_lists/sources_seed${PARAM_SEED}.txt" \
  --results_dir "${RESULTS_DIR}" \
  --condition "${PARAM_COND}" \
  --top_k ${TOPK} \
  --bottom_k ${BOTTOMK} \
  --alpha ${PARAM_ALPHA} \
  --max_samples 20000 \
  --epochs 50 \
  --batch_size 1024 \
  --lr 5e-3 \
  --seed ${PARAM_SEED}

echo "--- Done ---"

