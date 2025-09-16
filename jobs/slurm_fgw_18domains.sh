#!/bin/bash
#SBATCH --job-name=fgw18
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

# Paths
DATA_DIR="/work/hideki-h/jcomm/syn-dataset/18domains"
OUTPUT_DIR="/work/hideki-h/jcomm/syn-dataset/outputs/18domains"
VENV_ACTIVATE="/work/hideki-h/jcomm/env/bin/activate"

mkdir -p "${OUTPUT_DIR}" "logs/fgw18"
exec > "logs/fgw18/${SLURM_JOB_ID}.out" 2> "logs/fgw18/${SLURM_JOB_ID}.err"

echo "--- FGW 18domains ---"
echo "Timestamp: $(date)"

source "${VENV_ACTIVATE}"

PYTHONPATH=. python -m src.experiments.fgw_18domains \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --alphas 0.0 0.25 0.5 0.75 1.0

echo "--- Done ---"
echo "Timestamp: $(date)"

