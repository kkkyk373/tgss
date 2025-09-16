#!/bin/bash
JOB_NAME="fgw18"
LOG_DIR="logs/${JOB_NAME}"
mkdir -p "${LOG_DIR}"

DATA_DIR="/home/hayashi/projects/tgss/syn-dataset/18domains"
OUTPUT_DIR="/home/hayashi/projects/tgss/syn-dataset/outputs/18domains"
VENV_ACTIVATE="/home/hayashi/projects/tgss/env/bin/activate"

JOB_ID=$$

echo "--- FGW 18domains ---"
echo "Timestamp: $(date)"

# 仮想環境を有効化
source "${VENV_ACTIVATE}"

# ログリダイレクトはここで
exec > "${LOG_DIR}/${JOB_ID}.out" 2> "${LOG_DIR}/${JOB_ID}.err"

which python
which pip
pip list | grep numpy

PYTHONPATH=. python -m src.experiments.fgw_18domains \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --alphas 0.0 0.25 0.5 0.75 1.0
