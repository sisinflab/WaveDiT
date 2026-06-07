#!/usr/bin/env bash
# Launch a WaveDiT training run from a YAML config.
#
#   bash train.sh                     # uses configs/cfm.yaml
#   bash train.sh configs/ot_fm.yaml  # any experiment config
#
# Edit the chosen YAML to change data paths, architecture or hyper-parameters.
set -euo pipefail

CONFIG="${1:-configs/cfm.yaml}"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

run_name="$(basename "${CONFIG%.*}")"
log_file="${LOG_DIR}/train_${run_name}.log"

echo "Launching WaveDiT training | config=${CONFIG} | log=${log_file}"
PYTHONPATH=. nohup python3 -u scripts/train.py "${CONFIG}" > "${log_file}" 2>&1 &
echo "Started in background (PID $!). Follow with: tail -f ${log_file}"
