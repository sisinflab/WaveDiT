#!/usr/bin/env bash
# Generate samples from a trained WaveDiT checkpoint.
#
#   bash generate.sh checkpoints/WaveDiT_CFM/best.pth [output_dir]
#
# Checkpoints are self-contained, so no architecture flags are needed here.
set -euo pipefail

CHECKPOINT="${1:?usage: bash generate.sh <checkpoint.pth> [output_dir]}"
OUTPUT_DIR="${2:-generated_samples}"

# --- Edit these as needed ---
CFG_SCALE=1.0
NUM_FLOW_STEPS=10
SAMPLER=heun
SAVE_SIZE=(182 218 182)            # final crop (D H W); matches typical raw MRI size
CONDITIONS=("age=25.0" "age=45.0" "age=65.0" "age=85.0")
NUM_SAMPLES=5                       # per condition set

PYTHONPATH=. python3 scripts/generate.py "${CHECKPOINT}" "${OUTPUT_DIR}" \
    --cfg-scale "${CFG_SCALE}" --num-flow-steps "${NUM_FLOW_STEPS}" --sampler "${SAMPLER}" \
    --save-size "${SAVE_SIZE[@]}" \
    specific --num-samples "${NUM_SAMPLES}" --conditions "${CONDITIONS[@]}"

# --- Linear age interpolation alternative ---
# PYTHONPATH=. python3 scripts/generate.py "${CHECKPOINT}" "${OUTPUT_DIR}" \
#     --num-flow-steps "${NUM_FLOW_STEPS}" --sampler "${SAMPLER}" --save-size "${SAVE_SIZE[@]}" \
#     linear --condition age --min 6 --max 95 --num 100
