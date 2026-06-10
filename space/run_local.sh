#!/usr/bin/env bash
# Run the WaveDiT Space LOCALLY for testing (no ZeroGPU, no Hub round-trip).
#
# Prereqs: an env with the wavedit runtime deps + gradio (see below). The `spaces`
# package is NOT needed locally — app.py falls back to a no-op @spaces.GPU shim.
#
# It loads the local pre-release checkpoints via WAVEDIT_LOCAL_CKPT_DIR and serves
# Gradio on http://localhost:7860 — open that in a browser to test the Niivue 3D viewer
# (the one thing that can only be verified in a real browser).
set -euo pipefail

SPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${WAVEDIT_PY:-python}"                       # override with WAVEDIT_PY=/path/to/python
CKPT_DIR="${WAVEDIT_LOCAL_CKPT_DIR:-/lv_all/home/danilo/projects/HUGGINGFACE_SUBMISSION_2/WaveDiT/checkpoints/WaveDiT}"
GPU="${CUDA_VISIBLE_DEVICES:-1}"                  # use a free GPU (0/2 are training)

export WAVEDIT_NA_BACKEND=torch                  # no NATTEN locally
export WAVEDIT_LOCAL_CKPT_DIR="$CKPT_DIR"        # load checkpoints from disk
export CUDA_VISIBLE_DEVICES="$GPU"
export GRADIO_SERVER_NAME="0.0.0.0"              # reachable from the host browser
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"

echo "Space dir : $SPACE_DIR"
echo "Python    : $($PY -c 'import sys;print(sys.executable)')"
echo "Checkpoints: $CKPT_DIR"
echo "GPU        : $CUDA_VISIBLE_DEVICES"
echo "URL        : http://localhost:${GRADIO_SERVER_PORT}"
echo

cd "$SPACE_DIR"
exec "$PY" app.py
