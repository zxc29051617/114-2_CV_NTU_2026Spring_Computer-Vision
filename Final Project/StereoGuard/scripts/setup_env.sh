#!/usr/bin/env bash
# Create the conda environment, install the packages and download the model weights.
set -e
cd "$(dirname "$0")/.."
ENV=${1:-cvapp}
conda create -y -n "$ENV" -c conda-forge python=3.11 pip
PY="$(conda run -n "$ENV" which python)"
"$PY" -m pip install -r requirements.txt
# these two pin packages that would replace OpenCV / PyTorch, so they go in without dependencies
"$PY" -m pip install --no-deps uniception==0.1.7 \
  "vggt @ git+https://github.com/facebookresearch/vggt@a288dd0f14786c93483e45524328726ab7b1b4ce"
# with an NVIDIA GPU:  "$PY" -m pip uninstall -y onnxruntime && "$PY" -m pip install onnxruntime-gpu
# ultralytics / moge install opencv-python, which hides cv2.ximgproc used by the Team 18 matcher
"$PY" -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true
"$PY" -m pip install --force-reinstall --no-deps opencv-contrib-python-headless==4.13.0.92
"$PY" scripts/download_weights.py
echo "done: conda activate $ENV"
