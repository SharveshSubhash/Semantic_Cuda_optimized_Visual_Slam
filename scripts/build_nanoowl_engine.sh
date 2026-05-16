#!/usr/bin/env bash
# build_nanoowl_engine.sh
# =============================================================================
# One-shot helper to install NanoOWL and build its TensorRT vision-encoder
# engine on a Jetson. Re-running is safe (everything below is idempotent).
#
# What this does:
#   1. Ensure prerequisite system packages (TensorRT bindings, OpenCV).
#   2. Ensure torch2trt is installed.
#   3. Ensure the nanoowl Python package is installed in develop mode at
#      /opt/nanoowl (so the engine and the package live in a stable
#      location that doesn't depend on the user's home directory).
#   4. Build the OWL-ViT image encoder engine into
#         /opt/nanoowl/data/owl_image_encoder_patch32.engine
#      which is the default the inference node looks for.
#
# Expected runtime: ~5-15 minutes on Orin Nano (first time), ~20 s after
# that (the build step is skipped if the engine already exists).
#
# This script is meant to be run inside the Isaac ROS Docker container, but
# it also works on a bare-metal JetPack install. It does NOT need to be run
# every boot; once the engine is built it is reused.
# =============================================================================

set -euo pipefail

NANOOWL_DIR="${NANOOWL_DIR:-/opt/nanoowl}"
ENGINE_PATH="${NANOOWL_DIR}/data/owl_image_encoder_patch32.engine"
TORCH2TRT_DIR="${TORCH2TRT_DIR:-/opt/torch2trt}"
MODEL="${NANOOWL_MODEL:-google/owlvit-base-patch32}"

# Colour helpers
red()    { printf '\033[0;31m%s\033[0m\n' "$*"; }
green()  { printf '\033[0;32m%s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m%s\033[0m\n' "$*"; }
say()    { yellow "==> $*"; }

require_root_for() {
    # If we're root, run the command directly. Otherwise prepend sudo.
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

# -----------------------------------------------------------------------------
# Step 0: sanity checks
# -----------------------------------------------------------------------------
say "Checking Jetson + TensorRT environment"

if ! command -v python3 >/dev/null 2>&1; then
    red "python3 not found - run this on a JetPack-installed Jetson."
    exit 1
fi

# Check TensorRT availability via Python bindings.
if ! python3 -c "import tensorrt" >/dev/null 2>&1; then
    yellow "tensorrt Python module not importable - attempting to install bindings"
    require_root_for apt-get update
    require_root_for apt-get install -y python3-libnvinfer-dev || true
fi

if ! python3 -c "import tensorrt; print('TensorRT', tensorrt.__version__)"; then
    red "TensorRT Python bindings still missing."
    red "On JetPack: 'sudo apt install python3-libnvinfer-dev'."
    red "If the module lives in /usr/lib/python3.X/dist-packages, add it to PYTHONPATH:"
    red "  export PYTHONPATH=/usr/lib/python3.10/dist-packages:\$PYTHONPATH"
    exit 1
fi

if ! python3 -c "import torch; print('Torch', torch.__version__, 'CUDA', torch.cuda.is_available())"; then
    red "PyTorch missing or not CUDA-enabled. Install the Jetson-built wheel from"
    red "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: install python deps (transformers, pillow, etc)
# -----------------------------------------------------------------------------
say "Installing Python deps (transformers, pillow, etc)"
python3 -m pip install --user --upgrade \
    "transformers>=4.36.0,<5" \
    "Pillow>=10.0.0" \
    "matplotlib" \
    "aiohttp" \
    "onnx"

# -----------------------------------------------------------------------------
# Step 2: torch2trt
# -----------------------------------------------------------------------------
if ! python3 -c "import torch2trt" >/dev/null 2>&1; then
    say "Installing torch2trt into ${TORCH2TRT_DIR}"
    require_root_for mkdir -p "$(dirname "$TORCH2TRT_DIR")"
    if [[ ! -d "$TORCH2TRT_DIR/.git" ]]; then
        require_root_for git clone --depth 1 \
            https://github.com/NVIDIA-AI-IOT/torch2trt "$TORCH2TRT_DIR"
    fi
    (cd "$TORCH2TRT_DIR" && require_root_for python3 setup.py develop)
else
    green "torch2trt already installed"
fi

# -----------------------------------------------------------------------------
# Step 3: nanoowl python package
# -----------------------------------------------------------------------------
if ! python3 -c "import nanoowl" >/dev/null 2>&1; then
    say "Installing nanoowl into ${NANOOWL_DIR}"
    require_root_for mkdir -p "$(dirname "$NANOOWL_DIR")"
    if [[ ! -d "$NANOOWL_DIR/.git" ]]; then
        require_root_for git clone --depth 1 \
            https://github.com/NVIDIA-AI-IOT/nanoowl "$NANOOWL_DIR"
    fi
    (cd "$NANOOWL_DIR" && require_root_for python3 setup.py develop)
else
    green "nanoowl python package already importable"
fi

# Make sure the data dir exists and is writable.
require_root_for mkdir -p "${NANOOWL_DIR}/data"
require_root_for chmod a+rwx "${NANOOWL_DIR}/data" || true

# -----------------------------------------------------------------------------
# Step 4: build the TensorRT engine
# -----------------------------------------------------------------------------
if [[ -f "${ENGINE_PATH}" ]]; then
    green "Engine already exists at ${ENGINE_PATH} - skipping build."
    green "Delete it and re-run this script to rebuild."
else
    say "Building TensorRT engine: ${ENGINE_PATH}"
    say "Model: ${MODEL}"
    yellow "This will:"
    yellow "  - Download model weights from HuggingFace (~600 MB)"
    yellow "  - Export to ONNX"
    yellow "  - Build the TensorRT engine (slow: 5-15 min on Orin Nano)"
    yellow ""
    yellow "If this hangs at 'Building TRT engine' on an 8 GB Jetson, you"
    yellow "are running out of RAM. Free memory by closing other processes"
    yellow "and increase swap, e.g.:"
    yellow "  sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile"
    yellow "  sudo mkswap /swapfile && sudo swapon /swapfile"
    yellow ""

    (cd "${NANOOWL_DIR}" && \
        python3 -m nanoowl.build_image_encoder_engine \
            "${ENGINE_PATH}" \
            --model_name "${MODEL}")
fi

# -----------------------------------------------------------------------------
# Step 5: verification
# -----------------------------------------------------------------------------
say "Verifying installation by running a quick inference"
python3 - <<PYEOF
import sys
import numpy as np
from PIL import Image
from nanoowl.owl_predictor import OwlPredictor

predictor = OwlPredictor("${MODEL}",
                         image_encoder_engine="${ENGINE_PATH}")
img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
text = ["a chair", "a person"]
enc = predictor.encode_text(text)
out = predictor.predict(image=img, text=text, text_encodings=enc,
                        threshold=0.1, pad_square=False)
print("OK - predictor loaded, encoded ${MODEL} prompts, ran a dummy frame.")
print("    boxes:", getattr(out, "boxes", None))
PYEOF

green ""
green "=========================================================="
green "  NanoOWL ready."
green ""
green "  Engine path : ${ENGINE_PATH}"
green "  Python pkg  : ${NANOOWL_DIR}"
green ""
green "  Default inference-node arg:"
green "    image_encoder_engine := ${ENGINE_PATH}"
green "=========================================================="
