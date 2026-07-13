#!/usr/bin/env bash
# =============================================================================
# ALPR University Gate — Jetson Orin Nano install script
# Target: JetPack 7.2 | L4T 39.2 | Ubuntu 24.04 | CUDA 13.2 | Python 3.12
#
# Usage:
#   chmod +x install_jetson.sh
#   ./install_jetson.sh
#
# What this script does (in order):
#   1. Verifies JetPack version and Python 3.12
#   2. Installs system packages needed by OpenCV and SQLite
#   3. Installs TensorRT Python bindings (pre-built with JetPack)
#   4. Installs pycuda (needed by tensorrt_engine.py for Phase 2)
#   5. Installs all Python dependencies from requirements-jetson.txt
#   6. Creates required runtime directories
#   7. Runs a quick smoke-test to confirm key imports work
#
# What this script does NOT do:
#   - Install PyTorch (no CUDA 13.2 wheel for aarch64 yet)
#   - Install Real-ESRGAN / basicsr (depend on torch)
#   - Install PaddleOCR / EasyOCR (not used on Jetson)
#   These will be added in a future update once NVIDIA publishes JP7.2 wheels.
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Step 0: locate project root ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "Project root: $SCRIPT_DIR"

# ── Step 1: verify environment ────────────────────────────────────────────────
info "Step 1/7 — Verifying environment"

# Python 3.12
PY=$(python3 --version 2>&1 | awk '{print $2}')
PYMAJ=$(echo "$PY" | cut -d. -f1)
PYMIN=$(echo "$PY" | cut -d. -f2)
if [[ "$PYMAJ" -ne 3 || "$PYMIN" -lt 12 ]]; then
    error "Python 3.12+ required (found $PY). JetPack 7.2 ships Python 3.12."
fi
success "Python $PY"

# CUDA present
if ! command -v nvcc &>/dev/null; then
    warn "nvcc not found on PATH. Adding /usr/local/cuda/bin …"
    export PATH="/usr/local/cuda/bin:$PATH"
fi
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ,) || true
if [[ -z "$CUDA_VER" ]]; then
    warn "Could not detect CUDA version via nvcc. Continuing anyway."
else
    success "CUDA $CUDA_VER"
fi

# TensorRT — confirm it shipped with JetPack
if dpkg -l libnvinfer-dev &>/dev/null 2>&1; then
    TRT_VER=$(dpkg -l libnvinfer-dev 2>/dev/null | awk '/libnvinfer-dev/{print $3}' | head -1)
    success "TensorRT (libnvinfer-dev) $TRT_VER"
else
    warn "libnvinfer-dev not found via dpkg. TensorRT may still be present."
fi

# ── Step 2: system packages ───────────────────────────────────────────────────
info "Step 2/7 — Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    libsqlite3-dev \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    curl \
    git
success "System packages installed"

# ── Step 3: TensorRT Python bindings ─────────────────────────────────────────
info "Step 3/7 — Installing TensorRT Python bindings"
# These are shipped with JetPack but not always linked into Python path
if python3 -c "import tensorrt" &>/dev/null 2>&1; then
    TRT_PY=$(python3 -c "import tensorrt; print(tensorrt.__version__)")
    success "TensorRT Python bindings already available (v$TRT_PY)"
else
    info "Installing python3-libnvinfer via apt …"
    sudo apt-get install -y --no-install-recommends \
        python3-libnvinfer \
        python3-libnvinfer-dev \
        python3-libnvinfer-lean \
        uff-converter-tf 2>/dev/null || \
    sudo apt-get install -y --no-install-recommends \
        python3-libnvinfer \
        python3-libnvinfer-dev 2>/dev/null || \
    warn "Could not install TensorRT Python bindings via apt. " \
         "TensorRT OCR (Phase 2) will not work until they are installed. " \
         "Phase 1 (RapidOCR CPU) is unaffected."

    if python3 -c "import tensorrt" &>/dev/null 2>&1; then
        TRT_PY=$(python3 -c "import tensorrt; print(tensorrt.__version__)")
        success "TensorRT Python bindings installed (v$TRT_PY)"
    else
        warn "TensorRT Python bindings could not be verified. " \
             "Phase 1 (RapidOCR) will still work. " \
             "For Phase 2 (TensorRT PARSeq), manually run: " \
             "  sudo apt-get install python3-libnvinfer python3-libnvinfer-dev"
    fi
fi

# ── Step 4: pycuda ────────────────────────────────────────────────────────────
info "Step 4/7 — Installing pycuda (needed for TensorRT OCR Phase 2)"
if python3 -c "import pycuda" &>/dev/null 2>&1; then
    success "pycuda already installed"
else
    # pycuda needs CUDA headers — ensure path is set
    export CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"
    export PATH="$CUDA_ROOT/bin:$PATH"
    pip3 install --no-cache-dir pycuda 2>/dev/null && \
        success "pycuda installed" || \
        warn "pycuda install failed. Phase 1 (RapidOCR) is unaffected. " \
             "Fix for Phase 2: ensure CUDA headers are present then rerun."
fi

# ── Step 5: Python dependencies ───────────────────────────────────────────────
info "Step 5/7 — Installing Python packages from requirements-jetson.txt"

# Upgrade pip first — older pip on Ubuntu 24.04 may not resolve aarch64 wheels
pip3 install --upgrade pip setuptools wheel

# Install from the Jetson-specific requirements file
pip3 install --no-cache-dir -r requirements-jetson.txt

success "Python packages installed"

# ── Step 6: runtime directories ───────────────────────────────────────────────
info "Step 6/7 — Creating runtime directories"
# NOTE: data/alpr.db is a FILE created by SQLAlchemy at runtime — do NOT mkdir it.
# Only create the parent data/ directory.
mkdir -p data
mkdir -p data/plate_crops
mkdir -p logs
mkdir -p models/vehicle_detector
mkdir -p models/plate_detector
mkdir -p models/ocr
mkdir -p output
success "Directories created"

# ── Step 7: smoke test ────────────────────────────────────────────────────────
info "Step 7/7 — Smoke test"

SMOKE_FAIL=0

check_import() {
    local pkg="$1"
    local label="${2:-$1}"
    if python3 -c "import $pkg" &>/dev/null 2>&1; then
        success "$label"
    else
        warn "$label — IMPORT FAILED (check install logs above)"
        SMOKE_FAIL=1
    fi
}

check_import cv2            "opencv-python"
check_import numpy          "numpy"
check_import ultralytics    "ultralytics (YOLOv8)"
check_import supervision    "supervision (ByteTrack)"
check_import rapidocr       "rapidocr-onnxruntime"
check_import onnxruntime    "onnxruntime"
check_import sqlalchemy     "sqlalchemy"
check_import fastapi        "fastapi"
check_import uvicorn        "uvicorn"
check_import streamlit      "streamlit"
check_import yaml           "PyYAML"

# Optional — Phase 2
if python3 -c "import tensorrt" &>/dev/null 2>&1; then
    TRT_PY=$(python3 -c "import tensorrt; print(tensorrt.__version__)")
    success "tensorrt (Phase 2 GPU OCR) — v$TRT_PY"
else
    warn "tensorrt — not available yet (Phase 2 OCR will not work until installed)"
fi

if python3 -c "import pycuda" &>/dev/null 2>&1; then
    success "pycuda (Phase 2 GPU OCR)"
else
    warn "pycuda — not available yet (Phase 2 OCR will not work)"
fi

echo ""
if [[ $SMOKE_FAIL -eq 0 ]]; then
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Installation complete. All required imports OK.  ${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
else
    echo -e "${YELLOW}══════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}  Installation finished with warnings.            ${NC}"
    echo -e "${YELLOW}  Check items marked [WARN] above.                ${NC}"
    echo -e "${YELLOW}══════════════════════════════════════════════════${NC}"
fi

echo ""
echo -e "${CYAN}Next steps:${NC}"
echo "  1. Copy your model weights to the Jetson:"
echo "     scp models/plate_detector/best.pt  jetson:$(pwd)/models/plate_detector/"
echo "     scp models/vehicle_detector/yolov8n.pt  jetson:$(pwd)/models/vehicle_detector/"
echo ""
echo "  2. Run the pipeline (Phase 1 — RapidOCR CPU):"
echo "     python3 scripts/run_pipeline.py --source ALPR.mp4"
echo ""
echo "  3. Run the API server:"
echo "     uvicorn src.api.server:app --host 0.0.0.0 --port 8000"
echo ""
echo "  4. Run the dashboard:"
echo "     streamlit run src/dashboard/app.py --server.port 8501"
echo ""
echo "  5. Phase 2 (after PARSeq fine-tuning on dev machine):"
echo "     trtexec --onnx=models/ocr/parseq_plate.onnx \\"
echo "             --saveEngine=models/ocr/parseq_plate.engine \\"
echo "             --fp16 \\"
echo "             --minShapes=input:1x3x32x128 \\"
echo "             --optShapes=input:1x3x32x128 \\"
echo "             --maxShapes=input:8x3x32x128"
echo "     # Then set ocr.backend: 'tensorrt' in config/config.yaml"
echo ""
