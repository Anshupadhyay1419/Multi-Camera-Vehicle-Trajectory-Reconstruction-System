#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-models/ocr/parseq.onnx}"
ENGINE_PATH="${2:-models/ocr/parseq.engine}"
BATCH_MIN="${3:-1}"
BATCH_MAX="${4:-8}"
BATCH_OPT="${5:-1}"
HEIGHT="${6:-32}"
WIDTH="${7:-128}"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Install TensorRT and ensure trtexec is on PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$ENGINE_PATH")"

trtexec \
  --onnx="$ONNX_PATH" \
  --saveEngine="$ENGINE_PATH" \
  --fp16 \
  --minShapes=input:${BATCH_MIN}x3x${HEIGHT}x${WIDTH} \
  --optShapes=input:${BATCH_OPT}x3x${HEIGHT}x${WIDTH} \
  --maxShapes=input:${BATCH_MAX}x3x${HEIGHT}x${WIDTH}

echo "TensorRT engine written to $ENGINE_PATH"
