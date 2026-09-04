#!/usr/bin/env bash
# Export the vehicle/plate YOLO detectors to INT8 TensorRT engines for
# further latency reduction beyond FP16. OPTIONAL and riskier than the FP16
# export (export_detectors_trt.sh): INT8 quantizes weights/activations to
# 8-bit using a calibration dataset, and a calibration set that isn't
# representative of real traffic can silently degrade accuracy in ways that
# won't show up until real vehicles are missed or misread.
#
# Run export_detectors_trt.sh (FP16) first and confirm you're happy with its
# accuracy -- INT8 is an incremental speed gain on top of that, not a
# replacement, and you should A/B the two engines' stored-plate accuracy on
# the same video before deploying INT8.
#
# Usage:
#   bash scripts/export_detectors_int8.sh [vehicle_pt] [plate_pt] [vehicle_imgsz] [plate_imgsz] [calib_data_yaml]
#
# calib_data_yaml: an Ultralytics-format data.yaml whose train/val images
# are used to calibrate the INT8 quantization. Defaults to dataset/data.yaml
# if present (the plate detector's own training set is the most
# representative calibration data you have); otherwise falls back to
# Ultralytics' bundled coco128.yaml (downloads automatically), which is a
# reasonable generic calibration set for the COCO-pretrained vehicle
# detector but a poor match for the custom-trained plate detector -- if you
# don't have your own dataset/data.yaml, expect the plate detector's INT8
# engine to need real validation before trusting it.
set -euo pipefail

VEHICLE_PT="${1:-models/vehicle_detector/yolov8n.pt}"
PLATE_PT="${2:-models/plate_detector/best.pt}"
VEHICLE_IMGSZ="${3:-480}"
PLATE_IMGSZ="${4:-320}"
CALIB_DATA="${5:-}"

if ! python3 -c "import ultralytics" >/dev/null 2>&1; then
  echo "ultralytics not importable in this Python environment." >&2
  exit 1
fi

if [ -z "$CALIB_DATA" ]; then
  if [ -f "dataset/data.yaml" ]; then
    CALIB_DATA="dataset/data.yaml"
    echo "Using dataset/data.yaml for calibration."
  else
    CALIB_DATA="coco128.yaml"
    echo "No dataset/data.yaml found -- falling back to bundled coco128.yaml."
    echo "WARNING: this is a poor calibration match for the plate detector;"
    echo "validate its INT8 accuracy carefully before deploying it."
  fi
fi

export_one() {
  local pt_path="$1"
  local imgsz="$2"
  echo "Exporting ${pt_path} (imgsz=${imgsz}, int8, calib=${CALIB_DATA}) to TensorRT..."
  # Pass paths via the environment rather than interpolating them into the
  # embedded Python source -- a path containing a single quote would
  # otherwise break out of the string literal below.
  MODEL_PATH="$pt_path" IMGSZ="$imgsz" CALIB_DATA_PATH="$CALIB_DATA" python3 -c "
import os
from ultralytics import YOLO
model = YOLO(os.environ['MODEL_PATH'])
model.export(format='engine', imgsz=int(os.environ['IMGSZ']), int8=True, data=os.environ['CALIB_DATA_PATH'], device=0)
"
}

export_one "$VEHICLE_PT" "$VEHICLE_IMGSZ"
export_one "$PLATE_PT" "$PLATE_IMGSZ"

echo "Done. INT8 engines saved next to the FP16 ones (same .engine extension --"
echo "back up the FP16 engine first if you want to keep both, since this"
echo "export overwrites it):"
echo "  ${VEHICLE_PT%.pt}.engine"
echo "  ${PLATE_PT%.pt}.engine"
echo "Compare stored-plate accuracy against the FP16 run before trusting this in production."
