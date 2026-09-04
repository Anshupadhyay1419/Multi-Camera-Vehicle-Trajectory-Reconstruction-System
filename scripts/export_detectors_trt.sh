#!/usr/bin/env bash
# Export the vehicle and plate YOLO .pt weights to TensorRT .engine plans
# for fast inference on Jetson Orin Nano. Run this once on-device (TensorRT
# engines are hardware/version specific and are not portable between GPUs).
#
# After running, point config/config.yaml's vehicle_model_path /
# plate_model_path at the generated .engine files.
set -euo pipefail

VEHICLE_PT="${1:-models/vehicle_detector/yolov8n.pt}"
PLATE_PT="${2:-models/plate_detector/best.pt}"
VEHICLE_IMGSZ="${3:-480}"
PLATE_IMGSZ="${4:-320}"

if ! python3 -c "import ultralytics" >/dev/null 2>&1; then
  echo "ultralytics not importable in this Python environment." >&2
  exit 1
fi

export_one() {
  local pt_path="$1"
  local imgsz="$2"
  echo "Exporting ${pt_path} (imgsz=${imgsz}, fp16) to TensorRT..."
  # Pass paths via the environment rather than interpolating them into the
  # embedded Python source -- a path containing a single quote would
  # otherwise break out of the string literal below.
  MODEL_PATH="$pt_path" IMGSZ="$imgsz" python3 -c "
import os
from ultralytics import YOLO
model = YOLO(os.environ['MODEL_PATH'])
model.export(format='engine', imgsz=int(os.environ['IMGSZ']), half=True, device=0)
"
}

export_one "$VEHICLE_PT" "$VEHICLE_IMGSZ"
export_one "$PLATE_PT" "$PLATE_IMGSZ"

echo "Done. Update config/config.yaml:"
echo "  vehicle_model_path: \"${VEHICLE_PT%.pt}.engine\""
echo "  plate_model_path:   \"${PLATE_PT%.pt}.engine\""
