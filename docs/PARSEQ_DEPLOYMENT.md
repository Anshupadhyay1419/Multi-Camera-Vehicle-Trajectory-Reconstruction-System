# PARSeq TensorRT deployment for ALPR

## Overview

This project now supports a second OCR backend, `parseq_tensorrt`, while keeping `rapidocr` fully available. The rest of the ALPR pipeline remains unchanged.

## Phase 1 — Dataset preparation

1. Place cropped plate images under `OCR_data/images`.
2. Place matching labels under `OCR_data/labels` with one file per image.
3. Each label file contains the raw plate string, for example `AB1234`.

## Phase 2 — Fine-tuning

```bash
python training/train_parseq.py \
  --image-dir OCR_data/images \
  --label-dir OCR_data/labels \
  --epochs 5 \
  --batch-size 4 \
  --lr 1e-4 \
  --output-dir models/ocr
```

Artifacts:
- `models/ocr/last_parseq.pt`
- `models/ocr/best_parseq.pt`

## Phase 3 — Evaluation

```bash
python training/evaluate_parseq.py --checkpoint models/ocr/best_parseq.pt
```

## Phase 4 — ONNX export

```bash
python training/export_onnx.py \
  --checkpoint models/ocr/best_parseq.pt \
  --output models/ocr/parseq.onnx
```

## Phase 5 — TensorRT engine build

Build the engine on your Jetson Orin Nano target. TensorRT engines are device-specific.

```bash
bash scripts/build_parseq_engine.sh \
  models/ocr/parseq.onnx \
  models/ocr/parseq.engine \
  1 8 1 32 128
```

## Phase 6 — Switch backend

Update `config/config.yaml`:

```yaml
ocr:
  backend: parseq_tensorrt
  parseq:
    engine_path: models/ocr/parseq.engine
    input_size: [32, 128]
    charset: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    batch_size: 1
```

## Phase 7 — Run the Jetson pipeline

Use the existing ALPR pipeline entrypoint with the Jetson config:

```bash
python scripts/run_pipeline.py --config config/config.yaml --source /dev/video0
```

If you want CSV-style results instead of the full pipeline, use:

```bash
python scripts/run_alpr.py --config config/config.yaml --source /dev/video0
```

## Notes

- `rapidocr` remains available and is the default fallback.
- TensorRT engines are device-specific and should be built on the target Jetson.
- JetPack 7.2 users should verify `tensorrt`, `pycuda`, and the NVIDIA JetPack PyTorch stack before running inference.
