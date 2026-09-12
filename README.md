#  ALPR University Gate System

A production-quality **Automatic License Plate Recognition (ALPR)** system built for university gate access monitoring. It detects vehicles in video streams, reads Indian license plates using a custom-trained YOLOv8m model + PaddleOCR, classifies vehicle types, and logs every entry/exit event to a SQLite database with a live Streamlit dashboard and REST API.

---

##  Table of Contents

- [Features](#-features)
- [Pipeline Overview](#-pipeline-overview)
- [Project Structure](#-project-structure)
- [What Each File Does](#-what-each-file-does)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Model Setup](#-model-setup)
- [Configuration](#-configuration)
- [Usage — How to Run](#-usage--how-to-run)
- [Training Your Own Model](#-training-your-own-model)
- [API Reference](#-api-reference)
- [Streamlit Dashboard](#-streamlit-dashboard)
- [Running Tests](#-running-tests)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Troubleshooting](#-troubleshooting)

---

##  Features

- Real-time vehicle detection using YOLOv8n (COCO pretrained)
- Custom-trained YOLOv8m plate detector — **99.47% mAP50** on Indian plates
- ByteTrack vehicle tracking with consistent IDs across frames
- PaddleOCR by default, with optional TrOCR / ensemble OCR for harder cases
- Real-ESRGAN 4× super-resolution applied to every plate crop
- Multi-frame OCR fusion (majority voting over 5 reads)
- Indian plate validation — standard format + BH series
- HSV-based plate color → vehicle type classification
- Motion filter — only processes moving vehicles
- Duplicate suppression — one entry per vehicle per 30-second window
- Virtual line IN/OUT direction detection
- SQLite database + FastAPI REST API + Streamlit live dashboard
- CSV output mode for quick batch video processing

---

##  Pipeline Overview

```
Video / RTSP Input
        │
        ▼
Frame Capture          → src/capture/frame_capture.py
        │
        ▼
Vehicle Detection      → src/detection/vehicle_detector.py     (YOLOv8n COCO)
        │
        ▼
Vehicle Tracking       → src/tracking/vehicle_tracker.py       (ByteTrack)
        │
        ▼
Motion Filter          → src/utils/motion_filter.py            (skip stationary vehicles)
        │
        ▼
Plate Detection        → src/detection/plate_detector.py       (Custom YOLOv8m)
        │
        ▼
Preprocessing          → src/preprocessing/plate_preprocessor.py  (CLAHE + denoise + sharpen)
        │
        ▼
Super Resolution       → src/enhancement/super_resolution.py   (Real-ESRGAN ×4 — always applied)
        │
        ▼
OCR                    → src/ocr/paddle_ocr_engine.py          (PaddleOCR)
        │
        ▼
Post-processing        → src/ocr/ocr_postprocessor.py          (strip IND/INDIA, fix O↔0)
        │
        ▼
Validation             → src/validation/plate_validator.py     (Indian plate regex)
        │
        ▼
OCR Fusion             → src/validation/ocr_fusion.py          (majority voting, 5-frame window)
        │
        ▼
Classification         → src/classification/                    (plate color → vehicle type)
        │
        ▼
Duplicate Filter       → src/database/duplicate_filter.py      (30-second window)
        │
        ▼
Direction Detection    → src/utils/direction_detector.py       (virtual line crossing IN/OUT)
        │
        ▼
Database Storage       → src/database/db.py                    (SQLite via SQLAlchemy)
        │
        ▼
API / Dashboard        → src/api/server.py + src/dashboard/app.py
```

---

##  Project Structure

```
alpr-university-gate/
│
├── main.py                          # Entry point — runs the full pipeline
├── requirements.txt                 # All Python dependencies (pinned versions)
├── pytest.ini                       # Pytest configuration
├── conftest.py                      # Root-level test fixtures
├── .gitignore
│
├── config/
│   └── config.yaml                  # All configuration parameters (edit this)
│
├── src/                             # All source modules
│   ├── capture/
│   │   └── frame_capture.py         # RTSP/video reader with auto-reconnect
│   ├── detection/
│   │   ├── vehicle_detector.py      # YOLOv8n — detects cars, trucks, buses, motorcycles
│   │   └── plate_detector.py        # Custom YOLOv8m — detects license plate regions
│   ├── tracking/
│   │   └── vehicle_tracker.py       # ByteTrack — assigns consistent IDs to vehicles
│   ├── preprocessing/
│   │   └── plate_preprocessor.py    # Grayscale → CLAHE → Denoise → Sharpen
│   ├── enhancement/
│   │   └── super_resolution.py      # Real-ESRGAN ×4 upscaling (applied to all crops)
│   ├── ocr/
│   │   ├── base.py                  # Abstract OCREngine interface
│   │   ├── paddle_ocr_engine.py     # PaddleOCR backend (default)
│   │   ├── trocr_engine.py          # TrOCR backend (optional alternative)
│   │   ├── ocr_postprocessor.py     # Strip IND/INDIA, fix O↔0, remove noise chars
│   │   └── __init__.py              # create_ocr_engine() factory function
│   ├── validation/
│   │   ├── plate_validator.py       # Indian plate regex + OCR correction (O↔0, I↔1)
│   │   └── ocr_fusion.py            # Multi-frame majority voting for final plate number
│   ├── classification/
│   │   ├── color_classifier.py      # HSV-based plate color detection
│   │   └── vehicle_classifier.py    # Maps plate color → vehicle type (Private/Commercial/etc.)
│   ├── database/
│   │   ├── models.py                # SQLAlchemy ORM — VehicleEvent table schema
│   │   ├── db.py                    # Session management, CRUD, image saving
│   │   └── duplicate_filter.py      # Prevents duplicate entries within 30-second window
│   ├── utils/
│   │   ├── config.py                # YAML config loader with validation
│   │   ├── logger.py                # Rotating file + console logger
│   │   ├── direction_detector.py    # Virtual line crossing → IN or OUT
│   │   └── motion_filter.py         # Filters out stationary vehicles
│   ├── api/
│   │   ├── server.py                # FastAPI app — GET /logs, GET /search, POST /entry
│   │   └── schemas.py               # Pydantic request/response models
│   └── dashboard/
│       └── app.py                   # Streamlit live monitor (auto-refreshes every 3s)
│
├── scripts/
│   ├── run_pipeline.py              # Full pipeline with DB storage + tracking + SR
│   ├── run_alpr.py                  # Lightweight CSV output (no tracking, no DB)
│   ├── video_output.py              # Annotated video with bounding boxes
│   └── test_model.py                # Batch test on image folder → JSON + annotated images
│
├── training/
│   ├── dataset_converter.py         # Converts JSON labels → YOLO format + train/test split
│   ├── augmentation.py              # Brightness, blur, noise, fog, rotation augmentations
│   ├── trainer.py                   # YOLOv8m training script (100 epochs, RTX 3050)
│   └── evaluator.py                 # Computes mAP, precision, recall, FLOPs
│
├── tests/
│   ├── conftest.py                  # Shared fixtures (Hypothesis, DB session, image generators)
│   ├── unit/                        # Unit tests for each src/ module
│   │   ├── test_plate_validator.py
│   │   ├── test_ocr_fusion.py
│   │   ├── test_plate_preprocessor.py
│   │   ├── test_color_classifier.py
│   │   ├── test_vehicle_classifier.py
│   │   ├── test_duplicate_filter.py
│   │   ├── test_direction_detector.py
│   │   ├── test_super_resolution.py
│   │   ├── test_config.py
│   │   └── test_logger.py
│   └── integration/                 # Integration tests
│       ├── test_api.py
│       ├── test_database.py
│       └── test_dataset_converter.py
│
├── docs/
│   └── pipeline.md                  # Detailed pipeline documentation
│
└── models/                          # Model weights — NOT in git, download separately
    ├── plate_detector/
    │   ├── best.pt                  # ← Custom trained plate detector (REQUIRED)
    │   └── train/                   # Training graphs, confusion matrix, results.csv
    ├── vehicle_detector/
    │   └── yolov8n.pt               # ← Auto-downloaded by ultralytics on first run
    └── realesrgan/
        └── RealESRGAN_x4plus.pth    # ← SR weights (REQUIRED, download below)
```

---

##  What Each File Does

### Entry Points

| File | What it does |
|---|---|
| `main.py` | Thin wrapper — calls `scripts/run_pipeline.py`. Run this for the full system. |
| `scripts/run_pipeline.py` | Full pipeline: vehicle detection → tracking → plate detection → OCR → DB storage |
| `scripts/run_alpr.py` | Lightweight mode: detects plates frame-by-frame, saves unique plates to CSV. No tracking, no DB. |
| `scripts/video_output.py` | Reads a video, draws green boxes (vehicles) + orange boxes (plates) + plate text, saves annotated MP4. |
| `scripts/test_model.py` | Batch tests the plate detector on a folder of images. Saves a JSON + annotated image per input. |

### Source Modules (`src/`)

| Module | What it does |
|---|---|
| `capture/frame_capture.py` | Opens video files or RTSP streams. Handles reconnection on failure. Respects `frame_skip` config. |
| `detection/vehicle_detector.py` | Runs YOLOv8n on each frame. Returns bounding boxes for cars, trucks, buses, motorcycles. |
| `detection/plate_detector.py` | Runs custom YOLOv8m on a vehicle crop. Returns the plate region bounding box. |
| `tracking/vehicle_tracker.py` | ByteTrack wrapper. Assigns a stable integer ID to each vehicle across frames. |
| `preprocessing/plate_preprocessor.py` | Converts plate crop to grayscale, applies CLAHE contrast enhancement, denoises, sharpens. |
| `enhancement/super_resolution.py` | Runs Real-ESRGAN ×4 on every plate crop before OCR. Improves readability of small/blurry plates. |
| `ocr/paddle_ocr_engine.py` | Sends plate image to PaddleOCR. Returns (text, confidence). |
| `ocr/trocr_engine.py` | Alternative OCR using Microsoft TrOCR (transformer-based). Swap in `config.yaml`. |
| `ocr/ocr_postprocessor.py` | Strips `IND`/`INDIA` text (Ashoka Chakra area), removes noise characters, fixes common OCR errors. |
| `validation/plate_validator.py` | Validates OCR output against Indian plate regex. Corrects O↔0, I↔1 at known positions. Supports standard + BH series. |
| `validation/ocr_fusion.py` | Collects OCR reads for each tracked vehicle over 5 frames. Returns the majority-voted plate number. |
| `classification/color_classifier.py` | Detects plate background color (White/Yellow/Green/Blue/Black) using HSV ranges. |
| `classification/vehicle_classifier.py` | Maps plate color to vehicle type: White→Private, Yellow→Commercial, Green→EV, etc. |
| `database/models.py` | SQLAlchemy ORM model for the `vehicle_events` table. |
| `database/db.py` | Creates DB, manages sessions, inserts events, saves plate crop images to disk. |
| `database/duplicate_filter.py` | Tracks recently seen (plate, track_id) pairs. Blocks re-entry within 30 seconds. |
| `utils/config.py` | Loads and validates `config/config.yaml`. |
| `utils/logger.py` | Sets up rotating file logger + colored console output. |
| `utils/direction_detector.py` | Tracks vehicle centroid movement relative to a virtual line. Returns `IN` or `OUT`. |
| `utils/motion_filter.py` | Compares centroid positions across frames. Skips vehicles that haven't moved enough. |
| `api/server.py` | FastAPI app. Exposes `GET /logs`, `GET /search?plate=XX`, `POST /entry`. |
| `api/schemas.py` | Pydantic models for API request/response validation. |
| `dashboard/app.py` | Streamlit app. Shows live event table, plate crop images, stats. Auto-refreshes every 3 seconds. |

### Training (`training/`)

| File | What it does |
|---|---|
| `dataset_converter.py` | Reads JSON label files from `dataset/train/labels/`, converts to YOLO `.txt` format, splits into train/test. |
| `augmentation.py` | Applies random brightness, blur, noise, fog, and rotation to training images. |
| `trainer.py` | Trains YOLOv8m for 100 epochs on the converted dataset. Saves `best.pt` to `models/plate_detector/`. |
| `evaluator.py` | Runs the trained model on the test set. Computes mAP50, mAP50-95, precision, recall, FLOPs. |

---

##  Prerequisites

- Python **3.10** (required — PaddlePaddle 3.0.0 is not compatible with 3.11+)
- NVIDIA GPU with CUDA 11.8 (recommended for Real-ESRGAN + training; CPU works but is slow)
- ~8 GB disk space for models and dependencies

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/alpr-university-gate.git
cd alpr-university-gate
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install PyTorch (GPU)

```bash
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
```

> For CPU-only: `pip install torch==2.2.2 torchvision==0.17.2`

### 4. Install PaddlePaddle

```bash
# GPU (CUDA 11.8)
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# CPU only
pip install paddlepaddle==3.0.0
```

### 5. Install all other dependencies

```bash
pip install -r requirements.txt
```

---

##  Model Setup

Three model files are needed. None are included in the repository (they are in `.gitignore`).

### 1. Plate Detector — `models/plate_detector/best.pt` *(REQUIRED)*

This is the custom-trained YOLOv8m model. Download it from the project's release page or Google Drive link (provided separately) and place it at:

```
models/plate_detector/best.pt
```

### 2. Vehicle Detector — `models/vehicle_detector/yolov8n.pt` *(auto-downloaded)*

This is the standard YOLOv8n COCO model. It is **automatically downloaded** by the `ultralytics` library on first run. No manual action needed.

If you want to pre-download it:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# Then move it:
# Windows: move yolov8n.pt models\vehicle_detector\
# Linux:   mv yolov8n.pt models/vehicle_detector/
```

### 3. Real-ESRGAN Weights — `models/realesrgan/RealESRGAN_x4plus.pth` *(REQUIRED)*

```bash
python -c "
import urllib.request, os
os.makedirs('models/realesrgan', exist_ok=True)
urllib.request.urlretrieve(
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'models/realesrgan/RealESRGAN_x4plus.pth'
)
print('Downloaded RealESRGAN_x4plus.pth')
"
```

---

##  Configuration

All settings live in `config/config.yaml`. You should not need to change anything except the video source.

### Key settings to know

```yaml
video:
  source: "ALPR.mp4"        # ← Change this to your video file or RTSP URL
  frame_skip: 2             # Process every 2nd frame (set to 1 for every frame)

detection:
  vehicle_model_path: "models/vehicle_detector/yolov8n.pt"
  vehicle_confidence: 0.5
  plate_model_path: "models/plate_detector/best.pt"
  plate_confidence: 0.4

enhancement:
        sr_threshold_px: 180      # Real-ESRGAN only for small plates in live mode
  realesrgan_model_path: "models/realesrgan/RealESRGAN_x4plus.pth"

ocr:
        backend: "paddleocr"      # Fast default for live processing
        ensemble:
                backends: ["paddleocr", "trocr"]
                min_vote_count: 1
                use_variants: true

fusion:
  window_size: 5            # Majority vote over 5 OCR reads per vehicle
  min_confidence: 0.55

deduplication:
  window_seconds: 30        # Ignore same plate within 30 seconds

database:
  path: "data/alpr.db"
  image_save_path: "data/plate_crops/"
```

### Full config reference

| Section | Key | Description |
|---|---|---|
| `video` | `source` | Video file path or RTSP URL |
| `video` | `frame_skip` | Process every Nth frame |
| `detection` | `vehicle_confidence` | Min confidence for vehicle detection (0–1) |
| `detection` | `plate_confidence` | Min confidence for plate detection (0–1) |
| `tracking` | `lost_track_timeout` | Frames before a lost track is retired |
| `preprocessing` | `clahe_clip_limit` | CLAHE contrast limit (higher = more contrast) |
| `enhancement` | `sr_threshold_px` | Apply SR if plate width < this value |
| `ocr` | `backend` | `paddleocr`, `trocr`, or `ensemble` |
| `ocr` | `ensemble` | Ensemble OCR voting settings |
| `fusion` | `window_size` | Number of OCR reads to majority-vote over |
| `fusion` | `min_confidence` | Minimum OCR confidence to accept a read |
| `deduplication` | `window_seconds` | Duplicate suppression window |
| `direction` | `virtual_line` | Coordinates of the virtual IN/OUT line |
| `database` | `path` | SQLite database file path |
| `api` | `port` | FastAPI server port (default: 8000) |

---

##  Usage — How to Run

### Option 1: Full pipeline (recommended)

Runs the complete system: vehicle detection → tracking → plate OCR → database storage.

```bash
# Use the source defined in config.yaml
python main.py

# Override with a specific video file
python main.py --source ALPR.mp4

# Use an RTSP stream
python main.py --source rtsp://192.168.1.100/stream
```

Results are stored in `data/alpr.db`. Plate crop images are saved to `data/plate_crops/`.

---

### Option 2: Simple CSV output (no database, no tracking)

Faster and simpler. Detects plates frame-by-frame and writes unique plates to a CSV file.

```bash
python scripts/run_alpr.py --source ALPR.mp4

# Custom output path
python scripts/run_alpr.py --source ALPR.mp4 --output output/results.csv

# Require at least 3 consistent reads before saving a plate
python scripts/run_alpr.py --source ALPR.mp4 --min-reads 3
```

Output: `output/results.csv` with columns: `plate_number, vehicle_type, plate_color, series_type, confidence, total_reads, best_frame, timestamp`

---

### Option 3: Annotated video with bounding boxes

Draws green boxes around vehicles, orange boxes around plates, overlays plate numbers, and saves an annotated MP4.

```bash
python scripts/video_output.py --source ALPR.mp4

# Custom output path
python scripts/video_output.py --source ALPR.mp4 --output output/annotated.mp4
```

Output: `output/ALPR_annotated.mp4`

---

### Option 4: Batch test on image folder

Tests the plate detector on a folder of images. Saves a JSON file and annotated image for each input.

```bash
python scripts/test_model.py

# Custom input/output folders
python scripts/test_model.py --images path/to/images --output output/results
```

Output per image:
- `output/test_2_output/image_name.jpg` — annotated image with bounding box and plate number
- `output/test_2_output/image_name.json` — detection result:

```json
{
  "image": "1003.jpg",
  "image_width": 640,
  "image_height": 480,
  "plates_detected": 1,
  "detections": [
    {
      "bbox": {
        "xmin": 210, "ymin": 310, "xmax": 430, "ymax": 370,
        "x_center": 0.5, "y_center": 0.708,
        "width_norm": 0.344, "height_norm": 0.125
      },
      "plate_number": "KA19TR0234",
      "ocr_confidence": 0.923,
      "detection_confidence": 0.871
    }
  ]
}
```

Also saves `output/test_2_output/summary.json` with overall stats.

---

### Option 5: Start the REST API

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

---

### Option 6: Start the Streamlit dashboard

```bash
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501` in your browser. The dashboard auto-refreshes every 3 seconds and shows:
- Live event table (plate number, vehicle type, direction, timestamp)
- Plate crop images
- Detection statistics

---

##  Training Your Own Model

### Step 1 — Prepare your dataset

Place images in `dataset/train/images/` and JSON label files in `dataset/train/labels/`.

Label file format (`dataset/train/labels/1.json`):
```json
{
  "filename": "1.jpg",
  "image_width": 500,
  "image_height": 335,
  "annotations": [
    {
      "vehicle_number": "KA19TR0234",
      "class_id": 0,
      "x_center": 0.479,
      "y_center": 0.701,
      "width": 0.398,
      "height": 0.149
    }
  ]
}
```

### Step 2 — Convert to YOLO format

```bash
python -m training.dataset_converter
```

This creates `dataset/yolo/train/` and `dataset/yolo/test/` with YOLO `.txt` label files and a `dataset.yaml`.

### Step 3 — (Optional) Apply augmentation

```bash
python -m training.augmentation
```

### Step 4 — Train

```bash
python -m training.trainer
```

Trains YOLOv8m for 100 epochs. Saves `best.pt` to `models/plate_detector/best.pt`. Training graphs and logs go to `models/plate_detector/train/`.

### Step 5 — Evaluate

```bash
python -m training.evaluator
```

Saves `models/plate_detector/eval_report.json` with mAP, precision, recall, and FLOPs.

---

##  API Reference

Start the server: `uvicorn src.api.server:app --host 0.0.0.0 --port 8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/logs` | Returns all vehicle events ordered by timestamp |
| `GET` | `/search?plate=KA02MN1828` | Search events by plate number (partial match) |
| `POST` | `/entry` | Manually add a vehicle event |

Example response from `GET /logs`:
```json
[
  {
    "id": 1,
    "plate_number": "KA02MN1828",
    "vehicle_type": "Private",
    "plate_color": "White",
    "direction": "IN",
    "timestamp": "2026-05-03T05:20:51Z",
    "image_path": "data/plate_crops/KA02MN1828_20260503_052051.jpg",
    "camera_id": "GATE-01",
    "camera_name": "Main Gate",
    "latitude": 28.613939,
    "longitude": 77.209023
  }
]
```

### Camera attribution

Every event records which camera saw it and where that camera is, taken from
the `camera:` block of the capturing device's `config.yaml`:

```yaml
camera:
  camera_id: "GATE-01"        # short stable code, unique per deployment
  camera_name: "Main Gate"    # label shown in the dashboard
  latitude: 28.613939         # decimal degrees (WGS84), or null if unsurveyed
  longitude: 77.209023
```

Set these per Jetson when provisioning it; the pipeline reads them once at
startup. `POST /entry` fills them in from the server's own config when the
request omits them, so an existing client needs no changes.

All four columns are nullable: events recorded before camera attribution
existed, and gates whose coordinates haven't been surveyed, show as `—` on
the dashboard. `init_db()` adds the columns to an existing database in place
on first start, keeping the events already recorded.

Full interactive docs at `http://localhost:8000/docs`.

---

##  Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

- URL: `http://localhost:8501`
- Auto-refreshes every 3 seconds
- Shows: live event table, plate crop thumbnails, total count, IN/OUT breakdown,
  and the camera (id, name, coordinates) each event came from

---

##  Running Tests

```bash
# Run all tests
pytest tests/

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

---

##  Model Performance

### Training (100 epochs, 1398 images, NVIDIA RTX 3050)

| Metric | Value |
|---|---|
| mAP50 | **99.47%** |
| mAP50-95 | **93.0%** |
| Precision | **99.29%** |
| Recall | **99.91%** |
| Training time | 6.29 hours |
| Inference speed | 26.4 ms/image |

### Test set (300 held-out images)

| Metric | Value |
|---|---|
| mAP50 | **98.97%** |
| mAP50-95 | **88.26%** |
| Precision | **98.97%** |
| Recall | **99.00%** |
| Inference speed | 29.5 ms/image |

### Model info

| Property | Value |
|---|---|
| Architecture | YOLOv8m |
| Parameters | 25.8M |
| FLOPs | 78.7 GFLOPs |
| Input size | 640×640 |
| Training dataset | 1398 Indian plate images |

---

##  Tech Stack

| Component | Technology | Version |
|---|---|---|
| Vehicle detection | YOLOv8n (COCO pretrained) | ultralytics 8.2.0 |
| Plate detection | YOLOv8m (custom trained) | ultralytics 8.2.0 |
| Tracking | ByteTrack | supervision 0.20.0 |
| Super resolution | Real-ESRGAN ×4 | realesrgan 0.3.0 |
| OCR (default) | PaddleOCR | paddleocr 2.9.1 |
| OCR (optional) | TrOCR | transformers 4.40.0 |
| Database | SQLite + SQLAlchemy | sqlalchemy 2.0.30 |
| REST API | FastAPI | fastapi 0.111.0 |
| Dashboard | Streamlit | streamlit 1.34.0 |
| Language | Python | 3.10 |
| GPU | NVIDIA RTX 3050 | CUDA 11.8 |

---

##  Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**
Run all scripts from the project root directory, not from inside `scripts/`:
```bash
# Correct
python scripts/run_alpr.py --source ALPR.mp4

# Wrong
cd scripts && python run_alpr.py
```

**`Cannot open video: ALPR.mp4`**
Make sure the video file exists in the project root, or pass the full path:
```bash
python scripts/run_alpr.py --source "C:/Users/you/Videos/ALPR.mp4"
```

**PaddleOCR install fails**
Install PaddlePaddle separately before running `pip install -r requirements.txt`:
```bash
pip install paddlepaddle==3.0.0
```

**Real-ESRGAN is slow / out of memory**
Set `sr_threshold_px: 0` in `config.yaml` to disable super-resolution, or reduce `batch_size` in training config.

**`index 0 is out of bounds for axis 0 with size 0` warning**
This is a non-fatal ByteTrack warning that appears when no detections are found in a frame. It does not affect results.

**Only one vehicle detected in a multi-vehicle video**
Increase `lost_track_timeout` in `config.yaml` and lower `vehicle_confidence` to `0.3`.

---

##  License

This project is for educational and research purposes.

---

##  Author

Developed as part of a  project for automated gate access monitoring using computer vision and deep learning.
