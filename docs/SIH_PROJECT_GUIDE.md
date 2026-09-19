# ALPR University Gate + Multi-Camera Vehicle Tracking — Complete Project Guide (for the SIH PPT)

> **Who this is for:** the teammate building the Smart India Hackathon presentation.
> Everything here is written in simple words. Each section says **what it is**, **what it does**, and **why we chose it**.
> At the end there is a **suggested slide-by-slide order**.
>
> **Fill in yourself:** SIH Problem Statement ID / title, team name, team members, college.

---

## 1. The project in one line

**An AI camera system that reads vehicle number plates automatically. It records every vehicle entering or leaving the university gate. It also connects sightings from many cameras across a city to draw each vehicle's route on a map.**

Short name: **ALPR** = *Automatic License Plate Recognition*.

---

## 2. The problem we are solving

| Problem today | Why it matters |
|---|---|
| Gate guards write vehicle numbers **by hand** in a register | Slow, error-prone, handwriting unreadable, easy to skip vehicles |
| No digital record of **who entered and when** | Cannot search later ("did car X come in yesterday?") |
| CCTV only **records video**, nobody watches it live | A stolen or suspicious vehicle passes unnoticed |
| Cameras at different places are **not connected** | Nobody can tell where a vehicle went across the city |
| Indian plates are hard: different fonts, dirt, glare, night, **BH series**, "IND" sticker | Generic foreign ALPR software does badly on Indian plates |

---

## 3. Our solution (what the system does)

1. **Watches the camera feed** (a recorded video *or* a live RTSP IP camera).
2. **Finds every vehicle** in the frame (car, truck, bus, motorcycle).
3. **Follows each vehicle** across frames, giving it one ID, so the same car is counted only once.
4. **Finds the number plate** on that vehicle.
5. **Cleans and enlarges** the plate image so it is easier to read.
6. **Reads the characters** with a deep-learning OCR model trained on Indian plates.
7. **Checks the text** against official Indian plate formats (including BH series).
8. **Combines several readings** of the same plate (from many frames) into one final answer. This removes mistakes.
9. **Detects** vehicle type, body colour, plate colour and direction (**IN / OUT**).
10. **Saves** everything to a database with photos, time, camera and GPS location.
11. **Shows** it all on a live **dashboard**: camera wall, search, statistics, map, heatmap and blacklist alerts.
12. **Multi-camera:** joins the same plate seen at different cameras into a **trajectory** (route) drawn on a map.

---

## 4. Key features (good for a "Features" slide)

- ✅ **Real-time plate reading** from video files or **live RTSP IP cameras**
- ✅ **Custom-trained plate detector**: **99.47% mAP50** on Indian plates
- ✅ **Fine-tuned PARSeq OCR** (transformer model) for Indian plate text
- ✅ **Super-resolution** to enlarge small or far-away plates before reading
- ✅ **Multi-frame OCR fusion**: votes across 5 readings, so one bad frame cannot give a wrong plate
- ✅ **Indian plate validation**: standard `DL3CBJ1384` format and **BH series** `BH01AB1234`
- ✅ **Vehicle tracking (ByteTrack)**: each vehicle counted **exactly once**
- ✅ **Duplicate suppression**: the same vehicle is not stored again within 30 seconds
- ✅ **IN / OUT direction** detection using a virtual line
- ✅ **Vehicle profile**: type (car/truck/bus/bike), body colour, plate colour → category (private / commercial / EV / govt)
- ✅ **Multi-camera trajectory**: shows the path of a vehicle across the city on a map
- ✅ **Live RTSP cameras all run together in parallel, non-stop**, and reconnect automatically if a stream drops
- ✅ **Blacklist alerts**: flags stolen or wanted vehicles the moment they are seen
- ✅ **Natural-language search**: type *"red trucks after 9am today"* or *"white cars at Main Gate"*
- ✅ **Traffic heatmap**: shows which camera location is busiest
- ✅ **REST API**: other systems (security office, police software) can pull the data
- ✅ **Runs on the edge**: designed for the **NVIDIA Jetson Orin Nano** (small, low power, at the gate). It also runs on a normal Linux PC/server.

---

## 5. System architecture (big picture)

```
 ┌────────────────────┐     ┌────────────────────┐
 │  IP Camera (RTSP)  │     │ Uploaded video file│
 │  at gate / street  │     │   (.mp4 demo)      │
 └─────────┬──────────┘     └─────────┬──────────┘
           │   network (LAN / campus) │
           ▼                          ▼
 ┌──────────────────────────────────────────────────────┐
 │             EDGE / SERVER (Jetson or PC)             │
 │                                                      │
 │   Camera Manager  ──►  AI Pipeline (per camera)      │
 │   (runs cameras)       detect → track → plate →      │
 │                        enhance → OCR → validate →    │
 │                        fuse → classify → store       │
 │                               │                      │
 │                               ▼                      │
 │                     Database (SQLite)                │
 │                               │                      │
 │           ┌───────────────────┼──────────────────┐   │
 │           ▼                   ▼                  ▼   │
 │   Streamlit Dashboard    FastAPI REST API   MJPEG live│
 │   (main UI, maps)        (for other apps)   preview   │
 └──────────────────────────────────────────────────────┘
           │
           ▼
   Security staff / admin in a web browser
```

**Scheduling rule (important):**
- **Uploaded videos** are processed **one camera at a time**, in order. The GPU is shared, so this finishes faster.
- **Live RTSP cameras** all run **at the same time, continuously**, until STOP is pressed. A live feed cannot "wait its turn", because that footage would be lost.

---

## 6. The AI pipeline, step by step (the heart of the project)

This is the best diagram for a "How it works" slide.

```
Camera frame
   │
   ▼
① Frame Capture ──────── reads video / RTSP, auto-reconnects if the stream drops
   ▼
② Vehicle Detection ──── YOLOv8 finds cars, trucks, buses, bikes
   ▼
③ Vehicle Tracking ───── ByteTrack gives each vehicle a stable ID across frames
   ▼
④ Motion Filter ──────── ignores parked / stationary vehicles (optional)
   ▼
⑤ Plate Detection ────── custom YOLOv8 finds the number plate inside the vehicle
   ▼
⑥ Preprocessing ──────── CLAHE contrast, auto-brightness (day/night/glare), denoise, sharpen
   ▼
⑦ Super-Resolution ───── enlarges small plates ×4 (Real-ESRGAN / OpenCV fallback)
   ▼
⑧ OCR ────────────────── PARSeq (fine-tuned, TensorRT) reads the characters
   ▼
⑨ Post-processing ────── removes "IND", fixes O↔0, I↔1 style confusions
   ▼
⑩ Validation ─────────── must match the Indian plate format (normal or BH)
   ▼
⑪ OCR Fusion ─────────── votes over 5 readings of the same vehicle → final plate
   ▼
⑫ Classification ─────── plate colour → vehicle category, body colour, vehicle type
   ▼
⑬ Duplicate Filter ───── same plate within 30 s of video = same visit, not stored twice
   ▼
⑭ Direction ──────────── virtual line crossing → IN or OUT
   ▼
⑮ Database + Dashboard ─ stored with photos, time, camera, GPS
```

### What each step does, in plain words

| # | Step | Simple explanation | Why it matters |
|---|---|---|---|
| 1 | Frame capture | Takes pictures from the camera one after another | Works with files **and** live cameras; reconnects by itself |
| 2 | Vehicle detection | AI draws a box around each vehicle | We only search for plates inside vehicles, which is faster and gives fewer false plates |
| 3 | Tracking | Remembers "this box is the same car as last frame" | So one car = one record, not 50 |
| 4 | Motion filter | Skips cars that are not moving | Parked cars are not logged again and again |
| 5 | Plate detection | Second AI finds the plate on the car | Our own model, trained on Indian plates |
| 6 | Preprocessing | Makes the plate clearer (contrast, brightness, sharpness) | Helps at night, in glare and on dirty plates |
| 7 | Super-resolution | AI zooms the plate 4× without blur | Far-away plates become readable |
| 8 | OCR | AI reads the letters and numbers | Transformer model, fine-tuned on Indian plates |
| 9 | Post-processing | Fixes common mistakes | Letter O vs zero 0, removes "IND" |
| 10 | Validation | Checks the text looks like a real Indian plate | Rejects garbage reads |
| 11 | Fusion | Takes the best answer from 5 readings | One blurry frame cannot spoil the result |
| 12 | Classification | Colour of plate and body, type of vehicle | White = private, yellow = commercial, green = EV, etc. |
| 13 | Duplicate filter | Does not store the same car twice | Clean, correct counts |
| 14 | Direction | Did it enter or leave? | Gate IN/OUT log |
| 15 | Store and show | Saves to database, shows on dashboard | Searchable history |

---

## 7. AI models used (with versions and job)

| Model | Job in our system | Details | Why this model |
|---|---|---|---|
| **YOLOv8s** (COCO pretrained) — PC | Detects vehicles | Ultralytics YOLOv8, input 480 px, classes: car / truck / bus / motorcycle | Fast, accurate, state-of-the-art real-time detector |
| **YOLOv8n** → **TensorRT engine** — Jetson | Same job on the edge device | Nano version exported to TensorRT FP16 | Lightest YOLO; TensorRT makes it 3–5× faster on Jetson |
| **Custom YOLOv8m plate detector** (`best.pt` / `.onnx` / `.engine`) | Finds number plates | **Trained by us** on **1,398 Indian plate images**, 100 epochs | Generic models do not know Indian plates |
| **PARSeq-tiny** (fine-tuned) → **TensorRT** | Reads plate text (OCR) | Transformer-based scene-text recogniser, charset `0-9 A-Z`, input 32×128, **fine-tuned on our Indian plate crops** | Reads the whole sequence at once, is robust to fonts, very fast (≈2–5 ms/plate on GPU) |
| **Real-ESRGAN ×4** | Enlarges small plates | GAN-based super-resolution; falls back to OpenCV bicubic upscale if not available | Makes far or low-resolution plates readable |
| **ByteTrack** (via `supervision`) | Tracks vehicles | Kalman filter + IOU matching | Simple, fast and accurate multi-object tracker |
| *Optional OCR backends* | Backup / comparison | **RapidOCR** (PP-OCR ONNX), **PaddleOCR**, **EasyOCR**, **TrOCR** (Microsoft), and an **ensemble** mode | Lets us benchmark and switch backends from config |

### Plate detector performance (our trained model)

**Training** (100 epochs, 1,398 images, NVIDIA RTX 3050, 6.3 hours)

| Metric | Value |
|---|---|
| mAP50 | **99.47 %** |
| mAP50-95 | **93.0 %** |
| Precision | **99.29 %** |
| Recall | **99.91 %** |
| Inference speed | 26.4 ms / image |

**Test set** (300 unseen images)

| Metric | Value |
|---|---|
| mAP50 | **98.97 %** |
| mAP50-95 | **88.26 %** |
| Precision | **98.97 %** |
| Recall | **99.00 %** |

**Model size:** YOLOv8m, 25.8 M parameters, 78.7 GFLOPs, 640×640 input.

### End-to-end result on our gate test video

- The video has **4 vehicles**. The system recorded **exactly 4 plates, all correct** (`DL7CD5017`, `DL3CBJ1384`, `DL2CAT4762`, `HR26CQ6869`).
- We gave the same video to **2 cameras**: both gave **4 / 4**, with no duplicates and no wrong plates.
- On **live RTSP**, two cameras ran **in parallel** for 90+ seconds with **4 unique correct plates** each. A camera whose stream was cut **reconnected by itself** when the stream came back.

---

## 8. Complete tech stack

### 8.1 AI / Machine Learning / Computer Vision

| Technology | Version | Used for |
|---|---|---|
| **Ultralytics YOLOv8** | 8.2.0 (PC) / 8.4.x (Jetson & training) | Vehicle and plate detection, training |
| **PyTorch** + torchvision | 2.2.2 / 0.17.2 (runtime), 2.6.0 + CUDA 12.4 (training) | Deep-learning framework |
| **PARSeq** | tiny, fine-tuned | OCR model |
| **NVIDIA TensorRT** | JetPack / TRT 8–10 | Makes models run very fast on the GPU (FP16 / INT8) |
| **ONNX + ONNX Runtime** | onnxruntime 1.21.0 | Portable model format, CPU inference |
| **supervision** (Roboflow) | 0.25.0 | ByteTrack tracker |
| **OpenCV** | 4.x (FFMPEG backend) | Video / RTSP reading, image processing |
| **Real-ESRGAN** + basicsr | 0.3.0 / 1.4.2 | Super-resolution |
| **RapidOCR / PaddleOCR / EasyOCR / TrOCR** | 1.4.0 / 2.9.1 / 1.7.2 / transformers 4.40.0 | Alternative OCR engines |
| **Albumentations** | 1.4.10 | Data augmentation for training (blur, fog, brightness, rotation, noise) |
| **NumPy** | 1.26.4 | Array maths |
| **PyCUDA** | — | GPU memory / context handling for TensorRT |

### 8.2 Backend

| Technology | Version | Used for |
|---|---|---|
| **Python** | 3.10 – 3.12 | Main language |
| **FastAPI** | 0.111.0 | REST API server |
| **Uvicorn** | 0.29.0 | ASGI web server that runs FastAPI |
| **Pydantic** | 2.7.1 | Data validation for API requests and responses |
| **SQLAlchemy** (ORM) | 2.0.30 | Talks to the database with Python classes |
| **SQLite** | built-in | Default database (single file `data/alpr.db`) |
| **PostgreSQL** schema | ready | Production upgrade path (more users, indexing, full-text plate search) |
| **PyYAML** | 6.0.1 | Config files (`config.yaml`, `camera_config.yaml`, `blacklist.yaml`) |
| **Python threading** | — | Runs all live cameras in parallel |
| **MJPEG preview server** (custom, built-in HTTP) | — | Smooth live video of every camera in the browser |

**API endpoints** (FastAPI):

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/entry` | Record a vehicle event |
| GET | `/logs` | All events, newest first |
| GET | `/search?plate=` | Search by plate number |
| GET | `/stream` | Live annotated camera feed (MJPEG) |
| GET | `/snapshot` | Current frame as a JPEG |
| POST | `/clear` | Delete stored data |
| + | trajectory routes | Trajectory, sessions, GeoJSON for maps |

### 8.3 Frontend

| Technology | Version | Used for |
|---|---|---|
| **Streamlit** | 1.34.0 (PC) / 1.39.0 (Jetson) | Main operator dashboard (Python-based web UI) |
| **Leaflet.js** | 1.9.4 | Interactive maps (trajectory route, markers, arrows) |
| **Leaflet.heat** | 0.2.0 | Traffic heatmap |
| **OpenStreetMap** + **Esri satellite** tiles | — | Street and satellite map layers |
| **GeoJSON** | — | Standard map data format for routes |
| **Plotly** + **Pandas** | 5.24.1 / 2.2.3 | Charts and tables (Jetson build) |
| **HTML + CSS + JavaScript** (`src/dashboard_web/`) | — | Lightweight single-gate web dashboard: live feed, event table, search, detail view |

**Dashboard pages** (Streamlit, `trajectory_app.py`):
- **Home**: camera feeds (live camera wall), live processing status, statistics, traffic heatmap
- **Operations**: camera setup (upload video or set RTSP URL, add / remove / rename cameras), blacklist, search by plate, search by description (natural language)
- **Trajectory**: pick a plate and see its route on the map, timeline and every sighting
- **Sidebar**: controls, session scope and camera network

### 8.4 Hardware and deployment

| Item | Details |
|---|---|
| **Edge device (target)** | **NVIDIA Jetson Orin Nano**, JetPack 7.x, CUDA, TensorRT. Small, low-power, sits at the gate |
| **Server / dev PC** | Linux PC (currently NVIDIA RTX 5000 Ada GPU, 20 CPU threads) |
| **Training GPU** | NVIDIA RTX 3050 (4 GB), CUDA 12.4, mixed-precision (AMP) |
| **Cameras** | Any **IP camera with RTSP** (Hikvision / Dahua / CP Plus style), connected over campus LAN (Ethernet / PoE preferred) |
| **Install** | `install_jetson.sh` (one-shot Jetson setup), `requirements.txt` (PC), `requirements-jetson.txt` |
| **Model export** | `export_detectors_trt.sh`, `export_detectors_int8.sh`, `build_parseq_engine.sh` → TensorRT engines |

### 8.5 Testing and quality

| Tool | Version | Used for |
|---|---|---|
| **pytest** | 8.2.0 | Unit and integration tests (**715+ tests passing**) |
| **Hypothesis** | 6.100.0 | Property-based testing (auto-generates thousands of random test cases) |
| **pytest-cov** | 5.0.0 | Code coverage |
| Benchmark scripts | — | `benchmark_ocr_backends.py`, `--benchmark` flag for per-stage timings |

---

## 9. Database design (simple view)

**Table `vehicle_events`** has one row per sighting:
plate number, time, camera id / name, **latitude / longitude**, direction (IN/OUT), confidence, vehicle class (car/truck/…), body colour, plate colour, category, BH or normal, plate photo, vehicle photo, thumbnails, processing session, trajectory order.

**Table `vehicle_profiles`** has one row per unique vehicle:
first seen, last seen, total detections, number of cameras visited, best confidence, trajectory history, best photos.

→ Profiles make "tell me everything about vehicle X" instant.

---

## 10. Multi-camera trajectory (big "wow" feature)

```
CAM001 Main Gate  ──┐
CAM002 Connaught ───┤   each camera reads plates    ┌─▶ Trajectory engine
CAM003 Karol Bagh ──┤   independently, into ONE ────┤   (order · legs · GeoJSON)
CAM004 Kashmere ────┘   shared database              └─▶ Map + timeline
```

- Every camera has a **name and GPS location**.
- When plate `HR26CQ6869` is seen at camera 1, then camera 3, then camera 4, the **trajectory engine** puts the sightings in time order and draws the **route on a map** with arrows and numbered stops.
- Repeated sightings at the same camera close together in time are merged into one visit.
- **Use cases:** stolen-vehicle tracing, police investigation, traffic flow study, smart-city planning.

---

## 11. Live camera (RTSP) — how it is connected

```
IP Camera ──(Ethernet/PoE or Wi-Fi)──► Campus network ──► Server PC / Jetson
   rtsp://user:pass@<camera-ip>:554/stream1
```

- The IP camera runs its own small video server (RTSP). Our system **pulls** the stream using its URL.
- Camera and server must be on the **same network** (the campus LAN). For a remote site we would use a VPN.
- Enter the URL in the dashboard → **START**. All live cameras run **together** and **keep running**.
- If the network drops, the system **reconnects automatically** once the stream is back, and the other cameras keep working meanwhile.

---

## 12. What makes our project different (USP / innovation slide)

1. **Made for Indian plates**: our own trained detector (99.47% mAP) plus fine-tuned OCR plus Indian format and **BH-series** validation.
2. **Multi-frame fusion**: we do not trust one frame; we vote across several, which gives far fewer wrong reads.
3. **Edge AI**: runs on a **Jetson Orin Nano** at the gate with TensorRT, with no cloud needed, so video stays **private** and works **offline**.
4. **One vehicle = one record**: tracking + dedup gives correct counts (we fixed and tested this carefully).
5. **Multi-camera trajectory on a map**, not just a gate log: a city-scale idea.
6. **Natural-language search** that works offline (no API key, instant).
7. **Blacklist alerts** in real time.
8. **Live cameras in parallel with auto-reconnect**, built for 24×7 use.
9. **Pluggable OCR**: switch between 6 OCR engines from one config line.
10. **Well tested**: 715+ automated tests, including property-based tests.

---

## 13. Challenges we faced and how we solved them (judges like this)

| Challenge | Our solution |
|---|---|
| Small / far plates unreadable | Super-resolution ×4 plus adaptive preprocessing |
| Night, glare, dirty plates | CLAHE + auto gamma (adaptive lighting) + denoise + sharpen |
| OCR confuses O/0, I/1, B/8 | Post-processing rules + format validation + multi-frame voting |
| Same car counted many times | ByteTrack tracking + tuned matching threshold + duplicate filter |
| Counts changed with computer speed | Duplicate window measured in **video time**, not clock time |
| Jetson is slow for big models | TensorRT FP16 / INT8 engines, lighter YOLOv8n, shared model loading |
| Several live cameras at once | One thread per camera, shared GPU models behind a lock |
| Camera network drops | Automatic reconnect, never gives up until STOP |

---

## 14. Use cases / impact

- 🏫 **University / campus gate**: automatic entry–exit register
- 🚓 **Police**: stolen or wanted vehicle alerts and route tracing
- 🅿️ **Parking**: automatic entry, exit and billing
- 🛣️ **Toll / smart city**: traffic analytics and heatmaps
- 🏢 **Offices, societies, hospitals**: visitor vehicle log
- 🚨 **Emergencies**: find where a vehicle went across the city

---

## 15. Future scope

- Faster multi-camera search with **PostgreSQL** (schema is already written)
- **SMS / WhatsApp / email alerts** for blacklisted vehicles
- **Vehicle re-identification** (match cars by look when the plate is hidden)
- **Speed estimation** between two cameras
- Integration with **VAHAN / government RC database** (owner, insurance and PUC status)
- **Mobile app** for guards
- **Helmet / seat-belt detection** add-ons
- Scaling to **hundreds of cameras** with a central server and edge devices at each site

---

## 16. Suggested PPT slide order (SIH style)

1. **Title**: project name, team name, problem statement ID, college
2. **Problem statement**: section 2 (use the table)
3. **Proposed solution**: section 3 (one diagram + bullets)
4. **Key features**: section 4
5. **System architecture**: section 5 diagram
6. **AI pipeline / workflow**: section 6 diagram (most important slide)
7. **Models used + accuracy**: section 7 (tables + our results)
8. **Tech stack**: section 8 (one slide with logos: Python, PyTorch, YOLOv8, TensorRT, OpenCV, FastAPI, Streamlit, SQLite, Leaflet, Jetson)
9. **Multi-camera trajectory + map**: section 10 (screenshot of the map)
10. **Dashboard screenshots**: camera wall, search, heatmap, blacklist
11. **Innovation / USP**: section 12
12. **Challenges and solutions**: section 13
13. **Impact / use cases**: section 14
14. **Future scope**: section 15
15. **Thank you / Q&A**

### Tips for the PPT
- Put **screenshots** of the dashboard, map, heatmap and a detected plate with its box. Judges love visuals.
- Show the **99.47% mAP** and **"4 / 4 vehicles correct on both cameras"** in big bold numbers.
- Keep text short on slides; use this document for **speaking notes**.
- One-line summary for the judges: *"Indian-plate-trained, edge-AI, multi-camera ALPR that turns CCTV into a searchable, mapped vehicle record in real time."*
