# Multi-Camera Vehicle Trajectory Reconstruction System

Reconstructs where a vehicle travelled across a network of ALPR cameras.
Each camera recognises plates independently; the trajectory engine joins
those sightings back together into an ordered, mapped path.

This is an **extension** of the existing ALPR University Gate system. The
detection pipeline — YOLOv8 → ByteTrack → plate detection → PARSeq TensorRT
OCR → fusion → deduplication → SQLite — is unchanged. Every module described
here sits either upstream of it (choosing which camera runs next) or
downstream of it (reading what it stored).

---

## Contents

1. [Architecture](#1-architecture)
2. [Data flow](#2-data-flow)
3. [Camera configuration](#3-camera-configuration)
4. [Database](#4-database)
5. [Camera manager: files in turn, streams together](#5-camera-manager-files-in-turn-streams-together)
6. [Trajectory engine](#6-trajectory-engine)
7. [Map engine](#7-map-engine)
8. [Dashboard](#8-dashboard)
9. [API](#9-api)
10. [How to run](#10-how-to-run)
11. [Future RTSP migration](#11-future-rtsp-migration)
12. [Design decisions and assumptions](#12-design-decisions-and-assumptions)

---

## 1. Architecture

```
                        ┌──────────────────────────────────┐
                        │      config/camera_config.yaml   │
                        │  id · name · lat · lon · source  │
                        └────────────────┬─────────────────┘
                                         │ load_camera_registry()
                                         ▼
                        ┌──────────────────────────────────┐
                        │         CameraRegistry           │
                        │   cameras in processing order    │
                        └────────────────┬─────────────────┘
                                         │
                                         ▼
   ┌────────────────┐        ┌──────────────────────────────────┐
   │ Uploaded video │───────▶│          CameraManager           │
   │   or RTSP URL  │        │  builds the queue, runs it ONE   │
   └────────────────┘        │  CAMERA AT A TIME, publishes     │
                             │  progress to the status file     │
                             └────────────────┬─────────────────┘
                                              │ per camera
                                              ▼
                             ┌──────────────────────────────────┐
                             │           VideoSource            │
                             │   UploadSource  │  RTSPSource    │
                             └────────────────┬─────────────────┘
                                              ▼
        ╔═════════════════════════════════════════════════════════════╗
        ║        EXISTING ALPR PIPELINE — UNCHANGED                   ║
        ║  Frame → YOLOv8 vehicle → ByteTrack → YOLOv8 plate →        ║
        ║  preprocess → PARSeq TensorRT OCR → validate → OCR fusion   ║
        ║  → colour/type → duplicate filter → direction               ║
        ╚═══════════════════════════════┬═════════════════════════════╝
                                        │ + camera metadata
                                        │ + session context
                                        ▼
                             ┌──────────────────────────────────┐
                             │   SQLite / PostgreSQL            │
                             │   vehicle_events                 │
                             └────────────────┬─────────────────┘
                                              │
                    ┌─────────────────────────┴───────────────────────┐
                    ▼                                                 ▼
      ┌──────────────────────────┐                    ┌──────────────────────────┐
      │    TrajectoryEngine      │                    │        FastAPI           │
      │  group → order → legs    │───────────────────▶│   /trajectory-api/*      │
      └────────────┬─────────────┘                    └──────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌────────────────┐  ┌──────────────────┐
│    GeoJSON     │  │   Leaflet map    │
│  RFC 7946      │  │  OSM + satellite │
└────────────────┘  └──────────────────┘
                   │
                   ▼
      ┌──────────────────────────┐
      │  Streamlit dashboard     │
      │  trajectory_app.py       │
      └──────────────────────────┘
```

### Module map

| Module | Responsibility |
|---|---|
| `src/cameras/models.py` | `CameraConfig`, `CameraProgress`, `SessionStatus` — dependency-free value types |
| `src/cameras/registry.py` | Parse and validate `camera_config.yaml`; cameras in processing order |
| `src/cameras/sources.py` | `VideoSource` ABC → `UploadSource` / `RTSPSource`; the RTSP migration seam |
| `src/cameras/manager.py` | Builds the queue, runs recorded cameras **in turn** and live ones **together**, reports progress |
| `src/cameras/status_store.py` | Atomic JSON status file — cross-process progress handoff |
| `src/trajectory/models.py` | `Trajectory`, `TrajectoryPoint`, `TrajectoryLeg` |
| `src/trajectory/engine.py` | Group → order → measure. The reconstruction logic |
| `src/trajectory/geojson.py` | GeoJSON, ordered coordinates, ordered locations, bbox |
| `src/mapping/leaflet.py` | Self-contained Leaflet HTML with OSM + Esri satellite |
| `src/api/trajectory_routes.py` | The `/trajectory-api/*` endpoints |
| `src/dashboard/trajectory_app.py` | The multi-camera Streamlit console |

Everything follows the same separation: the **engine layer works on plain
dicts and dataclasses** and never imports SQLAlchemy, Streamlit or OpenCV.
That is what makes the ordering rules — the part with the real subtlety —
unit-testable without a database, a GPU or a browser.

---

## 2. Data flow

### Processing (write path)

```
Operator gives each camera ONE source (a video OR an RTSP stream)
        │
        ▼
save_upload() / set_rtsp_url()   mutually exclusive: setting one clears
        │                        the other, so a camera always has exactly
        │                        one source. Uploads land as their own file
        │                        per camera: data/uploads/CAM00N.mp4
        ▼
CameraManager.start()            background thread, returns immediately
        │
        ├── CAM001 ──▶ VideoSource.validate() ──▶ run_pipeline(...) ──▶ events
        │                                                      │
        │              ◀── completes fully ───────────────────┘
        ├── CAM002 ──▶ … same, only after CAM001 has finished
        ├── CAM003 ──▶ …
        └── CAM004 ──▶ …
                │
                ▼
        every stored event carries:
          camera_id, camera_name, latitude, longitude   (which camera, where)
          processing_session                            (which run)
          trajectory_order                              (queue position)
          video_source                                  (which clip/stream)
          confidence, ocr_text                          (how good the read was)
          image_path, vehicle_image_path                (plate + vehicle crop)
```

### Reconstruction (read path)

```
Search "DL8CA1234"
        │
        ▼
db.get_plate_detections(plate, session?)     indexed on plate_number
        │  [12 rows — 3 reads at each of 4 cameras]
        ▼
TrajectoryEngine.build()
        │
        ├─ group into visits      collapse repeat reads at one camera,
        │                         but keep a genuine A→B→A revisit as 3 visits
        ├─ resolve ordering       timestamp, else camera queue position
        ├─ sort                   with the other key as tiebreaker
        └─ measure legs           haversine distance, duration, speed, bearing
        │
        ▼
Trajectory  [4 points, 3 legs, 9.76 km, 48 min]
        │
        ├──▶ trajectory_to_geojson()   → FeatureCollection (LineString + Points)
        ├──▶ render_trajectory_map()   → Leaflet HTML with inlined thumbnails
        └──▶ ordered_locations()       → the timeline view
```

---

## 3. Camera configuration

`config/camera_config.yaml`. Nothing about the camera network is hardcoded
anywhere in the code — this file is the single source of truth for where the
cameras are.

```yaml
defaults:
  source_type: "upload"      # applied to any camera that omits the key
  enabled: true
  rtsp_url: null

cameras:
  - camera_id: "CAM001"
    camera_name: "India Gate"
    latitude: 28.6129
    longitude: 77.2295
    video_path: "data/uploads/CAM001.mp4"
    order: 1
  # CAM002 Connaught Place · CAM003 Karol Bagh · CAM004 Kashmere Gate
```

| Field | Meaning |
|---|---|
| `camera_id` | Stable unique code. Stored on every event — never renumber it after events exist |
| `camera_name` | Label shown in the dashboard, map popups and timeline |
| `latitude` / `longitude` | Decimal degrees, WGS84. `null` until surveyed |
| `video_path` | Recorded file, used when `source_type: upload` |
| `rtsp_url` | Live stream, used when `source_type: rtsp` |
| `source_type` | `upload` or `rtsp` — **the only field that changes to go live**. Exactly one of `video_path` / `rtsp_url` is ever populated |
| `enabled` | `false` skips the camera entirely |
| `order` | Queue position, and the fallback trajectory ordering key |

### Validation posture

The loader splits problems in two, deliberately:

**Fatal** (`CameraConfigError`) — the file is meaningless or dangerous:
no `cameras:` list, a duplicate `camera_id`, a missing `camera_id`.
A duplicate id merges every trajectory passing either site, irrecoverably,
so it is never tolerated.

**Degraded** (logged, camera still runs) — one field is unusable: an
out-of-range coordinate (almost always lat/lon swapped) becomes `None`, an
unknown `source_type` falls back to `upload`, a non-integer `order` falls
back to file position. A camera with no coordinates still records events and
still appears on the timeline; it just cannot be plotted. Losing a site's
whole feed to a typo would be worse than losing its map pin.

---

## 4. Database

The existing `vehicle_events` table is **extended in place**. No column was
removed, renamed or retyped.

### Columns

| Column | Type | Status |
|---|---|---|
| `id`, `plate_number`, `vehicle_type`, `plate_color`, `series_type`, `timestamp`, `direction`, `image_path` | — | **unchanged** |
| `camera_id`, `camera_name`, `latitude`, `longitude` | VARCHAR / DOUBLE | pre-existing |
| `vehicle_image_path` | VARCHAR | **new** — whole-vehicle crop, distinct from `image_path`'s plate crop |
| `trajectory_order` | INTEGER | **new** — capturing camera's queue position |
| `processing_session` | VARCHAR | **new** — groups one end-to-end run of the queue |
| `video_source` | VARCHAR | **new** — the file or URL this event came from |
| `confidence` | DOUBLE | **new** — fused OCR confidence for the stored read |
| `ocr_text` | VARCHAR | **new** — raw fused OCR string before validation |

Indexes: `idx_plate_number`, `idx_timestamp`, `idx_camera_id` (existing) plus
`idx_processing_session` (new).

**Every new column is nullable.** The single-gate pipeline sets none of them
and must keep inserting successfully; rows written before they existed read
back as `NULL`. The trajectory engine reads a `NULL` `trajectory_order` as
"not part of a multi-camera run", not as zero.

### Why these are denormalised onto the event

`trajectory_order`, `camera_name` and the coordinates are copied onto each
row rather than joined from a cameras table. A stored event should keep
saying where the vehicle actually was, even after the camera is renamed,
re-sited or reordered in the config. A live join would rewrite history.

### Migration

`db._migrate_schema()` runs on every `init_db()`. `create_all()` only creates
missing *tables* — it never alters an existing one — so a Jetson already in
service needs its table altered in place. The migration:

- inspects the live table and adds only the columns actually missing, so a
  database at **any** prior version (original 8 columns, or the
  4-camera-column version) converges in one pass;
- uses one transaction per statement, so a tolerated race does not roll back
  work that succeeded;
- treats "duplicate column" / "already exists" as success, because the
  pipeline and the API both call `init_db()` at startup and one of them
  loses that race;
- is a silent no-op on an already-current database.

Existing rows keep their data; the new columns read as `NULL`.

### New queries

| Function | Purpose |
|---|---|
| `get_plate_detections(plate, session?)` | One plate's detections, **oldest first** (a path is read forwards) |
| `get_multi_camera_plates(min_cameras)` | Plates seen at ≥N cameras — the vehicles with a path worth drawing |
| `get_session_stats(session?)` | Detection/plate counts, overall and per camera |
| `get_processing_sessions()` | Recent runs, for the session picker |

`get_session_stats` counts total unique plates in a **separate query** rather
than summing the per-camera counts: one vehicle seen at four cameras is 1
unique plate, not 4.

---

## 5. Camera manager: files in turn, streams together

A run has two schedules, chosen per camera by its source type:

* **Recorded videos** are processed **strictly one at a time**, in ascending
  `order`. Each camera's ALPR pass completes before the next opens its
  source.
* **Live RTSP streams** all start **together** and run **until the session is
  stopped**.

Sequential files are deliberate, for three reasons:

1. **Hardware.** The Jetson has one GPU and the ALPR pipeline already
   saturates it. Four concurrent files would contend for the same TensorRT
   execution context and finish *later* than four run in turn, while making
   every per-camera latency figure meaningless.
2. **Demo correctness.** With all four cameras replaying the same footage,
   sequential execution is what makes timestamps increase monotonically
   across the queue — so a reconstructed trajectory reads CAM001 → CAM004.
3. **Migration fidelity.** It matches how a real deployment is reasoned
   about: each camera is an independent observer with its own source.

Live streams cannot work that way, and §5's "Live streams run together"
below says why.

### Threading

`start()` returns immediately; the queue runs on one background daemon
thread. Progress is published to a JSON status file rather than held in
memory, because the process *watching* a run is usually not the process
*running* it (Streamlit dashboard vs. FastAPI vs. CLI).

Writes are atomic — temp file in the same directory, then `os.replace` — so
a reader polling every second can never see a half-written document. Reads
never raise: a missing file means nothing has run yet, and a corrupt one
degrades to "unknown" rather than taking the dashboard down.

### Live streams run together

A recorded file ends by itself; **an RTSP stream never does**. The first
design bounded each live camera with a dwell time
(`processing.live_duration_seconds`) so a sequential queue could move on.
That was wrong for live operation: a stream carries what is happening *now*,
so the minute camera 2 spends waiting its turn does not delay its footage,
it **discards** it. Every vehicle passing camera 2 during camera 1's turn was
simply never seen, and only one panel of the camera wall ever moved.

Live cameras therefore all start at once, each on its own thread, and run
until `stop()` — no dwell time, no queue position. The dwell setting is gone.

They share **one** set of GPU models, handed out through
`PipelineModels.shared()`: a thin proxy that takes a lock around the calls
that reach the GPU (`detect`, `recognize`, `enhance`). A TensorRT execution
context processes one inference at a time, and two threads inside the same
one corrupt each other's bindings. Everything else — decoding, tracking, OCR
fusion, colour, thumbnails, database writes — runs genuinely in parallel,
which is where a live stream spends most of its wall clock.

Measured on this Jetson, four cameras through the real pipeline sharing one
model bundle: ~197 frames each in 25 s (**~7.9 fps per camera, ~31 fps in
total**) with all 16 expected events stored, against ~20 fps for one camera
running alone.

Two concurrency faults this exposed, both fixed:

* `init_db()` was not thread-safe. Four pipelines opening the database at
  once raced inside `create_all()`, and three of them died with "table
  vehicle_events already exists" or "database is locked" — losing three
  cameras' detections. It is now guarded by a lock and returns immediately
  when the process is already connected to that database.
* SQLite now runs in **WAL** mode with a 30 s busy timeout, so concurrent
  writers wait instead of failing and the dashboard can read while four
  cameras write.

A mixed queue runs the recorded files first, then brings the live cameras up
together.

### Adding and removing camera sites

Cameras are managed from the dashboard ("Add camera", and "Remove camera" on
each card) and over the API (`POST /trajectory-api/cameras`,
`DELETE /trajectory-api/cameras/{id}`).

Adding asks for the **place, latitude and longitude**. Coordinates are
required rather than optional: a camera without them records events that no
map or heatmap can place, so it would look configured and then quietly
disappear from every view that matters. The new camera is complete from the
moment it exists -- its own upload/RTSP controls, its own panel on the camera
wall, its own marker on the map and heatmap, and the last place in the
processing queue. It simply has no source yet, which is the state a shipped
camera is in before anyone gives it one.

**Renaming** changes only the label: the camera keeps its id, coordinates,
queue position and source. Events already recorded keep the name they were
stamped with -- they are a record of what the site was called when the
vehicle passed, so the statistics table still reads the old name while the
camera card, its feed panel and every new event read the new one.

Removing takes the site's configuration, its uploaded video and its last
preview frame (so the wall cannot leave a still image behind under a camera
that no longer exists). It **keeps the detections it recorded**: those are
history, they carry their own camera id and coordinates, and a trajectory
that passed the site yesterday still plots. The last camera cannot be
removed -- a file with no cameras does not load, so writing one would strand
the deployment at the next start.

**Where this is stored.** `camera_config.yaml` is hand-written and heavily
commented, and rewriting it from a YAML dump would throw every comment away.
So the shipped file stays the declaration an engineer edits, and the
operator's camera list lives beside it in `config/cameras_runtime.yaml`,
written and read only by code (`CameraRegistry.save()`). When that file holds
a camera list it REPLACES the shipped one; `defaults`, `processing` and
`trajectory` still come from `camera_config.yaml`. Delete it to go back to
the shipped cameras. It is per-device runtime state, so it is git-ignored.

Neither operation is allowed while a session is running: swapping the
registry under a running queue would leave the worker processing cameras that
no longer match the status it publishes.

### Failure isolation

One camera's failure is contained to that camera. A missing upload is caught
by `VideoSource.validate()` before the pipeline is ever invoked; a pipeline
crash is caught around the runner call. Either way the camera is marked
`FAILED` with the reason on its own progress entry and the queue continues —
three good cameras still reconstruct a usable partial trajectory. Set
`processing.continue_on_error: false` to stop the whole run instead.

### Stopping

`stop()` is **cooperative, not a kill**. It sets an event that the pipeline
checks once per frame; the pipeline breaks its loop and still runs its final
OCR-fusion flush, so plates that were mid-vote are stored rather than
silently lost.

---

## 6. Trajectory engine

`TrajectoryEngine.build(plate, detections)` → `Trajectory`.

### Grouping into visits

Detections are first sorted by time, then folded into **visits** — one marker
on the map each. Consecutive detections at the same camera merge unless they
are more than `revisit_gap_seconds` apart.

Merging on a *time-sorted list* rather than grouping by `camera_id` is what
preserves a genuine revisit: a vehicle that goes A → B → A produces three
visits, two of them at A. Grouping by camera id alone would silently delete
the return leg.

A visit reports the **first** detection's time and position (when the vehicle
actually reached the site) but the **best-confidence** detection's images and
confidence — the clearest read is the one worth showing in a popup, and it is
rarely the frame the vehicle first appeared in.

### Ordering

| `order_by` | Behaviour |
|---|---|
| `timestamp` | Strictly chronological. Correct for live RTSP, where clocks are the only shared reference |
| `trajectory_order` | Camera queue position. Deterministic for the demo |
| `auto` *(default)* | Timestamp when **every** point has a usable one, else camera order |

`auto` is all-or-nothing on purpose. A path ordered half by clock and half by
config is not a sequence anyone can reason about, so a single unparseable
timestamp switches the whole trajectory to camera ordering. Each strategy
uses the other as a tiebreaker.

### Measurements

Legs are computed, not stored, so they are always consistent with the points
they join:

- **distance** — haversine on a sphere (±0.5%, far better than the positional
  uncertainty of a hand-entered coordinate)
- **duration** — from the two timestamps
- **speed** — distance ÷ duration, and only when both exist **and** duration
  is non-zero (two reads in the same second must not report infinite speed)
- **bearing** — initial great-circle bearing, used to rotate the map arrows

Every measurement is **independently optional**. An unsurveyed camera yields
no distance but still yields a duration. `total_distance_km` is `None`, not
`0.0`, when nothing is measurable — "we could not measure this" must stay
distinguishable from "the vehicle did not move".

> **Expected demo artifact: implausible speeds.** In the four-virtual-camera
> demo the cameras are kilometres apart but the same clip is replayed at each
> of them back to back, so consecutive sightings are ~37 s apart in wall-clock
> time. The engine divides a real 2.4 km by a real 37 s and correctly reports
> ~235 km/h. Nothing is wrong: the timestamps are genuine *processing* times,
> not the times of a genuine journey. With real cameras watching one real
> vehicle, the gaps become real travel times and the speeds become sensible.
> This is exactly why `speed_kmh` is derived rather than stored — it is always
> consistent with the two timestamps it came from, however those arose.

---

## 7. Map engine

`render_trajectory_map(trajectory)` → a complete, self-contained Leaflet HTML
document.

**Layers** — OpenStreetMap standard tiles (default) and Esri World Imagery
satellite, toggled by Leaflet's own layer control. Satellite is paired with
Esri's reference layer so place names stay visible; bare imagery makes a city
trajectory hard to place.

**Drawn**

- a colour-graded polyline (green at the first camera → red at the last), on
  a dark casing so it stays readable over both light streets and dark imagery
- direction arrows at each leg's midpoint, rotated to that leg's true bearing
- numbered markers; click for a popup with camera name, plate, timestamp,
  confidence, detection count, coordinates and the vehicle thumbnail

**Why Leaflet directly rather than Folium.** The page has to embed in a
Streamlit component iframe, which is sandboxed and cannot read the server's
filesystem — so thumbnails must travel inside the document as data URIs and
the popup markup has to be built here. Doing it directly also avoids adding
`folium` and `streamlit-folium`, neither of which is installed on the Jetson,
and keeps the same renderer usable from FastAPI.

**Axis order.** GeoJSON is `[longitude, latitude]`; Leaflet's API is
`[latitude, longitude]`. Getting this backwards plots Delhi in the Indian
Ocean, so the conversion happens exactly once, in `geojson.py`.

Points with no coordinates are omitted from the geometry rather than emitted
at `[0, 0]` — null island is a worse answer than a shorter line. They stay in
the timeline and history table, which still record that the camera saw the
vehicle.

---

## 8. Dashboard

`src/dashboard/trajectory_app.py` — the multi-camera console.

> The original single-gate dashboard (`src/dashboard/app.py`) is **untouched
> and still runs**. The two read the same database and neither depends on the
> other.

The console is split into three pages, each with one job:

| Page | What it is for | Sections |
|------|----------------|----------|
| **Home** | Watching: what is happening, and looking something up | Blacklisted vehicles · Camera feeds · Search by plate number · Search by description · Traffic heatmap |
| **Operations** | Running: sources, the run, and what it produced | Cameras · Live Processing Status · Statistics (with detections per camera) |
| **Trajectory** | One vehicle's route | Profile card · timeline · map · history table |

Home is the page somebody leaves open all day, so it carries nothing they
only touch when starting a run; Operations carries exactly those controls.

### Operations page

Stacked full width, in the order the work happens:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  CAMERAS            four camera cards per row, the rest on the next row   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                     │
│  │1. Gate A │ │2. Gate B │ │3. Square │ │4. Parking│                     │
│  │ (•)Upload│ │ ( )Upload│ │ (•)Upload│ │ (•)Upload│                     │
│  │ ( )RTSP  │ │ (•)RTSP  │ │ ( )RTSP  │ │ ( )RTSP  │                     │
│  │ [file]   │ │ rtsp://… │ │ [file]   │ │ [file]   │                     │
│  │ Clear·Rm │ │ Clear·Rm │ │ Clear·Rm │ │ Clear·Rm │                     │
│  │ Rename ▾ │ │ Rename ▾ │ │ Rename ▾ │ │ Rename ▾ │                     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                     │
│  ┌──────────┐   Add camera ▾                                             │
│  │5. Gate C │                                                            │
│  └──────────┘   ▶ START PROCESSING                                       │
├──────────────────────────────────────────────────────────────────────────┤
│  LIVE PROCESSING STATUS   current camera · FPS · progress · ETA          │
│                           per-camera queue, current plate, vehicle, log  │
├──────────────────────────────────────────────────────────────────────────┤
│  STATISTICS   detected · plates · time · camera · avg OCR · avg detect   │
│               detections per camera (table)                              │
└──────────────────────────────────────────────────────────────────────────┘
```

Four cameras to a row, like the camera wall: stacked in one narrow column,
five cameras were a metre of scrolling — each card's uploader, buttons and
rename control sat between it and the next one, and the deployment could not
be seen at once.

### Home page

```
┌──────────────────────────────────────────────────────────────────────────┐
│  BLACKLISTED VEHICLES    alert banner + watch list                       │
├──────────────────────────────────────────────────────────────────────────┤
│  CAMERA FEEDS            four live panels per row; a fifth wraps          │
├──────────────────────────────────────────────────────────────────────────┤
│  SEARCH BY PLATE NUMBER  [_______] [Search]   [2+ cameras ▾]             │
├──────────────────────────────────────────────────────────────────────────┤
│  SEARCH BY DESCRIPTION   "white cars after 9am today"  → vehicle cards    │
├──────────────────────────────────────────────────────────────────────────┤
│  TRAFFIC HEATMAP         where vehicles are now / total traffic          │
└──────────────────────────────────────────────────────────────────────────┘
```

Either search opens the vehicle on the **Trajectory** page.

**Each camera gets exactly one source.** A radio per camera picks the kind,
and the two are mutually exclusive — assigning a video clears the stream URL,
setting a stream clears the uploaded file. The exclusivity is enforced in
`CameraManager`, not just hidden in the UI, so "which source is this camera
on?" is always a one-field question.

| Source | How it is given | What happens |
|---|---|---|
| **Uploaded video** | File uploader per camera | Saved as `data/uploads/<CAM_ID>.<ext>` — its own file, so no two cameras ever share a `video_path`, even if the same recording is uploaded to all four |
| **RTSP stream** | URL box + *Set stream* | Validated immediately; a malformed URL is rejected at entry instead of failing minutes into a run |

*Clear source* detaches either one without needing a replacement first. The
uploaded file is left on disk — only the camera's reference to it is cleared.

The plate suggestion dropdown is populated from the database with vehicles
actually seen at 2+ cameras, so the operator never has to guess a plate.

### Trajectory tab

Vehicle image · plate · first/last seen · cameras visited · detection count ·
average confidence · travel time · path length, then:

- **Timeline** — numbered steps sharing the map's colour ramp, with each hop
  annotated `2.42 km · 12m · 12 km/h`
- **Map** — the Leaflet view, with a satellite toggle
- **History table** — every stored read: camera, time, confidence, latitude,
  longitude, direction, session, source clip
- **GeoJSON download**

**Tables are rendered as plain HTML, not `st.dataframe`.** This deployment
pairs **pyarrow 25 with numpy 1.26** — an ABI mismatch, since pyarrow 25 is
built against numpy 2.x. `st.dataframe` and `st.table` both serialise through
Arrow, and that conversion **segfaults** under some conditions on this box.
A segfault is not an exception Streamlit can catch and display: it kills the
server process outright, so the operator sees "Connection error" and loses
any run in progress.

Upgrading numpy is not the fix here — `requirements.txt` pins 1.26.4 and
torch, ultralytics and OpenCV are all built against it, so moving it to
satisfy a table renderer would put the ALPR pipeline itself at risk. These
tables are a handful of rows, so `_table()` renders them as escaped HTML:
no Arrow, nothing to crash. (The original single-gate dashboard still uses
`st.dataframe` and remains exposed to this.)

### Camera feeds

The Home page shows every camera's annotated video four panels to a row, played
as continuous **MJPEG streams** rather than still images.

Streamlit can only change the page by re-running it, so a still image updated
at most once per re-run — every 2 s while processing, a slideshow. Instead the
dashboard process runs a small stream server (`src/cameras/preview_server.py`,
port `api.preview_port`, default **8765**) and each panel is a plain `<img>`
the browser plays natively. Each player's HTML is identical on every re-run,
so the browser never reloads it and the video keeps playing while the status
text around it updates.

The pipeline publishes previews on a background thread
(`LiveFramePublisher(background=True)`), downscaled to
`api.live_frames_max_width` (960 px) at up to `api.live_frames_fps` (15 fps),
so smooth video costs ALPR little. Measured on the Orin:

| | Browser video | ALPR speed |
|---|---|---|
| Before (still image per page re-run) | 0.5 fps | — |
| Recorded clip, previews on | 14.1 fps | 19.6 fps (21.6 with previews off) |
| RTSP camera, previews on | 13.6 fps | 23.6 fps |

Notes:

- A recorded video's preview can never be smoother than the pipeline
  processes it — the frames shown are the annotated outputs.
- Viewing the dashboard from another machine requires port 8765 to be
  reachable as well as 8501.
- If the port is unavailable the wall falls back to still frames and says so.

The sidebar carries a **session scope** picker. Each run of the camera queue
is one session; scoping to a single run keeps repeated demos of the same
video from blending into one another.

While a session is running the page re-runs itself every 2 s. Nothing
long-lived is held in Streamlit state — the manager is a module-level
singleton in its own module (Streamlit re-runs the script on every
interaction, which would otherwise rebuild it and lose the worker thread),
and progress is read back from the status file.

---

## Vehicle profiles

Every stored detection now also records the vehicle's **YOLO class**, its
**body colour**, and two small **thumbnails** (vehicle, plate). Each plate
has one **vehicle profile** summarising all of its detections.

### What is stored

| Where | Field | Meaning |
|---|---|---|
| `vehicle_events` (new, nullable) | `vehicle_class` | YOLO class from the tracker: car / truck / bus / motorcycle |
| | `vehicle_color` | Dominant body colour (below) |
| | `vehicle_thumbnail_path` | ≤320 px JPEG of the vehicle, `data/thumbnails/vehicles/` |
| | `plate_thumbnail_path` | Raw colour plate crop, at least 176 px wide, `data/thumbnails/plates/` |
| `vehicle_profiles` (new table) | one row per plate | class, colour, registration category, plate colour, best images, last camera and location, session, latest and best OCR confidence, first/last seen, total detections, **total camera visits**, unique cameras, **trajectory history** (JSON camera-visit sequence) |

`vehicle_type` and `plate_color` keep their existing meaning — the
registration category and the plate's background colour, both inferred from
the plate. The YOLO class and the body colour are new fields beside them.

### How profiles are maintained

`src/database/vehicle_profiles.py`. `insert_event()` folds each detection into
its plate's profile inside the same transaction, in a savepoint — a profile
problem can never lose the detection. A profile can also be rebuilt from
`vehicle_events` at any time, through the same fold, so the two always agree.
Rebuilds happen when a session is deleted, when a detection arrives out of
time order, and once to backfill an existing database the first time the
table appears.

- **Class and colour** are the majority across detections, ignoring
  "Unknown", so one poor view cannot flip them.
- **Images** come from the highest-confidence detection.
- **A visit** is consecutive detections at one camera, in one session, no
  more than 300 s apart; a return to a camera later is a new visit.

### Body colour

`src/classification/vehicle_color.py` — White, Black, Silver, Gray, Blue, Red,
Green, Yellow, Brown, Orange. A fast HSV heuristic with no model to load:
it skips the top of the vehicle box (windscreen, roof), decides coloured vs
achromatic by the share of saturated pixels, then votes on hue or reads the
median panel brightness. Thresholds are in `config.yaml` → `vehicle_color:`.

Measured on 36 hand-labelled vehicle crops extracted from this site's two
videos by the real detector and tracker: **30/36 correct**. The misses are
mostly crops no colour heuristic can fix: a black car behind the video's
semi-transparent watermark, and crops containing more than one vehicle.
On the three plates stored from `ALPR.mp4`, the light-blue Honda reads Blue and
the dark Volvo reads Gray, but the dark-navy BMW reads **Silver**: its plate is
only readable in a distant, watermark-covered view. Treat colour as a
helpful description, not ground truth.

### Cost

Measured per stored vehicle (not per frame) on the Orin: colour 0.5 ms,
both thumbnails 1.3 ms, profile update +14 ms on the database insert. With a
handful of stored vehicles per clip this is well under 0.1% of a run;
pipeline speed was unchanged within run-to-run noise (19.5 fps).

### APIs

- Event responses (`/logs`, `/search`, `/live`, `/direction`, `/entry`) gain
  `vehicle_class`, `vehicle_color`, `vehicle_thumbnail_path`,
  `plate_thumbnail_path` (null for older detections).
- `/vehicles/{plate}` gains `profile`; `/trajectory-api/trajectory/{plate}`
  gains `profile`. Existing keys are unchanged.
- New: `GET /trajectory-api/vehicles`, `GET /trajectory-api/vehicles/{plate}`.
- Thumbnails are served at `/thumbnails/{vehicles,plates}/<file>`.

## 9. API

New routes, all under `/trajectory-api` so they cannot collide with the
existing single-gate routes or the static dashboard mounted at `/`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/trajectory-api/cameras` | Camera registry, in processing order |
| POST | `/trajectory-api/cameras` | Add a camera site (place + coordinates) |
| PATCH | `/trajectory-api/cameras/{id}` | Rename a camera site |
| DELETE | `/trajectory-api/cameras/{id}` | Remove a camera site; its events are kept |
| POST | `/trajectory-api/cameras/upload` | Upload every camera's video at once |
| POST | `/trajectory-api/cameras/{id}/upload` | Upload one camera's video |
| POST | `/trajectory-api/cameras/{id}/rtsp` | Switch a camera to a live stream |
| POST | `/trajectory-api/processing/start` | Start the run (files in turn, streams together) |
| POST | `/trajectory-api/processing/stop` | Request a cooperative stop |
| GET | `/trajectory-api/processing/status` | Live progress |
| GET | `/trajectory-api/trajectory/{plate}` | Reconstructed path (points + legs) |
| GET | `/trajectory-api/map/{plate}` | Path as GeoJSON |
| GET | `/trajectory-api/map/{plate}/html` | Path as a Leaflet page |
| GET | `/trajectory-api/statistics` | Session / per-camera aggregates |
| GET | `/trajectory-api/plates` | Plates seen at ≥N cameras |
| GET | `/trajectory-api/sessions` | Recent processing sessions |

Most accept `?session=<id>` to scope to one run.

**Every pre-existing endpoint is unchanged**: `/entry`, `/logs`, `/search`,
`/direction`, `/stats`, `/vehicles/{plate}`, `/live`, `/settings`, `/health`,
`/clear`, `/snapshot`, `/stream`, `/media/*` and the dashboard mount. The
`EntryRequest` / `EventResponse` schemas gained optional nullable fields
only, so existing clients continue to work.

---

## 10. How to run

### Prerequisites

No new dependencies. The system uses only what the project already installs
(`streamlit`, `fastapi`, `sqlalchemy`, `pandas`, `pyyaml`, `opencv`); Leaflet
loads from a CDN in the browser.

### Demo — four virtual cameras from one recording

```bash
cd /home/ansh/alpr/alpr-university-gate

# 1. Start the dashboard
streamlit run src/dashboard/trajectory_app.py

# 2. In the browser (http://localhost:8501), Operations page:
#      • for each camera pick its source — "Upload video" or "RTSP stream"
#        (one or the other; setting one replaces the other)
#      • upload the same recording to all four for the demo, or point a
#        camera at a real stream
#      • press START PROCESSING
#      • watch CAM001 → CAM002 → CAM003 → CAM004 (uploads run one at a time)
#
# 3. When the run finishes:
#      • on the Home page, pick a plate from the "2+ cameras" dropdown,
#        type one, or describe the vehicle
#      • the Trajectory page shows the timeline, map and history
```

Optionally run the API alongside it, for the REST endpoints and the existing
web dashboard:

```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 8000
# http://localhost:8000/docs
```

### The single-gate system is unchanged

```bash
python main.py                                  # the original pipeline
python scripts/run_pipeline.py --source ALPR.mp4
streamlit run src/dashboard/app.py              # the original dashboard
```

### Driving it from the API instead

```bash
# One video for every camera, in a single request
curl -F "files=@ALPR.mp4" \
     http://localhost:8000/trajectory-api/cameras/upload

# ...or one file per camera, assigned in queue order
# curl -F "files=@gate1.mp4" -F "files=@gate2.mp4" \
#      -F "files=@gate3.mp4" -F "files=@gate4.mp4" \
#      http://localhost:8000/trajectory-api/cameras/upload

curl -X POST http://localhost:8000/trajectory-api/processing/start \
     -H 'Content-Type: application/json' -d '{}'

curl http://localhost:8000/trajectory-api/processing/status
curl http://localhost:8000/trajectory-api/trajectory/DL8CA1234
```

### Pointing at a different database

```bash
export ALPR_DB_PATH=/path/to/alpr_demo.db   # honoured by pipeline, API, both dashboards
```

### Tests

```bash
python -m pytest tests/ -q
```

---

## 11. Future RTSP migration

The architecture assumes every camera is an independent observer, so
switching a site from a recorded clip to a live stream is a **configuration
change with no code change**.

### What changes

```diff
  cameras:
    - camera_id: "CAM001"
      camera_name: "India Gate"
      latitude: 28.6129
      longitude: 77.2295
-     source_type: "upload"
-     video_path: "data/uploads/CAM001.mp4"
+     source_type: "rtsp"
+     rtsp_url: "rtsp://192.168.1.50:554/stream1"
      order: 1
```

or at runtime, without a restart:

```bash
curl -X POST http://localhost:8000/trajectory-api/cameras/CAM001/rtsp \
     -H 'Content-Type: application/json' \
     -d '{"rtsp_url": "rtsp://192.168.1.50:554/stream1"}'
```

### What does not change

The ALPR pipeline, the database schema, the trajectory engine, the map
engine, the dashboard, and every API endpoint. `create_video_source()` reads
`source_type` and returns an `RTSPSource` instead of an `UploadSource`; both
satisfy the same interface, and `FrameCapture` already handles live streams —
background reader thread, newest-frame-only, automatic reconnection, stall
detection.

### Credentials

Never put a camera URL with embedded credentials in `camera_config.yaml` —
it is tracked in git. Use the existing `.env` mechanism (already gitignored)
and reference the variable.

### Operational differences once live

| | Upload | RTSP |
|---|---|---|
| `total_frames` | Real count | `0` — the dashboard shows an indeterminate bar |
| End of source | Clip ends | Never; stopped by the operator |
| Frame handling | Every frame, in order | Newest frame only; frames dropped under load |
| Heavy-track cap | Not applied | `tracking.live_max_heavy_tracks_per_frame` applies |
| Trajectory ordering | `auto` → timestamp | `auto` → timestamp (clocks are authoritative) |

Two things to plan for when going live:

1. **Clock sync.** Trajectory ordering depends on timestamps across cameras.
   Run NTP on every device, or ordering will be wrong in ways that look
   plausible.
2. **Concurrency.** Sequential processing suits one Jetson and N recorded
   clips. Continuous live cameras each need their own worker — run one
   process per camera (or per device), all writing to a shared PostgreSQL.
   The schema, engine and dashboard already support that: `processing_session`
   simply becomes less relevant, and `order_by: timestamp` becomes the
   correct explicit setting.

---

## 12. Design decisions and assumptions

### Assumptions made

1. **`confidence` and `ocr_text` columns were added** beyond the listed
   schema changes. The specification requires each recognition to store its
   confidence and OCR result, and the map popups, search page and history
   table all display confidence — there was nowhere to put it otherwise.
2. **A separate dashboard file.** `trajectory_app.py` is the redesigned
   multi-camera console; `app.py` is left intact so no existing functionality
   is removed. Both are documented and both run.
3. **Leaflet over Folium.** Neither `folium` nor `streamlit-folium` is
   installed on this Jetson, and the sandboxed Streamlit iframe cannot read
   local files anyway. Direct Leaflet adds no dependency and gives full
   control over popups and thumbnail inlining.
4. **One source per camera, chosen per camera.** Each camera takes either an
   uploaded video or an RTSP URL, never both; setting one clears the other.
   Uploads are still stored per camera, so cameras never share a
   `video_path`. A bulk-upload API route
   (`POST /trajectory-api/cameras/upload`) remains available for scripted
   demos.
5. **Live cameras run together and continuously.** An RTSP stream has no
   end and no reason to wait: queueing one behind another loses its footage
   rather than delaying it. They share the GPU models behind a lock. See §5.
6. **One visit per camera by default** (`collapse_per_camera: true`). A
   four-camera journey should draw four markers, not forty. Every individual
   read remains in the history table.
7. **Sequential processing of recorded files is a design constraint, not a
   limitation.** See §5.
8. **A file-backed status store, not a message broker.** One writer, a
   few-hundred-byte payload; adding Redis to a Jetson to move it would be out
   of proportion.
9. **`processing_session` scopes a demo run.** Because the same video is
   replayed repeatedly, without it a plate's trajectory would accumulate
   points across every past run.
10. **Vehicle crops go in `data/vehicle_crops/`**, separate from
   `data/plate_crops/`. Different sizes and lifetimes, and the API mounts
   only the plate-crop directory as a static route.

### Guarantees about the existing system

- The ALPR pipeline's per-frame logic is unchanged. `run_pipeline()` gained
  optional keyword arguments that all default to the previous behaviour.
- The only behavioural change to the CLI path: an unopenable video source now
  raises `VideoSourceUnavailable`, which `main()` catches and turns into
  `sys.exit(1)` — the same exit status as before.
- No database column was removed, renamed or retyped. Every addition is
  nullable.
- No API endpoint was removed or changed. Schemas gained optional fields.
- The original Streamlit dashboard is untouched.

### Performance

The pipeline pays essentially nothing for the extension:

- Camera metadata and session context are resolved **once per camera run**,
  not per event — they are static for the whole run.
- The progress reporter is called every 10th frame (two integer increments
  per frame otherwise), because each call ends in a file write.
- The vehicle crop is written **only when one is passed in**, so the
  single-gate path does no extra disk I/O.
- No extra database call per event: the new fields ride along in the existing
  `INSERT`.
- Trajectory reconstruction happens on the read path only, indexed on
  `plate_number`, and never during processing.
