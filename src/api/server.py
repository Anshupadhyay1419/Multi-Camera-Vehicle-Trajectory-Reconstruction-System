"""
FastAPI REST server for the ALPR University Gate system.

Endpoints:
  POST /entry          — Record a vehicle event
  GET  /logs           — Retrieve all events (newest first)
  GET  /search?plate=  — Search events by plate number
  GET  /stream         — Live annotated camera feed (MJPEG)
  GET  /snapshot       — Single current frame (JPEG)
  POST /clear          — Delete all stored events, images, and the live frame

Run with:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import EntryRequest, EventResponse
from src.database import db as database
from src.database.vehicle_profiles import get_profile
from src.utils.config import get_camera_metadata, load_config
from src.utils.data_reset import clear_all_data
from src.utils.logger import get_logger

_logger = get_logger("api.server")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_app_config() -> dict:
    try:
        return load_config(str(_REPO_ROOT / "config" / "config.yaml"))
    except Exception as exc:
        _logger.warning("Could not load config.yaml (%s); using defaults", exc)
        return {}


# Loaded at import time: lifespan() needs it to choose the database, and the
# module-level paths below need it too.
_config = _load_app_config()


def _resolve_db_url() -> str | None:
    """Pick the database this API should serve, matching the rest of the system.

    Order of precedence:
      1. $DB_URL   -- the explicit override, and the only way to point at
                      PostgreSQL. Passed through untouched by returning None,
                      which lets init_db() read it itself.
      2. config.yaml's `database.path` (which honours $ALPR_DB_PATH).
      3. init_db()'s own default.

    Step 2 is the fix for a real inconsistency: this used to call init_db()
    with no argument at all, so the API always opened sqlite:///data/alpr.db
    while the pipeline and both dashboards opened whatever `database.path`
    named. With the default config the two happen to coincide, which is why
    it went unnoticed -- but point the pipeline anywhere else and the API
    silently serves a different database, reporting "no detections" for
    plates that were definitely recorded.
    """
    if os.getenv("DB_URL"):
        return None
    return (_config.get("database") or {}).get("path") or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    try:
        database.init_db(_resolve_db_url())
        _logger.info(
            "Database initialized at startup: %s", database.get_engine().url
        )
    except Exception as exc:
        _logger.error("Failed to initialize database: %s", exc)
    yield
    # shutdown — nothing to release for SQLite; connection pool closes itself


app = FastAPI(
    title="ALPR University Gate API",
    description="License plate recognition entry/exit log API",
    version="1.0.0",
    lifespan=lifespan,
)


_live_frame_path = _REPO_ROOT / _config.get("api", {}).get("live_frame_path", "data/live_frame.jpg")
_plate_crops_dir = _REPO_ROOT / _config.get("database", {}).get("image_save_path", "data/plate_crops/")
_plate_crops_dir.mkdir(parents=True, exist_ok=True)

# This device's camera identity/location, used to stamp any event posted to
# /entry without its own camera fields (see config.yaml's `camera:` block).
_camera_meta = get_camera_metadata(_config)

# Plate crop thumbnails, served under /media/<filename> -- deliberately a
# narrow mount (just this one directory) rather than all of data/, which
# also holds the sqlite database file.
app.mount("/media", StaticFiles(directory=str(_plate_crops_dir)), name="media")

# Vehicle-profile thumbnails, served under /thumbnails/{vehicles,plates}/<file>.
# Another narrow mount -- only the thumbnail tree, never the rest of data/.
_thumbnail_dir = _REPO_ROOT / _config.get("database", {}).get("thumbnail_dir", "data/thumbnails")
for _kind in ("vehicles", "plates"):
    (_thumbnail_dir / _kind).mkdir(parents=True, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=str(_thumbnail_dir)), name="thumbnails")


def _get_db_session():
    """Helper to get a DB session, raising 503 if DB is unavailable."""
    try:
        return database.get_session()
    except RuntimeError as exc:
        _logger.error("Database unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


def _resolve_event_camera(request: EntryRequest) -> dict:
    """Decide which camera fields to store for a posted event.

    A caller on this device can omit the camera fields entirely and inherit
    this gate's configured identity -- the normal single-camera case, and
    why the fields are optional on EntryRequest at all.

    But inheritance only applies to *this* camera. If the request names a
    different camera_id, the rest of this device's block describes a
    different place, so filling in its name and coordinates would label the
    event with a gate it never passed. Those fields stay as sent (possibly
    None) instead.
    """
    sent = {
        "camera_id":   request.camera_id,
        "camera_name": request.camera_name,
        "latitude":    request.latitude,
        "longitude":   request.longitude,
    }

    is_other_camera = (
        request.camera_id is not None
        and request.camera_id != _camera_meta["camera_id"]
    )
    if is_other_camera:
        return sent

    # Same camera (or unspecified): fill each blank field from local config.
    return {
        key: _camera_meta[key] if value is None else value
        for key, value in sent.items()
    }


@app.post("/entry", response_model=EventResponse, status_code=201)
def create_entry(request: EntryRequest):
    """Record a new vehicle entry/exit event."""
    try:
        with database.get_session() as session:
            event_data = {
                "plate_number": request.plate_number,
                "vehicle_type": request.vehicle_type,
                "plate_color":  request.plate_color,
                "series_type":  request.series_type,
                "direction":    request.direction,
                "image_path":   request.image_path,
                "timestamp":    datetime.now(timezone.utc).isoformat(),
                # camera_id / camera_name / latitude / longitude
                **_resolve_event_camera(request),
            }
            event = database.insert_event(session, event_data)
            if event is None:
                raise HTTPException(status_code=500, detail="Failed to insert event")
            return EventResponse(**event.to_dict())
    except HTTPException:
        raise
    except RuntimeError as exc:
        _logger.error("Database unavailable on POST /entry: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on POST /entry: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/logs", response_model=list[EventResponse])
def get_logs():
    """Return all vehicle events ordered by timestamp descending."""
    try:
        with database.get_session() as session:
            events = database.get_all_events(session)
            return [EventResponse(**e) for e in events]
    except RuntimeError as exc:
        _logger.error("Database unavailable on GET /logs: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on GET /logs: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/search", response_model=list[EventResponse])
def search_by_plate(plate: str = Query(..., description="Plate number to search for")):
    """Search vehicle events by plate number."""
    try:
        with database.get_session() as session:
            events = database.search_events(session, plate_number=plate)
            return [EventResponse(**e) for e in events]
    except RuntimeError as exc:
        _logger.error("Database unavailable on GET /search: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on GET /search: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/direction")
def get_events_by_direction(direction: str = Query(..., description="'IN' or 'OUT'")):
    """Return recent events filtered by direction."""
    if direction.upper() not in ("IN", "OUT"):
        raise HTTPException(status_code=400, detail="Direction must be 'IN' or 'OUT'")

    try:
        with database.get_session() as session:
            events = database.get_events_by_direction(session, direction, limit=100)
            return [EventResponse(**e) for e in events]
    except RuntimeError as exc:
        _logger.error("Database unavailable on GET /direction: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on GET /direction: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats")
def get_traffic_statistics():
    """Return today's traffic statistics."""
    try:
        with database.get_session() as session:
            stats = database.get_daily_stats(session)
            return stats
    except RuntimeError as exc:
        _logger.error("Database unavailable on GET /stats: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on GET /stats: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/vehicles/{plate}")
def get_vehicle_history(plate: str):
    """Return full history for a specific vehicle by plate number."""
    try:
        plate = plate.upper()
        with database.get_session() as session:
            events = database.search_events(session, plate_number=plate)
            if not events:
                raise HTTPException(status_code=404, detail=f"No records for plate {plate}")

            return {
                "plate_number": plate,
                "total_events": len(events),
                "entries": sum(1 for e in events if e["direction"] == "IN"),
                "exits": sum(1 for e in events if e["direction"] == "OUT"),
                "first_seen": events[-1]["timestamp"] if events else None,
                "last_seen": events[0]["timestamp"] if events else None,
                "events": events,
                # Added: the vehicle profile (class, colour, thumbnails,
                # camera visits). Every key above is unchanged.
                "profile": get_profile(session, plate),
            }
    except HTTPException:
        raise
    except RuntimeError as exc:
        _logger.error("Database unavailable on GET /vehicles/{plate}: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on GET /vehicles/{plate}: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/live")
def get_live_feed(limit: int = Query(10, ge=1, le=100, description="Number of recent events")):
    """Return live feed of most recent events (auto-update on dashboard)."""
    try:
        with database.get_session() as session:
            events = database.get_all_events(session, limit=limit)
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_count": len(events),
                "events": [EventResponse(**e) for e in events],
            }
    except RuntimeError as exc:
        _logger.error("Database unavailable on GET /live: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")
    except Exception as exc:
        _logger.error("Unexpected error on GET /live: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/settings")
def get_system_settings():
    """Return current system configuration (read-only)."""
    try:
        from src.utils.config import load_config
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(str(repo_root / "config" / "config.yaml"))

        # Filter sensitive information
        safe_config = {
            "camera": get_camera_metadata(config),
            "video": config.get("video", {}),
            "detection": {
                "vehicle_confidence": config.get("detection", {}).get("vehicle_confidence"),
                "plate_confidence": config.get("detection", {}).get("plate_confidence"),
            },
            "ocr": {
                "backend": config.get("ocr", {}).get("backend"),
            },
            "fusion": config.get("fusion", {}),
            "tracking": config.get("tracking", {}),
        }

        return safe_config
    except Exception as exc:
        _logger.error("Failed to load settings: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to load settings")


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/clear")
def clear_all():
    """Delete every stored event, plate-crop image, and the live frame.

    Confirmation happens client-side (the dashboard asks before calling
    this) -- this endpoint itself performs the deletion unconditionally,
    same as scripts/clear_data.py --yes.
    """
    try:
        counts = clear_all_data(_config)
        _logger.info(
            "Cleared all data via dashboard: %d event(s), %d image(s)",
            counts["events"], counts["images"],
        )
        return counts
    except Exception as exc:
        _logger.error("Failed to clear data: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/snapshot")
def get_snapshot():
    """Return the pipeline's current camera frame as a single JPEG.

    The pipeline (scripts/run_pipeline.py) writes this file continuously
    while running -- see src/utils/live_frame.py. If it hasn't run yet, or
    isn't running right now, there is no file to serve.
    """
    if not _live_frame_path.exists():
        raise HTTPException(
            status_code=503,
            detail="No live frame available -- is the pipeline running?",
        )
    return FileResponse(str(_live_frame_path), media_type="image/jpeg")


@app.get("/stream")
async def get_stream():
    """Live camera feed as an MJPEG multipart stream.

    A plain <img src="/stream"> renders this as live video in every
    browser with no extra client-side code -- the standard simple way to
    show a Python vision pipeline's output in a web page, short of a full
    WebRTC setup.
    """
    boundary = "alprframe"
    poll_interval = 1.0 / max(
        float(_config.get("api", {}).get("live_frame_fps", 8.0)), 0.1
    )

    async def frame_generator():
        last_mtime = None
        while True:
            try:
                mtime = _live_frame_path.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    data = _live_frame_path.read_bytes()
                    yield (
                        f"--{boundary}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(data)}\r\n\r\n"
                    ).encode("ascii") + data + b"\r\n"
            except FileNotFoundError:
                pass
            await asyncio.sleep(poll_interval)

    return StreamingResponse(
        frame_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
    )


# Multi-camera trajectory routes, all under /trajectory-api. Registered
# before the "/" static mount below, since Starlette matches routes in
# registration order and that mount is the catch-all.
from src.api.trajectory_routes import router as trajectory_router  # noqa: E402

app.include_router(trajectory_router)


# Serves the dashboard's HTML/CSS/JS. Mounted last and at the root path so
# every API route above (all under distinct paths like /logs, /search)
# still matches first -- Starlette checks routes in the order they were
# added, and this mount is the fallback for anything that isn't one of
# them, including "/" itself (html=True serves index.html there).
_dashboard_dir = Path(__file__).resolve().parents[1] / "dashboard_web"
app.mount("/", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard")
