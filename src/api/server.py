"""
FastAPI REST server for the ALPR University Gate system.

Endpoints:
  POST /entry          — Record a vehicle event
  GET  /logs           — Retrieve all events (newest first)
  GET  /search?plate=  — Search events by plate number

Run with:
  uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.schemas import EntryRequest, EventResponse
from src.database import db as database
from src.utils.logger import get_logger

_logger = get_logger("api.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    try:
        database.init_db()
        _logger.info("Database initialized at startup")
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


def _get_db_session():
    """Helper to get a DB session, raising 503 if DB is unavailable."""
    try:
        return database.get_session()
    except RuntimeError as exc:
        _logger.error("Database unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Database unavailable")


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

        config = load_config("config/config.yaml")

        # Filter sensitive information
        safe_config = {
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
