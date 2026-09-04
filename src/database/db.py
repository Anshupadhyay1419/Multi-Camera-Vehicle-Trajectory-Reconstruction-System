"""
Database session management and CRUD operations for the ALPR system.

Supports both SQLite (development) and PostgreSQL (production).
Auto-creates SQLite schema on first run; PostgreSQL requires prior schema setup.

Environment variable: DB_URL
  SQLite:    "sqlite:///data/alpr.db"
  PostgreSQL: "postgresql://user:password@localhost:5432/alpr_db"
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional
from urllib.parse import urlparse

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, VehicleEvent
from src.utils.logger import get_logger

_logger = get_logger("database.db")

_engine = None
_SessionFactory = None
_db_type = None  # "sqlite" or "postgres"


def init_db(db_url: str = None) -> None:
    """Initialize database connection (SQLite or PostgreSQL).

    Args:
        db_url: Database URL.
               If None, uses DB_URL env var, or defaults to sqlite:///data/alpr.db
               
    Examples:
        init_db()  # Uses env var or default SQLite
        init_db("sqlite:///data/alpr.db")  # SQLite explicitly
        init_db("postgresql://user:pass@localhost/alpr_db")  # PostgreSQL
    """
    global _engine, _SessionFactory, _db_type

    # Resolve DB URL
    if db_url is None:
        db_url = os.getenv("DB_URL", "sqlite:///data/alpr.db")
    
    # If db_url is a plain file path (no scheme), convert to SQLite URL
    if "://" not in db_url:
        db_url = f"sqlite:///{Path(db_url).absolute()}"

    # Determine database type
    parsed = urlparse(db_url)
    scheme = parsed.scheme.lower()
    
    if scheme in ("postgresql", "postgres"):
        _db_type = "postgres"
        _logger.info("Using PostgreSQL backend")
    elif scheme == "sqlite":
        _db_type = "sqlite"
        # Ensure directory exists for SQLite
        db_path = db_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _logger.info("Using SQLite backend: %s", db_url)
    else:
        raise ValueError(f"Unsupported database scheme: {scheme}")

    # Create engine with connection pooling
    if _db_type == "postgres":
        engine_kwargs = {
            "echo": False,
            "pool_size": 20,
            "max_overflow": 40,
            "pool_pre_ping": True,  # Verify connections before use
        }
    else:
        # SQLite does not support pool_size / max_overflow
        engine_kwargs = {
            "echo": False,
            "connect_args": {"check_same_thread": False},
        }

    _engine = create_engine(db_url, **engine_kwargs)

    # Create all tables (SQLite) or verify they exist (PostgreSQL)
    try:
        Base.metadata.create_all(_engine)
        _SessionFactory = sessionmaker(bind=_engine)
        _logger.info("✓ Database connection established")
    except Exception as exc:
        _logger.error("Failed to initialize database: %s", exc)
        raise


def get_engine():
    """Return the current SQLAlchemy engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _engine


def get_db_type() -> str:
    """Return the database type: 'sqlite' or 'postgres'."""
    if _db_type is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db_type


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Context manager that yields a database session."""
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def insert_event(session: Session, event_data: dict) -> Optional[VehicleEvent]:
    """Insert a vehicle event record with one retry on failure.

    Args:
        session:    Active SQLAlchemy session.
        event_data: Dict with keys matching VehicleEvent columns.

    Returns:
        The inserted VehicleEvent, or None if both attempts failed.
    """
    for attempt in range(2):
        try:
            event = VehicleEvent(
                plate_number=event_data["plate_number"],
                vehicle_type=event_data["vehicle_type"],
                plate_color=event_data["plate_color"],
                series_type=event_data["series_type"],
                timestamp=event_data.get(
                    "timestamp",
                    datetime.now(timezone.utc).isoformat()
                ),
                direction=event_data["direction"],
                image_path=event_data.get("image_path", ""),
            )
            session.add(event)
            session.flush()
            return event
        except Exception as exc:
            if attempt == 0:
                _logger.error(
                    "DB insert failed (attempt 1), retrying: %s", exc
                )
                session.rollback()
            else:
                _logger.error(
                    "DB insert failed (attempt 2), discarding event: %s", exc
                )
    return None


def get_all_events(session: Session, limit: int = 1000) -> list[dict]:
    """Return all vehicle events ordered by timestamp descending."""
    events = (
        session.query(VehicleEvent)
        .order_by(VehicleEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [e.to_dict() for e in events]


def search_events(session: Session, plate_number: str) -> list[dict]:
    """Return all events matching the given plate number."""
    events = (
        session.query(VehicleEvent)
        .filter(VehicleEvent.plate_number == plate_number)
        .order_by(VehicleEvent.timestamp.desc())
        .all()
    )
    return [e.to_dict() for e in events]


def get_events_by_direction(session: Session, direction: str, limit: int = 100) -> list[dict]:
    """Return recent events filtered by direction (IN/OUT)."""
    events = (
        session.query(VehicleEvent)
        .filter(VehicleEvent.direction == direction.upper())
        .order_by(VehicleEvent.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [e.to_dict() for e in events]


def _event_local_date(timestamp: str):
    """Parse a stored ISO timestamp and return its date in the local
    timezone, or None if it can't be parsed (never let one bad row break
    the whole stats query).
    """
    try:
        ts = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        # Older/malformed rows without a UTC offset -- assume UTC, since
        # that's what every current write path uses.
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone().date()


def get_daily_stats(session: Session) -> dict:
    """Return today's traffic statistics.

    "Today" means the local calendar day (what a human operator means by
    it), even though events are stored with UTC timestamps
    (datetime.now(timezone.utc).isoformat() -- see insert_event()). The
    previous version compared local-date boundaries as plain strings
    against those UTC timestamps with no timezone conversion, which is
    wrong for roughly a third of the day (UTC and IST, for example, are
    5.5 hours apart -- from local midnight until UTC's own midnight, an
    event correctly stored for "today" sorts as "yesterday" against a
    same-looking-but-wrong string boundary). Parsing to real datetimes and
    comparing local calendar dates avoids the whole class of bug.
    """
    local_today = datetime.now().astimezone().date()

    events = [
        e for e in session.query(VehicleEvent).all()
        if _event_local_date(e.timestamp) == local_today
    ]

    in_count = sum(1 for e in events if e.direction == "IN")
    out_count = sum(1 for e in events if e.direction == "OUT")
    unique_vehicles = len(set(e.plate_number for e in events))

    return {
        "date": local_today.isoformat(),
        "entries": in_count,
        "exits": out_count,
        "unique_vehicles": unique_vehicles,
        "total_events": len(events),
    }


def save_plate_image(
    plate_crop: np.ndarray,
    plate_number: str,
    save_dir: str = "data/plate_crops/",
) -> str:
    """Save a plate crop image and return its relative path.

    Args:
        plate_crop:   NumPy array (grayscale or BGR).
        plate_number: Used to build the filename.
        save_dir:     Directory to save images.

    Returns:
        Relative path to the saved image, or "" on failure.
    """
    try:
        import cv2

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{plate_number}_{ts}.jpg"
        filepath = Path(save_dir) / filename

        cv2.imwrite(str(filepath), plate_crop)
        return str(filepath)
    except Exception as exc:
        _logger.warning("Failed to save plate image: %s", exc)
        return ""
