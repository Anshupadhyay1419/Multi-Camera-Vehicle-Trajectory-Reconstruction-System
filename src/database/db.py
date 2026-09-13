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
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base, VehicleEvent
from src.utils.logger import get_logger

_logger = get_logger("database.db")

_engine = None
_SessionFactory = None
_db_type = None  # "sqlite" or "postgres"


# Columns added to vehicle_events after the first deployment, as
# (name, SQL type) in the order they should be appended.
#
# Base.metadata.create_all() below only creates *missing tables* -- it never
# alters an existing one. So a Jetson that has already been running has a
# vehicle_events table with the original eight columns, and every query
# naming a new column would fail with "no such column: camera_id" until the
# table is actually altered. _migrate_schema() does that in place, keeping
# the rows already recorded (their new columns read as NULL, which is what
# nullable=True on the model is for).
#
# The types are spelled so one statement works on both backends: SQLite maps
# VARCHAR/DOUBLE PRECISION onto its TEXT/REAL affinities, and PostgreSQL
# takes them literally.
_ADDED_COLUMNS = (
    ("camera_id",   "VARCHAR"),
    ("camera_name", "VARCHAR"),
    ("latitude",    "DOUBLE PRECISION"),
    ("longitude",   "DOUBLE PRECISION"),
    # Multi-camera trajectory reconstruction. Appended to the same tuple
    # rather than given their own migration pass so a database at ANY prior
    # version -- the original eight columns, or the four-camera-column one --
    # converges on the current schema in a single init_db(). Order matters
    # only cosmetically (it is the order columns get appended in); the
    # inspector below adds whichever subset is actually missing.
    ("vehicle_image_path", "VARCHAR"),
    ("trajectory_order",   "INTEGER"),
    ("processing_session", "VARCHAR"),
    ("video_source",       "VARCHAR"),
    ("confidence",         "DOUBLE PRECISION"),
    ("ocr_text",           "VARCHAR"),
    # Vehicle profile attributes (see models.VehicleEvent).
    ("vehicle_class",          "VARCHAR"),
    ("vehicle_color",          "VARCHAR"),
    ("vehicle_thumbnail_path", "VARCHAR"),
    ("plate_thumbnail_path",   "VARCHAR"),
)


# Indexes that create_all() builds for a fresh database and that an altered
# one therefore needs created alongside the columns they cover. Kept as data
# next to _ADDED_COLUMNS so adding a column + its index is one edit in one
# place. IF NOT EXISTS works on both backends, and _apply_additive_ddl()
# tolerates the concurrent-startup race on top of that.
_ADDED_INDEXES = (
    ("idx_camera_id",          "vehicle_events (camera_id)"),
    ("idx_processing_session", "vehicle_events (processing_session)"),
)


# Substrings both backends use when asked to create something that is
# already there -- SQLite says "duplicate column name: camera_id", PostgreSQL
# says "column \"camera_id\" of relation ... already exists".
_ALREADY_APPLIED_MARKERS = ("duplicate column", "already exists")


def _apply_additive_ddl(conn, statement: str, description: str) -> None:
    """Run one additive DDL statement, tolerating "already applied".

    The pipeline and the API each call init_db() on startup, and on a Jetson
    they are usually started together. Both would inspect the old table,
    both would see the same column missing, and both would try to add it --
    so the loser of that race gets a "duplicate column" error for work that
    is now, in fact, done. Treating that as success keeps one process from
    dying at startup over a migration the other just completed.

    Scoped deliberately to additive DDL (ADD COLUMN / CREATE INDEX): for
    those, "it already exists" really is the desired end state. Any other
    failure propagates -- init_db() must not report a healthy database it
    could not actually migrate.
    """
    try:
        conn.execute(text(statement))
        _logger.info("Schema migration: %s", description)
    except OperationalError as exc:
        message = str(exc).lower()
        if any(marker in message for marker in _ALREADY_APPLIED_MARKERS):
            _logger.debug(
                "Schema migration: %s already applied by another process",
                description,
            )
            return
        raise


def _migrate_schema(engine) -> None:
    """Add any post-release columns missing from an existing vehicle_events.

    A no-op for a database create_all() just built from the current model
    (nothing is missing) and for one already migrated, so it is safe to run
    unconditionally on every startup.
    """
    inspector = inspect(engine)
    if "vehicle_events" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("vehicle_events")}
    missing = [(name, sql_type) for name, sql_type in _ADDED_COLUMNS
               if name not in existing]
    if not missing:
        return

    for name, sql_type in missing:
        # Always ADD COLUMN with no NOT NULL/DEFAULT -- existing rows have no
        # camera to attribute them to, so NULL is the correct value for them
        # and the cheapest ALTER on both backends.
        #
        # One transaction per statement, not one for the whole batch: if a
        # concurrent starter has already added some of these columns, the
        # tolerated error must not roll back the ones this process did add.
        with engine.begin() as conn:
            _apply_additive_ddl(
                conn,
                f"ALTER TABLE vehicle_events ADD COLUMN {name} {sql_type}",
                f"added vehicle_events.{name}",
            )

    for index_name, target in _ADDED_INDEXES:
        with engine.begin() as conn:
            _apply_additive_ddl(
                conn,
                f"CREATE INDEX IF NOT EXISTS {index_name} ON {target}",
                f"created index {index_name}",
            )


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

    # Create all tables (SQLite) or verify they exist (PostgreSQL), then
    # bring an older table up to the current model's column set.
    try:
        had_profiles_table = "vehicle_profiles" in inspect(_engine).get_table_names()
        Base.metadata.create_all(_engine)
        _migrate_schema(_engine)
        _SessionFactory = sessionmaker(bind=_engine)
        if not had_profiles_table:
            _backfill_vehicle_profiles()
        _logger.info("✓ Database connection established")
    except Exception as exc:
        _logger.error("Failed to initialize database: %s", exc)
        raise


def _backfill_vehicle_profiles() -> None:
    """Build profiles for detections recorded before the table existed.

    Runs once, on the start that creates vehicle_profiles, so an existing
    deployment's history gains profiles without a manual step. Never fails
    init_db(): profiles are a derived summary and can be rebuilt any time
    (database.vehicle_profiles.rebuild_all_profiles).
    """
    from src.database.vehicle_profiles import rebuild_all_profiles

    try:
        with get_session() as session:
            count = rebuild_all_profiles(session)
        if count:
            _logger.info("Backfilled vehicle profiles for %d plate(s)", count)
    except Exception as exc:
        _logger.warning("Could not backfill vehicle profiles: %s", exc)


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


def _as_optional_float(value) -> Optional[float]:
    """Coerce a coordinate to float, or None if it isn't a usable number.

    config.yaml is hand-edited, so a latitude can arrive as the string
    "28.6139" (quoted by the editor) or as an empty value. Neither should
    cost the gate an event, so an unusable coordinate is simply dropped --
    utils/config.get_camera_metadata() has already logged the reason for
    the pipeline's own path.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value) -> Optional[int]:
    """Coerce a queue position to int, or None if it isn't a usable number.

    Same tolerance as _as_optional_float above, and for the same reason: a
    trajectory_order that arrives as "2" from a YAML edit or as None from
    the single-gate pipeline must not cost the gate an event. A bad value
    degrades to None, which the trajectory engine reads as "not part of a
    multi-camera run" and falls back to timestamp ordering for.
    """
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def insert_event(session: Session, event_data: dict) -> Optional[VehicleEvent]:
    """Insert a vehicle event record with one retry on failure.

    Args:
        session:    Active SQLAlchemy session.
        event_data: Dict with keys matching VehicleEvent columns. The four
                    camera fields (camera_id, camera_name, latitude,
                    longitude) are optional -- omitted, they store as NULL,
                    so callers that predate camera attribution still work.

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
                camera_id=event_data.get("camera_id"),
                camera_name=event_data.get("camera_name"),
                latitude=_as_optional_float(event_data.get("latitude")),
                longitude=_as_optional_float(event_data.get("longitude")),
                # Multi-camera fields. All optional for the same reason the
                # camera block is: the single-gate pipeline never sets them,
                # and must keep inserting successfully without them.
                vehicle_image_path=event_data.get("vehicle_image_path"),
                trajectory_order=_as_optional_int(event_data.get("trajectory_order")),
                processing_session=event_data.get("processing_session"),
                video_source=event_data.get("video_source"),
                confidence=_as_optional_float(event_data.get("confidence")),
                ocr_text=event_data.get("ocr_text"),
                vehicle_class=event_data.get("vehicle_class"),
                vehicle_color=event_data.get("vehicle_color"),
                vehicle_thumbnail_path=event_data.get("vehicle_thumbnail_path"),
                plate_thumbnail_path=event_data.get("plate_thumbnail_path"),
            )
            session.add(event)
            session.flush()
            # Keep this plate's vehicle profile current, in the same
            # transaction. Isolated in a savepoint and never raising: a
            # profile problem must not cost the detection itself.
            from src.database.vehicle_profiles import update_profile_for_event

            update_profile_for_event(session, event)
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


def save_thumbnail(
    image: np.ndarray,
    plate_number: str,
    save_dir: str,
    max_side: int = 320,
    quality: int = 85,
    min_width: int = 0,
) -> str:
    """Save a small JPEG of *image* and return its path, or "" on failure.

    For cards and lists: the full-size crops already stored are often several
    hundred kilobytes, far more than a 200px card needs, and loading dozens of
    them makes a page slow. The longest side is scaled down to *max_side*
    (never up) with area averaging, which is the cheap and clean way to
    shrink an image.

    Never raises -- a failed thumbnail costs a picture, not the detection.
    """
    try:
        import cv2

        if image is None or getattr(image, "size", 0) == 0:
            return ""
        height, width = image.shape[:2]
        scale = min(1.0, float(max_side) / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        elif min_width and width < min_width:
            # Enlarge an image too small to read on a card. Plate crops are
            # often only ~40px wide at source (measured: 38x11); scaling for
            # display adds no detail, but it turns an unreadable speck into
            # something a person can actually look at.
            grow = float(min_width) / width
            image = cv2.resize(
                image, (int(min_width), max(1, int(round(height * grow)))),
                interpolation=cv2.INTER_CUBIC,
            )
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        path = Path(save_dir) / f"{plate_number}_{stamp}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]):
            return ""
        return str(path)
    except Exception as exc:
        _logger.warning("Failed to save thumbnail: %s", exc)
        return ""


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


# ---------------------------------------------------------------------------
# Multi-camera trajectory queries
# ---------------------------------------------------------------------------
#
# The trajectory engine needs three shapes of read that the single-gate
# queries above do not provide: one plate's detections across every camera,
# the set of plates that were seen at more than one camera, and per-session
# aggregates. They live here rather than in the engine so that all SQL stays
# in the database layer -- the engine works on dicts and never imports
# SQLAlchemy, which is what makes it unit-testable without a database.


def _scope_to_sessions(query, processing_session: Optional[str]):
    """Restrict a trajectory query to multi-camera session events.

    With a session id: that one run. Without one: EVERY run -- but only
    events that belong to some run.

    The second half is the point. "All sessions" used to mean "no filter at
    all", which swept in every event the single-gate pipeline had ever
    recorded (no session, and often no camera). Those rows are not part of
    the camera network, cannot be placed on a trajectory, and made the
    multi-camera dashboard show detections and a "Main Gate" camera even
    after every session had been deleted. They remain in the database and in
    the original single-gate dashboard; they just are not multi-camera data.
    """
    if processing_session:
        return query.filter(VehicleEvent.processing_session == processing_session)
    return query.filter(VehicleEvent.processing_session.isnot(None))


def get_plate_detections(
    session: Session,
    plate_number: str,
    processing_session: Optional[str] = None,
) -> list[dict]:
    """Return every detection of *plate_number*, oldest first.

    Ascending timestamp (the opposite of search_events(), which is
    newest-first for a log view) because a trajectory is read forwards: the
    first row is where the vehicle started.

    Args:
        session:            Active SQLAlchemy session.
        plate_number:       Exact plate to look up; matched case-insensitively
                            since plates are stored upper-cased but a search
                            box is not.
        processing_session: Restrict to one run of the camera queue. None
                            searches every run (events recorded outside any
                            run -- the single-gate pipeline -- are excluded;
                            see _scope_to_sessions).

    Returns:
        List of event dicts, ascending by (trajectory_order, timestamp).
    """
    query = session.query(VehicleEvent).filter(
        VehicleEvent.plate_number == plate_number.strip().upper()
    )
    query = _scope_to_sessions(query, processing_session)

    # Order in SQL by timestamp only. The composite (trajectory_order,
    # timestamp) ordering the engine actually applies is decided in Python
    # -- NULL ordering differs between SQLite and PostgreSQL, and the engine
    # has to re-sort anyway once it knows whether every point carries an
    # order. Sorting here is just to give a stable, useful default.
    events = query.order_by(VehicleEvent.timestamp.asc()).all()
    return [event.to_dict() for event in events]


def get_multi_camera_plates(
    session: Session,
    min_cameras: int = 2,
    processing_session: Optional[str] = None,
    limit: int = 200,
) -> list[dict]:
    """Return plates seen at *min_cameras* or more distinct cameras.

    This is the "which vehicles actually have a trajectory worth drawing?"
    query -- a plate seen at exactly one camera is a sighting, not a path.
    The dashboard uses it to offer real suggestions instead of making the
    operator guess a plate number.

    Args:
        min_cameras:        Minimum distinct cameras a plate must appear at.
        processing_session: Restrict to one run of the camera queue.
        limit:              Cap on returned plates, most-travelled first.

    Returns:
        List of {plate_number, camera_count, detection_count, first_seen,
        last_seen}, ordered by camera_count then detection_count descending.
    """
    from sqlalchemy import func

    query = session.query(
        VehicleEvent.plate_number,
        func.count(func.distinct(VehicleEvent.camera_id)).label("camera_count"),
        func.count(VehicleEvent.id).label("detection_count"),
        func.min(VehicleEvent.timestamp).label("first_seen"),
        func.max(VehicleEvent.timestamp).label("last_seen"),
    )
    query = _scope_to_sessions(query, processing_session)

    rows = (
        query.group_by(VehicleEvent.plate_number)
        .having(func.count(func.distinct(VehicleEvent.camera_id)) >= min_cameras)
        .order_by(
            func.count(func.distinct(VehicleEvent.camera_id)).desc(),
            func.count(VehicleEvent.id).desc(),
        )
        .limit(limit)
        .all()
    )

    return [
        {
            "plate_number": row.plate_number,
            "camera_count": int(row.camera_count or 0),
            "detection_count": int(row.detection_count or 0),
            "first_seen": row.first_seen,
            "last_seen": row.last_seen,
        }
        for row in rows
    ]


def get_session_stats(
    session: Session,
    processing_session: Optional[str] = None,
) -> dict:
    """Return aggregate counts for one processing session (or all sessions).

    One grouped query rather than a scan-and-count in Python: the dashboard
    polls this every couple of seconds while a run is in progress, and the
    events table grows without bound across runs.
    """
    from sqlalchemy import func

    query = session.query(
        VehicleEvent.camera_id,
        VehicleEvent.camera_name,
        func.count(VehicleEvent.id).label("detections"),
        func.count(func.distinct(VehicleEvent.plate_number)).label("unique_plates"),
        func.avg(VehicleEvent.confidence).label("avg_confidence"),
    )
    query = _scope_to_sessions(query, processing_session)

    rows = query.group_by(VehicleEvent.camera_id, VehicleEvent.camera_name).all()

    per_camera = [
        {
            "camera_id": row.camera_id,
            "camera_name": row.camera_name,
            "detections": int(row.detections or 0),
            "unique_plates": int(row.unique_plates or 0),
            "avg_confidence": float(row.avg_confidence) if row.avg_confidence is not None else None,
        }
        for row in rows
    ]

    # Total unique plates is deliberately a second query, not a sum of the
    # per-camera counts: one vehicle seen at all four cameras contributes 1
    # to the total but 4 to the sum, and the total is the number a dashboard
    # labelled "unique plates" has to show.
    total_query = session.query(
        func.count(VehicleEvent.id),
        func.count(func.distinct(VehicleEvent.plate_number)),
    )
    total_query = _scope_to_sessions(total_query, processing_session)
    total_detections, total_unique = total_query.one()

    return {
        "processing_session": processing_session,
        "total_detections": int(total_detections or 0),
        "unique_plates": int(total_unique or 0),
        "cameras_reporting": len(per_camera),
        "per_camera": sorted(per_camera, key=lambda item: item["camera_id"] or ""),
    }


def get_processing_sessions(session: Session, limit: int = 20) -> list[dict]:
    """Return recent processing sessions, newest first.

    Lets the dashboard offer "which run?" as a picker rather than requiring
    the operator to remember a generated session id.
    """
    from sqlalchemy import func

    rows = (
        session.query(
            VehicleEvent.processing_session,
            func.count(VehicleEvent.id).label("detections"),
            func.min(VehicleEvent.timestamp).label("started"),
            func.max(VehicleEvent.timestamp).label("ended"),
        )
        .filter(VehicleEvent.processing_session.isnot(None))
        .group_by(VehicleEvent.processing_session)
        .order_by(func.max(VehicleEvent.timestamp).desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "processing_session": row.processing_session,
            "detections": int(row.detections or 0),
            "started": row.started,
            "ended": row.ended,
        }
        for row in rows
    ]


def count_events_without_session(session: Session) -> int:
    """How many events were recorded outside any processing session.

    These come from the single-gate pipeline (scripts/run_pipeline.py run on
    its own) or predate multi-camera sessions. They are real data, which is
    exactly why deleting sessions never removes them -- but the dashboard
    needs the count to say so, or leftover totals look like a delete that
    did not work.
    """
    return int(
        session.query(VehicleEvent)
        .filter(VehicleEvent.processing_session.is_(None))
        .count()
    )


def delete_session(session, processing_session: str) -> dict:
    """Delete every event recorded in one processing session.

    Returns the images those events referenced so the caller can remove them
    too -- this function deliberately does NOT touch the filesystem.
    Deleting rows and deleting files are different failure modes (a locked or
    missing file must not roll back the database delete), so the two are kept
    separate and the caller decides how to handle each.

    Args:
        session:            Active SQLAlchemy session.
        processing_session: The run to delete. Must be a non-empty id --
                            deleting "everything with no session" would wipe
                            every single-gate event ever recorded, which is
                            never what a "delete this run" button means.

    Returns:
        {"events": N, "image_paths": [...]} -- how many rows were removed and
        the image files those rows leave orphaned.

    Raises:
        ValueError: No session id was given.
    """
    processing_session = (processing_session or "").strip()
    if not processing_session:
        raise ValueError(
            "delete_session() needs a session id; refusing to delete events "
            "that belong to no session"
        )

    rows = (
        session.query(VehicleEvent)
        .filter(VehicleEvent.processing_session == processing_session)
        .all()
    )
    image_paths = [
        path
        for event in rows
        for path in (event.image_path, event.vehicle_image_path,
                     event.vehicle_thumbnail_path, event.plate_thumbnail_path)
        if path
    ]
    affected_plates = {event.plate_number for event in rows}

    deleted = (
        session.query(VehicleEvent)
        .filter(VehicleEvent.processing_session == processing_session)
        .delete(synchronize_session=False)
    )
    _logger.info("Deleted %d event(s) from session %s", deleted, processing_session)

    # Those plates' profiles summarised events that no longer exist. Rebuild
    # them from what remains (which deletes a profile left with no events).
    if affected_plates:
        from src.database.vehicle_profiles import rebuild_profiles

        session.flush()
        rebuild_profiles(session, affected_plates)
    return {"events": int(deleted or 0), "image_paths": image_paths}


def save_vehicle_image(
    vehicle_crop: np.ndarray,
    plate_number: str,
    save_dir: str = "data/vehicle_crops/",
) -> str:
    """Save a whole-vehicle crop and return its relative path.

    Separate from save_plate_image() and writing to a different directory on
    purpose: the API mounts the plate-crop directory as a static route, and
    the two kinds of image have different lifetimes and sizes. Same
    failure posture though -- an unwritable image costs a thumbnail, never
    the event itself, so this returns "" instead of raising.
    """
    return save_plate_image(vehicle_crop, plate_number, save_dir)
