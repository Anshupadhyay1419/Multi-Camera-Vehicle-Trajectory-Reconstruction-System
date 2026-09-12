"""Clears all stored ALPR data: database rows, saved plate-crop images, and
the live dashboard frame. Shared by scripts/clear_data.py (CLI) and the
dashboard's "Clear all data" button (src/api/server.py's POST /clear), so
there's exactly one place this logic lives.
"""

from __future__ import annotations

from pathlib import Path

from src.database import db as database
from src.database.models import VehicleEvent

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ensure_database(config: dict) -> None:
    """Open the database only if one is not already open.

    Calling init_db() unconditionally re-points the process at whatever
    config.yaml names, which silently overrides a database the caller has
    already opened -- the API opens DB_URL at startup, then this would swap
    it for data/alpr.db and report "no events found" for rows that plainly
    exist. Respecting an existing connection keeps every caller operating on
    the database it is actually serving.
    """
    try:
        database.get_engine()
    except RuntimeError:
        database.init_db(config["database"]["path"])


def clear_all_data(config: dict) -> dict[str, int]:
    """Delete every stored event, plate-crop image, and the live frame.

    Args:
        config: The full loaded config dict (as returned by load_config()).

    Returns:
        {"events": N, "images": N} counts of what was deleted.
    """
    db_cfg = config["database"]
    _ensure_database(config)
    with database.get_session() as session:
        deleted_events = session.query(VehicleEvent).delete()
        session.commit()

    # Both crop directories: plate crops (every deployment) and whole-vehicle
    # crops (multi-camera runs only). Leaving the vehicle crops behind would
    # orphan them -- the rows referencing them have just been deleted.
    deleted_images = 0
    for setting, default in (
        ("image_save_path", "data/plate_crops/"),
        ("vehicle_image_save_path", "data/vehicle_crops/"),
    ):
        crops_dir = REPO_ROOT / db_cfg.get(setting, default)
        if crops_dir.is_dir():
            for image_path in crops_dir.glob("*.jpg"):
                image_path.unlink()
                deleted_images += 1

    live_frame_path = REPO_ROOT / config.get("api", {}).get("live_frame_path", "data/live_frame.jpg")
    if live_frame_path.exists():
        live_frame_path.unlink()

    # The processing status file describes a session whose events no longer
    # exist; leaving it would have the dashboard report detections that have
    # just been cleared.
    status_path = REPO_ROOT / "data" / "processing_status.json"
    if status_path.exists():
        status_path.unlink()

    return {"events": deleted_events, "images": deleted_images}


def delete_processing_session(config: dict, processing_session: str) -> dict[str, int]:
    """Delete one processing session: its events and their images.

    The per-session counterpart to clear_all_data(). Images are removed after
    the rows, and only the ones those rows referenced -- every event stores
    its own crop, so nothing a surviving event needs is touched.

    A missing or unreadable image counts as "not removed" and never raises:
    the events are already gone, and failing here would leave the caller
    unable to tell whether the delete actually happened.

    Args:
        config:             The full loaded config dict.
        processing_session: The run to delete.

    Returns:
        {"events": N, "images": N}

    Raises:
        ValueError: No session id was given.
    """
    from src.database.db import delete_session

    _ensure_database(config)
    with database.get_session() as session:
        result = delete_session(session, processing_session)

    removed_images = 0
    for raw_path in result["image_paths"]:
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        try:
            if path.is_file():
                path.unlink()
                removed_images += 1
        except OSError:
            # The row is already gone; a stuck file is not worth failing over.
            continue

    return {"events": result["events"], "images": removed_images}
