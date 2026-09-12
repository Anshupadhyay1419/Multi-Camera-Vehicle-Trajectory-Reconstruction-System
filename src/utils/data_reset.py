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


def clear_all_data(config: dict) -> dict[str, int]:
    """Delete every stored event, plate-crop image, and the live frame.

    Args:
        config: The full loaded config dict (as returned by load_config()).

    Returns:
        {"events": N, "images": N} counts of what was deleted.
    """
    db_cfg = config["database"]
    database.init_db(db_cfg["path"])
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
