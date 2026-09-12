"""
SQLAlchemy ORM models for the ALPR University Gate database.
"""

from __future__ import annotations

from sqlalchemy import Column, Float, Index, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class VehicleEvent(Base):
    """Persisted record of a vehicle entry/exit event."""

    __tablename__ = "vehicle_events"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String, nullable=False)
    vehicle_type = Column(String, nullable=False)
    plate_color  = Column(String, nullable=False)
    series_type  = Column(String, nullable=False)   # "BH" or "normal"
    timestamp    = Column(String, nullable=False)   # ISO 8601
    direction    = Column(String, nullable=False)   # "IN" or "OUT"
    image_path   = Column(String, nullable=False)   # relative path to plate crop

    # ── Which camera saw it, and where that camera is ─────────────────────
    # Copied from the capturing device's config.yaml `camera:` block at
    # insert time (see utils/config.get_camera_metadata) rather than joined
    # from a cameras table later, so a stored event keeps saying where the
    # vehicle actually was even after the camera is renamed or re-sited.
    #
    # All four are nullable, unlike every column above: rows written before
    # these columns existed have no camera to attribute them to, and a gate
    # whose coordinates haven't been surveyed yet legitimately has none.
    # Readers must handle None (the dashboard renders it as a dash).
    camera_id    = Column(String, nullable=True)    # e.g. "GATE-01", "CAM001"
    camera_name  = Column(String, nullable=True)    # e.g. "Main Gate", "India Gate"
    latitude     = Column(Float,  nullable=True)    # decimal degrees, WGS84
    longitude    = Column(Float,  nullable=True)    # decimal degrees, WGS84

    # ── Multi-camera trajectory reconstruction ────────────────────────────
    # Added for the multi-camera system. Every one is nullable for the same
    # reason the block above is: single-gate events (and every row written
    # before this existed) have no session and no queue position, and must
    # keep reading back correctly. The trajectory engine treats a NULL here
    # as "not part of a multi-camera run" rather than as a zero.
    #
    # vehicle_image_path  Crop of the whole VEHICLE, as distinct from
    #                     image_path's crop of just the plate. The map popups
    #                     and the search result page show the vehicle (you
    #                     can recognise a car in it); image_path stays the
    #                     plate crop every existing screen already renders.
    # trajectory_order    The capturing camera's position in the processing
    #                     queue (1..N). Denormalised onto the event on
    #                     purpose: it records the order that actually applied
    #                     at capture time, so re-ordering camera_config.yaml
    #                     afterwards cannot rewrite a stored trajectory.
    # processing_session  Groups every event from one END-TO-END run of the
    #                     camera queue. Lets the dashboard scope a
    #                     trajectory to "this demo run" instead of blending
    #                     it with every previous run of the same video.
    # video_source        The file path or RTSP URL this event came from --
    #                     the audit trail for "which clip produced this?",
    #                     which matters precisely because the demo feeds the
    #                     same video to four different cameras.
    # confidence          Fused OCR confidence for the stored read, surfaced
    #                     on map popups and the history table.
    # ocr_text            Raw fused OCR string BEFORE plate-format
    #                     validation/correction. Kept alongside the cleaned
    #                     plate_number so a misread can be diagnosed without
    #                     re-running the video.
    vehicle_image_path = Column(String,  nullable=True)
    trajectory_order   = Column(Integer, nullable=True)
    processing_session = Column(String,  nullable=True)
    video_source       = Column(String,  nullable=True)
    confidence         = Column(Float,   nullable=True)
    ocr_text           = Column(String,  nullable=True)

    __table_args__ = (
        Index("idx_plate_number", "plate_number"),
        Index("idx_timestamp",    "timestamp"),
        # Multi-camera deployments filter the log by gate; one index keeps
        # that cheap as history grows.
        Index("idx_camera_id",    "camera_id"),
        # Trajectory reconstruction always filters by plate and then orders
        # the hits; scoping a run to one session is the other hot filter.
        Index("idx_processing_session", "processing_session"),
    )

    def to_dict(self) -> dict:
        return {
            "id":           self.id,
            "plate_number": self.plate_number,
            "vehicle_type": self.vehicle_type,
            "plate_color":  self.plate_color,
            "series_type":  self.series_type,
            "timestamp":    self.timestamp,
            "direction":    self.direction,
            "image_path":   self.image_path,
            "camera_id":    self.camera_id,
            "camera_name":  self.camera_name,
            "latitude":     self.latitude,
            "longitude":    self.longitude,
            "vehicle_image_path": self.vehicle_image_path,
            "trajectory_order":   self.trajectory_order,
            "processing_session": self.processing_session,
            "video_source":       self.video_source,
            "confidence":         self.confidence,
            "ocr_text":           self.ocr_text,
        }
