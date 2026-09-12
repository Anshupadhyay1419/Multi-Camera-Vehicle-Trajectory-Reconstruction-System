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
    camera_id    = Column(String, nullable=True)    # e.g. "GATE-01"
    camera_name  = Column(String, nullable=True)    # e.g. "Main Gate"
    latitude     = Column(Float,  nullable=True)    # decimal degrees, WGS84
    longitude    = Column(Float,  nullable=True)    # decimal degrees, WGS84

    __table_args__ = (
        Index("idx_plate_number", "plate_number"),
        Index("idx_timestamp",    "timestamp"),
        # Multi-camera deployments filter the log by gate; one index keeps
        # that cheap as history grows.
        Index("idx_camera_id",    "camera_id"),
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
        }
