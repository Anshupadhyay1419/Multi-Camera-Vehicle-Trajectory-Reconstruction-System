"""
SQLAlchemy ORM models for the ALPR University Gate database.
"""

from __future__ import annotations

from sqlalchemy import Column, Float, Index, Integer, String, Text
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

    # ── Vehicle profile attributes ────────────────────────────────────────
    # Deliberately NEW columns rather than new meanings for old ones:
    # vehicle_type is the registration category inferred from the PLATE
    # colour ("Private", "Commercial"...) and plate_color is the plate's
    # background -- both keep meaning exactly what they always have.
    #
    # vehicle_class           YOLO detector class: car / truck / bus / motorcycle.
    # vehicle_color           Dominant BODY colour (classification.vehicle_color).
    # vehicle_thumbnail_path  Small JPEG of the vehicle, for cards and lists --
    # plate_thumbnail_path    and of the raw plate. The full-size crops
    #                         (vehicle_image_path, image_path) are unchanged.
    vehicle_class          = Column(String, nullable=True)
    vehicle_color          = Column(String, nullable=True)
    vehicle_thumbnail_path = Column(String, nullable=True)
    plate_thumbnail_path   = Column(String, nullable=True)

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
            "vehicle_class":          self.vehicle_class,
            "vehicle_color":          self.vehicle_color,
            "vehicle_thumbnail_path": self.vehicle_thumbnail_path,
            "plate_thumbnail_path":   self.plate_thumbnail_path,
        }


class VehicleProfile(Base):
    """Everything known about one vehicle, keyed by plate number.

    One row per plate, maintained incrementally as detections are stored
    (database.vehicle_profiles) and rebuildable from vehicle_events at any
    time -- the events table stays the source of truth, this is a
    summary of it. Created by create_all() on first start, so an existing
    database gains it without any migration step.

    Nullable wherever a detection may not supply the value: events from
    before this table existed have no class, colour or thumbnails.
    """

    __tablename__ = "vehicle_profiles"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String, nullable=False, unique=True)

    # Attributes -- the most frequent value across all detections (see
    # attribute_votes), so one misread camera cannot flip them.
    vehicle_class = Column(String, nullable=True)   # YOLO class
    vehicle_color = Column(String, nullable=True)   # body colour
    vehicle_type  = Column(String, nullable=True)   # registration category
    plate_color   = Column(String, nullable=True)

    # Images from the highest-confidence detection -- the clearest read.
    vehicle_image_path     = Column(String, nullable=True)
    plate_image_path       = Column(String, nullable=True)
    vehicle_thumbnail_path = Column(String, nullable=True)
    plate_thumbnail_path   = Column(String, nullable=True)

    # Where and when it was last seen.
    camera_id          = Column(String, nullable=True)
    camera_name        = Column(String, nullable=True)
    latitude           = Column(Float,  nullable=True)
    longitude          = Column(Float,  nullable=True)
    processing_session = Column(String, nullable=True)
    ocr_confidence     = Column(Float,  nullable=True)   # of the latest detection
    best_confidence    = Column(Float,  nullable=True)

    first_seen = Column(String, nullable=False)          # ISO 8601
    last_seen  = Column(String, nullable=False)

    total_detections    = Column(Integer, nullable=False, default=0)
    total_camera_visits = Column(Integer, nullable=False, default=0)
    unique_cameras      = Column(Integer, nullable=False, default=0)

    # JSON: ordered camera visits, oldest first. One entry per visit --
    # consecutive detections at the same camera in the same session close
    # together in time are one visit, not several.
    trajectory_history = Column(Text, nullable=False, default="[]")
    # JSON: {"vehicle_class": {"car": 3}, "vehicle_color": {...}, ...}
    attribute_votes    = Column(Text, nullable=False, default="{}")

    updated_at = Column(String, nullable=False)

    __table_args__ = (
        Index("idx_profile_last_seen", "last_seen"),
    )

    def to_dict(self) -> dict:
        import json

        try:
            history = json.loads(self.trajectory_history or "[]")
        except ValueError:
            history = []
        return {
            "plate_number":           self.plate_number,
            "vehicle_class":          self.vehicle_class,
            "vehicle_color":          self.vehicle_color,
            "vehicle_type":           self.vehicle_type,
            "plate_color":            self.plate_color,
            "vehicle_image_path":     self.vehicle_image_path,
            "plate_image_path":       self.plate_image_path,
            "vehicle_thumbnail_path": self.vehicle_thumbnail_path,
            "plate_thumbnail_path":   self.plate_thumbnail_path,
            "camera_id":              self.camera_id,
            "camera_name":            self.camera_name,
            "latitude":               self.latitude,
            "longitude":              self.longitude,
            "processing_session":     self.processing_session,
            "ocr_confidence":         self.ocr_confidence,
            "best_confidence":        self.best_confidence,
            "first_seen":             self.first_seen,
            "last_seen":              self.last_seen,
            "total_detections":       self.total_detections,
            "total_camera_visits":    self.total_camera_visits,
            "unique_cameras":         self.unique_cameras,
            "trajectory_history":     history,
            "updated_at":             self.updated_at,
        }
