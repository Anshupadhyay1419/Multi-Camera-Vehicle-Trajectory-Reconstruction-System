"""
Pydantic request/response schemas for the ALPR FastAPI server.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class EntryRequest(BaseModel):
    plate_number: str
    vehicle_type: str
    plate_color:  str
    series_type:  str
    direction:    Literal["IN", "OUT"]
    image_path:   str = ""

    # Which camera saw this vehicle, and where that camera is. Optional: a
    # caller on the capturing device can leave these out and the server
    # stamps its own config.yaml `camera:` values instead (see
    # server.create_entry), which is the normal single-gate case. Send them
    # explicitly only when posting on behalf of a *different* camera than
    # the one this API is running next to.
    camera_id:    Optional[str]   = None
    camera_name:  Optional[str]   = None
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None

    # Multi-camera trajectory fields. Optional for the same reason as the
    # camera block: a single-gate caller never sends them.
    vehicle_image_path: Optional[str]   = None
    trajectory_order:   Optional[int]   = None
    processing_session: Optional[str]   = None
    video_source:       Optional[str]   = None
    confidence:         Optional[float] = None
    ocr_text:           Optional[str]   = None


class EventResponse(BaseModel):
    id:           int
    plate_number: str
    vehicle_type: str
    plate_color:  str
    series_type:  str
    timestamp:    str
    direction:    str
    image_path:   str

    # Nullable in the database, so nullable here: events recorded before
    # camera attribution existed have none, and a gate that hasn't been
    # surveyed has no coordinates. The dashboard renders missing values as
    # a dash rather than hiding the column.
    camera_id:    Optional[str]   = None
    camera_name:  Optional[str]   = None
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None

    # Nullable for every row that is not part of a multi-camera run.
    vehicle_image_path: Optional[str]   = None
    trajectory_order:   Optional[int]   = None
    processing_session: Optional[str]   = None
    video_source:       Optional[str]   = None
    confidence:         Optional[float] = None
    ocr_text:           Optional[str]   = None

    # Vehicle profile attributes. Null for detections stored before these
    # existed. vehicle_type above keeps its meaning (registration category).
    vehicle_class:          Optional[str] = None
    vehicle_color:          Optional[str] = None
    vehicle_thumbnail_path: Optional[str] = None
    plate_thumbnail_path:   Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════
#  Multi-camera trajectory reconstruction
# ═══════════════════════════════════════════════════════════════════════════


class CameraResponse(BaseModel):
    """One configured camera, as the dashboard needs it."""

    camera_id:    str
    camera_name:  str
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None
    order:        int
    source_type:  str
    video_path:   Optional[str] = None
    rtsp_url:     Optional[str] = None
    enabled:      bool = True
    has_location: bool = False


class StartProcessingRequest(BaseModel):
    """Kick off a sequential run over the camera queue."""

    # None runs every enabled camera, which is the normal case; a list
    # restricts the run without changing the order they run in.
    camera_ids: Optional[list[str]] = None
    session_id: Optional[str] = None


class StartProcessingResponse(BaseModel):
    session_id: str
    queued_cameras: list[str]
    message: str


class RTSPRequest(BaseModel):
    """Point a camera at a live stream instead of an uploaded file."""

    rtsp_url: str


class RenameCameraRequest(BaseModel):
    """A camera's new place name. Nothing else about it changes."""

    camera_name: str


class AddCameraRequest(BaseModel):
    """A new camera site.

    Coordinates are required, not optional: a camera without them records
    events that no map or heatmap can place, so it would look configured and
    then quietly disappear from every view that matters.
    """

    camera_name: str
    latitude:    float
    longitude:   float
    camera_id:   Optional[str] = None


class TrajectoryPointResponse(BaseModel):
    """One camera visit on a reconstructed path."""

    sequence:           int
    camera_id:          str
    camera_name:        str
    latitude:           Optional[float] = None
    longitude:          Optional[float] = None
    timestamp:          Optional[str]   = None
    confidence:         Optional[float] = None
    detection_count:    int = 1
    plate_image_path:   Optional[str] = None
    vehicle_image_path: Optional[str] = None
    vehicle_type:       Optional[str] = None
    plate_color:        Optional[str] = None
    direction:          Optional[str] = None
    trajectory_order:   Optional[int] = None
    video_source:       Optional[str] = None
    ocr_text:           Optional[str] = None
    has_location:       bool = False
    all_timestamps:     list[str] = []


class TrajectoryLegResponse(BaseModel):
    """Measured movement between two consecutive points.

    Every measurement is independently optional -- an unsurveyed camera
    yields no distance, an unparseable timestamp no duration, and speed
    needs both.
    """

    from_camera_id:   str
    from_camera_name: str
    to_camera_id:     str
    to_camera_name:   str
    distance_km:      Optional[float] = None
    duration_seconds: Optional[float] = None
    speed_kmh:        Optional[float] = None
    bearing_degrees:  Optional[float] = None


class TrajectoryResponse(BaseModel):
    """A vehicle's reconstructed path across the camera network."""

    plate_number:       str
    processing_session: Optional[str] = None
    ordering:           str = "timestamp"
    points:             list[TrajectoryPointResponse] = []
    legs:               list[TrajectoryLegResponse] = []
    cameras_visited:    int = 0
    total_detections:   int = 0
    first_seen:         Optional[str] = None
    last_seen:          Optional[str] = None
    duration_seconds:   Optional[float] = None
    total_distance_km:  Optional[float] = None
    average_confidence: Optional[float] = None
    path_labels:        list[str] = []
    # The plate's vehicle profile, when one exists. Added alongside the
    # reconstruction; the reconstruction itself is unchanged.
    profile:            Optional[VehicleProfileResponse] = None


class CameraVisitResponse(BaseModel):
    """One visit in a vehicle profile's camera sequence."""

    camera_id:          Optional[str]   = None
    camera_name:        Optional[str]   = None
    latitude:           Optional[float] = None
    longitude:          Optional[float] = None
    processing_session: Optional[str]   = None
    first_seen:         Optional[str]   = None
    last_seen:          Optional[str]   = None
    detections:         int = 1
    confidence:         Optional[float] = None


class VehicleProfileResponse(BaseModel):
    """Everything known about one vehicle, across all its detections."""

    plate_number:           str
    vehicle_class:          Optional[str] = None
    vehicle_color:          Optional[str] = None
    vehicle_type:           Optional[str] = None
    plate_color:            Optional[str] = None
    vehicle_image_path:     Optional[str] = None
    plate_image_path:       Optional[str] = None
    vehicle_thumbnail_path: Optional[str] = None
    plate_thumbnail_path:   Optional[str] = None
    camera_id:              Optional[str]   = None
    camera_name:            Optional[str]   = None
    latitude:               Optional[float] = None
    longitude:              Optional[float] = None
    processing_session:     Optional[str]   = None
    ocr_confidence:         Optional[float] = None
    best_confidence:        Optional[float] = None
    first_seen:             str
    last_seen:              str
    total_detections:       int = 0
    total_camera_visits:    int = 0
    unique_cameras:         int = 0
    trajectory_history:     list[CameraVisitResponse] = []
    updated_at:             Optional[str] = None


TrajectoryResponse.model_rebuild()
