"""
Value types for the multi-camera layer.

Everything here is a plain, immutable-ish data holder: the camera registry
parses YAML into `CameraConfig`, the manager reports progress as
`CameraProgress` / `SessionStatus`, and both cross a process boundary as
JSON (the dashboard and the API read the same status file the worker
writes). Keeping them dependency-free -- no SQLAlchemy, no Streamlit, no
OpenCV -- is what lets the API import them without dragging in the vision
stack.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class SourceType(str, Enum):
    """Where a camera's frames come from.

    Deliberately a two-value enum rather than a bool: it is the single field
    that flips when a demo site is promoted to a real camera, and the code
    that branches on it reads better naming both cases. Subclassing `str`
    keeps it JSON-serialisable with no custom encoder.
    """

    UPLOAD = "upload"
    RTSP = "rtsp"


class CameraState(str, Enum):
    """Lifecycle of one camera inside a processing session."""

    PENDING = "pending"        # queued, not started
    RUNNING = "running"        # currently being processed
    COMPLETED = "completed"    # finished, source exhausted
    FAILED = "failed"          # errored out; see CameraProgress.error
    SKIPPED = "skipped"        # disabled, or no usable source


class SessionState(str, Enum):
    """Lifecycle of a whole multi-camera run."""

    IDLE = "idle"              # nothing has been started yet
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"    # stop() was requested mid-run


@dataclass(frozen=True)
class CameraConfig:
    """One camera site: who it is, where it stands, where its video is.

    `order` doubles as the position in the sequential processing queue and
    as the fallback ordering key for trajectory reconstruction, which is why
    it is part of the camera's identity rather than a scheduling detail
    invented by the manager.
    """

    camera_id: str
    camera_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    order: int
    source_type: SourceType = SourceType.UPLOAD
    video_path: Optional[str] = None
    rtsp_url: Optional[str] = None
    enabled: bool = True

    @property
    def has_location(self) -> bool:
        """True when this camera can actually be plotted on a map.

        Both coordinates or neither -- a lone latitude is not a position,
        so callers get one predicate instead of repeating the `and`.
        """
        return self.latitude is not None and self.longitude is not None

    @property
    def source_uri(self) -> Optional[str]:
        """The string to hand a VideoSource, or None if nothing is set.

        Resolves `source_type` to the matching field so callers never have
        to know which of video_path / rtsp_url is the live one.
        """
        if self.source_type is SourceType.RTSP:
            return self.rtsp_url
        return self.video_path

    def event_metadata(self) -> dict[str, Any]:
        """The camera fields to stamp onto every event this camera records.

        Keys match VehicleEvent's column names exactly so callers can splat
        this straight into an event_data dict -- the same contract
        utils.config.get_camera_metadata() already follows for the
        single-camera path.
        """
        return {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_type"] = self.source_type.value
        data["has_location"] = self.has_location
        return data


@dataclass
class CameraProgress:
    """Live progress for one camera within a session.

    Mutated in place by the worker and snapshotted to JSON on every update,
    so every field must stay JSON-friendly.
    """

    camera_id: str
    camera_name: str
    order: int
    state: CameraState = CameraState.PENDING
    frames_processed: int = 0
    total_frames: int = 0          # 0 when unknown (a live stream has no end)
    detections: int = 0            # events actually stored by this camera
    unique_plates: int = 0
    fps: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    last_plate: Optional[str] = None
    last_plate_image: Optional[str] = None
    last_vehicle_image: Optional[str] = None
    avg_ocr_ms: float = 0.0
    avg_detection_ms: float = 0.0

    @property
    def percent(self) -> float:
        """Completion 0-100.

        A live stream reports total_frames=0 and therefore has no meaningful
        percentage; it reads 0 while running and 100 once finished, which is
        the honest answer for an unbounded source.
        """
        if self.state in (CameraState.COMPLETED, CameraState.SKIPPED):
            return 100.0
        if self.total_frames <= 0:
            return 0.0
        return min(100.0, 100.0 * self.frames_processed / self.total_frames)

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        data["percent"] = self.percent
        data["elapsed_seconds"] = self.elapsed_seconds
        return data


@dataclass
class SessionStatus:
    """Snapshot of a whole sequential run, as written to the status file."""

    session_id: str
    state: SessionState = SessionState.IDLE
    cameras: list[CameraProgress] = field(default_factory=list)
    current_camera_id: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    logs: list[str] = field(default_factory=list)

    @property
    def total_detections(self) -> int:
        return sum(camera.detections for camera in self.cameras)

    @property
    def elapsed_seconds(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    @property
    def percent(self) -> float:
        """Whole-session completion: the mean of the per-camera percentages.

        Every camera counts equally rather than weighting by frame count --
        the queue is what the operator is watching progress through, and an
        equal-weight bar moves predictably even when one clip is much longer
        than the others.
        """
        if not self.cameras:
            return 0.0
        return sum(camera.percent for camera in self.cameras) / len(self.cameras)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "cameras": [camera.to_dict() for camera in self.cameras],
            "current_camera_id": self.current_camera_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "logs": list(self.logs),
            "total_detections": self.total_detections,
            "elapsed_seconds": self.elapsed_seconds,
            "percent": self.percent,
        }
