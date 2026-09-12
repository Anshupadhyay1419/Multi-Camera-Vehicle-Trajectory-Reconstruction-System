"""
Value types for reconstructed vehicle trajectories.

A `Trajectory` is the answer to "where did this vehicle go?": an ordered
list of `TrajectoryPoint`s, one per camera visit, plus the summary figures
the dashboard puts above the map. Like the camera models, these are plain
dataclasses with no database or UI dependency, so the engine can be tested
against dicts and the API can serialise them without a translation layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    """Parse a stored ISO-8601 timestamp into an aware datetime, or None.

    Every write path stores `datetime.now(timezone.utc).isoformat()`, but
    rows can predate that convention or arrive from an import, so a naive
    string is assumed to be UTC rather than rejected -- consistent with
    db._event_local_date(). Returning None for an unparseable value keeps
    one malformed row from breaking a whole trajectory.
    """
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


@dataclass
class TrajectoryPoint:
    """One camera visit on a vehicle's path.

    Note `detection_count` and `all_timestamps`: when several detections of
    the same plate at the same camera are collapsed into one visit, this
    point represents the FIRST of them (when the vehicle actually reached
    the site) while still reporting how many reads backed it. The map draws
    one marker per visit; the history table can still show every read.
    """

    sequence: int                       # 1-based position along the path
    camera_id: str
    camera_name: str
    latitude: Optional[float]
    longitude: Optional[float]
    timestamp: Optional[str]            # ISO 8601, the first sighting here
    confidence: Optional[float] = None  # best confidence among the reads
    detection_count: int = 1
    plate_image_path: Optional[str] = None
    vehicle_image_path: Optional[str] = None
    vehicle_type: Optional[str] = None
    plate_color: Optional[str] = None
    direction: Optional[str] = None
    trajectory_order: Optional[int] = None
    video_source: Optional[str] = None
    ocr_text: Optional[str] = None
    all_timestamps: list[str] = field(default_factory=list)

    @property
    def has_location(self) -> bool:
        """True when this point can be plotted on the map."""
        return self.latitude is not None and self.longitude is not None

    @property
    def datetime(self) -> Optional[datetime]:
        return parse_timestamp(self.timestamp)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["has_location"] = self.has_location
        return data


@dataclass
class TrajectoryLeg:
    """The movement between two consecutive points.

    Computed rather than stored: distance comes from the two cameras'
    coordinates and duration from their timestamps, so a leg is always
    consistent with the points it joins. `speed_kmh` is None whenever either
    input is missing -- an unsurveyed camera or an unparseable timestamp --
    rather than silently reported as zero.
    """

    from_camera_id: str
    from_camera_name: str
    to_camera_id: str
    to_camera_name: str
    distance_km: Optional[float] = None
    duration_seconds: Optional[float] = None
    speed_kmh: Optional[float] = None
    bearing_degrees: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    """A vehicle's reconstructed path across the camera network."""

    plate_number: str
    points: list[TrajectoryPoint] = field(default_factory=list)
    legs: list[TrajectoryLeg] = field(default_factory=list)
    processing_session: Optional[str] = None
    ordering: str = "timestamp"          # which key actually decided the order
    total_detections: int = 0            # every read, before collapsing

    # ── summary ───────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        return not self.points

    @property
    def cameras_visited(self) -> int:
        return len({point.camera_id for point in self.points})

    @property
    def first_seen(self) -> Optional[str]:
        return self.points[0].timestamp if self.points else None

    @property
    def last_seen(self) -> Optional[str]:
        return self.points[-1].timestamp if self.points else None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Wall-clock time from the first sighting to the last."""
        if len(self.points) < 2:
            return None
        start, end = self.points[0].datetime, self.points[-1].datetime
        if start is None or end is None:
            return None
        return max(0.0, (end - start).total_seconds())

    @property
    def total_distance_km(self) -> Optional[float]:
        """Path length: the sum of every leg with a measurable distance.

        None (not 0.0) when no leg has one, so "we could not measure this"
        stays distinguishable from "the vehicle did not move".
        """
        measured = [leg.distance_km for leg in self.legs if leg.distance_km is not None]
        return round(sum(measured), 4) if measured else None

    @property
    def average_confidence(self) -> Optional[float]:
        values = [p.confidence for p in self.points if p.confidence is not None]
        return round(sum(values) / len(values), 4) if values else None

    @property
    def mappable_points(self) -> list[TrajectoryPoint]:
        """Only the points that carry coordinates, in path order."""
        return [point for point in self.points if point.has_location]

    @property
    def path_labels(self) -> list[str]:
        """The route as camera names, e.g. ['India Gate', 'Karol Bagh']."""
        return [point.camera_name for point in self.points]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plate_number": self.plate_number,
            "processing_session": self.processing_session,
            "ordering": self.ordering,
            "points": [point.to_dict() for point in self.points],
            "legs": [leg.to_dict() for leg in self.legs],
            "cameras_visited": self.cameras_visited,
            "total_detections": self.total_detections,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_seconds": self.duration_seconds,
            "total_distance_km": self.total_distance_km,
            "average_confidence": self.average_confidence,
            "path_labels": self.path_labels,
        }
