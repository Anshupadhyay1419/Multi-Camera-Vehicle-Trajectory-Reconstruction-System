"""
Trajectory reconstruction: turn a plate's scattered detections into a path.

    detections (dicts)
        -> group per camera visit      (collapse repeat reads at one site)
        -> order                       (timestamp / camera queue position)
        -> legs                        (distance, duration, speed, bearing)
        -> Trajectory

The engine takes and returns plain data -- it is handed a list of event
dicts and never touches the database itself. That keeps the ordering rules
(the part with the real subtlety) testable without a schema, and lets the
same engine serve the API, the dashboard and a future batch export.

Nothing about the route is hardcoded. A point's position comes from the
coordinates stored on its own event, so re-siting a camera changes the map
for events recorded afterwards and leaves history where it happened.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

from src.trajectory.models import (
    Trajectory,
    TrajectoryLeg,
    TrajectoryPoint,
    parse_timestamp,
)
from src.utils.logger import get_logger

_logger = get_logger("trajectory.engine")

# Mean Earth radius (km), IUGG. Haversine on a sphere is accurate to ~0.5%
# at these distances -- far better than the positional uncertainty of a
# hand-entered camera coordinate, so an ellipsoidal formula would be false
# precision here.
_EARTH_RADIUS_KM = 6371.0088

# Ordering strategies accepted by `order_by`.
ORDER_AUTO = "auto"
ORDER_TIMESTAMP = "timestamp"
ORDER_TRAJECTORY = "trajectory_order"
_VALID_ORDERINGS = (ORDER_AUTO, ORDER_TIMESTAMP, ORDER_TRAJECTORY)


def haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance between two WGS84 points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing_degrees(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Compass bearing from point 1 to point 2, 0-360 degrees clockwise from north.

    Used to rotate the direction arrows drawn along the map polyline, so a
    viewer can see which way the vehicle travelled, not just where it went.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


class TrajectoryEngine:
    """Reconstructs vehicle paths from stored detections.

    Args:
        order_by: Which key decides the sequence of points.
            "timestamp"        -- strictly chronological. The right answer
                                  for live cameras, where clocks are the
                                  only shared reference.
            "trajectory_order" -- the capturing camera's queue position.
                                  Deterministic for the demo, where four
                                  cameras replay footage rather than
                                  observing one real journey.
            "auto" (default)   -- timestamp when every point has a usable
                                  one, else camera order. Degrades instead
                                  of producing a nonsense sequence.
        collapse_per_camera: Merge repeated reads of one plate at one camera
            into a single visit. On by default: the map should show four
            markers for a four-camera journey, not forty.
        revisit_gap_seconds: With collapsing on, two reads at the same camera
            further apart than this are treated as two separate VISITS
            rather than one. Without it, a vehicle that genuinely returns to
            a site hours later would fold into its first visit and the
            return trip would vanish from the path.
    """

    def __init__(
        self,
        order_by: str = ORDER_AUTO,
        collapse_per_camera: bool = True,
        revisit_gap_seconds: float = 300.0,
    ) -> None:
        if order_by not in _VALID_ORDERINGS:
            _logger.warning(
                "Unknown order_by %r -- falling back to %r. Valid: %s",
                order_by, ORDER_AUTO, ", ".join(_VALID_ORDERINGS),
            )
            order_by = ORDER_AUTO
        self.order_by = order_by
        self.collapse_per_camera = collapse_per_camera
        self.revisit_gap_seconds = max(0.0, float(revisit_gap_seconds))

    @classmethod
    def from_registry(cls, registry) -> "TrajectoryEngine":
        """Build from a CameraRegistry's `trajectory:` settings block."""
        settings = registry.trajectory
        return cls(
            order_by=str(settings.get("order_by", ORDER_AUTO)),
            collapse_per_camera=bool(settings.get("collapse_per_camera", True)),
            revisit_gap_seconds=float(settings.get("revisit_gap_seconds", 300)),
        )

    # ── public API ────────────────────────────────────────────────────────

    def build(
        self,
        plate_number: str,
        detections: Sequence[dict[str, Any]],
        processing_session: Optional[str] = None,
    ) -> Trajectory:
        """Reconstruct one vehicle's path.

        Args:
            plate_number:       The plate these detections belong to.
            detections:         Event dicts as returned by
                                db.get_plate_detections(). Any order; this
                                method sorts them.
            processing_session: Recorded on the result for provenance. The
                                caller is responsible for having filtered
                                the detections to that session -- passing it
                                here does not filter again.

        Returns:
            A Trajectory. Empty (is_empty) when there is nothing to plot,
            which is a normal answer for an unknown plate, not an error.
        """
        plate_number = (plate_number or "").strip().upper()
        usable = [d for d in detections if d]
        if not usable:
            _logger.info("No detections for plate %s -- empty trajectory", plate_number)
            return Trajectory(
                plate_number=plate_number,
                processing_session=processing_session,
                ordering=self.order_by,
            )

        visits = self._group_into_visits(usable)
        ordering = self._resolve_ordering(visits)
        visits.sort(key=self._sort_key(ordering))

        points = [
            self._to_point(sequence, visit)
            for sequence, visit in enumerate(visits, start=1)
        ]
        legs = self._build_legs(points)

        trajectory = Trajectory(
            plate_number=plate_number,
            points=points,
            legs=legs,
            processing_session=processing_session,
            ordering=ordering,
            total_detections=len(usable),
        )
        _logger.info(
            "Trajectory for %s: %d detection(s) -> %d point(s) across %d camera(s), "
            "ordered by %s",
            plate_number, len(usable), len(points),
            trajectory.cameras_visited, ordering,
        )
        return trajectory

    # ── grouping ──────────────────────────────────────────────────────────

    def _group_into_visits(self, detections: Iterable[dict]) -> list[list[dict]]:
        """Bucket detections into visits -- one bucket per marker on the map.

        With collapsing off, every detection is its own visit. With it on,
        consecutive detections at the same camera merge unless the gap
        between them exceeds revisit_gap_seconds.

        Merging is done on a time-sorted list, so a genuine A->B->A journey
        produces three visits (two of them at A) rather than two -- grouping
        by camera_id alone would silently delete the return leg.
        """
        ordered = sorted(detections, key=self._detection_time_key)
        if not self.collapse_per_camera:
            return [[detection] for detection in ordered]

        visits: list[list[dict]] = []
        for detection in ordered:
            if visits and self._continues_visit(visits[-1], detection):
                visits[-1].append(detection)
            else:
                visits.append([detection])
        return visits

    def _continues_visit(self, visit: list[dict], detection: dict) -> bool:
        """True if *detection* belongs to the visit currently being built."""
        previous = visit[-1]
        if previous.get("camera_id") != detection.get("camera_id"):
            return False

        previous_time = parse_timestamp(previous.get("timestamp"))
        current_time = parse_timestamp(detection.get("timestamp"))
        if previous_time is None or current_time is None:
            # No usable clock: fall back to "same camera, same visit". The
            # alternative -- splitting on every unparseable timestamp --
            # would scatter one site across several markers.
            return True
        return abs((current_time - previous_time).total_seconds()) <= self.revisit_gap_seconds

    # ── ordering ──────────────────────────────────────────────────────────

    @staticmethod
    def _detection_time_key(detection: dict) -> tuple[int, float]:
        """Sort key that puts parseable timestamps first, in order.

        The leading flag keeps rows with no usable timestamp at the end
        instead of letting them collide at epoch zero and reorder the
        good rows around them.
        """
        parsed = parse_timestamp(detection.get("timestamp"))
        if parsed is None:
            return (1, 0.0)
        return (0, parsed.timestamp())

    def _resolve_ordering(self, visits: list[list[dict]]) -> str:
        """Decide which key actually orders this trajectory.

        "auto" prefers timestamps and falls back to camera queue position
        only when at least one visit has no usable timestamp -- mixing the
        two would order some points by clock and others by config, which is
        not a sequence anyone can reason about.
        """
        if self.order_by != ORDER_AUTO:
            return self.order_by

        every_timestamp = all(
            parse_timestamp(visit[0].get("timestamp")) is not None for visit in visits
        )
        if every_timestamp:
            return ORDER_TIMESTAMP

        every_order = all(
            visit[0].get("trajectory_order") is not None for visit in visits
        )
        if every_order:
            _logger.info(
                "Some detections have no usable timestamp -- ordering by "
                "camera queue position instead"
            )
            return ORDER_TRAJECTORY

        _logger.warning(
            "Detections have neither complete timestamps nor complete camera "
            "ordering; the reconstructed sequence may not reflect the real journey"
        )
        return ORDER_TIMESTAMP

    def _sort_key(self, ordering: str):
        """Return the sort key for the chosen ordering.

        Both strategies fall back to the other as a tiebreaker, so two
        cameras that share a queue position still come out in clock order,
        and two detections sharing a timestamp still come out in queue order.
        """
        if ordering == ORDER_TRAJECTORY:
            def key(visit: list[dict]) -> tuple:
                head = visit[0]
                order = head.get("trajectory_order")
                return (
                    0 if order is not None else 1,
                    order if order is not None else 0,
                    self._detection_time_key(head),
                )
            return key

        def key(visit: list[dict]) -> tuple:
            head = visit[0]
            order = head.get("trajectory_order")
            return (
                self._detection_time_key(head),
                order if order is not None else 0,
            )
        return key

    # ── point + leg construction ──────────────────────────────────────────

    @staticmethod
    def _to_point(sequence: int, visit: list[dict]) -> TrajectoryPoint:
        """Fold one visit's detections into a single trajectory point.

        The visit's FIRST detection supplies time and position (when the
        vehicle reached the site), but the images and confidence come from
        its BEST-confidence detection -- the clearest read is the one worth
        showing in a popup, and it is rarely the first frame the vehicle
        appeared in.
        """
        first = visit[0]

        def confidence_of(detection: dict) -> float:
            value = detection.get("confidence")
            return float(value) if value is not None else -1.0

        best = max(visit, key=confidence_of)
        best_confidence = best.get("confidence")

        return TrajectoryPoint(
            sequence=sequence,
            camera_id=first.get("camera_id") or "UNKNOWN",
            camera_name=first.get("camera_name") or first.get("camera_id") or "Unknown camera",
            latitude=first.get("latitude"),
            longitude=first.get("longitude"),
            timestamp=first.get("timestamp"),
            confidence=float(best_confidence) if best_confidence is not None else None,
            detection_count=len(visit),
            plate_image_path=best.get("image_path") or first.get("image_path"),
            vehicle_image_path=best.get("vehicle_image_path") or first.get("vehicle_image_path"),
            vehicle_type=first.get("vehicle_type"),
            plate_color=first.get("plate_color"),
            direction=first.get("direction"),
            trajectory_order=first.get("trajectory_order"),
            video_source=first.get("video_source"),
            ocr_text=best.get("ocr_text") or first.get("ocr_text"),
            all_timestamps=[d.get("timestamp") for d in visit if d.get("timestamp")],
        )

    @staticmethod
    def _build_legs(points: Sequence[TrajectoryPoint]) -> list[TrajectoryLeg]:
        """Measure the movement between each consecutive pair of points.

        Every measurement is independently optional: a pair of surveyed
        cameras yields a distance even if a timestamp is missing, and a pair
        with timestamps yields a duration even if one camera has no
        coordinates. Speed needs both, plus a non-zero duration -- two reads
        in the same second would otherwise divide by zero and report an
        infinite speed.
        """
        legs: list[TrajectoryLeg] = []
        for start, end in zip(points, points[1:]):
            distance = bearing = duration = speed = None

            if start.has_location and end.has_location:
                distance = round(
                    haversine_km(start.latitude, start.longitude,
                                 end.latitude, end.longitude), 4
                )
                bearing = round(
                    initial_bearing_degrees(start.latitude, start.longitude,
                                            end.latitude, end.longitude), 1
                )

            start_time, end_time = start.datetime, end.datetime
            if start_time is not None and end_time is not None:
                duration = round((end_time - start_time).total_seconds(), 3)

            if distance is not None and duration is not None and duration > 0:
                speed = round(distance / (duration / 3600.0), 2)

            legs.append(
                TrajectoryLeg(
                    from_camera_id=start.camera_id,
                    from_camera_name=start.camera_name,
                    to_camera_id=end.camera_id,
                    to_camera_name=end.camera_name,
                    distance_km=distance,
                    duration_seconds=duration,
                    speed_kmh=speed,
                    bearing_degrees=bearing,
                )
            )
        return legs
