"""
Vehicle tracker for the ALPR University Gate system.

Wraps supervision's ByteTrack implementation to assign and maintain
consistent integer tracking IDs across frames for each detected vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.detection.vehicle_detector import Detection
from src.utils.logger import get_logger

_logger = get_logger("tracking.vehicle_tracker")


@dataclass
class Track:
    """A tracked vehicle with a consistent ID across frames."""
    track_id: int
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2
    class_label: str
    centroid: tuple[float, float]     # (cx, cy) in pixel coords


class VehicleTracker:
    """Assign and maintain consistent tracking IDs using ByteTrack.

    Args:
        lost_track_timeout:       Number of frames before a lost track is retired.
        minimum_matching_threshold: IOU threshold ByteTrack requires to keep
            associating a detection with an existing track. The supervision
            default (0.8) assumes small inter-frame motion; with frame_skip
            > 1 a vehicle's box can move enough between processed frames to
            drop under 0.8 IOU, which fragments one physical vehicle into
            several track IDs (each restarting OCR fusion from zero votes,
            so plates and cars go unstored). Lowering it tolerates that
            larger jump.
    """

    def __init__(
        self,
        lost_track_timeout: int = 30,
        minimum_matching_threshold: float = 0.5,
    ) -> None:
        self.lost_track_timeout = lost_track_timeout
        self.minimum_matching_threshold = minimum_matching_threshold
        self._tracker = None
        self._track_labels: dict[int, str] = {}  # track_id → class_label

    def _load_tracker(self) -> None:
        """Lazy-load ByteTrack on first use."""
        if self._tracker is not None:
            return
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack(
                lost_track_buffer=self.lost_track_timeout,
                minimum_matching_threshold=self.minimum_matching_threshold,
            )
            _logger.info(
                "ByteTrack tracker initialized (lost_track_timeout=%d, minimum_matching_threshold=%.2f)",
                self.lost_track_timeout, self.minimum_matching_threshold,
            )
        except Exception as exc:
            _logger.error("Failed to initialize ByteTrack: %s", exc)
            raise

    def update(
        self, detections: list[Detection], frame: np.ndarray
    ) -> list[Track]:
        """Update tracker with new detections and return active tracks.

        Args:
            detections: Vehicle detections from VehicleDetector for this frame.
            frame:      Current BGR frame (used for frame dimensions).

        Returns:
            List of active Track objects with consistent IDs.
        """
        self._load_tracker()

        if not detections:
            # Still update tracker with empty detections to age out lost tracks
            try:
                import supervision as sv
                empty = sv.Detections.empty()
                self._tracker.update_with_detections(empty)
            except Exception:
                pass
            return []

        try:
            import supervision as sv

            # Build supervision Detections object
            xyxy = np.array([[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]]
                              for d in detections], dtype=np.float32)
            confidence = np.array([d.confidence for d in detections], dtype=np.float32)
            class_id = np.zeros(len(detections), dtype=int)  # all same class for tracking

            sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id,
            )

            tracked = self._tracker.update_with_detections(sv_detections)

        except Exception as exc:
            _logger.warning("ByteTrack update failed: %s", exc)
            return []

        tracks: list[Track] = []
        for i in range(len(tracked)):
            try:
                track_id = int(tracked.tracker_id[i])
                x1, y1, x2, y2 = (int(tracked.xyxy[i][0]), int(tracked.xyxy[i][1]),
                                   int(tracked.xyxy[i][2]), int(tracked.xyxy[i][3]))

                # Map back to class label from nearest detection
                label = self._match_label(detections, (x1, y1, x2, y2))
                self._track_labels[track_id] = label

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                tracks.append(Track(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_label=label,
                    centroid=(cx, cy),
                ))
            except Exception as exc:
                _logger.warning("Error processing track %d: %s", i, exc)
                continue

        return tracks

    @staticmethod
    def _match_label(
        detections: list[Detection],
        bbox: tuple[int, int, int, int],
    ) -> str:
        """Find the class label of the detection closest to the tracked bbox."""
        if not detections:
            return "vehicle"

        tx1, ty1, tx2, ty2 = bbox
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0

        best_label = detections[0].class_label
        best_dist = float("inf")

        for det in detections:
            dx1, dy1, dx2, dy2 = det.bbox
            dcx = (dx1 + dx2) / 2.0
            dcy = (dy1 + dy2) / 2.0
            dist = (tcx - dcx) ** 2 + (tcy - dcy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_label = det.class_label

        return best_label
