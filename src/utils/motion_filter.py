"""
Motion filter for the ALPR University Gate system.

Tracks centroid movement per vehicle across frames.
Only vehicles that have moved more than a minimum pixel distance
are considered "moving" and eligible for database entry.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional


class MotionFilter:
    """Determine if a tracked vehicle is actually moving.

    Args:
        min_displacement_px: Minimum pixel movement to consider a vehicle moving.
        history_frames:      Number of frames to track centroid history.
        enabled:             When False, is_moving() always returns True --
                             i.e. every tracked vehicle is treated as moving.
                             Useful for testing against a stationary target
                             (e.g. a video played on a phone/laptop screen
                             held in front of the camera), where a real
                             vehicle's spatial movement never happens.
    """

    def __init__(
        self,
        min_displacement_px: float = 15.0,
        history_frames: int = 10,
        enabled: bool = True,
    ) -> None:
        self.min_displacement_px = min_displacement_px
        self.history_frames = history_frames
        self.enabled = enabled
        # track_id → deque of (cx, cy) centroids
        self._history: dict[int, deque[tuple[float, float]]] = {}

    @classmethod
    def from_config(cls, config: dict) -> "MotionFilter":
        """Construct from the full config dict."""
        motion_cfg = config.get("motion_filter", {})
        return cls(
            min_displacement_px=float(motion_cfg.get("min_displacement_px", 15.0)),
            history_frames=int(motion_cfg.get("history_frames", 10)),
            enabled=bool(motion_cfg.get("enabled", True)),
        )

    def update(self, track_id: int, centroid: tuple[float, float]) -> None:
        """Record a new centroid observation for a track."""
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.history_frames)
        self._history[track_id].append(centroid)

    def is_moving(self, track_id: int) -> bool:
        """Return True if the vehicle has moved enough to be considered moving."""
        if not self.enabled:
            return True
        history = self._history.get(track_id)
        if not history or len(history) < 2:
            return True

        points = list(history)
        first = points[0]
        last = points[-1]
        displacement = math.sqrt(
            (last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2
        )

        return displacement >= self.min_displacement_px

    def remove_track(self, track_id: int) -> None:
        """Clean up history for a retired track."""
        self._history.pop(track_id, None)

    def get_displacement(self, track_id: int) -> float:
        """Return the total displacement for a track (for debugging)."""
        history = self._history.get(track_id)
        if not history or len(history) < 2:
            return 0.0
        points = list(history)
        first, last = points[0], points[-1]
        return math.sqrt((last[0] - first[0]) ** 2 + (last[1] - first[1]) ** 2)
