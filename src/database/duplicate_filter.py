"""
Duplicate event filter for the ALPR University Gate system.

Suppresses repeated vehicle events for the same plate number or tracking ID
within a configurable time window (default 30 seconds).
"""

from __future__ import annotations

import time
from typing import Optional

# Reuse the existing edit-distance implementation (src.validation.ocr_fusion
# already has one for its own similarity-aware merge) instead of adding a
# third hand-rolled copy to the codebase.
from src.validation.ocr_fusion import _levenshtein


class DuplicateFilter:
    """Suppress duplicate vehicle events within a time window.

    Args:
        window_seconds:   Events for the same plate/track within this window
                          are considered duplicates (default 30).
        max_edit_distance: Two plate strings within this Levenshtein distance
                          are treated as the same plate. A single-character
                          OCR confusion (O/C, O/0, ...) between two track
                          fragments of the same physical vehicle otherwise
                          slips past exact-match dedup and gets stored twice.
    """

    def __init__(self, window_seconds: int = 30, max_edit_distance: int = 1) -> None:
        self.window_seconds = window_seconds
        self.max_edit_distance = max_edit_distance
        # plate_number → last recorded timestamp
        self._plate_times: dict[str, float] = {}
        # track_id → last recorded timestamp
        self._track_times: dict[int, float] = {}

    @classmethod
    def from_config(cls, config: dict) -> "DuplicateFilter":
        """Construct from the full config dict."""
        dedup_cfg = config.get("deduplication", {})
        return cls(
            window_seconds=int(dedup_cfg.get("window_seconds", 30)),
            max_edit_distance=int(dedup_cfg.get("max_edit_distance", 1)),
        )

    def is_duplicate(
        self,
        plate_number: str,
        track_id: int,
        now: Optional[float] = None,
    ) -> bool:
        """Check whether this event is a duplicate.

        Args:
            plate_number: Validated plate string.
            track_id:     Integer tracking ID.
            now:          Current timestamp (defaults to time.time()).

        Returns:
            True if the same plate (exact or within max_edit_distance) or
            track_id was recorded within the window.
        """
        t = now if now is not None else time.time()

        # Prune expired entries first. Nothing else in the pipeline calls
        # cleanup() periodically, so without this both _plate_times and
        # _track_times would otherwise grow for the life of a long-running
        # (24/7) process, and the fuzzy-match loop below would scan an
        # ever-growing number of already-expired entries on every call.
        self.cleanup(now=t)

        # Check plate — exact match first (cheap), then fuzzy match against
        # every recently-seen plate so a fragmented track that re-reads the
        # same physical plate with one OCR-confused character doesn't get
        # stored as a second event.
        for seen_plate, last_seen in self._plate_times.items():
            if (t - last_seen) >= self.window_seconds:
                continue
            if seen_plate == plate_number:
                return True
            if len(seen_plate) == len(plate_number) and _levenshtein(seen_plate, plate_number) <= self.max_edit_distance:
                return True

        # Check track_id
        last_track = self._track_times.get(track_id)
        if last_track is not None and (t - last_track) < self.window_seconds:
            return True

        return False

    def record(
        self,
        plate_number: str,
        track_id: int,
        now: Optional[float] = None,
    ) -> None:
        """Record a vehicle event to enable future duplicate detection.

        Args:
            plate_number: Validated plate string.
            track_id:     Integer tracking ID.
            now:          Current timestamp (defaults to time.time()).
        """
        t = now if now is not None else time.time()
        self._plate_times[plate_number] = t
        self._track_times[track_id] = t

    def cleanup(self, now: Optional[float] = None) -> None:
        """Remove expired entries to prevent unbounded memory growth."""
        t = now if now is not None else time.time()
        cutoff = t - self.window_seconds

        self._plate_times = {
            k: v for k, v in self._plate_times.items() if v >= cutoff
        }
        self._track_times = {
            k: v for k, v in self._track_times.items() if v >= cutoff
        }
