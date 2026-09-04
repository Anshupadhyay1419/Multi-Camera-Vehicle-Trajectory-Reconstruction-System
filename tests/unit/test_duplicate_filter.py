"""
Unit + property-based tests for database.duplicate_filter.DuplicateFilter.

Property 17: Duplicate Filter Within-Window Suppression — Validates: Req 16.1, 16.2
Property 18: Duplicate Filter Outside-Window Pass-Through — Validates: Req 16.3
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.database.duplicate_filter import DuplicateFilter


class TestDuplicateFilterExamples:
    def test_new_plate_not_duplicate(self):
        df = DuplicateFilter(window_seconds=30)
        assert df.is_duplicate("KA19TR0234", track_id=1) is False

    def test_recorded_plate_is_duplicate_within_window(self):
        df = DuplicateFilter(window_seconds=30)
        t0 = 1000.0
        df.record("KA19TR0234", track_id=1, now=t0)
        assert df.is_duplicate("KA19TR0234", track_id=99, now=t0 + 10) is True

    def test_recorded_plate_not_duplicate_after_window(self):
        df = DuplicateFilter(window_seconds=30)
        t0 = 1000.0
        df.record("KA19TR0234", track_id=1, now=t0)
        assert df.is_duplicate("KA19TR0234", track_id=99, now=t0 + 31) is False

    def test_same_track_id_is_duplicate_within_window(self):
        df = DuplicateFilter(window_seconds=30)
        t0 = 1000.0
        df.record("PLATE_A", track_id=5, now=t0)
        assert df.is_duplicate("PLATE_B", track_id=5, now=t0 + 5) is True

    def test_different_plate_and_track_not_duplicate(self):
        # Genuinely different plates (edit distance well above
        # max_edit_distance), not a one-character OCR-confusion pair --
        # see test_near_miss_plate_is_treated_as_duplicate below for that
        # intentionally-fuzzy case.
        df = DuplicateFilter(window_seconds=30)
        t0 = 1000.0
        df.record("KA19TR0234", track_id=1, now=t0)
        assert df.is_duplicate("DL05AB9876", track_id=2, now=t0 + 5) is False

    def test_near_miss_plate_is_treated_as_duplicate(self):
        # A single-character difference (the kind of OCR confusion a
        # fragmented track produces re-reading the same physical plate,
        # e.g. O/C, O/0) is intentionally treated as the same plate within
        # the default max_edit_distance=1 -- this is the fuzzy-match
        # behavior duplicate_filter.py documents, not a false positive.
        df = DuplicateFilter(window_seconds=30)
        t0 = 1000.0
        df.record("DL7CD5017", track_id=1, now=t0)
        assert df.is_duplicate("DD7CD5017", track_id=2, now=t0 + 5) is True

    def test_cleanup_removes_expired_entries(self):
        df = DuplicateFilter(window_seconds=30)
        t0 = 1000.0
        df.record("PLATE_A", track_id=1, now=t0)
        df.cleanup(now=t0 + 60)
        assert df.is_duplicate("PLATE_A", track_id=1, now=t0 + 60) is False


# ---------------------------------------------------------------------------
# Property 17: Within-Window Suppression
# ---------------------------------------------------------------------------

class TestDuplicateFilterProperty17:
    """Property 17: Within-Window Suppression — Validates: Req 16.1, 16.2"""

    @given(
        window=st.integers(min_value=1, max_value=300),
        t1=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        delta=st.floats(min_value=0.0, max_value=299.9, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_within_window_always_duplicate(
        self, window: int, t1: float, delta: float
    ):
        """If T2 - T1 < window_seconds, is_duplicate returns True."""
        if delta >= window:
            return
        t2 = t1 + delta
        df = DuplicateFilter(window_seconds=window)
        df.record("PLATE_X", track_id=1, now=t1)
        assert df.is_duplicate("PLATE_X", track_id=99, now=t2) is True


# ---------------------------------------------------------------------------
# Property 18: Outside-Window Pass-Through
# ---------------------------------------------------------------------------

class TestDuplicateFilterProperty18:
    """Property 18: Outside-Window Pass-Through — Validates: Req 16.3"""

    @given(
        window=st.integers(min_value=1, max_value=300),
        t1=st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
        extra=st.floats(min_value=0.0, max_value=300.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_outside_window_not_duplicate(
        self, window: int, t1: float, extra: float
    ):
        """If T2 - T1 >= window_seconds, is_duplicate returns False."""
        t2 = t1 + window + extra
        df = DuplicateFilter(window_seconds=window)
        df.record("PLATE_Y", track_id=1, now=t1)
        assert df.is_duplicate("PLATE_Y", track_id=99, now=t2) is False
