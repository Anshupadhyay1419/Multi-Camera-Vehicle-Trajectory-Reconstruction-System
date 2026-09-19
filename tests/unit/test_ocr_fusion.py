"""
Unit + property-based tests for validation.ocr_fusion.OCRFusion.

Properties tested:
  Property 11: OCR Fusion Majority Voting
  Property 12: OCR Fusion No-Majority Fallback
  Property 13: OCR Fusion Buffer Size Invariant
  Property 14: OCR Fusion Flush Clears Buffer
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.validation.ocr_fusion import OCRFusion


# ---------------------------------------------------------------------------
# Example-based tests
# ---------------------------------------------------------------------------

class TestOCRFusionBasic:
    def test_empty_buffer_returns_none(self):
        fusion = OCRFusion(window_size=5)
        assert fusion.get_result(track_id=1) is None

    def test_single_result_returned(self):
        fusion = OCRFusion(window_size=5)
        fusion.add_result(1, "KA19TR0234", 0.9)
        result = fusion.get_result(1)
        assert result is not None
        assert result[0] == "KA19TR0234"

    def test_majority_wins(self):
        fusion = OCRFusion(window_size=7)
        for _ in range(4):
            fusion.add_result(1, "KA19TR0234", 0.8)
        for _ in range(3):
            fusion.add_result(1, "KA19TR0235", 0.9)
        result = fusion.get_result(1)
        assert result[0] == "KA19TR0234"

    def test_no_majority_returns_highest_confidence(self):
        fusion = OCRFusion(window_size=4)
        fusion.add_result(1, "PLATE_A", 0.5)
        fusion.add_result(1, "PLATE_B", 0.95)
        fusion.add_result(1, "PLATE_C", 0.3)
        fusion.add_result(1, "PLATE_D", 0.4)
        result = fusion.get_result(1)
        assert result[0] == "PLATE_B"

    def test_flush_returns_result_and_clears(self):
        fusion = OCRFusion(window_size=5)
        fusion.add_result(1, "KA19TR0234", 0.9)
        result = fusion.flush(1)
        assert result is not None
        assert result[0] == "KA19TR0234"
        # Buffer should be cleared
        assert fusion.get_result(1) is None

    def test_flush_nonexistent_track_returns_none(self):
        fusion = OCRFusion(window_size=5)
        assert fusion.flush(999) is None

    def test_flush_all_clears_all_buffers(self):
        fusion = OCRFusion(window_size=5)
        fusion.add_result(1, "PLATE_A", 0.9)
        fusion.add_result(2, "PLATE_B", 0.8)
        results = fusion.flush_all()
        assert 1 in results
        assert 2 in results
        assert fusion.get_result(1) is None
        assert fusion.get_result(2) is None

    def test_buffer_does_not_exceed_window_size(self):
        window = 5
        fusion = OCRFusion(window_size=window)
        for i in range(20):
            fusion.add_result(1, f"PLATE_{i}", 0.5)
        assert len(fusion._buffers[1]) == window


# ---------------------------------------------------------------------------
# Property 11: OCR Fusion Majority Voting
# ---------------------------------------------------------------------------

class TestOCRFusionProperty11:
    """Property 11: Majority Voting — Validates: Req 13.2"""

    @given(
        majority_plate=st.text(min_size=1, max_size=12),
        majority_count=st.integers(min_value=1, max_value=10),
        other_count=st.integers(min_value=0, max_value=9),
    )
    @settings(max_examples=30)
    def test_majority_plate_always_wins(
        self, majority_plate: str, majority_count: int, other_count: int
    ):
        """If one plate appears > 50% of the time, it must be returned."""
        total = majority_count + other_count
        if majority_count <= total / 2:
            return  # not a majority — skip

        fusion = OCRFusion(window_size=total + 1)
        for _ in range(majority_count):
            fusion.add_result(1, majority_plate, 0.8)
        for i in range(other_count):
            fusion.add_result(1, f"OTHER_{i}", 0.7)

        result = fusion.get_result(1)
        assert result is not None
        assert result[0] == majority_plate


# ---------------------------------------------------------------------------
# Property 12: OCR Fusion No-Majority Fallback
# ---------------------------------------------------------------------------

class TestOCRFusionProperty12:
    """Property 12: No-Majority Fallback — Validates: Req 13.3"""

    @given(
        entries=st.lists(
            st.tuples(
                st.text(min_size=1, max_size=8),
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
            ),
            min_size=2,
            max_size=10,
        )
    )
    @settings(max_examples=30)
    def test_no_majority_returns_confidence_of_the_plate_it_returns(self, entries):
        """With no majority, the reported confidence describes the fused plate.

        Fusion scores *clusters* of similar reads, so the winner is not always
        the single highest-confidence entry. What must always hold is that the
        confidence returned alongside a plate came from reads OF THAT PLATE --
        otherwise a losing candidate donates its score and the stored event
        claims a certainty no read of it ever had.
        """
        from collections import Counter
        plates = [e[0] for e in entries]
        counter = Counter(plates)
        most_common, count = counter.most_common(1)[0]
        if count > len(entries) / 2:
            return  # majority exists — skip

        fusion = OCRFusion(window_size=len(entries) + 1)
        for plate, conf in entries:
            fusion.add_result(1, plate, conf)

        result = fusion.get_result(1)
        assert result is not None
        fused_plate, fused_conf = result
        own_confs = [c for p, c in entries if p == fused_plate]
        assert own_confs, f"fused plate {fused_plate!r} was never read"
        assert abs(fused_conf - max(own_confs)) < 1e-9


# ---------------------------------------------------------------------------
# Property 13: Buffer Size Invariant
# ---------------------------------------------------------------------------

class TestOCRFusionProperty13:
    """Property 13: Buffer Size Invariant — Validates: Req 13.1"""

    @given(
        window_size=st.integers(min_value=1, max_value=20),
        n_additions=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=30)
    def test_buffer_never_exceeds_window_size(self, window_size: int, n_additions: int):
        """Buffer for any track_id never exceeds window_size."""
        fusion = OCRFusion(window_size=window_size)
        for i in range(n_additions):
            fusion.add_result(1, f"PLATE_{i % 5}", 0.8)
            assert len(fusion._buffers.get(1, [])) <= window_size


# ---------------------------------------------------------------------------
# Property 14: Flush Clears Buffer
# ---------------------------------------------------------------------------

class TestOCRFusionProperty14:
    """Property 14: Flush Clears Buffer — Validates: Req 13.4"""

    @given(
        n_entries=st.integers(min_value=1, max_value=15),
        window_size=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30)
    def test_flush_always_clears_buffer(self, n_entries: int, window_size: int):
        """After flush(track_id), the buffer for that track_id is empty."""
        fusion = OCRFusion(window_size=window_size)
        for i in range(n_entries):
            fusion.add_result(1, f"PLATE_{i}", 0.8)
        fusion.flush(1)
        assert len(fusion._buffers.get(1, [])) == 0
