"""
RapidOCR's confidence must be the recognition score, not a stopwatch.

Different RapidOCR builds return their timings differently: some as a bare
float, some as a list of per-stage floats. The shape dispatch used to decide
"is this (txts, scores)?" by asking whether the second element was a list --
so the `(detections, [t_det, t_cls, t_rec])` build had every detection paired
with an elapsed time instead of its score. A plate recognised at 0.98 was
reported at 0.30, which falls under both ocr.confidence_threshold (0.70) and
fusion.min_confidence (0.55), so every read was silently discarded.

That path is not exotic: it is what any machine without CUDA now falls back
to, by design (see src/ocr/__init__.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ocr.rapidocr_engine import RapidOCREngine, _is_detection_list

BOX = [[0, 0], [10, 0], [10, 5], [0, 5]]
BLANK = np.zeros((32, 128, 3), dtype=np.uint8)


def engine_returning(payload) -> RapidOCREngine:
    """A RapidOCREngine whose backend returns *payload*, with no model load."""
    engine = RapidOCREngine.__new__(RapidOCREngine)
    engine._engine = lambda image, **kwargs: payload
    engine._init_failed = False
    engine.text_score = 0.5
    engine.use_det = True
    engine.use_cls = True
    engine.use_rec = True
    return engine


class TestTimingsAreNeverMistakenForScores:
    def test_a_list_of_timings_does_not_become_the_confidence(self):
        """The regression: this build reported 0.30 for a 0.98 read."""
        payload = ([[BOX, "HR26CQ6869", 0.9795]], [0.147, 0.0015, 0.022])
        text, confidence = engine_returning(payload).recognize(BLANK)
        assert text == "HR26CQ6869"
        assert confidence == pytest.approx(0.9795, abs=1e-4)

    def test_a_scalar_timing_still_works(self):
        """The shape the previous code did handle must keep working."""
        payload = ([[BOX, "KA19TR0234", 0.87]], 0.123)
        text, confidence = engine_returning(payload).recognize(BLANK)
        assert text == "KA19TR0234"
        assert confidence == pytest.approx(0.87, abs=1e-6)

    def test_the_confidence_clears_the_configured_thresholds(self):
        """The point of the fix: a good read must survive the gates.

        ocr.confidence_threshold is 0.70 and fusion.min_confidence is 0.55;
        a correct read scored as a timing cleared neither.
        """
        payload = ([[BOX, "DL7CD5017", 0.9512]], [0.14, 0.001, 0.02])
        _, confidence = engine_returning(payload).recognize(BLANK)
        assert confidence > 0.70

    def test_several_regions_average_their_scores_not_their_timings(self):
        payload = (
            [[BOX, "IND", 0.90], [BOX, "DL3CBJ1384", 0.98]],
            [0.1, 0.002, 0.03],
        )
        text, confidence = engine_returning(payload).recognize(BLANK)
        assert text == "INDDL3CBJ1384"
        assert confidence == pytest.approx(0.94, abs=1e-4)


class TestOtherShapesStillDispatch:
    def test_boxes_txts_scores(self):
        payload = ([BOX], ["MH12AB1234"], [0.91])
        text, confidence = engine_returning(payload).recognize(BLANK)
        assert text == "MH12AB1234"
        assert confidence == pytest.approx(0.91, abs=1e-6)

    def test_txts_and_scores(self):
        payload = (["MH12AB1234"], [0.77])
        text, confidence = engine_returning(payload).recognize(BLANK)
        assert text == "MH12AB1234"
        assert confidence == pytest.approx(0.77, abs=1e-6)

    def test_none_means_nothing_detected(self):
        assert engine_returning(None).recognize(BLANK) == ("", 0.0)

    def test_an_empty_detection_list_reads_nothing(self):
        assert engine_returning(([], [0.1, 0.002, 0.03])).recognize(BLANK) == ("", 0.0)


class TestShapeDiscriminator:
    """_is_detection_list is the discriminator the dispatch turns on."""

    def test_it_recognises_a_detection_list(self):
        assert _is_detection_list([[BOX, "ABC123", 0.9]]) is True

    @pytest.mark.parametrize(
        "value",
        [
            [0.147, 0.0015, 0.022],      # a timings list
            ["ABC123"],                  # a txts list
            [],                          # empty
            None,                        # not a sequence
            [[BOX, "ABC123"]],           # too short to carry a score
            [[BOX, 123, 0.9]],           # element 1 is not text
        ],
    )
    def test_it_rejects_everything_else(self, value):
        assert _is_detection_list(value) is False
