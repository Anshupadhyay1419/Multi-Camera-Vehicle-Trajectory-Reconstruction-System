"""
An OCR engine that exists must be an engine that works.

The PARSeq TensorRT engine used to record a failed load in a private flag
and return itself anyway. It satisfied the OCREngine interface in form only:
recognize() then returned ("", 0.0) for every frame, which the pipeline
cannot tell apart from "no plate in this crop". A real four-video session
stored zero plates and logged ocr_inference_ms(avg=0.00) -- i.e. the dead
engine reported as a fast one.

These tests run anywhere: the failure path under test is a missing engine
file, which needs no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ocr.base import OCREngine
from src.runtime.errors import ModelUnavailableError


class TestConstructionFailsLoudly:
    def test_a_missing_engine_file_raises(self, tmp_path):
        from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine

        with pytest.raises(ModelUnavailableError):
            PARSeqTensorRTOCREngine(engine_path=str(tmp_path / "absent.engine"))

    def test_the_error_names_the_path(self, tmp_path):
        from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine

        missing = tmp_path / "absent.engine"
        with pytest.raises(ModelUnavailableError) as excinfo:
            PARSeqTensorRTOCREngine(engine_path=str(missing))
        assert str(missing) in str(excinfo.value)

    def test_no_dead_engine_escapes_the_constructor(self, tmp_path):
        """The regression itself: construction used to SUCCEED here."""
        from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine

        engine = None
        try:
            engine = PARSeqTensorRTOCREngine(engine_path=str(tmp_path / "absent.engine"))
        except ModelUnavailableError:
            pass
        assert engine is None, (
            "a failed load handed back a usable-looking engine; it would "
            "return ('', 0.0) for every frame of the session"
        )


class TestLivenessIsObservable:
    def test_the_base_contract_reports_available_by_default(self):
        """Correct for any backend that cannot collapse after construction."""

        class PurePythonEngine(OCREngine):
            def recognize(self, image: np.ndarray) -> tuple[str, float]:
                return "", 0.0

        assert PurePythonEngine().is_available is True

    def test_a_gpu_backend_overrides_it(self):
        """So "read nothing" stays distinguishable from "cannot read"."""
        from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine

        assert "is_available" in vars(PARSeqTensorRTOCREngine)

    def test_an_engine_that_died_mid_run_reports_unavailable(self, tmp_path):
        """recognize() still degrades rather than raising -- a live session
        must survive one bad frame -- so the flag is how anyone finds out."""
        from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine

        engine = PARSeqTensorRTOCREngine.__new__(PARSeqTensorRTOCREngine)
        engine._init_failed = True
        assert engine.is_available is False
