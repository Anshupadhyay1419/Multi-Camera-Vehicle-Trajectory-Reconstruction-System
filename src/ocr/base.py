"""
Abstract base class for OCR engines in the ALPR University Gate system.

All OCR backends must implement the OCREngine interface so they can be
swapped via config without modifying pipeline code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class OCREngine(ABC):
    """Pluggable OCR engine interface.

    Concrete implementations: PaddleOCREngine, TrOCREngine.
    Selected via config['ocr']['backend'].
    """

    def close(self) -> None:
        """Release any resources the backend holds. Optional.

        The default is a no-op, so a pure-Python backend need not implement
        it. Backends holding GPU memory (TensorRT plans, CUDA buffers) MUST
        override it: the multi-camera manager runs one pipeline per camera in
        a single process, so an engine that is never released leaks its
        allocations once per camera in the queue. A single-run CLI invocation
        never noticed, because process exit cleaned up for it.

        Must be idempotent and must never raise -- it is called from a
        finally-block during shutdown, where an exception would mask whatever
        is already unwinding.
        """

    @property
    def is_available(self) -> bool:
        """Whether this engine can still read anything at all.

        Defaults to True, which is correct for any backend that cannot
        collapse after construction (a pure-Python or CPU backend).
        Backends holding a GPU context MUST override it: a CUDA fault can
        leave them permanently unable to read, and recognize() reports that
        the same way it reports an empty crop.

        This exists because the two states are otherwise identical from the
        outside, and the expensive one is invisible: a session whose OCR
        engine had died processed four videos, stored zero plates, and
        logged ocr_inference_ms(avg=0.00) -- which reads as "fast".
        """
        return True

    @abstractmethod
    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Extract text from a plate image.

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) NumPy array.

        Returns:
            Tuple of (recognized_text, confidence_score) where
            confidence_score is in [0.0, 1.0].

            ("", 0.0) means "no readable text in this crop" -- an ordinary,
            per-frame outcome, NOT an error. It must never be used to report
            that the engine itself is broken: a caller cannot distinguish
            that from an empty crop, so a broken engine reported this way
            stays invisible for an entire run.

        Raises:
            Implementations report a failure to COME UP by raising
            ModelUnavailableError from their constructor, so an engine that
            exists is an engine that works. Implementations do not raise
            from recognize(): by the time frames are flowing, a live
            multi-camera session must survive one bad frame. A backend that
            dies mid-run flips `is_available` to False instead.
        """
        ...
