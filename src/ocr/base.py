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

    @abstractmethod
    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Extract text from a plate image.

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) NumPy array.

        Returns:
            Tuple of (recognized_text, confidence_score) where
            confidence_score is in [0.0, 1.0].
            Returns ("", 0.0) on failure.
        """
        ...
