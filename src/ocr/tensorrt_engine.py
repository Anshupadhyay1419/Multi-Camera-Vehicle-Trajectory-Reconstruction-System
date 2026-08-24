"""Backward-compatible import for the PARSeq TensorRT OCR implementation.

All TensorRT execution lives in ``parseq_tensorrt_engine`` so legacy callers
cannot accidentally use deprecated TensorRT 8 binding APIs on TensorRT 10.
"""

from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine


class TensorRTOCREngine(PARSeqTensorRTOCREngine):
    """Deprecated name retained for callers using the original OCR backend."""

