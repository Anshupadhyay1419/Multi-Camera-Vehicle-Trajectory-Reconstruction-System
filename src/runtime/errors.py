"""Failures that mean a model must not be used."""

from __future__ import annotations


class ModelUnavailableError(RuntimeError):
    """A model could not be brought up, so it must not be used.

    Raised at CONSTRUCTION time, never per frame. The distinction matters:
    a model that answers "I read nothing" on a frame is doing its job, while
    a model that could not load answers "I read nothing" on *every* frame
    forever and is indistinguishable from a clean miss. A run that lost its
    OCR engine that way processed four videos and stored zero plates while
    reporting `ocr_inference_ms(avg=0.00)`, which read as "fast", not
    "dead". Refusing to hand back a broken object is what keeps those two
    states apart.

    Callers that have a genuine fallback (the OCR factory, which can drop a
    CUDA backend for a CPU one) may catch this and pick another
    implementation, provided they say so in the log. Callers with no
    fallback must let it propagate.
    """
