"""
OCR engine factory for the ALPR University Gate system.

Usage:
    from src.ocr import create_ocr_engine
    engine = create_ocr_engine(backend="paddleocr", config=cfg)
    text, confidence = engine.recognize(plate_image)

Backends live in a registry rather than in an if/elif chain, so adding one
means registering it, not editing the dispatch. `register_backend()` is part
of the public surface for exactly that reason.

The factory is also the one place that knows which backends need CUDA. It is
told the resolved `DeviceSpec` and refuses to build a CUDA-only backend into
a process that is running on the CPU -- because that combination does not
fail at construction, it fails silently on every frame afterwards. See
`src/runtime/device.py` for how the process lost its GPU in the first place.
"""

from __future__ import annotations

import importlib.util
from typing import Callable, Optional

from src.ocr.base import OCREngine
from src.runtime.device import DeviceSpec
from src.runtime.errors import ModelUnavailableError
from src.utils.logger import get_logger

_logger = get_logger("ocr")

# Backends that cannot run without a usable CUDA device.
CUDA_ONLY_BACKENDS = frozenset({"tensorrt", "parseq_tensorrt"})

# The backend everything degrades to: CPU-only, no native build step.
CPU_FALLBACK_BACKEND = "rapidocr"

# name -> (config, device) -> OCREngine
BackendBuilder = Callable[[dict, Optional[DeviceSpec]], OCREngine]
_BACKENDS: "dict[str, BackendBuilder]" = {}


def register_backend(name: str, builder: BackendBuilder) -> None:
    """Add or replace an OCR backend.

    Open for extension: a new engine registers itself here instead of
    appending another branch to this module's dispatch.
    """
    _BACKENDS[name] = builder


def available_backends() -> "list[str]":
    """Every registered backend name, sorted -- used in error messages."""
    return sorted(_BACKENDS)


# ── builders ──────────────────────────────────────────────────────────────
#
# Each one imports its engine lazily: loading this module must not drag in
# torch, paddle and tensorrt just to construct a RapidOCR engine.


def _build_rapidocr(config: dict, device: Optional[DeviceSpec] = None) -> OCREngine:
    from src.ocr.rapidocr_engine import RapidOCREngine

    cfg = config.get("ocr", {}).get("rapidocr", {})
    return RapidOCREngine(
        text_score=float(cfg.get("text_score", 0.5)),
        use_det=bool(cfg.get("use_det", True)),
        use_cls=bool(cfg.get("use_cls", True)),
        use_rec=bool(cfg.get("use_rec", True)),
    )


def _build_parseq_tensorrt(config: dict, device: Optional[DeviceSpec] = None) -> OCREngine:
    from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine

    cfg = config.get("ocr", {}).get("parseq", {})
    input_size = cfg.get("input_size", [32, 128])
    return PARSeqTensorRTOCREngine(
        engine_path=str(cfg.get("engine_path", "models/ocr/parseq.engine")),
        input_size=(int(input_size[0]), int(input_size[1])),
        charset=str(cfg.get("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
        onnx_path=str(cfg.get("onnx_path", "models/ocr/parseq.onnx")),
        decoder_mode=str(cfg.get("decoder_mode", "legacy")),
    )


def _build_paddleocr(config: dict, device: Optional[DeviceSpec] = None) -> OCREngine:
    from src.ocr.paddle_ocr_engine import PaddleOCREngine

    cfg = config.get("ocr", {}).get("paddleocr", {})
    return PaddleOCREngine(
        use_angle_cls=bool(cfg.get("use_angle_cls", True)),
        lang=str(cfg.get("lang", "en")),
    )


def _build_easyocr(config: dict, device: Optional[DeviceSpec] = None) -> OCREngine:
    from src.ocr.easyocr_engine import EasyOCREngine

    cfg = config.get("ocr", {}).get("easyocr", {})
    # EasyOCR picks its own device, so it is told what the rest of the
    # process resolved rather than being left to guess at a GPU that may
    # already have been hidden.
    use_gpu = bool(cfg.get("gpu", True))
    if device is not None and not device.cuda_enabled:
        use_gpu = False
    return EasyOCREngine(
        lang=list(cfg.get("lang", ["en"])),
        gpu=use_gpu,
        allowlist=str(cfg.get("allowlist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")),
    )


def _build_trocr(config: dict, device: Optional[DeviceSpec] = None) -> OCREngine:
    from src.ocr.trocr_engine import TrOCREngine

    cfg = config.get("ocr", {}).get("trocr", {})
    return TrOCREngine(model_name=str(cfg.get("model_name", "microsoft/trocr-base-printed")))


def _build_ensemble(config: dict, device: Optional[DeviceSpec] = None) -> OCREngine:
    from src.ocr.ensemble_engine import EnsembleOCREngine

    cfg = config.get("ocr", {}).get("ensemble", {})
    return EnsembleOCREngine.from_config(
        config=config,
        backends=[str(name) for name in cfg.get("backends", ["rapidocr"])],
        min_vote_count=int(cfg.get("min_vote_count", 1)),
        use_variants=bool(cfg.get("use_variants", True)),
    )


register_backend("rapidocr", _build_rapidocr)
register_backend("parseq_tensorrt", _build_parseq_tensorrt)
register_backend("tensorrt", _build_parseq_tensorrt)
register_backend("paddleocr", _build_paddleocr)
register_backend("easyocr", _build_easyocr)
register_backend("trocr", _build_trocr)
register_backend("ensemble", _build_ensemble)


# ── dispatch ──────────────────────────────────────────────────────────────


def _cuda_unavailable_reason(backend: str, device: Optional[DeviceSpec]) -> Optional[str]:
    """Why *backend* cannot run here, or None if nothing rules it out.

    Checked BEFORE construction. A CUDA-only engine built on a CPU-only
    process does not raise -- it comes up dead and returns ("", 0.0) for
    every frame of every camera, which is the exact failure that made a
    four-video session store zero plates.
    """
    if backend not in CUDA_ONLY_BACKENDS:
        return None
    if device is not None and not device.cuda_enabled:
        return (
            f"this process resolved to {device.describe()}, so its CUDA "
            f"context is unavailable"
        )
    if importlib.util.find_spec("tensorrt") is None:
        return "the TensorRT Python bindings are not installed"
    return None


def create_ocr_engine(
    backend: str,
    config: dict,
    device: Optional[DeviceSpec] = None,
) -> OCREngine:
    """Instantiate the OCR engine specified by *backend*.

    Args:
        backend: A registered backend name; see `available_backends()`.
        config:  The full ALPR config dict.
        device:  The DeviceSpec the process resolved, when the caller has
                 one. Optional so single-purpose scripts stay a one-liner,
                 but the pipeline always passes it -- without it the factory
                 cannot tell a CPU-only process from a GPU one, and will
                 build a CUDA backend that dies quietly.

    Returns:
        A live engine. Never a broken one: if the chosen backend cannot be
        brought up and no fallback applies, this raises instead of handing
        back an object that answers ("", 0.0) forever.

    Raises:
        ValueError:            unknown backend name.
        ModelUnavailableError: the backend could not be built and the CPU
                               fallback is unavailable or itself failed.
    """
    if backend not in _BACKENDS:
        msg = (
            f"Unsupported OCR backend: '{backend}'. "
            f"Valid options are: {', '.join(repr(n) for n in available_backends())}."
        )
        _logger.error(msg)
        raise ValueError(msg)

    chosen = backend
    blocked = _cuda_unavailable_reason(backend, device)
    if blocked is not None:
        _logger.warning(
            "OCR backend '%s' needs CUDA but %s; falling back to '%s'. "
            "Plate reads will be slower. Set config['ocr']['backend'] to "
            "'%s' to make this the intended configuration.",
            backend, blocked, CPU_FALLBACK_BACKEND, CPU_FALLBACK_BACKEND,
        )
        chosen = CPU_FALLBACK_BACKEND

    try:
        return _BACKENDS[chosen](config, device)
    except ModelUnavailableError as exc:
        if chosen == CPU_FALLBACK_BACKEND:
            # Nothing left to try. Raising beats returning a dead engine.
            _logger.error("OCR backend '%s' could not be loaded: %s", chosen, exc)
            raise
        _logger.error(
            "OCR backend '%s' could not be loaded (%s); falling back to '%s'.",
            chosen, exc, CPU_FALLBACK_BACKEND,
        )
        return _BACKENDS[CPU_FALLBACK_BACKEND](config, device)
