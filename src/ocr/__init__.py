"""
OCR engine factory for the ALPR University Gate system.

Usage:
    from src.ocr import create_ocr_engine
    engine = create_ocr_engine(backend="paddleocr", config=cfg)
    text, confidence = engine.recognize(plate_image)
"""

from __future__ import annotations

from src.ocr.base import OCREngine
from src.utils.logger import get_logger

_logger = get_logger("ocr")


def create_ocr_engine(backend: str, config: dict) -> OCREngine:
    """Instantiate the OCR engine specified by *backend*.

    Args:
        backend: One of "rapidocr", "tensorrt", "paddleocr", "easyocr",
                 "trocr", "ensemble".

                 JetPack 7.2 — Phase 1 (now):
                   "rapidocr" — PP-OCR models via onnxruntime CPU.
                   No GPU onnxruntime wheel exists for JP7.2 yet; CPU
                   runs at ~80-120ms/crop, sufficient with fusion window.

                 JetPack 7.2 — Phase 2 (after PARSeq fine-tuning):
                   "tensorrt" — compiled PARSeq .engine via TensorRT 10.16
                   (pre-installed with JetPack). ~5ms/crop GPU inference.
                   See src/ocr/tensorrt_engine.py for build instructions.

                 x86 dev machine:
                   "paddleocr" or "rapidocr" both work fine.

        config:  Full config dict (backend-specific settings read from
                 config['ocr'][backend]).

    Returns:
        An OCREngine instance.

    Raises:
        ValueError: If *backend* is not a supported value.
    """
    ocr_cfg = config.get("ocr", {})

    if backend == "rapidocr":
        from src.ocr.rapidocr_engine import RapidOCREngine
        rapid_cfg = ocr_cfg.get("rapidocr", {})
        return RapidOCREngine(
            text_score=float(rapid_cfg.get("text_score", 0.5)),
            use_det=bool(rapid_cfg.get("use_det", True)),
            use_cls=bool(rapid_cfg.get("use_cls", True)),
            use_rec=bool(rapid_cfg.get("use_rec", True)),
        )

    elif backend == "tensorrt":
        from src.ocr.tensorrt_engine import TensorRTOCREngine
        trt_cfg = ocr_cfg.get("tensorrt", {})
        input_size_raw = trt_cfg.get("input_size", [32, 128])
        return TensorRTOCREngine(
            engine_path=str(trt_cfg.get("engine_path", "models/ocr/parseq_plate.engine")),
            input_size=(int(input_size_raw[0]), int(input_size_raw[1])),
            charset=str(trt_cfg.get("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
        )

    elif backend == "paddleocr":
        from src.ocr.paddle_ocr_engine import PaddleOCREngine
        paddle_cfg = ocr_cfg.get("paddleocr", {})
        return PaddleOCREngine(
            use_angle_cls=bool(paddle_cfg.get("use_angle_cls", True)),
            lang=str(paddle_cfg.get("lang", "en")),
        )

    elif backend == "easyocr":
        from src.ocr.easyocr_engine import EasyOCREngine
        easy_cfg = ocr_cfg.get("easyocr", {})
        return EasyOCREngine(
            lang=list(easy_cfg.get("lang", ["en"])),
            gpu=bool(easy_cfg.get("gpu", True)),
            allowlist=str(easy_cfg.get("allowlist", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")),
        )

    elif backend == "trocr":
        from src.ocr.trocr_engine import TrOCREngine
        trocr_cfg = ocr_cfg.get("trocr", {})
        return TrOCREngine(
            model_name=str(trocr_cfg.get("model_name", "microsoft/trocr-base-printed")),
        )

    elif backend == "ensemble":
        from src.ocr.ensemble_engine import EnsembleOCREngine

        ensemble_cfg = ocr_cfg.get("ensemble", {})
        backends = ensemble_cfg.get("backends", ["rapidocr"])
        return EnsembleOCREngine.from_config(
            config=config,
            backends=[str(name) for name in backends],
            min_vote_count=int(ensemble_cfg.get("min_vote_count", 1)),
            use_variants=bool(ensemble_cfg.get("use_variants", True)),
        )

    else:
        msg = (
            f"Unsupported OCR backend: '{backend}'. "
            f"Valid options are: 'rapidocr', 'tensorrt', 'paddleocr', 'easyocr', 'trocr', 'ensemble'."
        )
        _logger.error(msg)
        raise ValueError(msg)
