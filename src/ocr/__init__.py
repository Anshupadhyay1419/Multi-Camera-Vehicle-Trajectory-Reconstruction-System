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

    Supported values: rapidocr, parseq_tensorrt, tensorrt, paddleocr,
    easyocr, trocr, ensemble.
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

    elif backend in {"tensorrt", "parseq_tensorrt"}:
        import importlib.util

        if importlib.util.find_spec("tensorrt") is None:
            _logger.warning(
                "TensorRT Python bindings are unavailable; falling back to RapidOCR for OCR. "
                "Install TensorRT or switch config['ocr']['backend'] to 'rapidocr'."
            )
            from src.ocr.rapidocr_engine import RapidOCREngine
            rapid_cfg = ocr_cfg.get("rapidocr", {})
            return RapidOCREngine(
                text_score=float(rapid_cfg.get("text_score", 0.5)),
                use_det=bool(rapid_cfg.get("use_det", True)),
                use_cls=bool(rapid_cfg.get("use_cls", True)),
                use_rec=bool(rapid_cfg.get("use_rec", True)),
            )

        from src.ocr.parseq_tensorrt_engine import PARSeqTensorRTOCREngine
        parseq_cfg = ocr_cfg.get("parseq", {})
        input_size_raw = parseq_cfg.get("input_size", [32, 128])
        return PARSeqTensorRTOCREngine(
            engine_path=str(parseq_cfg.get("engine_path", "models/ocr/parseq.engine")),
            input_size=(int(input_size_raw[0]), int(input_size_raw[1])),
            charset=str(parseq_cfg.get("charset", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")),
            onnx_path=str(parseq_cfg.get("onnx_path", "models/ocr/parseq.onnx")),
            decoder_mode=str(parseq_cfg.get("decoder_mode", "legacy")),
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
            f"Valid options are: 'rapidocr', 'parseq_tensorrt', 'tensorrt', 'paddleocr', 'easyocr', 'trocr', 'ensemble'."
        )
        _logger.error(msg)
        raise ValueError(msg)
