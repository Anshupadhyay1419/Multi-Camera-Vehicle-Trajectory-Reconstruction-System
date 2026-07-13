"""
RapidOCR backend for the ALPR University Gate OCR engine.

RapidOCR runs PaddleOCR's PP-OCR models (v3/v4) through ONNX Runtime
instead of the PaddlePaddle framework. This removes the PaddlePaddle
dependency entirely, which matters because PaddlePaddle has no GPU
support on Jetson aarch64.

─── Deployment status by platform ──────────────────────────────────────────

  x86 dev machine:
    onnxruntime-gpu is available from PyPI.
    CUDAExecutionProvider active → full GPU inference.

  Jetson Orin — JetPack 6.x (L4T 36, CUDA 12.6, Python 3.10):
    PyPI onnxruntime-gpu has no aarch64 wheel.
    Use NVIDIA's own index:
      pip install onnxruntime-gpu==1.20.2 \
        --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
    CUDAExecutionProvider active after that install.

  Jetson Orin — JetPack 7.2 (L4T 39.2, CUDA 13.2, Python 3.12):
    No onnxruntime-gpu wheel for CUDA 13.2 / Python 3.12 exists yet
    (confirmed as of July 2026 — https://pypi.jetson-ai-lab.dev/jp7/).
    CPU-only onnxruntime is used instead (~80–120ms/crop on Orin Nano).
    The OCRFusion window of 5 frames and multi-frame voting in run_alpr.py
    compensate well for the slower per-frame latency.

    ── Upgrade path to full GPU OCR on JetPack 7.2 ──────────────────────
    TensorRT 10.16 IS pre-installed with JetPack 7.2 and works today.
    After fine-tuning PARSeq on your Indian plate dataset:
      1. Export to ONNX on dev machine
      2. Compile to TensorRT .engine on Jetson with trtexec --fp16
      3. Set config['ocr']['backend'] = "tensorrt" in config.yaml
      4. The TensorRTOCREngine (src/ocr/tensorrt_engine.py) handles the rest
    Expected latency after switch: ~5ms/crop (GPU, FP16).
    ─────────────────────────────────────────────────────────────────────

─── Installation ────────────────────────────────────────────────────────────

  Jetson (any):   pip install rapidocr-onnxruntime==1.4.0 onnxruntime==1.20.1
                  (install_jetson.sh handles this automatically)

  x86 dev:        pip install rapidocr-onnxruntime==1.4.0 onnxruntime-gpu==1.18.0

To enable: set config['ocr']['backend'] = 'rapidocr'
"""

from __future__ import annotations

import re

import numpy as np

from src.ocr.base import OCREngine
from src.utils.logger import get_logger

_logger = get_logger("ocr.rapidocr_engine")


class RapidOCREngine(OCREngine):
    """OCR engine backed by RapidOCR (PP-OCR ONNX models via onnxruntime).

    RapidOCR runs the same PP-OCRv3/v4 detection + recognition pipeline
    as PaddleOCR but uses ONNX Runtime as the inference backend, which
    supports CUDAExecutionProvider on Jetson Orin without requiring the
    PaddlePaddle framework.

    Args:
        text_score:     Minimum confidence for a text region to be accepted.
                        0.5 is a good balance; lower = more detections but
                        noisier results.
        use_det:        Run text detection inside the crop. For plate crops
                        (already tightly cropped) this can be False to skip
                        detection and go straight to recognition — faster,
                        but detection=True is safer for slightly loose crops.
        use_cls:        Run angle classifier to handle upside-down plates.
        use_rec:        Run text recognition (always True).
    """

    def __init__(
        self,
        text_score: float = 0.5,
        use_det: bool = True,
        use_cls: bool = True,
        use_rec: bool = True,
    ) -> None:
        self.text_score = text_score
        self.use_det = use_det
        self.use_cls = use_cls
        self.use_rec = use_rec
        self._engine = None
        self._init_failed = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize RapidOCR engine. Logs error and sets flag on failure."""
        try:
            from rapidocr import RapidOCR

            _logger.info("Initializing RapidOCR engine...")
            self._engine = RapidOCR()

            # Log which ONNX Runtime execution providers are actually active.
            # This is critical on Jetson where GPU availability depends on
            # how onnxruntime was installed (see requirements.txt for details).
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    _logger.info("RapidOCR: GPU inference active (CUDAExecutionProvider)")
                else:
                    _logger.warning(
                        "RapidOCR: CUDAExecutionProvider NOT available — "
                        "running on CPU. Available: %s. "
                        "JetPack 6: install via --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126 . "
                        "JetPack 7.2: no GPU ort wheel yet — CPU is expected. "
                        "Upgrade path: fine-tune PARSeq → trtexec → set backend='tensorrt'.",
                        providers,
                    )
            except Exception:
                pass  # onnxruntime import check is best-effort

            _logger.info("RapidOCR initialized successfully.")

        except ImportError as exc:
            _logger.error(
                "RapidOCR is not installed: %s. "
                "Install with: pip install rapidocr-onnxruntime==1.4.0",
                exc,
            )
            self._init_failed = True
        except Exception as exc:
            _logger.error("RapidOCR initialization failed: %s", exc)
            self._init_failed = True

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Run RapidOCR on a plate crop image.

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) NumPy array.
                   Should already be the cropped plate region from PlateDetector.

        Returns:
            (text, confidence) tuple.
            text is uppercase alphanumeric only (noise stripped).
            confidence is in [0.0, 1.0].
            Returns ("", 0.0) on any failure.
        """
        if self._init_failed or self._engine is None:
            return ("", 0.0)

        if image is None or image.size == 0:
            return ("", 0.0)

        try:
            import cv2

            # RapidOCR expects a BGR uint8 image (same as OpenCV default)
            if len(image.shape) == 2:
                # Grayscale → BGR so the PP-OCR pipeline works correctly
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = image.copy()

            if img_bgr.dtype != np.uint8:
                img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)

            # Call RapidOCR. Returns a RapidOCROutput object.
            #   result.boxes  → list of bounding boxes
            #   result.txts   → list of recognized text strings
            #   result.scores → list of confidence floats
            result = self._engine(
                img_bgr,
                use_det=self.use_det,
                use_cls=self.use_cls,
                use_rec=self.use_rec,
                text_score=self.text_score,
            )

            # RapidOCR returns None when nothing is detected
            if result is None or result.txts is None or len(result.txts) == 0:
                return ("", 0.0)

            texts: list[str] = []
            scores: list[float] = []

            for txt, score in zip(result.txts, result.scores):
                if txt is None:
                    continue
                cleaned = re.sub(r"[^A-Z0-9]", "", str(txt).upper())
                if cleaned:
                    texts.append(cleaned)
                    scores.append(float(score) if score is not None else 0.0)

            if not texts:
                return ("", 0.0)

            # Join all detected text regions.
            # For a tightly-cropped plate there's usually one region; two for
            # plates where the state emblem/IND text appears on a separate row.
            combined = "".join(texts)
            avg_score = float(np.mean(scores))
            avg_score = float(np.clip(avg_score, 0.0, 1.0))

            return (combined, avg_score)

        except Exception as exc:
            _logger.warning("RapidOCR inference failed: %s", exc)
            return ("", 0.0)
