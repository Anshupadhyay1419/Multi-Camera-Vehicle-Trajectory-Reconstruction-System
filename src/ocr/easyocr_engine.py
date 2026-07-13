"""
EasyOCR backend for the ALPR University Gate OCR engine.

EasyOCR uses PyTorch under the hood, which means it gets full GPU
acceleration on Jetson Orin via NVIDIA's JetPack PyTorch wheels —
unlike PaddleOCR which has no GPU support on Jetson aarch64.

Installation on Jetson Orin:
    # 1. Install NVIDIA's Jetson PyTorch wheel first (from your JetPack version):
    #    pip install torch torchvision --index-url <nvidia-jetson-wheel-url>
    # 2. Then install easyocr (it will use the already-installed PyTorch):
    #    pip install easyocr==1.7.2

On x86 (dev machine):
    pip install easyocr==1.7.2

To enable: set config['ocr']['backend'] = 'easyocr'
"""

from __future__ import annotations

import re

import numpy as np

from src.ocr.base import OCREngine
from src.utils.logger import get_logger

_logger = get_logger("ocr.easyocr_engine")


class EasyOCREngine(OCREngine):
    """OCR engine backed by EasyOCR with GPU support on Jetson Orin.

    EasyOCR wraps a CRAFT text-detector + CRNN recognizer. Since we
    pass already-cropped plate images, text detection is suppressed
    (allowlist + single-line mode) so only the recognizer runs —
    this cuts latency roughly in half.

    Args:
        lang:         Language codes list. ['en'] covers all Indian plates.
        gpu:          Use GPU if available. True by default.
        allowlist:    Characters to allow in output. Defaults to A-Z and 0-9
                      which is exactly the Indian plate character set.
        min_confidence: Minimum per-character confidence floor (internal).
    """

    def __init__(
        self,
        lang: list[str] | None = None,
        gpu: bool = True,
        allowlist: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        min_confidence: float = 0.1,
    ) -> None:
        self.lang = lang or ["en"]
        self.gpu = gpu
        self.allowlist = allowlist
        self.min_confidence = min_confidence
        self._reader = None
        self._init_failed = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize EasyOCR reader. Logs error and sets flag on failure."""
        try:
            import easyocr

            _logger.info(
                "Initializing EasyOCR (lang=%s, gpu=%s)...", self.lang, self.gpu
            )
            # verbose=False suppresses EasyOCR's own download/init logs
            self._reader = easyocr.Reader(self.lang, gpu=self.gpu, verbose=False)
            _logger.info("EasyOCR initialized successfully (gpu=%s)", self.gpu)

        except ImportError as exc:
            _logger.error(
                "EasyOCR is not installed: %s. "
                "Install with: pip install easyocr==1.7.2",
                exc,
            )
            self._init_failed = True
        except Exception as exc:
            _logger.error("EasyOCR initialization failed: %s", exc)
            self._init_failed = True

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Run EasyOCR on a plate crop image.

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) NumPy array.
                   Should already be the cropped plate region.

        Returns:
            (text, confidence) — ("", 0.0) on any failure.
        """
        if self._init_failed or self._reader is None:
            return ("", 0.0)

        if image is None or image.size == 0:
            return ("", 0.0)

        try:
            import cv2

            # EasyOCR expects an RGB image (or grayscale is fine too)
            if len(image.shape) == 2:
                # Grayscale — EasyOCR handles this fine
                ocr_image = image
            else:
                # BGR → RGB
                ocr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # readtext parameters tuned for license plate crops:
            #   - detail=1           → returns bbox + text + confidence
            #   - allowlist          → restrict to plate characters only
            #   - paragraph=False    → treat each detection as a separate word
            #   - min_size=5         → don't skip tiny characters
            #   - contrast_ths=0.1   → lower threshold for low-contrast crops
            #   - adjust_contrast=0.5 → auto contrast adjustment
            #   - text_threshold=0.5 → character confidence threshold
            #   - low_text=0.3       → text region threshold
            results = self._reader.readtext(
                ocr_image,
                detail=1,
                allowlist=self.allowlist,
                paragraph=False,
                min_size=5,
                contrast_ths=0.1,
                adjust_contrast=0.5,
                text_threshold=0.5,
                low_text=0.3,
            )

            if not results:
                return ("", 0.0)

            # Collect all detected text regions with their confidences.
            # For plate crops, there's usually one result; sometimes two
            # when the plate has two rows (e.g. state name + number).
            texts: list[str] = []
            confidences: list[float] = []

            for _bbox, text, conf in results:
                cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
                if cleaned:
                    texts.append(cleaned)
                    confidences.append(float(conf))

            if not texts:
                return ("", 0.0)

            # Join all text regions (handles two-row plates)
            combined = "".join(texts)
            avg_conf = float(np.mean(confidences))
            avg_conf = float(np.clip(avg_conf, 0.0, 1.0))

            return (combined, avg_conf)

        except Exception as exc:
            _logger.warning("EasyOCR inference failed: %s", exc)
            return ("", 0.0)
