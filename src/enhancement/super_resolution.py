"""
Super-resolution enhancer for the ALPR University Gate system.

Two upscale backends are supported, selected automatically at runtime:

  1. Real-ESRGAN (HIGH QUALITY) — requires torch + basicsr + realesrgan.
     Available on x86 dev machines. NOT available on JetPack 7.2 because
     PyTorch has no CUDA 13.2 / Python 3.12 aarch64 wheel yet.

  2. OpenCV INTER_CUBIC (FAST FALLBACK) — pure OpenCV, no extra deps.
     Used automatically when Real-ESRGAN dependencies are missing.
     Quality is lower than Real-ESRGAN but sufficient for RapidOCR input,
     especially given the multi-frame voting in OCRFusion.

The backend is chosen lazily on first enhance() call — no config change
needed when deploying on Jetson. Once PyTorch wheels for JetPack 7.2
become available, Real-ESRGAN will activate automatically.

Setup (x86 / JetPack 6 only):
    pip install basicsr realesrgan
    # Download RealESRGAN_x4plus.pth to models/realesrgan/
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils.logger import get_logger

_logger = get_logger("enhancement.super_resolution")

# Scale factor used by both backends.
_SR_SCALE = 4


class SuperResolutionEnhancer:
    """Conditionally upscale plate crops using Real-ESRGAN or OpenCV fallback.

    Enhancement is only applied when plate width < sr_threshold_px.
    When Real-ESRGAN is unavailable (e.g. JetPack 7.2 without PyTorch),
    OpenCV INTER_CUBIC upscaling is used transparently.

    Args:
        model_path:      Path to RealESRGAN_x4plus.pth weights file.
        sr_threshold_px: Only upscale plates narrower than this (pixels).
                         Set to 0 in config to disable SR entirely.
    """

    def __init__(self, model_path: str, sr_threshold_px: int = 80) -> None:
        self.model_path = model_path
        self.sr_threshold_px = sr_threshold_px
        self._upsampler = None
        self._realesrgan_available: bool | None = None  # None = not yet checked

    # ------------------------------------------------------------------
    # Real-ESRGAN backend
    # ------------------------------------------------------------------

    def _try_load_realesrgan(self) -> bool:
        """Attempt to load Real-ESRGAN. Returns True on success."""
        if self._realesrgan_available is True:
            return True
        if self._realesrgan_available is False:
            return False

        try:
            import torch  # noqa: F401  — confirms PyTorch is present
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            model_file = Path(self.model_path)
            if not model_file.exists():
                _logger.warning(
                    "Real-ESRGAN weights not found at '%s'. "
                    "Falling back to OpenCV upscale. "
                    "Download: https://github.com/xinntao/Real-ESRGAN/releases",
                    self.model_path,
                )
                self._realesrgan_available = False
                return False

            arch = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23, num_grow_ch=32, scale=_SR_SCALE,
            )
            self._upsampler = RealESRGANer(
                scale=_SR_SCALE,
                model_path=str(model_file),
                model=arch,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
            )
            self._realesrgan_available = True
            _logger.info("Real-ESRGAN loaded from '%s'", self.model_path)
            return True

        except ImportError:
            # torch / basicsr / realesrgan not installed — expected on Jetson 7.2
            _logger.info(
                "Real-ESRGAN dependencies unavailable (torch/basicsr/realesrgan "
                "not installed). Using OpenCV INTER_CUBIC upscale instead. "
                "This is normal on JetPack 7.2 — PyTorch wheels are not yet "
                "available for CUDA 13.2 / Python 3.12 aarch64."
            )
            self._realesrgan_available = False
            return False
        except Exception as exc:
            _logger.warning(
                "Real-ESRGAN failed to load (%s). "
                "Falling back to OpenCV upscale.", exc
            )
            self._realesrgan_available = False
            return False

    # ------------------------------------------------------------------
    # OpenCV fallback backend
    # ------------------------------------------------------------------

    @staticmethod
    def _opencv_upscale(image: np.ndarray, scale: int = _SR_SCALE) -> np.ndarray:
        """Upscale image using bicubic interpolation (no deps beyond OpenCV)."""
        h, w = image.shape[:2]
        return cv2.resize(
            image,
            (w * scale, h * scale),
            interpolation=cv2.INTER_CUBIC,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, plate_crop: np.ndarray) -> np.ndarray:
        """Upscale a plate crop if its width is below the threshold.

        Selects Real-ESRGAN when available, OpenCV bicubic otherwise.
        Always returns the original crop if sr_threshold_px == 0 or
        if the crop is already wide enough.

        Args:
            plate_crop: Grayscale (H, W) or BGR (H, W, 3) plate image.

        Returns:
            Upscaled image, or the original crop if no upscaling is needed.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        width = plate_crop.shape[1] if plate_crop.ndim >= 2 else 0

        # SR disabled or crop already large enough
        if self.sr_threshold_px == 0 or width >= self.sr_threshold_px:
            return plate_crop

        # Ensure BGR uint8 for processing
        if plate_crop.ndim == 2:
            bgr = cv2.cvtColor(plate_crop, cv2.COLOR_GRAY2BGR)
        else:
            bgr = plate_crop.copy()
        if bgr.dtype != np.uint8:
            bgr = np.clip(bgr, 0, 255).astype(np.uint8)

        # Try Real-ESRGAN first, fall back to OpenCV
        if self._try_load_realesrgan():
            try:
                enhanced_bgr, _ = self._upsampler.enhance(bgr, outscale=_SR_SCALE)
                _logger.debug(
                    "Real-ESRGAN: %dx%d → %dx%d",
                    width, plate_crop.shape[0],
                    enhanced_bgr.shape[1], enhanced_bgr.shape[0],
                )
            except Exception as exc:
                _logger.warning(
                    "Real-ESRGAN inference failed (%s); falling back to OpenCV.", exc
                )
                enhanced_bgr = self._opencv_upscale(bgr)
        else:
            enhanced_bgr = self._opencv_upscale(bgr)
            _logger.debug(
                "OpenCV upscale: %dx%d → %dx%d",
                width, plate_crop.shape[0],
                enhanced_bgr.shape[1], enhanced_bgr.shape[0],
            )

        # Return in same channel format as input
        if plate_crop.ndim == 2:
            return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        return enhanced_bgr
