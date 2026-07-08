"""
Plate image preprocessor for the ALPR University Gate system.

Applies a fixed preprocessing pipeline to plate crops before OCR:
  1. Grayscale conversion
  2. CLAHE (contrast limited adaptive histogram equalization)
  3. Non-local means denoising
  4. Unsharp mask sharpening

All parameters are read from the 'preprocessing' section of config.yaml.
"""

from __future__ import annotations

import numpy as np
import cv2

from src.utils.logger import get_logger

_logger = get_logger("preprocessing.plate_preprocessor")


class PlatePreprocessor:
    """Preprocess a plate crop for OCR.

    Args:
        clahe_clip_limit:  CLAHE clip limit (default 2.0).
        clahe_tile_size:   CLAHE tile grid size (default 8).
        denoise_h:         Non-local means filter strength h (default 10).
        sharpen_strength:  Unsharp mask blend weight (default 1.5).
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        denoise_h: int = 10,
        sharpen_strength: float = 1.5,
    ) -> None:
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise_h = denoise_h
        self.sharpen_strength = sharpen_strength

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
        )

    @classmethod
    def from_config(cls, config: dict) -> "PlatePreprocessor":
        """Construct from the full config dict."""
        pp = config.get("preprocessing", {})
        return cls(
            clahe_clip_limit=float(pp.get("clahe_clip_limit", 2.0)),
            clahe_tile_size=int(pp.get("clahe_tile_size", 8)),
            denoise_h=int(pp.get("denoise_h", 10)),
            sharpen_strength=float(pp.get("sharpen_strength", 1.5)),
        )

    def process(self, plate_crop: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing pipeline to a plate crop.

        Args:
            plate_crop: BGR or grayscale plate image as NumPy array.

        Returns:
            2D grayscale NumPy array (H, W) with dtype uint8, pixel values
            in [0, 255]. On invalid input, logs a WARNING and returns the
            original crop unchanged.
        """
        if plate_crop is None or plate_crop.size == 0:
            _logger.warning("PlatePreprocessor received empty/None input; returning as-is.")
            return plate_crop

        try:


            # Step 1: Grayscale
            if len(plate_crop.shape) == 3:
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_crop.copy()

            # Ensure uint8
            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            # Step 2: CLAHE
            clahe_img = self._clahe.apply(gray)

            # Step 3: Non-local means denoising
            denoised = cv2.fastNlMeansDenoising(
                clahe_img,
                h=float(self.denoise_h),
                templateWindowSize=7,
                searchWindowSize=21,
            )

            # Step 4: Unsharp mask sharpening
            blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
            sharpened = cv2.addWeighted(
                denoised, self.sharpen_strength,
                blurred, -(self.sharpen_strength - 1.0),
                0,
            )
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            return sharpened

        except Exception as exc:
            _logger.warning(
                "PlatePreprocessor failed (%s); returning original crop.", exc
            )
            try:
                if len(plate_crop.shape) == 3:
                    return cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            except Exception:
                pass
            return plate_crop
