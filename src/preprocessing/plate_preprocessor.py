"""
Plate image preprocessor for the ALPR University Gate system.

Applies a preprocessing pipeline to plate crops before OCR:
  1. Grayscale conversion
  2. Adaptive gamma correction (brightens night/underexposed crops,
     recovers detail from glare/overexposed crops)
  3. CLAHE (contrast limited adaptive histogram equalization), with the
     clip limit adapted to the crop's own contrast after gamma correction
  4. Bilateral filter denoising
  5. Unsharp mask sharpening

All parameters are read from the 'preprocessing' section of config.yaml.
"""

from __future__ import annotations

import math

import numpy as np
import cv2

from src.utils.logger import get_logger

_logger = get_logger("preprocessing.plate_preprocessor")

# Steps 2-3 target this mean brightness (mid-gray). Crops already within
# _GAMMA_SKIP_BAND of it are left alone -- gamma correction on an
# already-well-exposed crop only adds noise/artifacts for no benefit.
_GAMMA_TARGET = 128.0
_GAMMA_SKIP_BAND = (100.0, 160.0)
_GAMMA_MIN, _GAMMA_MAX = 0.35, 3.0

# CLAHE clip-limit adaptation thresholds, measured on post-gamma std-dev.
_LOW_CONTRAST_STD = 30.0    # hazy / heavily glare-washed
_HIGH_CONTRAST_STD = 70.0   # already sharp/high-contrast (or noisy)


class PlatePreprocessor:
    """Preprocess a plate crop for OCR.

    Args:
        clahe_clip_limit:  Base CLAHE clip limit (default 2.0). Adapted up
                           for low-contrast (hazy/glare) crops and down for
                           already high-contrast crops -- see process().
        clahe_tile_size:   CLAHE tile grid size (default 8).
        denoise_h:         Non-local means filter strength h (default 10).
        sharpen_strength:  Unsharp mask blend weight (default 1.5).
        adaptive_lighting: Enable gamma correction + CLAHE clip-limit
                           adaptation. On by default; disable to fall back
                           to the previous fixed-parameter behavior.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        denoise_h: int = 10,
        sharpen_strength: float = 1.5,
        adaptive_lighting: bool = True,
    ) -> None:
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise_h = denoise_h
        self.sharpen_strength = sharpen_strength
        self.adaptive_lighting = adaptive_lighting

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
        )
        # Gamma LUTs are cheap to build but there are only a handful of
        # distinct gamma values in practice (rounded below) -- cache them
        # instead of rebuilding a 256-entry table on every single crop.
        self._gamma_lut_cache: dict[float, np.ndarray] = {}

    @classmethod
    def from_config(cls, config: dict) -> "PlatePreprocessor":
        """Construct from the full config dict."""
        pp = config.get("preprocessing", {})
        return cls(
            clahe_clip_limit=float(pp.get("clahe_clip_limit", 2.0)),
            clahe_tile_size=int(pp.get("clahe_tile_size", 8)),
            denoise_h=int(pp.get("denoise_h", 10)),
            sharpen_strength=float(pp.get("sharpen_strength", 1.5)),
            adaptive_lighting=bool(pp.get("adaptive_lighting", True)),
        )

    def _gamma_correct(self, gray: np.ndarray) -> np.ndarray:
        """Auto gamma-correct so the crop's mean brightness moves toward
        mid-gray, using log(target)/log(mean) -- the standard closed-form
        gamma for hitting a target mean in one shot. gamma < 1 brightens
        (dark/night crops), gamma > 1 darkens (glare/overexposed crops).
        """
        mean = float(np.mean(gray))
        if _GAMMA_SKIP_BAND[0] <= mean <= _GAMMA_SKIP_BAND[1]:
            return gray
        # Clamp away from both 0 and 1: log(mean_norm) is undefined at 0
        # and exactly 0 at 1.0 (pure white crop), which would divide by zero.
        mean_norm = float(np.clip(mean / 255.0, 1e-3, 1.0 - 1e-3))
        gamma = math.log(_GAMMA_TARGET / 255.0) / math.log(mean_norm)
        gamma = float(np.clip(gamma, _GAMMA_MIN, _GAMMA_MAX))
        gamma_key = round(gamma, 2)

        lut = self._gamma_lut_cache.get(gamma_key)
        if lut is None:
            # Apply the exponent directly: corrected_norm = pixel_norm**gamma.
            # gamma was solved from mean_norm**gamma == target_norm above,
            # so this is the same relationship applied pixel-by-pixel.
            lut = (((np.arange(256) / 255.0) ** gamma_key) * 255.0).astype(np.uint8)
            self._gamma_lut_cache[gamma_key] = lut

        return cv2.LUT(gray, lut)

    def _adaptive_clip_limit(self, gray: np.ndarray) -> float:
        """Scale CLAHE's clip limit to the crop's own contrast: push it up
        for hazy/washed-out crops (more aggressive local contrast is
        needed to reveal characters) and down for already high-contrast
        crops (avoid amplifying noise on top of real detail).
        """
        std = float(np.std(gray))
        if std < _LOW_CONTRAST_STD:
            return min(self.clahe_clip_limit * 2.0, 6.0)
        if std > _HIGH_CONTRAST_STD:
            return max(self.clahe_clip_limit * 0.75, 1.0)
        return self.clahe_clip_limit

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

            # Step 2: Adaptive gamma correction (night/glare robustness)
            if self.adaptive_lighting:
                gray = self._gamma_correct(gray)

            # Step 3: CLAHE, with clip limit adapted to this crop's contrast
            if self.adaptive_lighting:
                clip_limit = self._adaptive_clip_limit(gray)
                clahe = (
                    self._clahe if clip_limit == self.clahe_clip_limit
                    else cv2.createCLAHE(
                        clipLimit=clip_limit,
                        tileGridSize=(self.clahe_tile_size, self.clahe_tile_size),
                    )
                )
            else:
                clahe = self._clahe
            clahe_img = clahe.apply(gray)

            # Step 4: Denoise. fastNlMeansDenoising (searchWindowSize=21) is
            # ~10-20x slower than a bilateral filter for a near-identical
            # result on small plate crops, and dominated OCR preprocessing
            # latency on Jetson CPU. Bilateral filtering smooths noise while
            # keeping character edges sharp, which is what OCR needs here.
            denoised = cv2.bilateralFilter(
                clahe_img,
                d=5,
                sigmaColor=float(self.denoise_h) * 5,
                sigmaSpace=float(self.denoise_h) * 5,
            )

            # Step 5: Unsharp mask sharpening
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
