"""
Robust OCR ensemble for the ALPR University Gate system.

This engine combines multiple OCR backends and multiple image variants,
then validates and votes on the best plate string rather than trusting a
single OCR result.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from src.ocr.base import OCREngine
from src.ocr.ocr_postprocessor import correct_ocr_text, remove_noise_characters
from src.validation.plate_validator import PlateValidator
from src.utils.logger import get_logger

_logger = get_logger("ocr.ensemble_engine")


@dataclass(frozen=True)
class _Candidate:
    plate: str
    confidence: float
    source: str


class EnsembleOCREngine(OCREngine):
    """OCR engine that fuses multiple OCR backends and image variants."""

    def __init__(
        self,
        engines: list[OCREngine],
        min_vote_count: int = 1,
        use_variants: bool = True,
    ) -> None:
        self._engines = engines
        self.min_vote_count = max(1, int(min_vote_count))
        self.use_variants = bool(use_variants)
        self._validator = PlateValidator()

    @classmethod
    def from_config(
        cls,
        config: dict,
        backends: list[str] | None = None,
        min_vote_count: int = 1,
        use_variants: bool = True,
    ) -> "EnsembleOCREngine":
        """Build the ensemble from config while skipping unavailable backends."""
        from src.ocr import create_ocr_engine

        ocr_cfg = config.get("ocr", {})
        backend_names = backends or ["paddleocr", "trocr"]
        engines: list[OCREngine] = []

        for backend in backend_names:
            try:
                if backend == "ensemble":
                    continue
                engines.append(create_ocr_engine(backend, config))
            except Exception as exc:
                _logger.warning("Skipping OCR backend '%s': %s", backend, exc)

        if not engines:
            # Fall back to a single concrete backend so the pipeline still runs.
            engines.append(create_ocr_engine("paddleocr", config))

        return cls(
            engines=engines,
            min_vote_count=min_vote_count,
            use_variants=use_variants,
        )

    def recognize(self, image: np.ndarray) -> tuple[str, float]:
        """Run multiple OCR passes and return the best validated plate."""
        if image is None or image.size == 0:
            return ("", 0.0)

        candidates: list[_Candidate] = []
        variants = self._build_variants(image) if self.use_variants else [image]

        for variant_index, variant in enumerate(variants):
            for engine_index, engine in enumerate(self._engines):
                try:
                    raw_text, confidence = engine.recognize(variant)
                except Exception as exc:
                    _logger.warning(
                        "OCR engine %s failed on variant %d: %s",
                        engine.__class__.__name__,
                        variant_index,
                        exc,
                    )
                    continue

                plate = self._normalize_plate(raw_text)
                if not plate:
                    continue

                if confidence <= 0:
                    continue

                candidates.append(
                    _Candidate(
                        plate=plate,
                        confidence=float(confidence),
                        source=f"{engine.__class__.__name__}:v{variant_index}",
                    )
                )

        if not candidates:
            return ("", 0.0)

        grouped: dict[str, list[_Candidate]] = defaultdict(list)
        for candidate in candidates:
            grouped[candidate.plate].append(candidate)

        ranked = []
        for plate, items in grouped.items():
            vote_count = len(items)
            avg_conf = sum(item.confidence for item in items) / vote_count
            best_conf = max(item.confidence for item in items)
            score = (avg_conf * 0.65) + (best_conf * 0.25) + (min(vote_count, 5) * 0.10)
            ranked.append((plate, vote_count, avg_conf, best_conf, score))

        ranked.sort(key=lambda item: (item[1], item[4], item[2], item[3]), reverse=True)
        best_plate, vote_count, avg_conf, best_conf, _ = ranked[0]

        if vote_count < self.min_vote_count:
            # Still return the top result, but only if it looks like a valid plate.
            if not self._looks_like_plate(best_plate):
                return ("", 0.0)

        combined_conf = min(1.0, (avg_conf * 0.7) + (best_conf * 0.3) + (0.03 * (vote_count - 1)))
        return (best_plate, combined_conf)

    @staticmethod
    def _build_variants(image: np.ndarray) -> list[np.ndarray]:
        """Generate a small set of OCR-friendly image variants."""
        variants: list[np.ndarray] = [image]

        try:
            import cv2

            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if gray.dtype != np.uint8:
                gray = np.clip(gray, 0, 255).astype(np.uint8)

            variants.append(gray)

            upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            variants.append(upscaled)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            variants.append(enhanced)

        except Exception:
            # If OpenCV preprocessing fails, just use the original image.
            return [image]

        return variants

    def _normalize_plate(self, raw_text: str) -> str:
        """Clean OCR text and keep only outputs that validate as plates."""
        if not raw_text:
            return ""

        cleaned = remove_noise_characters(raw_text)
        plate_number, _ = self._validator.validate(cleaned)
        if plate_number:
            return plate_number

        corrected = correct_ocr_text(cleaned)
        plate_number, _ = self._validator.validate(corrected)
        if plate_number:
            return plate_number

        return ""

    def _looks_like_plate(self, text: str) -> bool:
        plate_number, _ = self._validator.validate(text)
        return plate_number is not None
