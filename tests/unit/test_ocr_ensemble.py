"""
Unit tests for the OCR ensemble engine.
"""

from __future__ import annotations

import numpy as np

from src.ocr.ensemble_engine import EnsembleOCREngine


class _FakeEngine:
    def __init__(self, text: str, confidence: float) -> None:
        self.text = text
        self.confidence = confidence

    def recognize(self, image):
        return (self.text, self.confidence)


class TestEnsembleOCREngine:
    def test_valid_plate_wins_over_high_confidence_invalid_text(self):
        engine = EnsembleOCREngine(
            engines=[
                _FakeEngine("KA19TR0234", 0.72),
                _FakeEngine("KAI9TRO234", 0.96),
            ],
            min_vote_count=1,
            use_variants=False,
        )

        plate, confidence = engine.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

        assert plate == "KA19TR0234"
        assert confidence > 0.0

    def test_multiple_valid_votes_outweigh_single_outlier(self):
        engine = EnsembleOCREngine(
            engines=[
                _FakeEngine("KA19TR0234", 0.78),
                _FakeEngine("KA19TR0234", 0.80),
                _FakeEngine("KA19TRO234", 0.99),
            ],
            min_vote_count=2,
            use_variants=False,
        )

        plate, confidence = engine.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

        assert plate == "KA19TR0234"
        assert confidence > 0.0
