import numpy as np

from src.ocr.rapidocr_engine import RapidOCREngine


def test_recognize_handles_detections_list_and_time():
    engine = RapidOCREngine.__new__(RapidOCREngine)
    # Simulate RapidOCR returning (detections_list, time_float)
    # detections_list: list of [box, text, score]
    dets = [
        ([[1, 2], [3, 4]], 'KA19TR0234', 0.87),
    ]
    engine._engine = lambda image, **kwargs: (dets, 0.123)
    engine._init_failed = False
    engine.text_score = 0.5
    engine.use_det = True
    engine.use_cls = True
    engine.use_rec = True

    text, confidence = engine.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

    assert text == "KA19TR0234"
    assert abs(confidence - 0.87) < 1e-6
