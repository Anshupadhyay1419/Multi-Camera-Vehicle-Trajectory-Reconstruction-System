import numpy as np

from src.ocr.rapidocr_engine import RapidOCREngine


def test_recognize_handles_tuple_results():
    engine = RapidOCREngine.__new__(RapidOCREngine)
    engine._engine = lambda image, **kwargs: ([], ["KA19TR0234"], [0.91])
    engine._init_failed = False
    engine.text_score = 0.5
    engine.use_det = True
    engine.use_cls = True
    engine.use_rec = True

    text, confidence = engine.recognize(np.zeros((32, 128, 3), dtype=np.uint8))

    assert text == "KA19TR0234"
    assert confidence == 0.91
