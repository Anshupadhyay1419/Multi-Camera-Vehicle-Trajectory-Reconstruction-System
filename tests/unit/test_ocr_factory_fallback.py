import importlib.util

from src.ocr import create_ocr_engine


class FakeRapidOCREngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_create_ocr_engine_falls_back_to_rapidocr_when_tensorrt_is_unavailable(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "tensorrt" else object())

    import src.ocr as ocr_module

    monkeypatch.setattr(ocr_module, "get_logger", lambda *_args, **_kwargs: None)

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "src.ocr.rapidocr_engine":
            module = type("Module", (), {})()
            module.RapidOCREngine = FakeRapidOCREngine
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    engine = create_ocr_engine("parseq_tensorrt", {"ocr": {"rapidocr": {}}})

    assert isinstance(engine, FakeRapidOCREngine)
