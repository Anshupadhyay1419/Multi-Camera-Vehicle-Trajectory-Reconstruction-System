"""
Unit tests for OCR backend selection.

Two rules are being defended, both learned from the same outage:

  1. A CUDA-only backend must never be BUILT into a process that resolved to
     the CPU. Doing so does not raise -- the engine comes up dead and then
     returns ("", 0.0) for every frame, which is indistinguishable from "no
     plate in shot". A four-video session stored zero plates that way while
     reporting ocr_inference_ms(avg=0.00), i.e. it looked fast.

  2. The factory must never hand back a broken engine. If nothing can be
     built, it raises.
"""

from __future__ import annotations

import pytest

import src.ocr as ocr_module
from src.ocr import available_backends, create_ocr_engine, register_backend
from src.runtime.device import DeviceSpec
from src.runtime.errors import ModelUnavailableError

CPU_DEVICE = DeviceSpec(kind="cpu", index=None, use_half=False)
CUDA_DEVICE = DeviceSpec(kind="cuda", index=0, use_half=True)


class FakeEngine:
    def __init__(self, name: str, device=None):
        self.name = name
        self.device = device

    def recognize(self, image):
        return "", 0.0


@pytest.fixture()
def fake_backends(monkeypatch):
    """Replace the real builders so no test touches a GPU or a model file."""
    def install(name: str, builder):
        monkeypatch.setitem(ocr_module._BACKENDS, name, builder)

    install("rapidocr", lambda cfg, dev=None: FakeEngine("rapidocr", dev))
    install("parseq_tensorrt", lambda cfg, dev=None: FakeEngine("parseq_tensorrt", dev))
    return install


class TestCudaBackendOnACpuProcess:
    """Rule 1: the exact combination that caused the outage."""

    def test_a_cuda_backend_is_not_built_on_a_cpu_process(self, fake_backends):
        engine = create_ocr_engine("parseq_tensorrt", {"ocr": {}}, CPU_DEVICE)
        assert engine.name == "rapidocr"

    def test_the_diversion_is_logged_as_a_warning(self, fake_backends, monkeypatch):
        """Silently swapping the backend would hide a real misconfiguration."""
        warnings: list[str] = []
        monkeypatch.setattr(
            ocr_module._logger,
            "warning",
            lambda msg, *a, **k: warnings.append(str(msg) % a if a else str(msg)),
        )
        create_ocr_engine("parseq_tensorrt", {"ocr": {}}, CPU_DEVICE)
        assert warnings and "rapidocr" in warnings[0]

    def test_a_cuda_backend_is_built_when_the_process_has_cuda(self, fake_backends):
        engine = create_ocr_engine("parseq_tensorrt", {"ocr": {}}, CUDA_DEVICE)
        assert engine.name == "parseq_tensorrt"

    def test_a_cpu_backend_is_unaffected_by_a_cpu_process(self, fake_backends):
        engine = create_ocr_engine("rapidocr", {"ocr": {}}, CPU_DEVICE)
        assert engine.name == "rapidocr"

    def test_the_resolved_device_reaches_the_builder(self, fake_backends):
        """Backends that pick their own device must be told, not left to guess."""
        engine = create_ocr_engine("rapidocr", {"ocr": {}}, CPU_DEVICE)
        assert engine.device is CPU_DEVICE


class TestNeverReturnsABrokenEngine:
    """Rule 2."""

    def test_a_failed_cuda_backend_falls_back(self, fake_backends):
        def explode(cfg, dev=None):
            raise ModelUnavailableError("engine file is corrupt")

        fake_backends("parseq_tensorrt", explode)
        engine = create_ocr_engine("parseq_tensorrt", {"ocr": {}}, CUDA_DEVICE)
        assert engine.name == "rapidocr"

    def test_it_raises_when_the_fallback_itself_fails(self, fake_backends):
        """Better a hard stop than a session that reads nothing all day."""
        def explode(cfg, dev=None):
            raise ModelUnavailableError("nothing works here")

        fake_backends("rapidocr", explode)
        fake_backends("parseq_tensorrt", explode)
        with pytest.raises(ModelUnavailableError):
            create_ocr_engine("parseq_tensorrt", {"ocr": {}}, CUDA_DEVICE)

    def test_the_fallback_is_not_retried_in_a_loop(self, fake_backends):
        calls: list[str] = []

        def explode(cfg, dev=None):
            calls.append("rapidocr")
            raise ModelUnavailableError("nope")

        fake_backends("rapidocr", explode)
        with pytest.raises(ModelUnavailableError):
            create_ocr_engine("rapidocr", {"ocr": {}}, CPU_DEVICE)
        assert calls == ["rapidocr"], "the fallback must be attempted exactly once"


class TestRegistry:
    """Open for extension: a new backend registers, it does not edit dispatch."""

    def test_a_new_backend_can_be_registered(self, monkeypatch):
        monkeypatch.setitem(
            ocr_module._BACKENDS, "my_backend", lambda cfg, dev=None: FakeEngine("mine", dev)
        )
        assert create_ocr_engine("my_backend", {"ocr": {}}).name == "mine"

    def test_an_unknown_backend_is_a_config_error(self):
        with pytest.raises(ValueError) as excinfo:
            create_ocr_engine("does_not_exist", {"ocr": {}})
        assert "does_not_exist" in str(excinfo.value)

    def test_the_error_lists_what_is_valid(self):
        with pytest.raises(ValueError) as excinfo:
            create_ocr_engine("does_not_exist", {"ocr": {}})
        assert "rapidocr" in str(excinfo.value)

    def test_the_shipped_backends_are_all_registered(self):
        for name in ("rapidocr", "parseq_tensorrt", "tensorrt", "paddleocr",
                     "easyocr", "trocr", "ensemble"):
            assert name in available_backends()

    def test_register_backend_is_the_public_entry_point(self, monkeypatch):
        monkeypatch.setitem(ocr_module._BACKENDS, "temp", lambda cfg, dev=None: None)
        register_backend("temp", lambda cfg, dev=None: FakeEngine("temp", dev))
        assert create_ocr_engine("temp", {"ocr": {}}).name == "temp"


class TestEasyOCRDeviceAgreement:
    """EasyOCR chooses its own device, so it must be told what we resolved."""

    def test_gpu_is_forced_off_on_a_cpu_process(self, monkeypatch):
        captured = {}

        class FakeEasyOCR:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        module = type("Module", (), {"EasyOCREngine": FakeEasyOCR})
        monkeypatch.setitem(__import__("sys").modules, "src.ocr.easyocr_engine", module)

        create_ocr_engine("easyocr", {"ocr": {"easyocr": {"gpu": True}}}, CPU_DEVICE)
        assert captured["gpu"] is False

    def test_gpu_is_left_alone_on_a_cuda_process(self, monkeypatch):
        captured = {}

        class FakeEasyOCR:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        module = type("Module", (), {"EasyOCREngine": FakeEasyOCR})
        monkeypatch.setitem(__import__("sys").modules, "src.ocr.easyocr_engine", module)

        create_ocr_engine("easyocr", {"ocr": {"easyocr": {"gpu": True}}}, CUDA_DEVICE)
        assert captured["gpu"] is True
