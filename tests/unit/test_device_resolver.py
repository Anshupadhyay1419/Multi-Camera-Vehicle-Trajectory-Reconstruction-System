"""
Unit tests for the single owner of device selection.

The incident these guard against: `detection.device: "cpu"` did not merely
move detection to the CPU. ultralytics' select_device() sets
CUDA_VISIBLE_DEVICES=-1 for the whole process, so the TensorRT OCR engine
loaded moments later died with "cuInit failed: no CUDA-capable device is
detected" -- and then answered ("", 0.0) for every frame of a four-video
session without anything looking wrong. Detection ran ~58x slower and zero
plates were stored.

Every test here injects its own hardware probe, so the suite gives the same
answers on a GPU box and a CPU-only one.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.runtime.device import (
    DeviceSpec,
    cuda_hidden_process_wide,
    device_from_config,
    resolve_device,
)
from src.runtime.errors import ModelUnavailableError


def gpu(count: int = 1):
    """A probe reporting *count* usable CUDA devices."""
    return lambda: (True, count)


def no_gpu():
    """A probe reporting a CPU-only machine."""
    return lambda: (False, 0)


class TestRequestParsing:
    """Every spelling a config file might reasonably use."""

    @pytest.mark.parametrize(
        "requested,expected_index",
        [(0, 0), ("0", 0), (1, 1), ("cuda", 0), ("cuda:0", 0), ("cuda:1", 1)],
    )
    def test_cuda_spellings_resolve_to_that_index(self, requested, expected_index):
        spec = resolve_device(requested, probe=gpu(count=2))
        assert spec.kind == "cuda"
        assert spec.index == expected_index

    @pytest.mark.parametrize("requested", ["cpu", "CPU", " cpu ", -1, "-1"])
    def test_cpu_spellings_resolve_to_cpu(self, requested):
        assert resolve_device(requested, probe=gpu()).kind == "cpu"

    @pytest.mark.parametrize("requested", [None, "auto", "AUTO", ""])
    def test_auto_spellings_take_the_gpu_when_there_is_one(self, requested):
        assert resolve_device(requested, probe=gpu()).kind == "cuda"

    @pytest.mark.parametrize("requested", ["gpu", "cuda:", "cuda:x", "first", 1.5, "0.5"])
    def test_nonsense_is_a_config_error(self, requested):
        """A typo must not quietly become the CPU -- that is the whole bug."""
        with pytest.raises(ValueError):
            resolve_device(requested, probe=gpu())

    def test_booleans_are_rejected(self):
        """`device: true` is a YAML accident, not CUDA device 1."""
        with pytest.raises(ValueError):
            resolve_device(True, probe=gpu())


class TestAutoDegradesLoudly:
    def test_auto_falls_back_to_cpu_without_a_gpu(self):
        assert resolve_device("auto", probe=no_gpu()).kind == "cpu"

    def test_auto_fallback_is_logged_as_a_warning(self, monkeypatch):
        """INFO would be lost in the noise; this is a 50x slowdown.

        Spies the module logger rather than using caplog: this project's
        loggers set propagate=False (src/utils/logger.py), so caplog's root
        handler never sees these records.
        """
        import src.runtime.device as device_module

        warnings: list[str] = []
        monkeypatch.setattr(
            device_module._logger,
            "warning",
            lambda msg, *args, **kwargs: warnings.append(str(msg) % args if args else str(msg)),
        )
        resolve_device("auto", probe=no_gpu())
        assert warnings, "a silent fallback to the CPU is the bug, not the fix"
        assert "CPU" in warnings[0]


class TestExplicitCudaFailsFast:
    """An explicit `device: 0` that silently became the CPU IS the incident."""

    def test_demanding_cuda_without_a_gpu_raises(self):
        with pytest.raises(ModelUnavailableError):
            resolve_device(0, probe=no_gpu())

    def test_an_out_of_range_index_raises(self):
        with pytest.raises(ModelUnavailableError) as excinfo:
            resolve_device(3, probe=gpu(count=2))
        assert "0..1" in str(excinfo.value)

    def test_the_error_names_the_process_wide_cause_when_that_is_it(self, monkeypatch):
        """The actionable detail: something already hid the GPU."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        with pytest.raises(ModelUnavailableError) as excinfo:
            resolve_device(0, probe=no_gpu())
        assert "CUDA_VISIBLE_DEVICES" in str(excinfo.value)

    def test_the_error_suggests_the_graceful_alternative(self):
        with pytest.raises(ModelUnavailableError) as excinfo:
            resolve_device(0, probe=no_gpu())
        assert "auto" in str(excinfo.value)


class TestHalfPrecision:
    def test_fp16_is_the_default_on_cuda(self):
        assert resolve_device(0, probe=gpu()).use_half is True

    def test_fp16_can_be_declined_on_cuda(self):
        assert resolve_device(0, half=False, probe=gpu()).use_half is False

    def test_fp16_is_forced_off_on_cpu(self):
        """fp16 on the CPU is emulated: slower than fp32, not faster."""
        assert resolve_device("cpu", half=True, probe=gpu()).use_half is False


class TestSpecIsAUsableValueObject:
    def test_the_spec_is_immutable(self):
        """Every model in the process shares this; none may edit it."""
        spec = resolve_device(0, probe=gpu())
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.kind = "cpu"          # type: ignore[misc]

    def test_cuda_enabled_tells_the_ocr_factory_what_it_needs(self):
        assert resolve_device(0, probe=gpu()).cuda_enabled is True
        assert resolve_device("cpu", probe=gpu()).cuda_enabled is False

    def test_it_speaks_ultralytics(self):
        assert resolve_device(1, probe=gpu(count=2)).ultralytics == 1
        assert resolve_device("cpu", probe=gpu()).ultralytics == "cpu"

    def test_it_speaks_torch(self):
        assert resolve_device(1, probe=gpu(count=2)).torch == "cuda:1"
        assert resolve_device("cpu", probe=gpu()).torch == "cpu"

    def test_describe_is_human_readable(self):
        assert resolve_device(0, probe=gpu()).describe() == "CUDA:0 (fp16)"
        assert resolve_device("cpu", probe=gpu()).describe() == "CPU (fp32)"

    def test_equal_specs_compare_equal(self):
        assert DeviceSpec("cuda", 0, True) == DeviceSpec("cuda", 0, True)


class TestProbeIsNeverAllowedToExplode:
    def test_a_broken_probe_is_not_swallowed_into_a_wrong_answer(self):
        """A probe that raises is a bug in the probe, not a CPU machine.

        The real torch probe catches its own exceptions and reports
        "no CUDA"; an injected probe that raises should surface, so a test
        helper's mistake is never read as hardware.
        """
        def explode():
            raise RuntimeError("probe is broken")
        with pytest.raises(RuntimeError):
            resolve_device("auto", probe=explode)


class TestFromConfig:
    def test_it_reads_the_detection_block(self):
        config = {"detection": {"device": "cpu", "half": True}}
        assert device_from_config(config, probe=gpu()).kind == "cpu"

    def test_a_missing_detection_block_is_auto(self):
        assert device_from_config({}, probe=gpu()).kind == "cuda"

    def test_a_null_device_is_auto(self):
        config = {"detection": {"device": None}}
        assert device_from_config(config, probe=no_gpu()).kind == "cpu"

    def test_the_shipped_config_resolves(self):
        """Guards the real file against a typo that only bites at runtime."""
        from src.utils.config import load_config

        spec = device_from_config(load_config("config/config.yaml"), probe=gpu())
        assert spec.kind in {"cuda", "cpu"}


class TestProcessWideCudaVisibility:
    def test_it_detects_the_hidden_gpu(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        assert cuda_hidden_process_wide() is True

    def test_unset_is_not_hidden(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert cuda_hidden_process_wide() is False

    def test_a_real_device_list_is_not_hidden(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        assert cuda_hidden_process_wide() is False
