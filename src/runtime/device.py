"""The one place that decides which device this process computes on.

Why this module exists
----------------------
`config['detection']['device']` used to be read straight out of the dict by
each detector (`det_cfg.get("device", 0)`), and the OCR engine never read it
at all -- it simply assumed CUDA. Nothing reconciled the two, and ultralytics
turns that disagreement into a process-wide fault:

    ultralytics.utils.torch_utils.select_device(device="cpu")
        -> os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

That assignment is global and irreversible for the life of the process. The
session loads detection BEFORE OCR, so `device: "cpu"` did not merely move
detection to the CPU -- it hid the GPU from the TensorRT PARSeq engine that
loaded moments later:

    ERROR | ocr.parseq_tensorrt_engine | Failed to load PARSeq TensorRT
            engine: cuInit failed: no CUDA-capable device is detected

Detection then ran at ~125 ms/frame instead of ~2 ms, and OCR silently read
nothing at all. One config key, two subsystems, no single owner.

So device selection is resolved ONCE, up front, into an immutable
`DeviceSpec`, and that object is handed to every model that needs it. A
model no longer parses config; it is told. Anything that cannot honour the
spec says so at construction time rather than degrading quietly.

Accepted `device` values
------------------------
    "auto"            use the GPU when one is really present, else the CPU
    "cpu"             force the CPU (and, implicitly, CPU-only OCR)
    0 / "0" / 1 ...   a specific CUDA index; an error if it is not there
    "cuda"            CUDA index 0
    "cuda:1"          CUDA index 1

`half` (fp16) is a request, not a command: it is forced off on the CPU,
where fp16 is emulated and slower than fp32 rather than faster.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Union

from src.runtime.errors import ModelUnavailableError
from src.utils.logger import get_logger

_logger = get_logger("runtime.device")

# What a caller may pass as the requested device.
DeviceRequest = Union[str, int, None]

# Probe signature: () -> (cuda_is_usable, visible_device_count)
DeviceProbe = Callable[[], "tuple[bool, int]"]

_CUDA_INDEX = re.compile(r"^cuda:(\d+)$")

CUDA = "cuda"
CPU = "cpu"


@dataclass(frozen=True)
class DeviceSpec:
    """An immutable, already-validated answer to "where do the models run?".

    Frozen on purpose. This is passed to every model in the process, and a
    value that one of them could edit would put us back where we started --
    with two subsystems disagreeing about the hardware.
    """

    kind: str                  # CUDA or CPU
    index: Optional[int]       # CUDA ordinal; None on the CPU
    use_half: bool             # fp16; never true on the CPU

    @property
    def cuda_enabled(self) -> bool:
        """True when CUDA backends (TensorRT, pycuda) can actually work.

        The OCR factory consults this instead of assuming a GPU, which is
        what stops a CUDA-only backend from being built into a process that
        has already given the GPU away.
        """
        return self.kind == CUDA

    @property
    def ultralytics(self) -> Union[str, int]:
        """The value ultralytics' `device=` argument expects."""
        return self.index if self.kind == CUDA else CPU

    @property
    def torch(self) -> str:
        """The value `torch.device()` expects."""
        return f"cuda:{self.index}" if self.kind == CUDA else CPU

    def describe(self) -> str:
        if self.kind == CPU:
            return "CPU (fp32)"
        return f"CUDA:{self.index} ({'fp16' if self.use_half else 'fp32'})"


def _torch_probe() -> "tuple[bool, int]":
    """Ask torch what is really available, without ever raising.

    A probe that raised would turn a missing/broken CUDA install into a
    crash at import time on machines that are legitimately CPU-only.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False, 0
        return True, int(torch.cuda.device_count())
    except Exception as exc:                     # noqa: BLE001 - probe must not raise
        _logger.debug("CUDA probe failed, treating this machine as CPU-only: %s", exc)
        return False, 0


def cuda_hidden_process_wide() -> bool:
    """True when something has already hidden the GPU from this process.

    `CUDA_VISIBLE_DEVICES=-1` is what ultralytics sets for `device="cpu"`.
    Once set, no later call can undo it, so this is reported rather than
    repaired -- it exists so the failure is named in the log instead of
    surfacing as an unexplained `cuInit failed` several seconds later.
    """
    return os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1"


def _parse(requested: DeviceRequest) -> "tuple[str, Optional[int]]":
    """Normalise a config value to (kind, index). Raises on nonsense."""
    if requested is None:
        return "auto", None

    if isinstance(requested, bool):              # bool is an int subclass; reject it
        raise ValueError(f"Invalid detection.device: {requested!r}")

    if isinstance(requested, int):
        if requested < 0:
            return CPU, None                     # -1 is the conventional "no GPU"
        return CUDA, requested

    text = str(requested).strip().lower()
    if text in {"auto", ""}:
        return "auto", None
    if text in {CPU, "-1"}:
        return CPU, None
    if text == CUDA:
        return CUDA, 0
    if text.isdigit():
        return CUDA, int(text)
    match = _CUDA_INDEX.match(text)
    if match:
        return CUDA, int(match.group(1))

    raise ValueError(
        f"Invalid detection.device: {requested!r}. "
        f"Use 'auto', 'cpu', a CUDA index such as 0, or 'cuda:0'."
    )


def resolve_device(
    requested: DeviceRequest = None,
    half: Optional[bool] = None,
    *,
    probe: Optional[DeviceProbe] = None,
) -> DeviceSpec:
    """Turn a config value into a validated DeviceSpec, or fail loudly.

    Args:
        requested: The `detection.device` config value. See module docstring.
        half:      Whether fp16 was asked for. None means "use the default
                   for the resolved device" -- fp16 on CUDA, fp32 on CPU.
        probe:     Injectable hardware probe, so this is testable on a
                   machine with no GPU (and on one with a GPU, testable as
                   though it had none).

    Raises:
        ValueError:             the config value is not a device at all.
        ModelUnavailableError:  a SPECIFIC CUDA device was demanded and is
                                not there. Deliberately fatal: the incident
                                this module exists for was a silent 58x
                                slowdown, and an explicit `device: 0` that
                                quietly becomes the CPU is that same failure
                                wearing a different hat. Callers who want
                                graceful degradation ask for "auto", which
                                degrades loudly.
    """
    probe = probe or _torch_probe
    kind, index = _parse(requested)
    available, count = probe()

    if kind == "auto":
        if available and count > 0:
            kind, index = CUDA, 0
        else:
            reason = (
                "CUDA_VISIBLE_DEVICES=-1 is set for this process"
                if cuda_hidden_process_wide()
                else "no CUDA device is visible to torch"
            )
            # WARNING, not INFO: on a machine that is supposed to have a GPU
            # this is the difference between noticing a driver problem and
            # wondering for a week why processing got slower.
            _logger.warning(
                "detection.device='auto' resolved to the CPU because %s. "
                "Detection will run roughly 50x slower and CUDA-only OCR "
                "backends will be unavailable.",
                reason,
            )
            kind, index = CPU, None

    if kind == CUDA:
        if not available or count == 0:
            hint = (
                " CUDA_VISIBLE_DEVICES=-1 is set for this process, which "
                "ultralytics does when some earlier model was built with "
                "device='cpu'."
                if cuda_hidden_process_wide()
                else ""
            )
            raise ModelUnavailableError(
                f"detection.device={requested!r} asks for CUDA, but no CUDA "
                f"device is available.{hint} Set detection.device to 'auto' "
                f"to fall back to the CPU, or to 'cpu' to require it."
            )
        if index is None or index >= count:
            raise ModelUnavailableError(
                f"detection.device={requested!r} asks for CUDA device "
                f"{index}, but this machine has {count} (valid: 0..{count - 1})."
            )

    if kind == CPU:
        if half:
            _logger.info("Ignoring half=true: fp16 is not a speed-up on the CPU.")
        return DeviceSpec(kind=CPU, index=None, use_half=False)

    return DeviceSpec(kind=CUDA, index=index, use_half=True if half is None else bool(half))


def device_from_config(
    config: dict,
    *,
    probe: Optional[DeviceProbe] = None,
) -> DeviceSpec:
    """Resolve the device described by a full ALPR config dict.

    The single call every entry point should make, exactly once, before any
    model is constructed.
    """
    detection = config.get("detection", {}) or {}
    spec = resolve_device(
        detection.get("device"),
        detection.get("half"),
        probe=probe,
    )
    _logger.info("Models will run on %s.", spec.describe())
    return spec
