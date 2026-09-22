"""Process-wide runtime facts the models depend on.

Everything here answers questions that must have exactly ONE answer per
process -- which compute device the models run on, and what to do when one
cannot be brought up. Those answers used to be re-derived independently by
each detector and assumed outright by the OCR engine, which is how a
process could end up running detection on the CPU while the OCR engine
still believed it had a GPU.
"""

from src.runtime.device import DeviceSpec, resolve_device, device_from_config
from src.runtime.errors import ModelUnavailableError

__all__ = [
    "DeviceSpec",
    "ModelUnavailableError",
    "device_from_config",
    "resolve_device",
]
