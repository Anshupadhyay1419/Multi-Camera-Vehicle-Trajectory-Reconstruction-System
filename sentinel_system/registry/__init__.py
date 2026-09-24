"""Camera registry: the system of record for what cameras exist and where."""

from sentinel_system.registry.enums import (
    CameraStatus,
    CameraType,
    Codec,
    HealthStatus,
    MaintenanceStatus,
    Protocol,
)
from sentinel_system.registry.exceptions import (
    CameraNotFoundError,
    DuplicateCameraCodeError,
    RegistryError,
)
from sentinel_system.registry.models import Camera
from sentinel_system.registry.repository import CameraRepository
from sentinel_system.registry.schemas import (
    CameraCreate,
    CameraFilter,
    CameraPage,
    CameraRead,
    CameraUpdate,
)
from sentinel_system.registry.service import CameraService

__all__ = [
    "Camera",
    "CameraCreate",
    "CameraFilter",
    "CameraNotFoundError",
    "CameraPage",
    "CameraRead",
    "CameraRepository",
    "CameraService",
    "CameraStatus",
    "CameraType",
    "CameraUpdate",
    "Codec",
    "DuplicateCameraCodeError",
    "HealthStatus",
    "MaintenanceStatus",
    "Protocol",
    "RegistryError",
]
