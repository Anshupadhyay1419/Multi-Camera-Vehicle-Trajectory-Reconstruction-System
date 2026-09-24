"""Registry domain errors.

The service layer raises these instead of returning None or leaking
SQLAlchemy's IntegrityError. Two reasons: the caller should not have to
know which ORM is underneath, and "no such camera" and "that code is
already taken" are different outcomes that an API layer must map to
different HTTP statuses (404 vs 409). Returning None collapses them.
"""

from __future__ import annotations

import uuid

from sentinel_system.core.exceptions import SentinelError


class RegistryError(SentinelError):
    """Base class for camera-registry errors."""

    code = "registry_error"


class CameraNotFoundError(RegistryError):
    """No camera matches the given identifier."""

    code = "camera_not_found"

    def __init__(
        self,
        *,
        camera_id: uuid.UUID | None = None,
        camera_code: str | None = None,
    ) -> None:
        if camera_id is not None:
            message = f"No camera with id {camera_id}"
        elif camera_code is not None:
            message = f"No camera with code {camera_code!r}"
        else:  # pragma: no cover - defensive
            message = "No such camera"
        super().__init__(message)
        self.camera_id = camera_id
        self.camera_code = camera_code


class DuplicateCameraCodeError(RegistryError):
    """A camera with that code already exists.

    camera_code is the one field operators type by hand, so colliding on it
    is an ordinary mistake rather than an exceptional one, and the message
    names the offending value so the caller can show it without unpicking
    a database constraint string.
    """

    code = "duplicate_camera_code"

    def __init__(self, camera_code: str) -> None:
        super().__init__(f"Camera code {camera_code!r} is already registered")
        self.camera_code = camera_code


__all__ = ["CameraNotFoundError", "DuplicateCameraCodeError", "RegistryError"]
