"""Base exception type for the platform.

Every domain error inherits from SentinelError, so a caller that does not
care which module raised can still catch everything this platform throws
without also swallowing ValueError, OSError and friends.
"""

from __future__ import annotations


class SentinelError(Exception):
    """Base class for all Sentinel domain errors."""

    #: Short, stable, machine-readable identifier. API layers (M1.2) will
    #: map this onto an error envelope, which is why it lives on the
    #: exception rather than being reconstructed from the class name later.
    code: str = "sentinel_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code

    def __str__(self) -> str:
        return self.message
