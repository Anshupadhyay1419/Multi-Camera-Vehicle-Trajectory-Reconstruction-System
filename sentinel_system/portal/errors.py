"""Failures talking to the official Sentinel portal.

One class per thing an operator does differently next. The API layer maps
each to its own status and stable code, so "your account cannot see that
camera" never reads the same as "the portal is down".
"""

from __future__ import annotations

from sentinel_system.core.exceptions import SentinelError


class PortalError(SentinelError):
    """Base class for Sentinel-portal integration errors."""

    code = "portal_error"


class PortalNotConfiguredError(PortalError):
    """The integration lacks something it needs before it can make a request.

    Raised instead of guessing. Carries the list of what is missing, so the
    answer to "why doesn't it work?" is a checklist rather than a stack trace.
    """

    code = "portal_not_configured"

    def __init__(self, missing: list[str]) -> None:
        self.missing = list(missing)
        super().__init__(
            "The Sentinel portal integration is not configured. Missing: "
            + "; ".join(self.missing)
        )


class AuthenticationError(PortalError):
    """The portal rejected the configured credentials."""

    code = "portal_auth_failed"


class AuthExpiredError(AuthenticationError):
    """A previously valid session or token is no longer accepted.

    Separate from AuthenticationError because the right response differs:
    an expired session is retried once after re-authenticating, a wrong
    password is not retried at all.
    """

    code = "portal_auth_expired"


class PermissionDeniedError(PortalError):
    """The account is authenticated but not allowed to see this resource."""

    code = "portal_permission_denied"


class CameraUnavailableError(PortalError):
    """The camera is not in this account's catalogue."""

    code = "portal_camera_not_found"


class StreamOfflineError(PortalError):
    """The camera is listed, but none of its streams is currently available."""

    code = "portal_stream_offline"


class NoSupportedStreamError(PortalError):
    """The camera offers no stream type this system can play (RTSP/WHEP/HLS)."""

    code = "portal_no_supported_stream"


class PortalTimeoutError(PortalError):
    """The portal did not answer in time."""

    code = "portal_timeout"


__all__ = [
    "AuthExpiredError",
    "AuthenticationError",
    "CameraUnavailableError",
    "NoSupportedStreamError",
    "PermissionDeniedError",
    "PortalError",
    "PortalNotConfiguredError",
    "PortalTimeoutError",
    "StreamOfflineError",
]
