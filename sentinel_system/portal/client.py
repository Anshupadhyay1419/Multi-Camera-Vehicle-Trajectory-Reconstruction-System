"""The contract any Sentinel-portal client must satisfy, and what stands in
for one until the official API is documented to us.

There is deliberately NO HTTP client for sentinel.gujarat.gov.in in this
module. Writing one needs the portal's documented authentication scheme,
catalogue endpoint and response format, none of which we have; the
alternative -- reading the portal's web frontend to find its internal calls
-- is scraping by another name and is out of bounds. Until the documentation
arrives, `UnconfiguredPortalClient` is the client, and every request answers
with the checklist below instead of a guess.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.utils.logger import get_logger

from sentinel_system.portal.errors import (
    AuthExpiredError,
    CameraUnavailableError,
    PortalNotConfiguredError,
)
from sentinel_system.portal.models import PortalCamera

_logger = get_logger("sentinel.portal")

#: What the official Sentinel API documentation must tell us before a real
#: client can be written. Returned verbatim by the unconfigured client and
#: by the API, so the gap is visible from the running system, not only here.
REQUIRED_INFORMATION: tuple[str, ...] = (
    "The documented API base URL (distinct from the /resource web page).",
    "The authentication scheme: login endpoint and request format, or OAuth2 "
    "flow, or API-key header -- the issued credential has the shape of an "
    "access key (XXXX-XXXX-XXXX), which suggests key-based auth.",
    "Token/session lifetime and how it is refreshed.",
    "The camera catalogue endpoint, its pagination, and its response fields "
    "for id, name, department, district/location, coordinates and status.",
    "How stream URLs are issued per camera, which types are offered "
    "(RTSP / WHEP / HLS), and whether they are signed or expire.",
    "Status codes the API uses for expired sessions, missing permissions and "
    "offline cameras.",
    "Rate limits and the terms under which programmatic access is permitted.",
    "Network prerequisites (IP allowlisting, VPN) for this server to reach it.",
)


@runtime_checkable
class PortalClient(Protocol):
    """What the rest of the integration needs from a portal client."""

    def authenticate(self) -> None:
        """Establish (or re-establish) a session. Raises AuthenticationError."""

    def list_cameras(self) -> list[PortalCamera]:
        """Every camera visible to the authenticated account."""

    def get_camera(self, portal_id: str) -> PortalCamera:
        """One camera, including its current stream offers.

        Raises CameraUnavailableError if the account cannot see it.
        """


class UnconfiguredPortalClient:
    """The client in use until the official API is documented.

    Every method raises PortalNotConfiguredError carrying both what is
    missing from configuration and what is missing from the documentation,
    so a caller always learns exactly why nothing was fetched.
    """

    def __init__(self, missing_settings: list[str] | None = None) -> None:
        self.missing = list(missing_settings or []) + [
            f"Official API documentation: {item}" for item in REQUIRED_INFORMATION
        ]

    def authenticate(self) -> None:
        raise PortalNotConfiguredError(self.missing)

    def list_cameras(self) -> list[PortalCamera]:
        raise PortalNotConfiguredError(self.missing)

    def get_camera(self, portal_id: str) -> PortalCamera:
        raise PortalNotConfiguredError(self.missing)


class ResilientPortalSession:
    """Wraps any PortalClient and survives one session expiry per call.

    Authenticates lazily on first use. If a call fails with AuthExpiredError,
    it re-authenticates once and retries that call once. A second expiry is
    re-raised rather than looped on: an account whose fresh session is
    immediately rejected has a real problem, and retrying forever would hide
    it while hammering the portal.

    A wrong password (AuthenticationError that is NOT an expiry) is never
    retried.
    """

    def __init__(self, client: PortalClient) -> None:
        self.client = client
        self._authenticated = False

    def _ensure_session(self) -> None:
        if not self._authenticated:
            self.client.authenticate()
            self._authenticated = True

    def _call(self, operation, *args):
        self._ensure_session()
        try:
            return operation(*args)
        except AuthExpiredError:
            _logger.info("Sentinel portal session expired; re-authenticating once")
            self._authenticated = False
            self._ensure_session()
            return operation(*args)

    def list_cameras(self) -> list[PortalCamera]:
        return self._call(self.client.list_cameras)

    def get_camera(self, portal_id: str) -> PortalCamera:
        camera = self._call(self.client.get_camera, portal_id)
        if camera is None:  # a client that returns None instead of raising
            raise CameraUnavailableError(f"No camera {portal_id!r} in this account's catalogue")
        return camera


__all__ = [
    "PortalClient",
    "REQUIRED_INFORMATION",
    "ResilientPortalSession",
    "UnconfiguredPortalClient",
]
