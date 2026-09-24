"""Integration with the official Gujarat Sentinel portal.

Portal-independent parts only: credentials, errors, the client contract,
stream selection and registry sync. No client for the live portal exists yet
-- see client.REQUIRED_INFORMATION for what the official API documentation
must provide before one can be written without guessing.
"""

from sentinel_system.portal.client import (
    REQUIRED_INFORMATION,
    PortalClient,
    ResilientPortalSession,
    UnconfiguredPortalClient,
)
from sentinel_system.portal.credentials import PortalCredentials, load_credentials
from sentinel_system.portal.errors import (
    AuthExpiredError,
    AuthenticationError,
    CameraUnavailableError,
    NoSupportedStreamError,
    PermissionDeniedError,
    PortalError,
    PortalNotConfiguredError,
    PortalTimeoutError,
    StreamOfflineError,
)
from sentinel_system.portal.models import PortalCamera, StreamOffer, StreamType
from sentinel_system.portal.streams import STREAM_PRIORITY, select_stream
from sentinel_system.portal.sync import CatalogueSync, SyncReport, derive_camera_code

__all__ = [
    "AuthExpiredError", "AuthenticationError", "CameraUnavailableError",
    "CatalogueSync", "NoSupportedStreamError", "PermissionDeniedError",
    "PortalCamera", "PortalClient", "PortalCredentials", "PortalError",
    "PortalNotConfiguredError", "PortalTimeoutError", "REQUIRED_INFORMATION",
    "ResilientPortalSession", "STREAM_PRIORITY", "StreamOffer", "StreamOfflineError",
    "StreamType", "SyncReport", "UnconfiguredPortalClient", "derive_camera_code",
    "load_credentials", "select_stream",
]
