"""Pydantic v2 schemas for the camera registry.

Four shapes, because they have four different contracts:

  CameraCreate  what a caller must supply to register a camera
  CameraUpdate  a partial patch; every field optional, unset != null
  CameraRead    what the platform returns, including server-owned fields
  CameraFilter  query parameters for listing

The create/update split matters. A single schema with everything optional
cannot express "latitude is required to register a camera", and a single
schema with everything required cannot express "just change the firmware
version". Separating them is what lets the update path distinguish a field
the caller omitted from a field the caller explicitly set to null -- see
`changed_fields()`.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Annotated, Any, ClassVar
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from sentinel_system.registry.enums import (
    CameraStatus,
    CameraType,
    Codec,
    HealthStatus,
    MaintenanceStatus,
    Protocol,
)

# Uppercase alphanumeric, with dashes/underscores inside. Matches the
# AHM-SAT-0142 style used on enclosure labels, and rules out whitespace and
# punctuation that would make the code unusable in a URL path or a log grep.
CAMERA_CODE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9_-]{1,62}[A-Z0-9]$")
RESOLUTION_PATTERN = re.compile(r"^\d{2,5}x\d{2,5}$")

Latitude = Annotated[float, Field(ge=-90, le=90, description="WGS84 degrees")]
Longitude = Annotated[float, Field(ge=-180, le=180, description="WGS84 degrees")]
Bearing = Annotated[
    float, Field(ge=0, lt=360, description="Compass degrees, 0=north, clockwise")
]


def _normalise_code(value: str) -> str:
    """Uppercase and trim a camera code, then check its shape.

    Normalising before validating (rather than rejecting lowercase) is
    deliberate: operators type these by hand off a label, and 'ahm-sat-0142'
    is not a different camera from 'AHM-SAT-0142'. Folding case here is also
    what makes the uniqueness check in the service layer meaningful.
    """
    code = value.strip().upper()
    if not CAMERA_CODE_PATTERN.match(code):
        raise ValueError(
            "camera_code must be 3-64 characters of A-Z, 0-9, '-' or '_', "
            "starting and ending alphanumeric"
        )
    return code


def _check_stream_url(url: str, protocol: Protocol) -> str:
    """Reject a stream URL whose scheme contradicts the declared protocol.

    A row saying protocol=RTSP with stream_url='https://...' is not a
    harmless inconsistency: the stream reader in a later module picks its
    client off `protocol` and would fail at connect time, far from the
    mistake. Catching it at registration costs nothing.
    """
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        raise ValueError("stream_url must include a scheme, e.g. rtsp://host/path")
    if not parsed.netloc:
        raise ValueError("stream_url must include a host")
    allowed = protocol.url_schemes
    if parsed.scheme.lower() not in allowed:
        raise ValueError(
            f"stream_url scheme {parsed.scheme!r} does not match protocol "
            f"{protocol.value!r}; expected one of {', '.join(allowed)}"
        )
    return url.strip()


class _CameraFields(BaseModel):
    """Field definitions shared by create and read.

    Not a public schema on its own -- it exists so the descriptions and
    constraints are written once instead of drifting between the shapes.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        use_enum_values=False,
        extra="forbid",
        validate_assignment=True,
    )

    # Internal
    camera_code: str = Field(description="Operator-facing code, e.g. AHM-SAT-0142")
    camera_name: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=4000)

    # Ownership
    department: str = Field(min_length=1, max_length=120)
    owner: str | None = Field(default=None, max_length=120)
    zone: str | None = Field(default=None, max_length=120)
    district: str = Field(min_length=1, max_length=120)

    # Location
    latitude: Latitude
    longitude: Longitude
    address: str | None = Field(default=None, max_length=4000)

    # Camera details
    vendor: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    serial_number: str | None = Field(default=None, max_length=120)
    camera_type: CameraType
    protocol: Protocol
    stream_url: str = Field(min_length=1)
    codec: Codec | None = None
    resolution: str | None = Field(default=None, description="WIDTHxHEIGHT, e.g. 1920x1080")
    fps: int | None = Field(default=None, gt=0, le=240)

    # Status
    status: CameraStatus = CameraStatus.PLANNED
    health: HealthStatus = HealthStatus.UNKNOWN
    maintenance_status: MaintenanceStatus = MaintenanceStatus.NONE
    last_seen: datetime | None = None
    last_health_check: datetime | None = None

    # Installation
    installed_on: datetime | None = None
    last_service: datetime | None = None
    firmware_version: str | None = Field(default=None, max_length=64)

    # Capabilities
    supports_ptz: bool = False
    supports_audio: bool = False
    supports_nightvision: bool = False
    supports_analytics: bool = False

    # GIS
    coverage_radius_m: float | None = Field(default=None, ge=0, le=100_000)
    bearing_deg: Bearing | None = None

    @field_validator("camera_code")
    @classmethod
    def _validate_code(cls, value: str) -> str:
        return _normalise_code(value)

    @field_validator("resolution")
    @classmethod
    def _validate_resolution(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower().replace("*", "x").replace("X", "x")
        if not RESOLUTION_PATTERN.match(cleaned):
            raise ValueError("resolution must look like '1920x1080'")
        return cleaned

    @model_validator(mode="after")
    def _validate_stream(self) -> _CameraFields:
        # object.__setattr__ rather than plain assignment: validate_assignment
        # is on, so assigning here would re-enter this validator.
        object.__setattr__(
            self, "stream_url", _check_stream_url(self.stream_url, self.protocol)
        )
        return self

    @model_validator(mode="after")
    def _validate_ptz_consistency(self) -> _CameraFields:
        """A PTZ-type camera that cannot pan/tilt/zoom is a data-entry slip."""
        if self.camera_type is CameraType.PTZ and not self.supports_ptz:
            raise ValueError(
                "camera_type 'ptz' requires supports_ptz=true; set one or the other"
            )
        return self


class CameraCreate(_CameraFields):
    """Payload to register a new camera."""


class CameraRead(BaseModel):
    """A camera as the platform reports it, including server-owned fields."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    camera_code: str
    camera_name: str
    description: str | None

    department: str
    owner: str | None
    zone: str | None
    district: str

    latitude: float
    longitude: float
    address: str | None

    vendor: str | None
    model: str | None
    serial_number: str | None
    camera_type: CameraType
    protocol: Protocol
    stream_url: str
    codec: Codec | None
    resolution: str | None
    fps: int | None

    status: CameraStatus
    health: HealthStatus
    maintenance_status: MaintenanceStatus
    last_seen: datetime | None
    last_health_check: datetime | None

    installed_on: datetime | None
    last_service: datetime | None
    firmware_version: str | None

    supports_ptz: bool
    supports_audio: bool
    supports_nightvision: bool
    supports_analytics: bool

    coverage_radius_m: float | None
    bearing_deg: float | None

    created_at: datetime
    updated_at: datetime


class CameraUpdate(BaseModel):
    """A partial update. Every field optional; omitted fields are untouched.

    `camera_code` is intentionally absent. It is the identifier printed on
    the hardware and quoted in incident reports, so re-pointing it at a
    different physical unit silently invalidates historical references. A
    code change means retiring the record and registering a new one.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True, extra="forbid", validate_assignment=True
    )

    camera_name: str | None = Field(default=None, min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=4000)

    department: str | None = Field(default=None, min_length=1, max_length=120)
    owner: str | None = Field(default=None, max_length=120)
    zone: str | None = Field(default=None, max_length=120)
    district: str | None = Field(default=None, min_length=1, max_length=120)

    latitude: Latitude | None = None
    longitude: Longitude | None = None
    address: str | None = Field(default=None, max_length=4000)

    vendor: str | None = Field(default=None, max_length=120)
    model: str | None = Field(default=None, max_length=120)
    serial_number: str | None = Field(default=None, max_length=120)
    camera_type: CameraType | None = None
    protocol: Protocol | None = None
    stream_url: str | None = Field(default=None, min_length=1)
    codec: Codec | None = None
    resolution: str | None = None
    fps: int | None = Field(default=None, gt=0, le=240)

    status: CameraStatus | None = None
    health: HealthStatus | None = None
    maintenance_status: MaintenanceStatus | None = None
    last_seen: datetime | None = None
    last_health_check: datetime | None = None

    installed_on: datetime | None = None
    last_service: datetime | None = None
    firmware_version: str | None = Field(default=None, max_length=64)

    supports_ptz: bool | None = None
    supports_audio: bool | None = None
    supports_nightvision: bool | None = None
    supports_analytics: bool | None = None

    coverage_radius_m: float | None = Field(default=None, ge=0, le=100_000)
    bearing_deg: Bearing | None = None

    @field_validator("resolution")
    @classmethod
    def _validate_resolution(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower().replace("*", "x").replace("X", "x")
        if not RESOLUTION_PATTERN.match(cleaned):
            raise ValueError("resolution must look like '1920x1080'")
        return cleaned

    def changed_fields(self) -> dict[str, Any]:
        """Only the fields the caller actually supplied.

        `exclude_unset` is the whole point: with it, `{"owner": null}`
        clears the owner and `{}` leaves it alone. Without it, every
        unmentioned field would be sent as its default and a patch would
        silently blank the rest of the record.
        """
        return self.model_dump(exclude_unset=True)


class CameraFilter(BaseModel):
    """Filters and pagination for listing cameras."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    department: str | None = None
    district: str | None = None
    zone: str | None = None
    status: CameraStatus | None = None
    health: HealthStatus | None = None
    camera_type: CameraType | None = None
    protocol: Protocol | None = None
    vendor: str | None = None
    supports_analytics: bool | None = None
    #: Decommissioned cameras are hidden by default. A soft delete that
    #: left the camera in every listing would not be a delete at all. An
    #: explicit `status=decommissioned` filter still finds them, so nothing
    #: becomes unreachable -- it just stops being the default view.
    include_decommissioned: bool = False
    #: Case-insensitive substring match over camera_code and camera_name.
    search: str | None = Field(default=None, max_length=120)

    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    order_by: str = Field(default="camera_code")
    descending: bool = False

    #: Allow-list, because order_by reaches a SQL ORDER BY clause. Anything
    #: not named here is rejected rather than interpolated. ClassVar, so
    #: Pydantic treats it as a constant instead of a filter field.
    ORDERABLE: ClassVar[frozenset[str]] = frozenset(
        {
            "camera_code",
            "camera_name",
            "district",
            "department",
            "status",
            "health",
            "last_seen",
            "created_at",
            "updated_at",
        }
    )

    @field_validator("order_by")
    @classmethod
    def _validate_order_by(cls, value: str) -> str:
        allowed = CameraFilter.ORDERABLE
        if value not in allowed:
            raise ValueError(
                f"order_by must be one of: {', '.join(sorted(allowed))}"
            )
        return value


class CameraPage(BaseModel):
    """One page of results, plus enough context to request the next."""

    model_config = ConfigDict(from_attributes=True)

    items: list[CameraRead]
    total: int = Field(description="Rows matching the filter, ignoring pagination")
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


__all__ = [
    "CameraCreate",
    "CameraFilter",
    "CameraPage",
    "CameraRead",
    "CameraUpdate",
]
