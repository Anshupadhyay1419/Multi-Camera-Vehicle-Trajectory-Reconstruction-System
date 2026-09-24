"""Closed vocabularies for the camera registry.

What is an enum here and what is a plain string is a deliberate split.

Enums are for sets whose members the *code* reasons about: a scheduler
checks `status is CameraStatus.ACTIVE`, a health monitor compares against
`HealthStatus.UNHEALTHY`, a stream reader switches on `Protocol.RTSP`.
Adding a member to one of these means new behaviour, so it belongs in a
migration and a code review.

Department, owner, zone and district are NOT enums. They are the Gujarat
Police organisational structure, which changes by administrative order
rather than by software release -- districts get reorganised, ranges get
renamed. Baking that into a PostgreSQL ENUM would turn a clerical update
into a schema migration, so they are validated, indexed strings instead.
"""

from __future__ import annotations

from enum import Enum


class _ValueEnum(str, Enum):
    """String enum whose `str()` is its value.

    Inheriting from `str` keeps these directly usable as SQLAlchemy and
    Pydantic values, and `__str__` stops f-strings rendering the awkward
    'CameraStatus.ACTIVE' form in logs and error messages.
    """

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]


class CameraStatus(_ValueEnum):
    """Administrative lifecycle of a camera record."""

    PLANNED = "planned"              # approved, not yet physically installed
    COMMISSIONING = "commissioning"  # installed, under acceptance testing
    ACTIVE = "active"                # in service, expected to stream
    INACTIVE = "inactive"            # deliberately switched off
    SUSPENDED = "suspended"          # withheld from service (legal/admin hold)
    DECOMMISSIONED = "decommissioned"  # permanently retired


class HealthStatus(_ValueEnum):
    """Observed runtime condition, as last measured by a health check.

    Distinct from CameraStatus on purpose: an ACTIVE camera can be
    UNREACHABLE, and that combination -- expected to work, isn't -- is
    precisely what an operations dashboard needs to surface.
    """

    UNKNOWN = "unknown"          # never checked, or check results expired
    HEALTHY = "healthy"
    DEGRADED = "degraded"        # streaming, but below spec (fps/bitrate/loss)
    UNHEALTHY = "unhealthy"      # reachable, not usable
    UNREACHABLE = "unreachable"  # no network response


class MaintenanceStatus(_ValueEnum):
    """Where the camera sits in the maintenance cycle."""

    NONE = "none"                    # nothing scheduled or outstanding
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    OVERDUE = "overdue"
    AWAITING_PARTS = "awaiting_parts"


class CameraType(_ValueEnum):
    """Physical form factor / role of the unit."""

    FIXED = "fixed"
    DOME = "dome"
    BULLET = "bullet"
    PTZ = "ptz"
    ANPR = "anpr"            # number-plate capture optics
    THERMAL = "thermal"
    PANORAMIC = "panoramic"  # multi-sensor / 180-360 degree
    BODY_WORN = "body_worn"
    DASHBOARD = "dashboard"


class Protocol(_ValueEnum):
    """Transport the stream is published over."""

    RTSP = "rtsp"
    RTMP = "rtmp"
    ONVIF = "onvif"
    HTTP = "http"
    HTTPS = "https"
    HLS = "hls"
    WEBRTC = "webrtc"
    SRT = "srt"

    @property
    def url_schemes(self) -> tuple[str, ...]:
        """URL schemes that are legitimate for this protocol.

        ONVIF is a device-management standard rather than a transport: a
        camera is discovered and configured over ONVIF's HTTP service, and
        the media it hands back is RTSP. So an ONVIF camera's stream_url
        may legitimately be any of the three.
        """
        return _PROTOCOL_SCHEMES[self]


_PROTOCOL_SCHEMES: dict[Protocol, tuple[str, ...]] = {
    Protocol.RTSP: ("rtsp", "rtsps"),
    Protocol.RTMP: ("rtmp", "rtmps"),
    Protocol.ONVIF: ("onvif", "http", "https", "rtsp"),
    Protocol.HTTP: ("http",),
    Protocol.HTTPS: ("https",),
    Protocol.HLS: ("http", "https"),
    Protocol.WEBRTC: ("http", "https", "ws", "wss"),
    Protocol.SRT: ("srt",),
}


class Codec(_ValueEnum):
    """Video compression the stream is encoded with."""

    H264 = "h264"
    H265 = "h265"
    MJPEG = "mjpeg"
    MPEG4 = "mpeg4"
    AV1 = "av1"
    VP8 = "vp8"
    VP9 = "vp9"


__all__ = [
    "CameraStatus",
    "CameraType",
    "Codec",
    "HealthStatus",
    "MaintenanceStatus",
    "Protocol",
]
