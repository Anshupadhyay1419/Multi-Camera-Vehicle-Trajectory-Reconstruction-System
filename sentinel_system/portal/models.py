"""The integration's own description of a portal camera.

This is OUR contract, not the portal's response format. The official
catalogue's field names and structure are not documented to us yet, so
nothing here pretends to know them. Whatever client is eventually written
against the documented API is responsible for mapping the portal's response
INTO these types; everything downstream (stream selection, registry sync,
the API) depends only on this shape and never on the portal's.

That split is what lets the portal-independent parts be built and tested
now, and keeps the eventual client a thin, reviewable mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sentinel_system.registry.enums import CameraType, _ValueEnum


class StreamType(_ValueEnum):
    """Stream kinds this system can consume, in no particular order.

    Priority lives in streams.py, not in declaration order here, so that
    reordering the enum can never silently change which stream is opened.
    """

    RTSP = "rtsp"
    WHEP = "whep"   # WebRTC-HTTP Egress Protocol; plays natively in browsers
    HLS = "hls"


@dataclass(frozen=True)
class StreamOffer:
    """One way the portal says a camera can be watched."""

    stream_type: StreamType
    url: str
    #: False when the portal lists the stream but reports it down.
    available: bool = True


@dataclass(frozen=True)
class PortalCamera:
    """A camera as the portal describes it, normalised.

    Required registry fields (district, coordinates) are optional HERE on
    purpose: if the portal does not provide them, the registry sync reports
    the camera as skipped with the exact reason rather than inventing a
    district or a location to get it through validation.
    """

    portal_id: str
    name: str
    department: str | None = None
    district: str | None = None
    location: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    #: The portal's own status word, kept verbatim for display.
    status: str | None = None
    online: bool | None = None
    streams: tuple[StreamOffer, ...] = field(default_factory=tuple)
    #: The catalogue does not describe optics. FIXED is the registry's most
    #: neutral type (it claims nothing about pan/tilt); ops can correct it.
    camera_type: CameraType = CameraType.FIXED

    def availability(self) -> dict[str, bool]:
        """Which stream types are offered AND currently up, for display."""
        return {
            kind.value: any(
                offer.stream_type is kind and offer.available
                for offer in self.streams
            )
            for kind in StreamType
        }


__all__ = ["PortalCamera", "StreamOffer", "StreamType"]
