"""Choosing which of a camera's streams to open.

Priority, per the brief: RTSP, then WHEP, then HLS. RTSP first because it is
what the existing ALPR pipeline and stream_probe already consume, so an RTSP
camera needs no new playback path at all; WHEP next because it plays
natively in a browser at low latency; HLS last because its segment
buffering adds seconds of delay.

Two failures are kept distinct on purpose: a camera that offers streams but
has none up right now (StreamOfflineError -- try again later) is a different
fact from one that offers nothing this system can play
(NoSupportedStreamError -- no amount of waiting will help).
"""

from __future__ import annotations

from sentinel_system.portal.errors import NoSupportedStreamError, StreamOfflineError
from sentinel_system.portal.models import PortalCamera, StreamOffer, StreamType
from sentinel_system.registry.enums import Protocol

#: Explicit, not derived from StreamType's declaration order.
STREAM_PRIORITY: tuple[StreamType, ...] = (
    StreamType.RTSP,
    StreamType.WHEP,
    StreamType.HLS,
)

#: How each stream type is recorded in the camera registry's `protocol`.
REGISTRY_PROTOCOL: dict[StreamType, Protocol] = {
    StreamType.RTSP: Protocol.RTSP,
    StreamType.WHEP: Protocol.WEBRTC,
    StreamType.HLS: Protocol.HLS,
}


def select_stream(camera: PortalCamera, *, require_available: bool = True) -> StreamOffer:
    """The best stream for a camera.

    `require_available=False` picks by priority even among streams the portal
    reports as down. The registry sync uses that: a camera whose stream is
    offline right now is still a camera, and belongs in the registry marked
    unreachable -- dropping it would make a temporary outage look like a
    decommissioning.

    Raises:
        NoSupportedStreamError: it offers none of RTSP/WHEP/HLS.
        StreamOfflineError: it offers supported types, but none is up (only
            when `require_available` is true).
    """
    supported = [o for o in camera.streams if o.stream_type in STREAM_PRIORITY and o.url]
    if not supported:
        raise NoSupportedStreamError(
            f"Camera {camera.portal_id!r} offers no RTSP, WHEP or HLS stream."
        )
    for kind in STREAM_PRIORITY:
        for offer in supported:
            if offer.stream_type is kind and (offer.available or not require_available):
                return offer
    raise StreamOfflineError(
        f"Camera {camera.portal_id!r} lists "
        f"{', '.join(sorted({o.stream_type.value for o in supported}))} "
        f"but none is currently available."
    )


__all__ = ["REGISTRY_PROTOCOL", "STREAM_PRIORITY", "select_stream"]
