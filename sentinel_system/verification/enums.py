"""Outcome vocabulary for camera verification."""

from __future__ import annotations

from sentinel_system.registry.enums import _ValueEnum


class ConnectionStatus(_ValueEnum):
    """Why a verification ended the way it did.

    One value per thing an operator would actually do next, which is what
    makes this worth storing rather than just keeping the message:

      UNREACHABLE      the network/firewall/address is wrong  -> ask the network team
      TIMEOUT          it answers but too slowly to use       -> check load/bandwidth
      STREAM_ERROR     it answers, the stream will not open   -> check path/credentials
      NO_FRAMES        it opens and then sends nothing        -> check encoder/channel
      INVALID_URL      the stream_url is not usable at all    -> fix the registration
      REACHABLE        frames arrived and were measured
    """

    REACHABLE = "reachable"
    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    STREAM_ERROR = "stream_error"
    NO_FRAMES = "no_frames"
    INVALID_URL = "invalid_url"


__all__ = ["ConnectionStatus"]
