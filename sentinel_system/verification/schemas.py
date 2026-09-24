"""Pydantic v2 schemas for camera verification."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from sentinel_system.verification.enums import ConnectionStatus


class VerificationResult(BaseModel):
    """The outcome of one verification attempt.

    `verified` is the single boolean an operator acts on; everything else
    explains it. A failed verification is still a RESULT, not an error --
    it is stored, returned with HTTP 200, and carries the same shape as a
    success with the measurements left null. The alternative, raising on
    failure, would make "this camera is unreachable" indistinguishable from
    "the verification endpoint is broken".
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    camera_id: uuid.UUID
    verified: bool = Field(description="Whether usable video was actually read")
    connection_status: ConnectionStatus
    verification_message: str = Field(description="One line, written for an operator")

    measured_resolution: str | None = Field(
        default=None, description="As delivered, e.g. '1920x1080'"
    )
    measured_fps: float | None = Field(
        default=None, description="From the gaps between frames, not the stream's claim"
    )
    measured_latency_ms: float | None = Field(
        default=None, description="Connect + open + first frame"
    )
    connect_latency_ms: float | None = None
    first_frame_latency_ms: float | None = None
    frames_sampled: int = 0
    codec_detected: str | None = Field(
        default=None, description="Often absent over RTSP, which carries no FOURCC"
    )
    thumbnail_path: str | None = Field(
        default=None, description="Served under /thumbnails/cameras/"
    )
    errors: list[str] = Field(default_factory=list)
    checked_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "9d2c1f6e-0d9a-4a21-9b53-5f3e2a7c1b44",
                "camera_id": "72168e0b-e9be-48cd-a5da-ef395e951000",
                "verified": True,
                "connection_status": "reachable",
                "verification_message": "Read 5 frame(s) at 1920x1080.",
                "measured_resolution": "1920x1080",
                "measured_fps": 24.9,
                "measured_latency_ms": 42.0,
                "connect_latency_ms": 3.1,
                "first_frame_latency_ms": 38.9,
                "frames_sampled": 5,
                "codec_detected": "hevc",
                "thumbnail_path": "/thumbnails/cameras/72168e0b.jpg",
                "errors": [],
                "checked_at": "2026-09-23T14:33:01Z",
            }
        },
    )


__all__ = ["VerificationResult"]
