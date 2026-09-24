"""Where verification results are stored.

A table of its own, deliberately. The brief's rule -- do not overwrite
camera metadata -- is the important one here, and it is not merely about
tidiness: `Camera.resolution`, `Camera.fps` and `Camera.codec` are what the
camera was REGISTERED as, taken from a purchase order or a survey sheet,
while these columns are what it actually DELIVERED when someone last
looked. Folding the measurement back into the registration would destroy
the only thing that makes either useful, which is the discrepancy between
them: a camera commissioned as 1080p25 that verifies at 704x576 is exactly
the finding this milestone exists to surface.

Append-only, one row per attempt. Verification is explicitly operator-
triggered (M1.3 is not monitoring), so the table grows a row per click, and
keeping the history costs nothing while answering "was this working last
week?" -- which is the first question asked when a camera goes dark.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from sentinel_system.core.database import Base
from sentinel_system.verification.enums import ConnectionStatus


class CameraVerification(Base):
    """One recorded attempt to prove a camera is usable."""

    __tablename__ = "camera_verifications"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # CASCADE: a verification is meaningless once its camera is gone, and
    # the registry's hard-delete path would otherwise be blocked by this
    # foreign key.
    camera_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
    )

    verified: Mapped[bool] = mapped_column(Boolean, nullable=False)
    connection_status: Mapped[ConnectionStatus] = mapped_column(
        SAEnum(
            ConnectionStatus,
            name="camera_connection_status",
            native_enum=True,
            validate_strings=True,
            values_callable=lambda cls: [member.value for member in cls],
        ),
        nullable=False,
    )
    verification_message: Mapped[str] = mapped_column(Text, nullable=False)

    # ── what was measured ────────────────────────────────────────────────
    # All nullable: a verification that fails at the TCP connect has nothing
    # to report but the failure, and storing zeroes there would be a lie
    # that later averages would happily consume.
    measured_resolution: Mapped[str | None] = mapped_column(String(32), nullable=True)
    measured_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Headline latency: TCP connect plus stream open plus first frame --
    #: what the operator waited for before seeing a picture.
    measured_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: The same figure split up, because "slow" has different causes.
    connect_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_frame_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    frames_sampled: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    codec_detected: Mapped[str | None] = mapped_column(String(32), nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    #: Zero or more operator-facing problems. JSON rather than a child
    #: table: nothing queries an individual error, they are read as a set.
    errors: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "measured_fps IS NULL OR measured_fps >= 0", name="measured_fps_non_negative"
        ),
        CheckConstraint(
            "measured_latency_ms IS NULL OR measured_latency_ms >= 0",
            name="measured_latency_non_negative",
        ),
        CheckConstraint("frames_sampled >= 0", name="frames_sampled_non_negative"),
        # The only query this table serves: "the latest result for this
        # camera", and its history in the same order.
        Index("ix_camera_verifications_camera_checked", "camera_id", "checked_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<CameraVerification camera={self.camera_id} "
            f"verified={self.verified} status={self.connection_status}>"
        )
