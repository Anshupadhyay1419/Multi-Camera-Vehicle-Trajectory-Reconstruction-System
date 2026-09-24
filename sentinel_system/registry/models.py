"""SQLAlchemy model for the camera registry.

Written against the PostgreSQL feature set but kept dialect-portable so the
suite can run on SQLite: `sqlalchemy.Uuid` becomes a native `uuid` column on
PostgreSQL and CHAR(32) on SQLite, and the enums become native PostgreSQL
ENUM types and VARCHAR + CHECK respectively. Nothing here needs a dialect
branch at the call site.

Invariants that can be expressed in the schema are expressed in the schema.
Pydantic validates what reaches the service layer, but the database is what
several services, a migration and the occasional DBA all share, so latitude
being a real latitude is a CHECK constraint and not only a validator.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum as SAEnum,
    Float,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from sentinel_system.core.database import Base
from sentinel_system.registry.enums import (
    CameraStatus,
    CameraType,
    Codec,
    HealthStatus,
    MaintenanceStatus,
    Protocol,
)


def _pg_enum(enum_cls: type, name: str) -> SAEnum:
    """A PostgreSQL ENUM that stores member VALUES, not member NAMES.

    Without `values_callable` SQLAlchemy persists `"ACTIVE"` (the Python
    attribute) while every API payload, log line and hand-written SQL query
    says `"active"` (the value). The mismatch stays invisible until someone
    writes a raw query against the table, so it is pinned here once.
    """
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=True,
        validate_strings=True,
        values_callable=lambda cls: [member.value for member in cls],
    )


class Camera(Base):
    """A single registered camera.

    The registry is the system of record for *what cameras exist and where*.
    It deliberately holds no frames, no detections and no analytics results
    -- those belong to later modules and would give this table a write rate
    it is not designed for.
    """

    __tablename__ = "cameras"

    # ── Internal ──────────────────────────────────────────────────────────
    # A UUID rather than a serial: camera records are created by several
    # district systems that must not have to coordinate over an ID sequence,
    # and a non-guessable identifier is the right default for anything that
    # will eventually be addressable over an API.
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # The human-facing handle, printed on the enclosure and used in radio
    # traffic ("check AHM-SAT-0142"). Unique and immutable in practice.
    camera_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    camera_name: Mapped[str] = mapped_column(String(160), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Ownership ─────────────────────────────────────────────────────────
    # Strings, not enums -- see the note at the top of enums.py.
    department: Mapped[str] = mapped_column(String(120), nullable=False)
    owner: Mapped[str | None] = mapped_column(String(120), nullable=True)
    zone: Mapped[str | None] = mapped_column(String(120), nullable=True)
    district: Mapped[str] = mapped_column(String(120), nullable=False)

    # ── Location ──────────────────────────────────────────────────────────
    # Plain float columns rather than PostGIS geography. The registry needs
    # "where is this camera" and bounding-box queries; adding a PostGIS
    # dependency to get that would be a deployment cost with no M1 payoff.
    # The column pair is indexed together so a bbox scan stays cheap, and
    # migrating to geography(Point) later does not change this model's API.
    latitude: Mapped[float] = mapped_column(Float, nullable=False)
    longitude: Mapped[float] = mapped_column(Float, nullable=False)
    address: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ── Camera details ────────────────────────────────────────────────────
    vendor: Mapped[str | None] = mapped_column(String(120), nullable=True)
    model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    serial_number: Mapped[str | None] = mapped_column(String(120), nullable=True)
    camera_type: Mapped[CameraType] = mapped_column(
        _pg_enum(CameraType, "camera_type"), nullable=False
    )
    protocol: Mapped[Protocol] = mapped_column(
        _pg_enum(Protocol, "camera_protocol"), nullable=False
    )
    # Text, not String(n): RTSP URLs carry credentials, query strings and
    # channel selectors and routinely run past any length worth guessing at.
    stream_url: Mapped[str] = mapped_column(Text, nullable=False)
    codec: Mapped[Codec | None] = mapped_column(
        _pg_enum(Codec, "camera_codec"), nullable=True
    )
    # "1920x1080". Kept as text rather than two integers because it is
    # displayed and matched far more often than it is arithmetic.
    resolution: Mapped[str | None] = mapped_column(String(32), nullable=True)
    fps: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # ── Status ────────────────────────────────────────────────────────────
    status: Mapped[CameraStatus] = mapped_column(
        _pg_enum(CameraStatus, "camera_status"),
        nullable=False,
        default=CameraStatus.PLANNED,
        server_default=CameraStatus.PLANNED.value,
    )
    health: Mapped[HealthStatus] = mapped_column(
        _pg_enum(HealthStatus, "camera_health"),
        nullable=False,
        default=HealthStatus.UNKNOWN,
        server_default=HealthStatus.UNKNOWN.value,
    )
    maintenance_status: Mapped[MaintenanceStatus] = mapped_column(
        _pg_enum(MaintenanceStatus, "camera_maintenance_status"),
        nullable=False,
        default=MaintenanceStatus.NONE,
        server_default=MaintenanceStatus.NONE.value,
    )
    # Every timestamp is timezone-aware. Gujarat is a single timezone today,
    # but naive timestamps are a one-way door: once rows are stored without
    # an offset there is no way to tell what they meant.
    last_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_health_check: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # ── Installation ──────────────────────────────────────────────────────
    installed_on: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_service: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    firmware_version: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ── Capabilities ──────────────────────────────────────────────────────
    # NOT NULL with a default: "we know this camera cannot pan" and "nobody
    # has recorded whether it can" are different facts, but a nullable
    # boolean makes every consumer write `is True` to stay correct. Default
    # false, and let the survey update it.
    supports_ptz: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=func.false()
    )
    supports_audio: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=func.false()
    )
    supports_nightvision: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=func.false()
    )
    supports_analytics: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=func.false()
    )

    # ── GIS ───────────────────────────────────────────────────────────────
    # How far the camera usefully sees, and which way it points. Together
    # these turn a point on a map into a coverage wedge, which is what makes
    # "which cameras could have seen this junction" answerable.
    coverage_radius_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    bearing_deg: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ── Audit ─────────────────────────────────────────────────────────────
    # server_default so rows created by a migration or by hand in psql are
    # stamped too, not only rows that happen to go through this ORM.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "latitude >= -90 AND latitude <= 90", name="latitude_range"
        ),
        CheckConstraint(
            "longitude >= -180 AND longitude <= 180", name="longitude_range"
        ),
        CheckConstraint("fps IS NULL OR fps > 0", name="fps_positive"),
        CheckConstraint(
            "coverage_radius_m IS NULL OR coverage_radius_m >= 0",
            name="coverage_radius_non_negative",
        ),
        # 0 <= bearing < 360: 360 and 0 are the same direction, so allowing
        # both would let two rows describe an identical heading differently.
        CheckConstraint(
            "bearing_deg IS NULL OR (bearing_deg >= 0 AND bearing_deg < 360)",
            name="bearing_range",
        ),
        CheckConstraint("length(camera_code) >= 3", name="camera_code_min_length"),
        # Operations filters by "whose is it and is it working"; the map view
        # filters by bounding box. These two cover both without indexing
        # columns nothing queries on.
        Index("ix_cameras_district_status", "district", "status"),
        Index("ix_cameras_department", "department"),
        Index("ix_cameras_zone", "zone"),
        Index("ix_cameras_status_health", "status", "health"),
        Index("ix_cameras_location", "latitude", "longitude"),
        Index("ix_cameras_last_seen", "last_seen"),
    )

    def __repr__(self) -> str:
        return (
            f"<Camera {self.camera_code!r} {self.camera_name!r} "
            f"status={self.status} health={self.health}>"
        )
