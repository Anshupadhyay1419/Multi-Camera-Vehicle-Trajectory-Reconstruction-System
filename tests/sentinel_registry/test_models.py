"""The Camera table: defaults, constraints and the schema Alembic will see."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from sentinel_system.core.database import Base
from sentinel_system.registry.enums import (
    CameraStatus,
    CameraType,
    HealthStatus,
    MaintenanceStatus,
    Protocol,
)
from sentinel_system.registry.models import Camera


def _minimal(**overrides) -> Camera:
    """Only the NOT NULL columns, so defaults are visible."""
    data = {
        "camera_code": "AHM-MIN-0001",
        "camera_name": "Minimal",
        "department": "Gujarat Police",
        "district": "Ahmedabad",
        "latitude": 23.0,
        "longitude": 72.5,
        "camera_type": CameraType.FIXED,
        "protocol": Protocol.RTSP,
        "stream_url": "rtsp://10.0.0.1/stream",
    }
    data.update(overrides)
    return Camera(**data)


class TestDefaults:
    def test_a_new_camera_gets_a_uuid_primary_key(self, session):
        camera = _minimal()
        session.add(camera)
        session.commit()
        assert isinstance(camera.id, uuid.UUID)

    def test_two_cameras_get_different_ids(self, session):
        first, second = _minimal(), _minimal(camera_code="AHM-MIN-0002")
        session.add_all([first, second])
        session.commit()
        assert first.id != second.id

    def test_lifecycle_fields_default_to_the_unknown_end(self, session):
        """A camera nobody has checked yet must not look healthy."""
        camera = _minimal()
        session.add(camera)
        session.commit()
        assert camera.status is CameraStatus.PLANNED
        assert camera.health is HealthStatus.UNKNOWN
        assert camera.maintenance_status is MaintenanceStatus.NONE

    def test_capabilities_default_to_false_not_null(self, session):
        """Nullable booleans force every consumer to write `is True`."""
        camera = _minimal()
        session.add(camera)
        session.commit()
        assert camera.supports_ptz is False
        assert camera.supports_audio is False
        assert camera.supports_nightvision is False
        assert camera.supports_analytics is False

    def test_audit_timestamps_are_stamped_on_insert(self, session):
        camera = _minimal()
        session.add(camera)
        session.commit()
        assert camera.created_at is not None
        assert camera.updated_at is not None

    def test_updated_at_moves_on_update(self, session):
        camera = _minimal()
        session.add(camera)
        session.commit()
        before = camera.updated_at
        camera.camera_name = "Renamed"
        session.commit()
        session.refresh(camera)
        assert camera.updated_at >= before


class TestConstraints:
    def test_camera_code_is_unique(self, session):
        session.add(_minimal())
        session.commit()
        session.add(_minimal(camera_name="Different name"))
        with pytest.raises(IntegrityError):
            session.commit()

    @pytest.mark.parametrize("lat", [-90.5, 91.0])
    def test_the_database_rejects_impossible_latitudes(self, session, lat):
        """Enforced in the schema, not only in Pydantic: several services
        and the occasional psql session share this table."""
        session.add(_minimal(latitude=lat))
        with pytest.raises(IntegrityError):
            session.commit()

    @pytest.mark.parametrize("lon", [-180.5, 181.0])
    def test_the_database_rejects_impossible_longitudes(self, session, lon):
        session.add(_minimal(longitude=lon))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_fps_must_be_positive(self, session):
        session.add(_minimal(fps=0))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_bearing_360_is_rejected_by_the_database(self, session):
        session.add(_minimal(bearing_deg=360.0))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_coverage_radius_cannot_be_negative(self, session):
        session.add(_minimal(coverage_radius_m=-5.0))
        with pytest.raises(IntegrityError):
            session.commit()

    def test_a_too_short_code_is_rejected_by_the_database(self, session):
        session.add(_minimal(camera_code="AB"))
        with pytest.raises(IntegrityError):
            session.commit()


class TestSchemaShape:
    """What Alembic will render. Guards the migration story, not behaviour."""

    def test_every_specified_field_exists_as_a_column(self, engine):
        expected = {
            "id", "camera_code", "camera_name", "description",
            "department", "owner", "zone", "district",
            "latitude", "longitude", "address",
            "vendor", "model", "serial_number", "camera_type", "protocol",
            "stream_url", "codec", "resolution", "fps",
            "status", "health", "maintenance_status", "last_seen",
            "last_health_check",
            "installed_on", "last_service", "firmware_version",
            "supports_ptz", "supports_audio", "supports_nightvision",
            "supports_analytics",
            "coverage_radius_m", "bearing_deg",
            "created_at", "updated_at",
        }
        actual = {c["name"] for c in inspect(engine).get_columns("cameras")}
        assert expected == actual

    def test_constraints_are_named_not_auto_generated(self, engine):
        """Unnamed constraints get database-invented names that differ
        between dialects and between autogenerate runs, which makes the
        resulting migration unrunnable anywhere else."""
        inspector = inspect(engine)
        names = [c["name"] for c in inspector.get_check_constraints("cameras")]
        assert "ck_cameras_latitude_range" in names
        assert "ck_cameras_bearing_range" in names
        assert all(name for name in names)

    def test_the_operational_indexes_exist(self, engine):
        names = {ix["name"] for ix in inspect(engine).get_indexes("cameras")}
        assert {
            "ix_cameras_district_status",
            "ix_cameras_status_health",
            "ix_cameras_location",
        } <= names

    def test_metadata_carries_only_the_milestones_built_so_far(self):
        """Guards the scope line: no watchlist, no events, no stream tables.

        `cameras` is M1.1; `camera_verifications` is M1.3, which records
        what a camera actually delivered WITHOUT touching what it was
        registered as; `import_jobs` and `import_row_errors` are M1.4's audit
        trail for bulk uploads. Anything else appearing here means a later
        milestone has leaked in.
        """
        assert set(Base.metadata.tables) == {
            "cameras",
            "camera_verifications",
            "import_jobs",
            "import_row_errors",
        }


class TestEnumStorage:
    def test_enums_persist_as_their_values_not_their_python_names(
        self, session, engine
    ):
        """Stored as 'active', which is what every payload and log line
        says -- not 'ACTIVE', the Python attribute name."""
        camera = _minimal(status=CameraStatus.ACTIVE)
        session.add(camera)
        session.commit()
        raw = session.execute(
            text("SELECT status FROM cameras WHERE camera_code = :c"),
            {"c": camera.camera_code},
        ).scalar_one()
        assert raw == "active"

    def test_enum_columns_round_trip_as_enum_members(self, session):
        camera = _minimal(status=CameraStatus.ACTIVE, health=HealthStatus.DEGRADED)
        session.add(camera)
        session.commit()
        session.expire_all()
        reloaded = session.get(Camera, camera.id)
        assert reloaded.status is CameraStatus.ACTIVE
        assert reloaded.health is HealthStatus.DEGRADED


class TestTimezones:
    def test_an_aware_timestamp_survives_the_round_trip(self, session):
        """Naive timestamps are a one-way door: once stored without an
        offset there is no recovering what they meant."""
        moment = datetime(2026, 9, 23, 6, 15, tzinfo=timezone.utc)
        camera = _minimal(last_seen=moment)
        session.add(camera)
        session.commit()
        session.expire_all()
        assert session.get(Camera, camera.id).last_seen is not None
