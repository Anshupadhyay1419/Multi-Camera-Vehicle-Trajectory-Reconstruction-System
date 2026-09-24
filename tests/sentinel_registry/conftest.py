"""Fixtures for the camera-registry tests.

Each test gets its own in-memory SQLite database, created from the same
declarative metadata the real migrations are generated from. In-memory
keeps the suite fast and leaves nothing behind; StaticPool is what makes a
single in-memory database visible to every connection in the test, since
SQLite otherwise gives each connection its own private one.

SQLite is not the deployment target, so anything that genuinely needs
PostgreSQL (native ENUM types, concurrent inserts racing the unique index)
is called out where it is tested rather than silently assumed to hold.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from sentinel_system.core.database import Base
from sentinel_system.registry.enums import (
    CameraStatus,
    CameraType,
    Codec,
    HealthStatus,
    Protocol,
)
from sentinel_system.registry.repository import CameraRepository
from sentinel_system.registry.schemas import CameraCreate
from sentinel_system.registry.service import CameraService


@pytest.fixture()
def engine():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        future=True,
    )
    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture()
def session(engine) -> Iterator[Session]:
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = factory()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def repository(session: Session) -> CameraRepository:
    return CameraRepository(session)


@pytest.fixture()
def service(session: Session) -> CameraService:
    return CameraService(session)


def camera_payload(**overrides) -> CameraCreate:
    """A valid CameraCreate, with any field overridable.

    A helper rather than a fixture so a single test can build several
    cameras that differ in one field, which is most of what the repository
    tests need.
    """
    data = {
        "camera_code": "AHM-SAT-0142",
        "camera_name": "Satellite Circle North",
        "description": "Junction overview, facing the underpass",
        "department": "Gujarat Police",
        "owner": "Ahmedabad City Police",
        "zone": "Zone-1",
        "district": "Ahmedabad",
        "latitude": 23.0225,
        "longitude": 72.5714,
        "address": "Satellite Circle, Ahmedabad, Gujarat",
        "vendor": "Hikvision",
        "model": "DS-2CD2T47G2",
        "serial_number": "HK-2291-8841",
        "camera_type": CameraType.BULLET,
        "protocol": Protocol.RTSP,
        "stream_url": "rtsp://10.20.4.11:554/Streaming/Channels/101",
        "codec": Codec.H265,
        "resolution": "1920x1080",
        "fps": 25,
        "status": CameraStatus.ACTIVE,
        "health": HealthStatus.HEALTHY,
        "last_seen": datetime(2026, 9, 23, 10, 30, tzinfo=timezone.utc),
        "installed_on": datetime(2025, 3, 14, tzinfo=timezone.utc),
        "firmware_version": "V5.7.3",
        "supports_ptz": False,
        "supports_audio": True,
        "supports_nightvision": True,
        "supports_analytics": True,
        "coverage_radius_m": 80.0,
        "bearing_deg": 135.0,
    }
    data.update(overrides)
    return CameraCreate(**data)


@pytest.fixture()
def payload() -> CameraCreate:
    return camera_payload()
