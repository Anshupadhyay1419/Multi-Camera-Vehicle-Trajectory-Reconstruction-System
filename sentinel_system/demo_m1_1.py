"""M1.1 verification: create a camera, insert it, read it back.

Runnable end-to-end against a throwaway database:

    SENTINEL_DATABASE_URL="sqlite+pysqlite:///./demo.db" \
        alpr/bin/python sentinel_system/demo_m1_1.py

Nothing here is imported by the module itself -- it exists so the milestone
can be demonstrated without a REST API, which is M1.2.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Run as a script, the interpreter puts THIS directory on sys.path, not the
# repository root -- so `import sentinel_system` would miss. Same fix main.py
# uses for `src`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault(
    "SENTINEL_DATABASE_URL",
    f"sqlite+pysqlite:///{tempfile.mkdtemp()}/demo.db",
)

from sentinel_system.core.database import Base, get_engine, session_scope  # noqa: E402
from sentinel_system.registry import (  # noqa: E402
    CameraCreate,
    CameraFilter,
    CameraService,
    CameraStatus,
    CameraType,
    CameraUpdate,
    Codec,
    DuplicateCameraCodeError,
    HealthStatus,
    Protocol,
)


def rule(title: str) -> None:
    print(f"\n{'─' * 68}\n{title}\n{'─' * 68}")


def main() -> None:
    Base.metadata.create_all(get_engine())
    print(f"database: {os.environ['SENTINEL_DATABASE_URL']}")

    # ── 1. Build and validate a camera, without touching the database ────
    rule("1. Object creation (Pydantic validates before anything is stored)")
    payload = CameraCreate(
        camera_code="  ahm-sat-0142  ",          # normalised below
        camera_name="Satellite Circle North",
        description="Junction overview, facing the underpass",
        department="Gujarat Police",
        owner="Ahmedabad City Police",
        zone="Zone-1",
        district="Ahmedabad",
        latitude=23.0225,
        longitude=72.5714,
        address="Satellite Circle, Ahmedabad, Gujarat",
        vendor="Hikvision",
        model="DS-2CD2T47G2",
        serial_number="HK-2291-8841",
        camera_type=CameraType.BULLET,
        protocol=Protocol.RTSP,
        stream_url="rtsp://10.20.4.11:554/Streaming/Channels/101",
        codec=Codec.H265,
        resolution="1920X1080",                   # normalised below
        fps=25,
        status=CameraStatus.ACTIVE,
        health=HealthStatus.HEALTHY,
        last_seen=datetime(2026, 9, 23, 10, 30, tzinfo=timezone.utc),
        installed_on=datetime(2025, 3, 14, tzinfo=timezone.utc),
        firmware_version="V5.7.3",
        supports_audio=True,
        supports_nightvision=True,
        supports_analytics=True,
        coverage_radius_m=80.0,
        bearing_deg=135.0,
    )
    print(f"  camera_code  '  ahm-sat-0142  ' -> {payload.camera_code!r}")
    print(f"  resolution   '1920X1080'        -> {payload.resolution!r}")
    print(f"  camera_type  {payload.camera_type}   protocol {payload.protocol}")

    # ── 2. Insert ─────────────────────────────────────────────────────────
    rule("2. Database insertion")
    with session_scope() as session:
        service = CameraService(session)
        created = service.create(payload)
        camera_id = created.id
        print(f"  inserted id          {created.id}")
        print(f"  server-set created_at {created.created_at}")
        print(f"  defaulted supports_ptz={created.supports_ptz} "
              f"maintenance={created.maintenance_status}")

        service.create(
            CameraCreate(
                camera_code="AHM-SAT-0143",
                camera_name="Satellite Circle South",
                department="Gujarat Police",
                district="Ahmedabad",
                latitude=23.0219,
                longitude=72.5702,
                camera_type=CameraType.PTZ,
                supports_ptz=True,
                protocol=Protocol.ONVIF,
                stream_url="rtsp://10.20.4.12:554/onvif1",
                status=CameraStatus.ACTIVE,
                health=HealthStatus.UNREACHABLE,
            )
        )
        service.create(
            CameraCreate(
                camera_code="SUR-RNG-0007",
                camera_name="Ring Road Gate 2",
                department="Gujarat Police",
                district="Surat",
                latitude=21.1702,
                longitude=72.8311,
                camera_type=CameraType.ANPR,
                protocol=Protocol.RTSP,
                stream_url="rtsp://10.30.1.7:554/live",
                status=CameraStatus.COMMISSIONING,
            )
        )
        print("  inserted 2 more cameras")

    # ── 3. Retrieve ───────────────────────────────────────────────────────
    rule("3. Retrieval")
    with session_scope() as session:
        service = CameraService(session)

        by_id = service.get(camera_id)
        print(f"  by id    {by_id.camera_code}  {by_id.camera_name}")

        by_code = service.get_by_code("ahm-sat-0142")   # case-insensitive
        print(f"  by code  {by_code.camera_code}  lat/lon "
              f"{by_code.latitude},{by_code.longitude}")

        page = service.list(CameraFilter(district="Ahmedabad"))
        print(f"  district=Ahmedabad -> {page.total} camera(s): "
              f"{[c.camera_code for c in page.items]}")

        stuck = service.list(
            CameraFilter(status=CameraStatus.ACTIVE, health=HealthStatus.UNREACHABLE)
        )
        print(f"  active but unreachable -> {[c.camera_code for c in stuck.items]}")

        page1 = service.list(CameraFilter(limit=2, offset=0))
        page2 = service.list(CameraFilter(limit=2, offset=2))
        print(f"  page 1 {[c.camera_code for c in page1.items]} has_more={page1.has_more}")
        print(f"  page 2 {[c.camera_code for c in page2.items]} has_more={page2.has_more}")

    # ── 4. Update and the rules ───────────────────────────────────────────
    rule("4. Partial update and enforced rules")
    with session_scope() as session:
        service = CameraService(session)

        patched = service.update(camera_id, CameraUpdate(firmware_version="V6.0.0"))
        print(f"  patched firmware -> {patched.firmware_version}, "
              f"name untouched: {patched.camera_name!r}")

        cleared = service.update(camera_id, CameraUpdate(owner=None))
        print(f"  explicit null cleared owner -> {cleared.owner!r}")

        try:
            service.create(CameraCreate(**{**payload.model_dump(),
                                           "camera_name": "Duplicate attempt"}))
        except DuplicateCameraCodeError as exc:
            print(f"  duplicate code refused -> [{exc.code}] {exc}")

        try:
            service.update(camera_id, CameraUpdate(camera_type=CameraType.PTZ))
        except ValueError as exc:
            print(f"  ptz without ptz support refused -> {exc}")

        try:
            service.update(camera_id, CameraUpdate(stream_url="https://cam/stream"))
        except ValueError as exc:
            print(f"  url/protocol mismatch refused -> {exc}")

    rule("5. Delete")
    with session_scope() as session:
        service = CameraService(session)
        service.delete(camera_id)
        print(f"  deleted {camera_id}")
        print(f"  remaining: {service.list().total} camera(s)")

    print("\nM1.1 verified.\n")


if __name__ == "__main__":
    main()
