"""Populate the camera registry from the portal's catalogue.

Every discovered camera goes through `CameraService` -- the same schema and
the same service call as `POST /api/v1/cameras` and the bulk importer. There
is still exactly one way into the registry.

Three rules keep a sync from doing damage:

  * Portal cameras are namespaced: their registry code is `SEN-<portal id>`,
    so a sync can never collide with, or overwrite, a camera an operator
    registered by hand.
  * Nothing is invented. A camera the portal describes without a district
    or coordinates is SKIPPED with that reason -- the registry requires them,
    and a placeholder location on a police map is worse than no marker.
  * An operator's decision outranks the portal. A re-sync refreshes name,
    stream and health, but never touches `status`, so a camera someone
    decommissioned stays decommissioned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from sqlalchemy.orm import Session

from src.cameras.stream_probe import mask_credentials
from src.utils.logger import get_logger

from sentinel_system.portal.errors import NoSupportedStreamError
from sentinel_system.portal.models import PortalCamera
from sentinel_system.portal.streams import REGISTRY_PROTOCOL, select_stream
from sentinel_system.registry.enums import CameraStatus, HealthStatus
from sentinel_system.registry.exceptions import DuplicateCameraCodeError
from sentinel_system.registry.schemas import CameraCreate, CameraUpdate
from sentinel_system.registry.service import CameraService

_logger = get_logger("sentinel.portal.sync")

CODE_PREFIX = "SEN-"
_UNSAFE = re.compile(r"[^A-Z0-9_-]+")


def derive_camera_code(portal_id: str) -> str | None:
    """`SEN-` plus the portal id, folded into the registry's code alphabet.

    Returns None when nothing usable is left, so the caller skips the camera
    with a reason instead of registering it under an empty-looking code.
    """
    body = _UNSAFE.sub("-", str(portal_id or "").strip().upper()).strip("-_")
    body = re.sub(r"-{2,}", "-", body)
    if not body:
        return None
    return (CODE_PREFIX + body)[:64].rstrip("-_")


def _health(camera: PortalCamera) -> HealthStatus:
    if camera.online is True:
        return HealthStatus.HEALTHY
    if camera.online is False:
        return HealthStatus.UNREACHABLE
    return HealthStatus.UNKNOWN


@dataclass
class SyncReport:
    """What a sync did, camera by camera."""

    discovered: int = 0
    created: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovered": self.discovered,
            "created": self.created,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "errors": self.errors,
        }


class CatalogueSync:
    """Mirror the portal catalogue into the registry, idempotently."""

    def __init__(self, session: Session, portal: Any, camera_service: CameraService | None = None):
        self.portal = portal
        self.cameras = camera_service or CameraService(session)

    def run(self) -> SyncReport:
        report = SyncReport()
        catalogue = self.portal.list_cameras()   # portal errors propagate to the caller
        report.discovered = len(catalogue)
        _logger.info("Sentinel catalogue sync started: %s camera(s) discovered", len(catalogue))

        for camera in catalogue:
            self._sync_one(camera, report)

        _logger.info(
            "Sentinel catalogue sync completed: created=%s updated=%s unchanged=%s "
            "skipped=%s errors=%s",
            report.created, report.updated, report.unchanged,
            len(report.skipped), len(report.errors),
        )
        return report

    def _sync_one(self, camera: PortalCamera, report: SyncReport) -> None:
        code = derive_camera_code(camera.portal_id)
        if code is None:
            report.skipped.append({"portal_id": str(camera.portal_id),
                                   "reason": "The portal id cannot form a camera code."})
            return

        missing = [
            name for name, value in (
                ("department", camera.department),
                ("district", camera.district),
                ("latitude", camera.latitude),
                ("longitude", camera.longitude),
            ) if value in (None, "")
        ]
        if missing:
            report.skipped.append({
                "portal_id": camera.portal_id,
                "reason": "The portal did not provide: " + ", ".join(missing)
                          + ". Not registered rather than given a made-up value.",
            })
            return

        try:
            offer = select_stream(camera, require_available=False)
        except NoSupportedStreamError as exc:
            report.skipped.append({"portal_id": camera.portal_id, "reason": exc.message})
            return

        fields = {
            "camera_name": camera.name,
            "address": camera.location,
            "protocol": REGISTRY_PROTOCOL[offer.stream_type],
            "stream_url": offer.url,
            "health": _health(camera),
        }

        existing = self._existing(code)
        try:
            if existing is None:
                self.cameras.create(CameraCreate(
                    camera_code=code,
                    department=camera.department,
                    district=camera.district,
                    latitude=camera.latitude,
                    longitude=camera.longitude,
                    camera_type=camera.camera_type,
                    status=CameraStatus.ACTIVE,
                    **fields,
                ))
                report.created += 1
                _logger.info("Portal camera registered: %s stream=%s",
                             code, mask_credentials(offer.url))
                return

            changed = {k: v for k, v in fields.items() if getattr(existing, k) != v}
            if not changed:
                report.unchanged += 1
                return
            self.cameras.update(existing.id, CameraUpdate(**changed))
            report.updated += 1
            _logger.info("Portal camera refreshed: %s fields=%s", code, sorted(changed))
        except (ValidationError, ValueError, DuplicateCameraCodeError) as exc:
            report.errors.append({"portal_id": camera.portal_id, "camera_code": code,
                                  "reason": str(exc).splitlines()[0]})
            _logger.warning("Portal camera %s rejected by the registry: %s",
                            code, str(exc).splitlines()[0])

    def _existing(self, code: str):
        from sentinel_system.registry.exceptions import CameraNotFoundError

        try:
            return self.cameras.get_by_code(code)
        except CameraNotFoundError:
            return None


__all__ = ["CODE_PREFIX", "CatalogueSync", "SyncReport", "derive_camera_code"]
