"""Camera registry use-cases.

This is the layer an API endpoint (M1.2) will call. It owns the rules that
are true regardless of who is asking: codes are unique, you cannot update
or delete a camera that is not there, and a caller gets a validated schema
object back rather than a live ORM instance.

Returning CameraRead rather than Camera is a deliberate boundary. A
detached ORM instance whose session has closed raises on attribute access,
and handing one to a template or a serialiser is how `DetachedInstanceError`
ends up in a request log. The schema is a plain value object; it cannot do
that.

Transactions: each method commits, because each is a complete use-case. A
caller that needs several of these in one transaction should use the
repository directly inside its own `session_scope()`.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from sentinel_system.registry.enums import CameraStatus
from sentinel_system.registry.exceptions import (
    CameraNotFoundError,
    DuplicateCameraCodeError,
)
from sentinel_system.registry.models import Camera
from sentinel_system.registry.repository import CameraRepository
from sentinel_system.registry.schemas import (
    CameraCreate,
    CameraFilter,
    CameraPage,
    CameraRead,
    CameraUpdate,
)


class CameraService:
    """Use-cases for registering and maintaining cameras."""

    def __init__(self, session: Session, repository: CameraRepository | None = None):
        self.session = session
        # Injectable so a test can substitute a fake, but defaulted so the
        # ordinary caller does not have to build one.
        self.repository = repository or CameraRepository(session)

    # ── create ────────────────────────────────────────────────────────────

    def create(self, payload: CameraCreate) -> CameraRead:
        """Register a new camera.

        Raises:
            DuplicateCameraCodeError: the code is already registered.
        """
        if self.repository.exists_by_code(payload.camera_code):
            raise DuplicateCameraCodeError(payload.camera_code)

        camera = Camera(**payload.model_dump())
        try:
            self.repository.add(camera)
            self.session.commit()
        except IntegrityError as exc:
            # The check above is a race, not a guarantee: two requests can
            # both pass it before either inserts. The unique index is what
            # actually enforces uniqueness, so translate its violation into
            # the same domain error rather than letting a 500 escape.
            self.session.rollback()
            if self._is_camera_code_conflict(exc):
                raise DuplicateCameraCodeError(payload.camera_code) from exc
            raise
        return CameraRead.model_validate(camera)

    # ── read ──────────────────────────────────────────────────────────────

    def get(self, camera_id: uuid.UUID) -> CameraRead:
        """Fetch one camera by id.

        Raises:
            CameraNotFoundError: no camera with that id.
        """
        camera = self.repository.get_by_id(camera_id)
        if camera is None:
            raise CameraNotFoundError(camera_id=camera_id)
        return CameraRead.model_validate(camera)

    def get_by_code(self, camera_code: str) -> CameraRead:
        """Fetch one camera by its operator-facing code.

        Raises:
            CameraNotFoundError: no camera with that code.
        """
        camera = self.repository.get_by_code(camera_code)
        if camera is None:
            raise CameraNotFoundError(camera_code=camera_code)
        return CameraRead.model_validate(camera)

    def list(self, filters: CameraFilter | None = None) -> CameraPage:
        """A filtered, ordered, paginated page of cameras.

        `total` is the count matching the filter, not the page size, so a
        caller can render "showing 1-50 of 812" and know whether to offer a
        next page without a second round trip.
        """
        active = filters or CameraFilter()
        rows = self.repository.list(active)
        total = self.repository.count(active)
        return CameraPage(
            items=[CameraRead.model_validate(row) for row in rows],
            total=total,
            limit=active.limit,
            offset=active.offset,
        )

    def exists(self, camera_code: str) -> bool:
        """Whether a code is already registered. Never raises."""
        return self.repository.exists_by_code(camera_code)

    # ── update ────────────────────────────────────────────────────────────

    def update(self, camera_id: uuid.UUID, payload: CameraUpdate) -> CameraRead:
        """Apply a partial update.

        Only fields the caller actually supplied are touched -- omitting a
        field leaves it alone, and setting it to null clears it.

        Raises:
            CameraNotFoundError: no camera with that id.
        """
        camera = self.repository.get_by_id(camera_id)
        if camera is None:
            raise CameraNotFoundError(camera_id=camera_id)

        changes = payload.changed_fields()
        if not changes:
            # Nothing to do. Returning early keeps `updated_at` honest: an
            # empty PATCH should not make the record look freshly edited.
            return CameraRead.model_validate(camera)

        self._validate_transition(camera, changes)
        self.repository.update(camera, changes)
        self.session.commit()
        return CameraRead.model_validate(camera)

    # ── retire ────────────────────────────────────────────────────────────

    def decommission(self, camera_id: uuid.UUID) -> CameraRead:
        """Retire a camera without destroying its record: the soft delete.

        This is what the API's DELETE does. A camera that has been in
        service is referenced by incident reports and, once later modules
        land, by stored detections -- so removing the row would strand
        every one of those references and lose the audit trail of what was
        watching a junction and when.

        Decommissioned cameras drop out of the default listing, so the
        effect looks like a delete to anyone using the registry, while
        `status=decommissioned` still retrieves them.

        Idempotent: retiring an already-retired camera is not an error,
        because a retry of a request that succeeded should not fail.

        Raises:
            CameraNotFoundError: no camera with that id.
        """
        camera = self.repository.get_by_id(camera_id)
        if camera is None:
            raise CameraNotFoundError(camera_id=camera_id)
        if camera.status is not CameraStatus.DECOMMISSIONED:
            self.repository.update(
                camera, {"status": CameraStatus.DECOMMISSIONED}
            )
            self.session.commit()
        return CameraRead.model_validate(camera)

    # ── delete ────────────────────────────────────────────────────────────

    def delete(self, camera_id: uuid.UUID) -> None:
        """Permanently remove a camera row.

        Kept for administrative cleanup -- correcting a mistyped
        registration that was never in service -- and deliberately NOT what
        the API's DELETE calls. Use `decommission()` for anything that has
        ever watched a junction; see the note there.

        Raises:
            CameraNotFoundError: no camera with that id.
        """
        camera = self.repository.get_by_id(camera_id)
        if camera is None:
            raise CameraNotFoundError(camera_id=camera_id)
        self.repository.delete(camera)
        self.session.commit()

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _validate_transition(camera: Camera, changes: dict[str, Any]) -> None:
        """Cross-field rules that a partial update can only break in combination.

        The schema validates a payload in isolation. It cannot see that
        `{"camera_type": "ptz"}` is invalid because the *stored* camera has
        supports_ptz=false, or that a new stream_url contradicts the
        *stored* protocol. Those need the merged picture, so they live here.
        """
        from sentinel_system.registry.enums import CameraType
        from sentinel_system.registry.schemas import _check_stream_url

        merged_type = changes.get("camera_type", camera.camera_type)
        merged_ptz = changes.get("supports_ptz", camera.supports_ptz)
        if merged_type is CameraType.PTZ and not merged_ptz:
            raise ValueError(
                "camera_type 'ptz' requires supports_ptz=true; set one or the other"
            )

        if "stream_url" in changes or "protocol" in changes:
            merged_url = changes.get("stream_url", camera.stream_url)
            merged_protocol = changes.get("protocol", camera.protocol)
            if merged_url is not None:
                _check_stream_url(merged_url, merged_protocol)

    @staticmethod
    def _is_camera_code_conflict(exc: IntegrityError) -> bool:
        """Whether an IntegrityError is the camera_code unique violation.

        Matched on the constraint name, which the naming convention in
        core/database.py pins to `uq_cameras_camera_code` on every dialect.
        That is the point of having the convention: without it this check
        would need a different string per database.
        """
        detail = str(getattr(exc, "orig", exc)).lower()
        return "camera_code" in detail


__all__ = ["CameraService"]
