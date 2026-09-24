"""Data access for cameras. SQL lives here and nowhere else.

The repository knows how to read and write rows. It does not know what a
duplicate camera code means for the business, it does not decide what to
do when a camera is missing, and it does not commit -- the caller owns the
transaction boundary, because a later module will want to register a camera
and write an audit entry in the same transaction, and a repository that
commits on its own makes that impossible.

So: methods return `Camera | None` and never raise domain errors. Turning
"None" into CameraNotFoundError is the service layer's job.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from sentinel_system.registry.enums import CameraStatus
from sentinel_system.registry.models import Camera
from sentinel_system.registry.schemas import CameraFilter


class CameraRepository:
    """CRUD and queries over the `cameras` table."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ── writes ────────────────────────────────────────────────────────────

    def add(self, camera: Camera) -> Camera:
        """Stage a new camera and flush it.

        Flush, not commit: this assigns the primary key and surfaces any
        constraint violation right here, where the caller can still attach
        meaning to it, while leaving the transaction open.
        """
        self.session.add(camera)
        self.session.flush()
        return camera

    def update(self, camera: Camera, changes: dict[str, object]) -> Camera:
        """Apply `changes` to an already-loaded camera."""
        for field_name, value in changes.items():
            setattr(camera, field_name, value)
        self.session.flush()
        return camera

    def delete(self, camera: Camera) -> None:
        self.session.delete(camera)
        self.session.flush()

    # ── reads ─────────────────────────────────────────────────────────────

    def get_by_id(self, camera_id: uuid.UUID) -> Camera | None:
        return self.session.get(Camera, camera_id)

    def get_by_code(self, camera_code: str) -> Camera | None:
        """Look up by code, case-insensitively.

        Codes are normalised to uppercase on the way in, so an exact match
        would almost always do. Almost: rows predating the normalisation,
        or written by a bulk import that bypassed the schema, would be
        missed, and a lookup that silently fails to find an existing camera
        is how duplicates get created.
        """
        statement = select(Camera).where(
            func.upper(Camera.camera_code) == camera_code.strip().upper()
        )
        return self.session.execute(statement).scalar_one_or_none()

    def exists_by_code(
        self, camera_code: str, *, exclude_id: uuid.UUID | None = None
    ) -> bool:
        """Whether a code is taken, optionally ignoring one camera.

        `exclude_id` is what lets an update re-save a camera without its own
        existing code counting as a collision with itself.
        """
        statement = select(func.count()).select_from(Camera).where(
            func.upper(Camera.camera_code) == camera_code.strip().upper()
        )
        if exclude_id is not None:
            statement = statement.where(Camera.id != exclude_id)
        return bool(self.session.execute(statement).scalar_one())

    def list(self, filters: CameraFilter) -> Sequence[Camera]:
        """Cameras matching `filters`, ordered and paginated."""
        statement = self._apply_filters(select(Camera), filters)
        column = getattr(Camera, filters.order_by)
        statement = statement.order_by(
            column.desc() if filters.descending else column.asc()
        )
        # Secondary sort on the primary key. Without it, rows that tie on
        # the sort column come back in whatever order the planner chose,
        # which differs between pages -- so a row can appear on page 1 and
        # again on page 2, or be skipped entirely.
        statement = statement.order_by(Camera.id.asc())
        statement = statement.limit(filters.limit).offset(filters.offset)
        return self.session.execute(statement).scalars().all()

    def count(self, filters: CameraFilter) -> int:
        """How many cameras match `filters`, ignoring limit/offset."""
        statement = self._apply_filters(
            select(func.count()).select_from(Camera), filters
        )
        return int(self.session.execute(statement).scalar_one())

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _apply_filters(statement: Select, filters: CameraFilter) -> Select:
        """Add WHERE clauses. Shared so list() and count() cannot disagree.

        If these were written out twice, a filter added to one and not the
        other would produce a page of 10 rows reporting a total of 4000 --
        the kind of bug that looks like a pagination problem for a week.
        """
        if filters.department is not None:
            statement = statement.where(Camera.department == filters.department)
        if filters.district is not None:
            statement = statement.where(Camera.district == filters.district)
        if filters.zone is not None:
            statement = statement.where(Camera.zone == filters.zone)
        if filters.status is not None:
            statement = statement.where(Camera.status == filters.status)
        if filters.health is not None:
            statement = statement.where(Camera.health == filters.health)
        if filters.camera_type is not None:
            statement = statement.where(Camera.camera_type == filters.camera_type)
        if filters.protocol is not None:
            statement = statement.where(Camera.protocol == filters.protocol)
        if filters.vendor is not None:
            statement = statement.where(Camera.vendor == filters.vendor)
        # Only when the caller has not asked for a specific status: asking
        # for `status=decommissioned` must still return them.
        if not filters.include_decommissioned and filters.status is None:
            statement = statement.where(
                Camera.status != CameraStatus.DECOMMISSIONED
            )
        if filters.supports_analytics is not None:
            statement = statement.where(
                Camera.supports_analytics == filters.supports_analytics
            )
        if filters.search:
            # Escaped so an operator pasting a code containing % or _ gets
            # a literal search instead of an accidental wildcard scan.
            needle = (
                filters.search.strip()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            pattern = f"%{needle}%"
            statement = statement.where(
                or_(
                    Camera.camera_code.ilike(pattern, escape="\\"),
                    Camera.camera_name.ilike(pattern, escape="\\"),
                )
            )
        return statement


__all__ = ["CameraRepository"]
