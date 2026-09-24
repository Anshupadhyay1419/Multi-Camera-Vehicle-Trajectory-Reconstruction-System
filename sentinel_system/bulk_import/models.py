"""Import job history.

Two tables, because row errors and job summaries have different shapes and
very different cardinalities: a 5,000-row spreadsheet can carry thousands of
errors, and a JSON blob of that size on the job row would be re-read in full
every time anybody listed recent imports.

The job row is what an operator sees first ("250 rows, 241 in, 9 rejected");
the error rows are what they open next to find out which nine and why.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
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
from sqlalchemy.orm import Mapped, mapped_column, relationship

from sentinel_system.core.database import Base
from sentinel_system.bulk_import.enums import ImportFormat, ImportStatus


def _enum(enum_cls: type, name: str) -> SAEnum:
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=True,
        validate_strings=True,
        values_callable=lambda cls: [member.value for member in cls],
    )


class ImportJob(Base):
    """One upload, whether it was committed or only validated."""

    __tablename__ = "import_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_format: Mapped[ImportFormat] = mapped_column(
        _enum(ImportFormat, "import_format"), nullable=False
    )
    status: Mapped[ImportStatus] = mapped_column(
        _enum(ImportStatus, "import_status"), nullable=False
    )
    #: True when the job deliberately wrote nothing.
    dry_run: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=func.false()
    )
    #: Nullable: this deployment has no authentication yet (M1.2 left it
    #: alone on purpose), so there is often nobody to attribute an import
    #: to. The column exists now so that adding auth later is a backfill
    #: rather than a migration of a table that is already large.
    uploaded_by: Mapped[str | None] = mapped_column(String(120), nullable=True)

    total_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    successful_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: File-level notes that did not stop the import.
    warnings: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    #: Wall-clock seconds. Stored rather than derived from timestamps so a
    #: later change to how they are recorded cannot silently rewrite history.
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    errors: Mapped[list["ImportRowError"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="ImportRowError.row_number",
        lazy="selectin",
    )

    __table_args__ = (Index("ix_import_jobs_uploaded_at", "uploaded_at"),)

    def __repr__(self) -> str:
        return (
            f"<ImportJob {self.filename!r} {self.status} "
            f"{self.successful_rows}/{self.total_rows}>"
        )


class ImportRowError(Base):
    """One rejected row, named by the field that rejected it."""

    __tablename__ = "import_row_errors"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("import_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    #: 1-based and counted as the OPERATOR sees it -- the line number in the
    #: spreadsheet, header included. An error saying "row 17" has to point
    #: at row 17 of the file they are looking at, not at index 15 of an
    #: array they cannot see.
    row_number: Mapped[int] = mapped_column(Integer, nullable=False)
    camera_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: The offending column, or None for whole-row problems.
    field: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    job: Mapped[ImportJob] = relationship(back_populates="errors")

    __table_args__ = (Index("ix_import_row_errors_job", "job_id", "row_number"),)

    def __repr__(self) -> str:
        return f"<ImportRowError row={self.row_number} field={self.field!r}>"
