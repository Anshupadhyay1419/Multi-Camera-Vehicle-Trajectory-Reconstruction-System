"""Bulk camera import.

The rule this module is built around: there is exactly ONE way to create a
camera, and it is `CameraService.create()`. Every row of every spreadsheet
goes through the same `CameraCreate` schema and the same service call that
`POST /api/v1/cameras` uses. The importer contributes no validation of its
own beyond what a file has that a single JSON body does not -- which columns
are present, and whether the file repeats a code inside itself.

That constraint is the whole design. A bulk path with its own "lenient"
parsing is how a registry ends up with 4,000 cameras that the API would
have rejected one at a time.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.cameras.stream_probe import mask_credentials
from src.utils.logger import get_logger

from sentinel_system.bulk_import.enums import ImportFormat, ImportStatus
from sentinel_system.bulk_import.models import ImportJob, ImportRowError
from sentinel_system.bulk_import.parsers import (
    ParsedFile,
    ParseError,
    detect_format,
    parse,
)
from sentinel_system.bulk_import.schemas import ImportJobSummary, ImportReport
from sentinel_system.bulk_import.template import required_columns
from sentinel_system.registry.exceptions import DuplicateCameraCodeError
from sentinel_system.registry.schemas import CameraCreate, CameraFilter
from sentinel_system.registry.service import CameraService

_logger = get_logger("sentinel.import")

#: Above this, an upload is refused rather than parsed. A spreadsheet of
#: cameras is kilobytes; anything approaching this is a mistake or an
#: attempt to exhaust the server, and both are better answered immediately.
MAX_ROWS = 20_000


class ImportJobNotFoundError(Exception):
    """No import job with that id."""

    code = "import_job_not_found"

    def __init__(self, job_id: uuid.UUID) -> None:
        super().__init__(f"No import job with id {job_id}")
        self.job_id = job_id
        self.message = str(self)


class _Collector:
    """Accumulates row errors while an import runs."""

    def __init__(self) -> None:
        self.entries: list[tuple[int, str | None, str | None, str]] = []
        self.failed_rows = 0

    def add(self, row: int, code: str | None, field: str | None, message: str) -> None:
        self.entries.append((row, code, field, message))


class CameraImportService:
    """Parse an upload, validate every row, and import the good ones."""

    def __init__(self, session: Session, camera_service: CameraService | None = None):
        self.session = session
        # The one and only camera-creation path. Injectable for tests, but
        # never replaced by a bulk-specific shortcut.
        self.cameras = camera_service or CameraService(session)

    # ── the use-case ──────────────────────────────────────────────────────

    def run(
        self,
        *,
        filename: str,
        content: bytes,
        dry_run: bool = False,
        uploaded_by: str | None = None,
    ) -> ImportReport:
        """Import (or, with `dry_run`, only validate) an uploaded file.

        Never raises for a bad file or a bad row: an unreadable upload and a
        spreadsheet of nonsense are both reported as a job with a status and
        an error list, because "your file is wrong" is an answer, not a
        server fault.
        """
        started = time.perf_counter()
        _logger.info(
            "Import started: %s (dry_run=%s, by=%s)",
            filename, dry_run, uploaded_by or "unknown",
        )

        job = ImportJob(
            filename=filename,
            file_format=ImportFormat.CSV,  # replaced below once known
            status=ImportStatus.FAILED,
            dry_run=dry_run,
            uploaded_by=uploaded_by,
            uploaded_at=datetime.now(timezone.utc),
            warnings=[],
        )

        try:
            file_format = detect_format(filename, content)
            job.file_format = file_format
            parsed = parse(content, file_format)
        except ParseError as exc:
            return self._finish(job, _Collector(), started, message=str(exc))

        job.warnings = list(parsed.warnings)

        if not parsed.rows:
            return self._finish(
                job, _Collector(), started,
                message="The file contains no data rows.",
            )
        if len(parsed.rows) > MAX_ROWS:
            return self._finish(
                job, _Collector(), started,
                message=(
                    f"The file has {len(parsed.rows):,} rows; the limit is "
                    f"{MAX_ROWS:,}. Split it and import the parts."
                ),
            )

        collector = _Collector()
        missing = self._missing_columns(parsed)
        if missing:
            return self._finish(
                job, collector, started,
                total_rows=len(parsed.rows),
                message=(
                    "The file is missing required column(s): "
                    + ", ".join(sorted(missing))
                    + ". Download the template for the expected headings."
                ),
            )

        self._warn_about_unknown_columns(parsed, job)
        imported, skipped = self._process_rows(parsed, collector, dry_run)

        return self._finish(
            job, collector, started,
            total_rows=len(parsed.rows),
            imported=imported,
            skipped=skipped,
        )

    # ── reads ─────────────────────────────────────────────────────────────

    def get_report(self, job_id: uuid.UUID) -> ImportReport:
        """A past job's full report, error list included.

        Raises:
            ImportJobNotFoundError: no job with that id.
        """
        job = self.session.get(ImportJob, job_id)
        if job is None:
            raise ImportJobNotFoundError(job_id)
        return self._to_report(job)

    def history(self, limit: int = 20) -> list[ImportJobSummary]:
        """Recent import jobs, newest first, without their error lists."""
        statement = (
            select(ImportJob)
            .order_by(ImportJob.uploaded_at.desc(), ImportJob.id.desc())
            .limit(limit)
        )
        rows = self.session.execute(statement).scalars().all()
        return [ImportJobSummary.model_validate(row) for row in rows]

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _missing_columns(parsed: ParsedFile) -> set[str]:
        """Required columns absent from the file's headings.

        Checked against the columns, not the first row: a file whose header
        is right but whose first row is blank should report bad rows, not a
        bad file.
        """
        present = set(parsed.columns)
        for row in parsed.rows:  # JSON has no header; take the union of keys
            present |= set(row)
        return set(required_columns()) - present

    @staticmethod
    def _warn_about_unknown_columns(parsed: ParsedFile, job: ImportJob) -> None:
        """Note columns the schema will reject, before every row fails on them."""
        known = set(CameraCreate.model_fields)
        unknown = sorted({c for c in parsed.columns if c and c not in known})
        if unknown:
            job.warnings.append(
                "Unrecognised column(s) ignored by the schema: "
                + ", ".join(unknown)
            )

    def _process_rows(
        self, parsed: ParsedFile, collector: _Collector, dry_run: bool
    ) -> tuple[int, int]:
        """Validate and (unless dry run) create each row. Returns (imported, skipped)."""
        imported = 0
        skipped = 0
        seen_codes: dict[str, int] = {}

        for row, line in zip(parsed.rows, parsed.line_numbers):
            raw_code = str(row.get("camera_code") or "").strip().upper() or None

            # A file that repeats a code cannot be imported twice; the
            # second occurrence is SKIPPED rather than failed, because the
            # camera does get created -- from the first row.
            if raw_code and raw_code in seen_codes:
                skipped += 1
                collector.add(
                    line, raw_code, "camera_code",
                    f"Duplicate of row {seen_codes[raw_code]} in this file; "
                    f"this row was skipped.",
                )
                continue

            payload = self._validate_row(row, line, raw_code, collector)
            if payload is None:
                collector.failed_rows += 1
                continue
            if raw_code:
                seen_codes[raw_code] = line

            if dry_run:
                # The same uniqueness question the real create would ask,
                # asked without writing -- so a dry run reports the conflict
                # the commit would hit rather than passing and then failing.
                if self.cameras.exists(payload.camera_code):
                    collector.failed_rows += 1
                    collector.add(
                        line, payload.camera_code, "camera_code",
                        f"Camera code {payload.camera_code!r} is already registered",
                    )
                else:
                    imported += 1
                    seen_codes[payload.camera_code] = line
                continue

            if self._create(payload, line, collector):
                imported += 1
            else:
                collector.failed_rows += 1

        return imported, skipped

    def _validate_row(
        self,
        row: dict[str, Any],
        line: int,
        raw_code: str | None,
        collector: _Collector,
    ) -> CameraCreate | None:
        """Run the row through the SAME schema the single-camera POST uses."""
        # Unknown keys would trip the schema's extra="forbid"; they are
        # already reported once as a file-level warning, so drop them rather
        # than failing every row for the same reason.
        known = set(CameraCreate.model_fields)
        cleaned = {k: v for k, v in row.items() if k in known and v is not None}
        try:
            return CameraCreate(**cleaned)
        except ValidationError as exc:
            for error in exc.errors():
                message = self._clean_message(error.get("msg", "invalid value"))
                location = error.get("loc") or ()
                field = str(location[0]) if location else self._field_named_in(message)
                collector.add(line, raw_code, field, message)
            return None

    @staticmethod
    def _clean_message(message: str) -> str:
        """Drop Pydantic's 'Value error, ' prefix, which means nothing to an operator."""
        prefix = "Value error, "
        return message[len(prefix):] if message.startswith(prefix) else message

    @staticmethod
    def _field_named_in(message: str) -> str | None:
        """Attribute a cross-field error to the column it names first.

        Cross-field rules (protocol vs stream_url, PTZ type vs support) are
        model-level validators, so Pydantic reports them with no location --
        and a report saying `field: null` next to "stream_url scheme 'https'
        does not match protocol" makes the operator do the attribution.
        Every such message in the schema opens with the field at fault, so
        the first known column mentioned is the one to point at. This labels
        an error the schema already raised; it validates nothing itself.
        """
        positions = {
            name: message.find(name)
            for name in CameraCreate.model_fields
            if name in message
        }
        return min(positions, key=positions.get) if positions else None

    def _create(
        self, payload: CameraCreate, line: int, collector: _Collector
    ) -> bool:
        """Create one camera through CameraService. True if it landed."""
        try:
            self.cameras.create(payload)
            return True
        except DuplicateCameraCodeError as exc:
            collector.add(line, payload.camera_code, "camera_code", exc.message)
        except ValueError as exc:
            collector.add(line, payload.camera_code, None, str(exc))
        except Exception as exc:
            # An unexpected fault must not take the whole import with it.
            # Roll this row back so the session is usable for the next one;
            # rows already committed stay committed, which is what partial
            # success means.
            self.session.rollback()
            _logger.exception(
                "Import row %s failed unexpectedly for %s", line, payload.camera_code
            )
            collector.add(line, payload.camera_code, None, f"Unexpected error: {exc}")
        return False

    def _finish(
        self,
        job: ImportJob,
        collector: _Collector,
        started: float,
        *,
        total_rows: int = 0,
        imported: int = 0,
        skipped: int = 0,
        message: str | None = None,
    ) -> ImportReport:
        """Record the job, log the outcome, and build the report."""
        job.total_rows = total_rows
        job.successful_rows = imported
        job.failed_rows = collector.failed_rows
        job.skipped_rows = skipped
        job.duration_seconds = round(time.perf_counter() - started, 3)
        job.message = message
        job.status = self._status_for(job, collector, message)

        for row, code, field, text in collector.entries:
            job.errors.append(
                ImportRowError(
                    row_number=row, camera_code=code, field=field, message=text
                )
            )

        self.session.add(job)
        self.session.commit()

        if job.status is ImportStatus.FAILED:
            _logger.error(
                "Import failed: %s (%s rows) -- %s",
                job.filename, job.total_rows, message or "no rows imported",
            )
        else:
            _logger.info(
                "Import completed: %s status=%s imported=%s failed=%s skipped=%s in %ss",
                job.filename, job.status, job.successful_rows,
                job.failed_rows, job.skipped_rows, job.duration_seconds,
            )
        return self._to_report(job)

    @staticmethod
    def _status_for(
        job: ImportJob, collector: _Collector, message: str | None
    ) -> ImportStatus:
        if message and job.successful_rows == 0:
            return ImportStatus.FAILED
        if job.dry_run:
            return ImportStatus.VALIDATED
        if job.successful_rows == 0:
            return ImportStatus.FAILED
        if collector.failed_rows or job.skipped_rows:
            return ImportStatus.PARTIAL
        return ImportStatus.COMPLETED

    @staticmethod
    def _to_report(job: ImportJob) -> ImportReport:
        return ImportReport(
            job_id=job.id,
            filename=job.filename,
            file_format=job.file_format,
            status=job.status,
            dry_run=job.dry_run,
            total_rows=job.total_rows,
            imported=job.successful_rows,
            failed=job.failed_rows,
            skipped=job.skipped_rows,
            warnings=list(job.warnings or []),
            errors=[
                {
                    "row": e.row_number,
                    "camera_code": e.camera_code,
                    "field": e.field,
                    "message": e.message,
                }
                for e in sorted(job.errors, key=lambda e: (e.row_number, e.field or ""))
            ],
            duration_seconds=job.duration_seconds,
            message=job.message,
            uploaded_at=job.uploaded_at,
            uploaded_by=job.uploaded_by,
        )


__all__ = ["CameraImportService", "ImportJobNotFoundError", "MAX_ROWS"]
