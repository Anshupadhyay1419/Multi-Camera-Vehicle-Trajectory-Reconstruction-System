"""Pydantic v2 schemas for bulk import reports."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from sentinel_system.bulk_import.enums import ImportFormat, ImportStatus


class RowError(BaseModel):
    """One reason one row was rejected."""

    model_config = ConfigDict(from_attributes=True)

    row: int = Field(description="Line number in the uploaded file, header included")
    camera_code: str | None = Field(
        default=None, description="As given in the file, if it had one"
    )
    field: str | None = Field(default=None, description="The offending column")
    message: str

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "row": 17,
                "camera_code": "AHM-019",
                "field": "latitude",
                "message": "Input should be less than or equal to 90",
            }
        },
    )


class ImportReport(BaseModel):
    """What one import or validation run did.

    `imported`, `failed` and `skipped` count ROWS; `errors` can be longer
    than `failed`, because a single row with a bad latitude AND a bad
    protocol earns one entry per problem -- fixing a spreadsheet one error
    at a time is how a 250-row import turns into five uploads.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID
    filename: str
    file_format: ImportFormat
    status: ImportStatus
    dry_run: bool = Field(description="True when nothing was written to the registry")

    total_rows: int
    imported: int
    failed: int
    skipped: int = Field(
        description="Rows deliberately not attempted, e.g. a duplicate of an "
        "earlier row in the same file"
    )

    warnings: list[str] = Field(default_factory=list)
    errors: list[RowError] = Field(default_factory=list)

    duration_seconds: float | None = None
    message: str | None = None
    uploaded_at: datetime
    uploaded_by: str | None = None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "job_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "filename": "ahmedabad-cameras.csv",
                "file_format": "csv",
                "status": "partial",
                "dry_run": False,
                "total_rows": 250,
                "imported": 241,
                "failed": 9,
                "skipped": 0,
                "warnings": [],
                "errors": [
                    {
                        "row": 17,
                        "camera_code": "AHM-019",
                        "field": "latitude",
                        "message": "Input should be less than or equal to 90",
                    }
                ],
                "duration_seconds": 1.84,
                "uploaded_at": "2026-09-23T15:04:11Z",
                "uploaded_by": None,
            }
        },
    )


class ImportJobSummary(BaseModel):
    """An import job without its error list, for history listings."""

    model_config = ConfigDict(from_attributes=True)

    job_id: uuid.UUID = Field(validation_alias="id")
    filename: str
    file_format: ImportFormat
    status: ImportStatus
    dry_run: bool
    total_rows: int
    successful_rows: int
    failed_rows: int
    skipped_rows: int
    duration_seconds: float | None
    uploaded_at: datetime
    uploaded_by: str | None


__all__ = ["ImportJobSummary", "ImportReport", "RowError"]
