"""Vocabularies for bulk camera import."""

from __future__ import annotations

from sentinel_system.registry.enums import _ValueEnum


class ImportFormat(_ValueEnum):
    """File formats the importer accepts."""

    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"


class ImportStatus(_ValueEnum):
    """How an import job ended.

    PARTIAL is its own outcome rather than a flavour of COMPLETED because it
    is the one an operator must actually look at: some cameras are in the
    registry and some are not, and which is which is in the error list.
    Collapsing it into "completed" would hide exactly the rows that need a
    human.
    """

    VALIDATED = "validated"   # dry run: nothing was written, by design
    COMPLETED = "completed"   # every row imported
    PARTIAL = "partial"       # some rows imported, some rejected
    FAILED = "failed"         # nothing imported


__all__ = ["ImportFormat", "ImportStatus"]
