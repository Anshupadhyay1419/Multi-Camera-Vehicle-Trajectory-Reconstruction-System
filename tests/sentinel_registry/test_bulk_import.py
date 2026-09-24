"""Bulk camera import (M1.4): parsers and the import service.

The rule under test throughout is that there is ONE way to create a camera.
Anything a spreadsheet can express that `POST /api/v1/cameras` would refuse
must be refused here too, by the same schema, with the row and column named.
"""

from __future__ import annotations

import io
import json
import uuid

import pytest

from sentinel_system.bulk_import import CameraImportService, ImportStatus
from sentinel_system.bulk_import.parsers import (
    ParseError,
    detect_format,
    parse,
    parse_csv,
    parse_json,
    parse_xlsx,
)
from sentinel_system.bulk_import.enums import ImportFormat
from sentinel_system.bulk_import.service import ImportJobNotFoundError, MAX_ROWS
from sentinel_system.bulk_import.template import (
    build_template_csv,
    required_columns,
    template_columns,
)

HEADER = (
    "camera_code,camera_name,department,district,latitude,longitude,"
    "camera_type,protocol,stream_url"
)


def row(code="AHM-001", name="Gate One", lat="23.02", lon="72.57",
        kind="bullet", proto="rtsp", url=None, district="Ahmedabad"):
    url = url or f"rtsp://10.0.0.1/{code}"
    return f"{code},{name},Gujarat Police,{district},{lat},{lon},{kind},{proto},{url}"


def csv_bytes(*rows: str, header: str = HEADER) -> bytes:
    return ("\n".join([header, *rows]) + "\n").encode()


def xlsx_bytes(rows: list[list], header: list[str] | None = None) -> bytes:
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(header or HEADER.split(","))
    for values in rows:
        sheet.append(values)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


@pytest.fixture()
def importer(session) -> CameraImportService:
    return CameraImportService(session)


# ── format detection and parsing ─────────────────────────────────────────


class TestFormatDetection:
    def test_csv_by_extension(self):
        assert detect_format("cams.csv", csv_bytes(row())) is ImportFormat.CSV

    def test_json_by_content_even_without_an_extension(self):
        assert detect_format("upload", b'[{"camera_code": "A"}]') is ImportFormat.JSON

    def test_xlsx_by_magic_bytes_even_when_named_csv(self):
        """A spreadsheet saved as .csv is a common enough mistake that
        failing with 'line 1 is not valid CSV' would be a poor answer."""
        assert detect_format("cams.csv", xlsx_bytes([])) is ImportFormat.XLSX

    @pytest.mark.parametrize("name", ["cams.pdf", "cams.txt", "cams"])
    def test_an_unidentifiable_file_is_refused(self, name):
        with pytest.raises(ParseError, match="Unsupported file type"):
            detect_format(name, b"\x00\x01binary junk")


class TestCsvParsing:
    def test_headers_are_matched_forgivingly(self):
        parsed = parse_csv(b"Camera Code,Camera-Name\nAHM-001,Gate\n")
        assert parsed.columns == ["camera_code", "camera_name"]

    def test_blank_cells_become_none_not_empty_strings(self):
        """int('') is a crash where the operator meant 'not specified'."""
        parsed = parse_csv(b"camera_code,fps\nAHM-001,\n")
        assert parsed.rows[0]["fps"] is None

    def test_line_numbers_are_what_the_operator_sees(self):
        """An error saying 'row 3' must point at line 3 of their file."""
        parsed = parse_csv(csv_bytes(row("A-1"), row("A-2")))
        assert parsed.line_numbers == [2, 3]

    def test_trailing_blank_lines_are_ignored(self):
        parsed = parse_csv(csv_bytes(row("A-1")) + b",,,,,,,,\n")
        assert len(parsed.rows) == 1

    def test_an_empty_file_is_refused(self):
        with pytest.raises(ParseError, match="no header row"):
            parse_csv(b"")

    def test_a_utf8_bom_is_stripped(self):
        """Excel writes one, and it would otherwise corrupt the first heading."""
        parsed = parse_csv(b"\xef\xbb\xbfcamera_code\nAHM-001\n")
        assert parsed.columns == ["camera_code"]


class TestXlsxParsing:
    def test_rows_are_read(self):
        parsed = parse_xlsx(xlsx_bytes([row().split(",")]))
        assert parsed.rows[0]["camera_code"] == "AHM-001"

    def test_native_numbers_survive(self):
        """Excel returns real floats where CSV returns strings; both must
        end up in one coercion path."""
        values = ["AHM-001", "Gate", "GP", "Ahmedabad", 23.02, 72.57,
                  "bullet", "rtsp", "rtsp://10.0.0.1/s"]
        parsed = parse_xlsx(xlsx_bytes([values]))
        assert parsed.rows[0]["latitude"] == 23.02

    def test_an_empty_workbook_is_refused(self):
        from openpyxl import Workbook

        book = Workbook()
        buffer = io.BytesIO()
        book.save(buffer)
        with pytest.raises(ParseError, match="no header row"):
            parse_xlsx(buffer.getvalue())

    def test_a_corrupt_workbook_is_refused_cleanly(self):
        with pytest.raises(ParseError, match="could not be opened"):
            parse_xlsx(b"PK\x03\x04 not really a workbook")


class TestJsonParsing:
    def test_a_bare_list_is_accepted(self):
        parsed = parse_json(b'[{"camera_code": "AHM-001"}]')
        assert parsed.rows[0]["camera_code"] == "AHM-001"

    def test_a_cameras_envelope_is_accepted(self):
        parsed = parse_json(b'{"cameras": [{"camera_code": "AHM-001"}]}')
        assert len(parsed.rows) == 1

    def test_an_object_without_a_list_is_refused(self):
        with pytest.raises(ParseError, match="must be a list"):
            parse_json(b'{"nope": 1}')

    def test_malformed_json_is_refused_cleanly(self):
        with pytest.raises(ParseError, match="could not be parsed"):
            parse_json(b"{not json")

    def test_non_object_entries_are_warned_about_not_fatal(self):
        parsed = parse_json(b'[{"camera_code": "A"}, "junk"]')
        assert len(parsed.rows) == 1 and parsed.warnings


# ── the import itself ────────────────────────────────────────────────────


class TestSuccessfulImport:
    def test_csv_rows_become_cameras(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("AHM-001"), row("AHM-002"))
        )
        assert report.status is ImportStatus.COMPLETED
        assert (report.total_rows, report.imported, report.failed) == (2, 2, 0)
        assert importer.cameras.list().total == 2

    def test_xlsx_rows_become_cameras(self, importer):
        content = xlsx_bytes([row("AHM-001").split(","), row("AHM-002").split(",")])
        report = importer.run(filename="c.xlsx", content=content)
        assert report.imported == 2
        assert report.file_format is ImportFormat.XLSX

    def test_json_rows_become_cameras(self, importer):
        payload = json.dumps([
            {
                "camera_code": "AHM-001", "camera_name": "Gate One",
                "department": "Gujarat Police", "district": "Ahmedabad",
                "latitude": 23.02, "longitude": 72.57,
                "camera_type": "bullet", "protocol": "rtsp",
                "stream_url": "rtsp://10.0.0.1/s", "supports_analytics": True,
            }
        ]).encode()
        report = importer.run(filename="c.json", content=payload)
        assert report.imported == 1
        assert importer.cameras.get_by_code("AHM-001").supports_analytics is True

    def test_imported_cameras_are_normalised_by_the_same_schema(self, importer):
        """Lower-case codes are folded exactly as the single-camera POST does."""
        importer.run(filename="c.csv", content=csv_bytes(row("ahm-001")))
        assert importer.cameras.get_by_code("AHM-001").camera_code == "AHM-001"


class TestPartialSuccess:
    def test_one_bad_row_never_aborts_the_import(self, importer):
        report = importer.run(
            filename="c.csv",
            content=csv_bytes(row("AHM-001"), row("AHM-002", lat="91.0"), row("AHM-003")),
        )
        assert report.status is ImportStatus.PARTIAL
        assert (report.imported, report.failed) == (2, 1)
        assert importer.cameras.list().total == 2

    def test_the_error_names_the_row_and_the_column(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("AHM-001"), row("AHM-002", lat="91.0"))
        )
        bad = [e for e in report.errors if e.field == "latitude"]
        assert len(bad) == 1
        assert bad[0].row == 3 and bad[0].camera_code == "AHM-002"

    def test_a_row_with_two_problems_reports_both(self, importer):
        """Fixing a spreadsheet one error at a time is how one import
        becomes five uploads."""
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("AHM-001", lat="91.0", lon="181.0"))
        )
        assert {e.field for e in report.errors} == {"latitude", "longitude"}
        assert report.failed == 1  # counts ROWS, not errors

    def test_nothing_valid_means_failed_not_partial(self, importer):
        report = importer.run(filename="c.csv", content=csv_bytes(row(lat="91.0")))
        assert report.status is ImportStatus.FAILED
        assert report.imported == 0


class TestDuplicates:
    def test_a_repeat_inside_the_file_is_skipped_not_failed(self, importer):
        """The camera IS created -- from the first row -- so the second
        occurrence is a skip, not a failure."""
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("AHM-001"), row("AHM-001", name="Again"))
        )
        assert (report.imported, report.skipped, report.failed) == (1, 1, 0)
        assert importer.cameras.get_by_code("AHM-001").camera_name == "Gate One"

    def test_the_skip_message_points_at_the_first_occurrence(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("AHM-001"), row("AHM-001"))
        )
        assert "row 2" in report.errors[0].message

    def test_a_duplicate_differing_in_case_is_still_a_duplicate(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("AHM-001"), row("ahm-001"))
        )
        assert report.skipped == 1

    def test_a_code_already_in_the_database_is_rejected(self, importer):
        importer.run(filename="first.csv", content=csv_bytes(row("AHM-001")))
        report = importer.run(filename="second.csv", content=csv_bytes(row("AHM-001")))
        assert report.imported == 0 and report.failed == 1
        assert "already registered" in report.errors[0].message


class TestDryRun:
    def test_a_dry_run_writes_no_cameras(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("AHM-001")), dry_run=True
        )
        assert report.status is ImportStatus.VALIDATED
        assert report.imported == 1          # would import
        assert importer.cameras.list().total == 0   # but did not

    def test_a_dry_run_still_reports_bad_rows(self, importer):
        report = importer.run(
            filename="c.csv",
            content=csv_bytes(row("AHM-001"), row("AHM-002", lat="91.0")),
            dry_run=True,
        )
        assert report.failed == 1 and report.errors[0].field == "latitude"

    def test_a_dry_run_reports_a_conflict_the_commit_would_hit(self, importer):
        """Otherwise a dry run passes and the real import then fails."""
        importer.run(filename="first.csv", content=csv_bytes(row("AHM-001")))
        report = importer.run(
            filename="again.csv", content=csv_bytes(row("AHM-001")), dry_run=True
        )
        assert report.failed == 1
        assert "already registered" in report.errors[0].message

    def test_a_dry_run_is_still_retrievable_afterwards(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row()), dry_run=True
        )
        assert importer.get_report(report.job_id).dry_run is True


class TestBadFiles:
    def test_a_missing_required_column_fails_the_file_not_every_row(self, importer):
        content = b"camera_code,camera_name\nAHM-001,Gate One\n"
        report = importer.run(filename="c.csv", content=content)
        assert report.status is ImportStatus.FAILED
        assert "missing required column" in (report.message or "").lower()
        assert "district" in report.message

    def test_an_unsupported_file_type_is_reported_not_raised(self, importer):
        report = importer.run(filename="c.pdf", content=b"\x00\x01junk")
        assert report.status is ImportStatus.FAILED
        assert "Unsupported file type" in report.message

    def test_a_file_with_no_data_rows_is_reported(self, importer):
        report = importer.run(filename="c.csv", content=(HEADER + "\n").encode())
        assert report.status is ImportStatus.FAILED
        assert "no data rows" in report.message

    def test_unknown_columns_are_a_warning_not_a_per_row_failure(self, importer):
        content = csv_bytes(row("AHM-001") + ",oops", header=HEADER + ",nonsense")
        report = importer.run(filename="c.csv", content=content)
        assert report.imported == 1
        assert any("nonsense" in w for w in report.warnings)

    def test_an_oversized_file_is_refused_before_parsing(self, importer, monkeypatch):
        import sentinel_system.bulk_import.service as service_module

        monkeypatch.setattr(service_module, "MAX_ROWS", 2)
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("A-1"), row("A-2"), row("A-3"))
        )
        assert report.status is ImportStatus.FAILED
        assert "the limit is" in report.message


class TestInternalFailure:
    def test_an_unexpected_error_rolls_back_that_row_and_continues(
        self, importer, monkeypatch
    ):
        """Partial success must survive a bug, not just a bad row."""
        real_create = importer.cameras.create
        calls = {"n": 0}

        def flaky(payload):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("database went sideways")
            return real_create(payload)

        monkeypatch.setattr(importer.cameras, "create", flaky)
        report = importer.run(
            filename="c.csv",
            content=csv_bytes(row("AHM-001"), row("AHM-002"), row("AHM-003")),
        )
        assert report.imported == 2 and report.failed == 1
        assert "Unexpected error" in report.errors[0].message
        assert importer.cameras.list().total == 2

    def test_the_session_is_usable_after_a_rolled_back_row(self, importer, monkeypatch):
        def boom(payload):
            raise RuntimeError("nope")

        monkeypatch.setattr(importer.cameras, "create", boom)
        importer.run(filename="c.csv", content=csv_bytes(row("AHM-001")))
        # A fresh, ordinary query must still work on this session.
        assert importer.cameras.list().total == 0


class TestLargeImport:
    def test_a_thousand_rows_import(self, importer):
        rows = [row(f"AHM-{n:05d}") for n in range(1000)]
        report = importer.run(filename="big.csv", content=csv_bytes(*rows))
        assert report.total_rows == 1000
        assert report.imported == 1000
        assert report.status is ImportStatus.COMPLETED
        assert importer.cameras.list(
            __import__("sentinel_system.registry.schemas", fromlist=["CameraFilter"])
            .CameraFilter(limit=1)
        ).total == 1000

    def test_a_large_partly_bad_import_reports_every_bad_row(self, importer):
        rows = [
            row(f"AHM-{n:05d}", lat="91.0" if n % 10 == 0 else "23.02")
            for n in range(200)
        ]
        report = importer.run(filename="big.csv", content=csv_bytes(*rows))
        assert report.imported == 180 and report.failed == 20
        assert len(report.errors) == 20


class TestHistory:
    def test_a_job_is_retrievable_by_id(self, importer):
        report = importer.run(filename="c.csv", content=csv_bytes(row()))
        again = importer.get_report(report.job_id)
        assert again.job_id == report.job_id and again.imported == 1

    def test_the_stored_report_keeps_its_errors(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row("A-1"), row("A-2", lat="91.0"))
        )
        assert len(importer.get_report(report.job_id).errors) == 1

    def test_an_unknown_job_raises(self, importer):
        with pytest.raises(ImportJobNotFoundError):
            importer.get_report(uuid.uuid4())

    def test_history_is_newest_first(self, importer):
        importer.run(filename="one.csv", content=csv_bytes(row("A-1")))
        importer.run(filename="two.csv", content=csv_bytes(row("A-2")))
        assert [j.filename for j in importer.history()] == ["two.csv", "one.csv"]

    def test_the_job_records_what_it_was_told(self, importer):
        report = importer.run(
            filename="ahmedabad.csv", content=csv_bytes(row()), uploaded_by="insp.sharma"
        )
        stored = importer.get_report(report.job_id)
        assert stored.filename == "ahmedabad.csv"
        assert stored.uploaded_by == "insp.sharma"
        assert stored.duration_seconds is not None


class TestTemplate:
    def test_every_required_column_comes_first(self):
        columns = template_columns()
        assert columns[: len(required_columns())] == required_columns()

    def test_the_template_covers_every_schema_field(self):
        from sentinel_system.registry.schemas import CameraCreate

        assert set(template_columns()) == set(CameraCreate.model_fields)

    def test_the_template_round_trips_through_the_importer(self, importer):
        """The strongest check there is: the file we hand out must import."""
        report = importer.run(
            filename="template.csv", content=build_template_csv().encode()
        )
        assert report.imported == 1, report.errors
        assert importer.cameras.get_by_code("AHM-SAT-0142") is not None


class TestLogging:
    @pytest.fixture()
    def captured(self):
        import logging

        records: list[logging.LogRecord] = []

        class Collector(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Collector()
        logger = logging.getLogger("sentinel.import")
        logger.addHandler(handler)
        try:
            yield records
        finally:
            logger.removeHandler(handler)

    def test_start_and_completion_are_logged(self, importer, captured):
        importer.run(filename="c.csv", content=csv_bytes(row()))
        text = " ".join(r.getMessage() for r in captured)
        assert "Import started" in text and "Import completed" in text

    def test_a_bad_file_is_logged_as_failed(self, importer, captured):
        importer.run(filename="c.pdf", content=b"\x00junk")
        assert "Import failed" in " ".join(r.getMessage() for r in captured)

    def test_no_stream_password_reaches_the_log(self, importer, captured):
        content = csv_bytes(row(url="rtsp://admin:hunter2@10.0.0.9/live"))
        importer.run(filename="c.csv", content=content)
        assert "hunter2" not in " ".join(r.getMessage() for r in captured)


class TestErrorAttribution:
    """Cross-field rules are model-level validators, so Pydantic reports them
    with no field. The report must still name the column at fault."""

    def test_a_protocol_url_mismatch_points_at_stream_url(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row(url="https://cam/stream"))
        )
        assert report.errors[0].field == "stream_url"

    def test_a_ptz_conflict_points_at_camera_type(self, importer):
        report = importer.run(filename="c.csv", content=csv_bytes(row(kind="ptz")))
        assert report.errors[0].field == "camera_type"

    def test_pydantics_prefix_is_not_shown_to_operators(self, importer):
        report = importer.run(
            filename="c.csv", content=csv_bytes(row(url="https://cam/stream"))
        )
        assert not report.errors[0].message.startswith("Value error")

    def test_a_plain_field_error_keeps_its_own_field(self, importer):
        report = importer.run(filename="c.csv", content=csv_bytes(row(lat="91")))
        assert report.errors[0].field == "latitude"
