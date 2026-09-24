"""Turn an uploaded file into rows of plain strings.

Parsing only. Nothing here knows what a camera is or which fields are
required -- it produces `{column: value}` dictionaries and the service hands
each one to the same `CameraCreate` schema that the single-camera POST uses.
Keeping the two apart is what stops a second, weaker set of validation rules
growing inside the importer.

Three jobs the formats genuinely need doing before validation can start:

  * empty cells must become None, not "". A spreadsheet's blank `fps` cell
    arrives as an empty string, and `int("")` is a crash where the operator
    meant "not specified".
  * headers must be matched forgivingly. `Camera Code`, `camera_code` and
    `CAMERA CODE ` are the same column to anyone filling in a template.
  * Excel gives real types back (numbers, dates, bools) while CSV gives
    strings, so both are normalised to strings and left for Pydantic to
    coerce -- one coercion path, not two.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any

from sentinel_system.bulk_import.enums import ImportFormat


class ParseError(Exception):
    """The file could not be read at all. Distinct from a row being invalid."""


@dataclass
class ParsedFile:
    """Rows lifted out of an upload, plus anything odd noticed on the way."""

    rows: list[dict[str, Any]]
    #: Header names as they appeared, for reporting unknown columns.
    columns: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    #: File line number for each row, so an error can say "row 17" and mean
    #: the seventeenth line of the operator's spreadsheet.
    line_numbers: list[int] = field(default_factory=list)


_EXTENSIONS = {
    ".csv": ImportFormat.CSV,
    ".xlsx": ImportFormat.XLSX,
    ".xlsm": ImportFormat.XLSX,
    ".json": ImportFormat.JSON,
}

#: Excel files are ZIP archives; JSON starts with a bracket or brace. Used
#: only when the filename does not settle it, because a browser will happily
#: send application/octet-stream for all three.
_XLSX_MAGIC = b"PK\x03\x04"


def detect_format(filename: str, content: bytes) -> ImportFormat:
    """Work out which of the three formats this is.

    The extension is trusted first because it is what the operator chose
    deliberately. Content sniffing is the fallback for uploads that arrive
    without one, and it is also what catches a `.csv` that is really a
    spreadsheet -- a common enough mistake that failing on it with "line 1
    is not valid CSV" would be a poor answer.
    """
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if content.startswith(_XLSX_MAGIC):
        return ImportFormat.XLSX

    stripped = content.lstrip()[:1]
    if stripped in (b"[", b"{"):
        return ImportFormat.JSON

    if suffix in _EXTENSIONS:
        return _EXTENSIONS[suffix]

    raise ParseError(
        "Unsupported file type. Upload a .csv, .xlsx or .json file "
        "(this one could not be identified as any of them)."
    )


def normalise_header(name: Any) -> str:
    """Fold a column heading to its canonical field name."""
    text = str(name or "").strip().lower()
    return text.replace(" ", "_").replace("-", "_")


def _clean(value: Any) -> Any:
    """Blank-ish cells become None; everything else becomes a trimmed string.

    None rather than "" because the schema's optional fields accept None and
    reject "" -- an operator who leaves `vendor` blank means "unknown", and
    that should not read as a validation error.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return value


def parse_csv(content: bytes) -> ParsedFile:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except Exception as exc:  # pragma: no cover - defensive
            raise ParseError(f"The file is not readable text: {exc}") from exc

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise ParseError("The CSV file is empty -- it has no header row.")

    columns = [normalise_header(name) for name in reader.fieldnames]
    rows: list[dict[str, Any]] = []
    lines: list[int] = []
    warnings: list[str] = []

    for offset, raw in enumerate(reader, start=2):  # row 1 is the header
        # csv.DictReader puts surplus cells under a None key; that means the
        # row has more values than headings, which is nearly always a stray
        # comma inside an unquoted field.
        if None in raw:
            warnings.append(
                f"Row {offset}: more values than columns -- check for an "
                f"unquoted comma. The extra values were ignored."
            )
            raw.pop(None, None)
        row = {normalise_header(k): _clean(v) for k, v in raw.items()}
        if not any(value is not None for value in row.values()):
            continue  # a blank line at the end of a spreadsheet
        rows.append(row)
        lines.append(offset)

    return ParsedFile(rows=rows, columns=columns, warnings=warnings, line_numbers=lines)


def parse_xlsx(content: bytes) -> ParsedFile:
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - dependency is installed
        raise ParseError(
            "Excel support needs the 'openpyxl' package on the server."
        ) from exc

    try:
        # read_only + data_only: stream rather than build the whole sheet in
        # memory, and take formula RESULTS rather than the formula text,
        # which is what a cell computing a stream URL should contribute.
        workbook = load_workbook(
            io.BytesIO(content), read_only=True, data_only=True
        )
    except Exception as exc:
        raise ParseError(f"The Excel file could not be opened: {exc}") from exc

    try:
        sheet = workbook.worksheets[0]
        warnings: list[str] = []
        if len(workbook.worksheets) > 1:
            warnings.append(
                f"The workbook has {len(workbook.worksheets)} sheets; only the "
                f"first ({sheet.title!r}) was imported."
            )

        iterator = sheet.iter_rows(values_only=True)
        try:
            header = next(iterator)
        except StopIteration:
            raise ParseError("The spreadsheet is empty -- it has no header row.")

        columns = [normalise_header(name) for name in header]
        rows: list[dict[str, Any]] = []
        lines: list[int] = []
        for offset, values in enumerate(iterator, start=2):
            row = {
                column: _clean(value)
                for column, value in zip(columns, values)
                if column
            }
            if not any(value is not None for value in row.values()):
                continue
            rows.append(row)
            lines.append(offset)
        return ParsedFile(
            rows=rows, columns=columns, warnings=warnings, line_numbers=lines
        )
    finally:
        workbook.close()


def parse_json(content: bytes) -> ParsedFile:
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParseError(f"The JSON file could not be parsed: {exc}") from exc

    # Accept both a bare list and the {"cameras": [...]} envelope, because
    # both are what people actually export.
    if isinstance(payload, dict):
        for key in ("cameras", "items", "data", "rows"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            raise ParseError(
                "The JSON file must be a list of cameras, or an object with a "
                "'cameras' list."
            )
    if not isinstance(payload, list):
        raise ParseError("The JSON file must contain a list of cameras.")

    rows: list[dict[str, Any]] = []
    lines: list[int] = []
    warnings: list[str] = []
    columns: list[str] = []
    for index, entry in enumerate(payload, start=1):
        if not isinstance(entry, dict):
            warnings.append(f"Entry {index} is not an object and was ignored.")
            continue
        row = {normalise_header(k): _clean(v) for k, v in entry.items()}
        for key in row:
            if key not in columns:
                columns.append(key)
        rows.append(row)
        # JSON has no line numbers an operator could act on, so entries are
        # numbered from 1 -- which is what a "row" means in a JSON array.
        lines.append(index)

    return ParsedFile(rows=rows, columns=columns, warnings=warnings, line_numbers=lines)


_PARSERS = {
    ImportFormat.CSV: parse_csv,
    ImportFormat.XLSX: parse_xlsx,
    ImportFormat.JSON: parse_json,
}


def parse(content: bytes, file_format: ImportFormat) -> ParsedFile:
    """Parse `content` in the given format. Raises ParseError on a bad file."""
    return _PARSERS[file_format](content)


__all__ = [
    "ParseError",
    "ParsedFile",
    "detect_format",
    "normalise_header",
    "parse",
    "parse_csv",
    "parse_json",
    "parse_xlsx",
]
