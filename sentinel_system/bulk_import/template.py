"""The downloadable CSV template.

Columns are taken from `CameraCreate` rather than typed out here, so a field
added to the schema cannot quietly go missing from the template that
operators fill in. Required columns come first and are the ones the schema
says are required -- again derived, not restated.
"""

from __future__ import annotations

import csv
import io

from sentinel_system.registry.schemas import CameraCreate

#: One filled-in row, so the file explains its own formats -- what a bearing
#: looks like, that booleans are true/false, that resolution is WIDTHxHEIGHT.
#: An empty template makes every one of those a guess.
EXAMPLE_ROW: dict[str, str] = {
    "camera_code": "AHM-SAT-0142",
    "camera_name": "Satellite Circle North",
    "description": "Junction overview, facing the underpass",
    "department": "Gujarat Police",
    "owner": "Ahmedabad City Police",
    "zone": "Zone-1",
    "district": "Ahmedabad",
    "latitude": "23.0225",
    "longitude": "72.5714",
    "address": "Satellite Circle, Ahmedabad",
    "vendor": "Hikvision",
    "model": "DS-2CD2T47G2",
    "serial_number": "HK-2291-8841",
    "camera_type": "bullet",
    "protocol": "rtsp",
    "stream_url": "rtsp://10.20.4.11:554/Streaming/Channels/101",
    "codec": "h265",
    "resolution": "1920x1080",
    "fps": "25",
    "status": "active",
    "health": "unknown",
    "maintenance_status": "none",
    "firmware_version": "V5.7.3",
    "supports_ptz": "false",
    "supports_audio": "true",
    "supports_nightvision": "true",
    "supports_analytics": "true",
    "coverage_radius_m": "80",
    "bearing_deg": "135",
}


def required_columns() -> list[str]:
    """Columns the schema will refuse a row without."""
    return [
        name
        for name, field in CameraCreate.model_fields.items()
        if field.is_required()
    ]


def template_columns() -> list[str]:
    """Every column the importer understands, required ones first."""
    required = required_columns()
    optional = [
        name for name in CameraCreate.model_fields if name not in set(required)
    ]
    return required + optional


def build_template_csv() -> str:
    """The template file's contents: a header row and one example row."""
    columns = template_columns()
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerow({column: EXAMPLE_ROW.get(column, "") for column in columns})
    return buffer.getvalue()


__all__ = [
    "EXAMPLE_ROW",
    "build_template_csv",
    "required_columns",
    "template_columns",
]
