"""REST API for the camera registry (M1.2).

These drive the real FastAPI application from src/api/server.py -- the same
app that serves /entry and /logs -- with its lifespan running, so they cover
the wiring as well as the handlers: that the router is registered before the
catch-all static mount, and that the registry lands in the database the ALPR
API opened rather than one of its own.

DB_URL points that at a temp file, which is how the existing trajectory API
tests isolate themselves too.
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

CAMERA = {
    "camera_code": "AHM-SAT-0142",
    "camera_name": "Satellite Circle North",
    "department": "Gujarat Police",
    "owner": "Ahmedabad City Police",
    "zone": "Zone-1",
    "district": "Ahmedabad",
    "latitude": 23.0225,
    "longitude": 72.5714,
    "vendor": "Hikvision",
    "camera_type": "bullet",
    "protocol": "rtsp",
    "stream_url": "rtsp://10.20.4.11:554/Streaming/Channels/101",
    "codec": "h265",
    "resolution": "1920x1080",
    "fps": 25,
    "status": "active",
    "health": "healthy",
    "supports_analytics": True,
}


def payload(**overrides) -> dict:
    return {**CAMERA, **overrides}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path}/api.db")
    from sentinel_system.core import database as registry_db
    from src.api.server import app

    with TestClient(app) as test_client:
        yield test_client
    # The registry engine is process-global; leaving it pointed at a deleted
    # temp file would break whatever test ran next.
    registry_db.reset_engine()


def create(client, **overrides):
    return client.post("/api/v1/cameras", json=payload(**overrides))


class TestCreate:
    def test_a_valid_camera_is_created(self, client):
        response = create(client)
        assert response.status_code == 201
        body = response.json()
        assert body["camera_code"] == "AHM-SAT-0142"
        assert uuid.UUID(body["id"])
        assert body["created_at"] and body["updated_at"]

    def test_the_code_is_normalised_on_the_way_in(self, client):
        assert create(client, camera_code="  ahm-sat-0142 ").json()["camera_code"] == (
            "AHM-SAT-0142"
        )

    def test_server_owned_defaults_are_applied(self, client):
        body = create(client, camera_code="AHM-MIN-0001").json()
        assert body["supports_ptz"] is False
        assert body["maintenance_status"] == "none"

    def test_a_duplicate_code_is_a_409_with_a_stable_code(self, client):
        create(client)
        response = create(client, camera_name="Another camera")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "duplicate_camera_code"

    def test_a_duplicate_differing_only_in_case_still_conflicts(self, client):
        create(client)
        assert create(client, camera_code="ahm-sat-0142").status_code == 409


class TestCreateValidation:
    """Every one of these is rejected before anything is written."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("latitude", 91.0),
            ("latitude", -91.0),
            ("longitude", 181.0),
            ("bearing_deg", 360.0),
            ("coverage_radius_m", -1.0),
            ("fps", 0),
            ("camera_code", "AB"),
            ("resolution", "1080p"),
            ("protocol", "carrier-pigeon"),
            ("camera_type", "telescope"),
        ],
    )
    def test_invalid_values_are_rejected(self, client, field, value):
        assert create(client, **{field: value}).status_code == 422

    def test_a_url_contradicting_the_protocol_is_rejected(self, client):
        response = create(client, protocol="rtsp", stream_url="https://cam/stream")
        assert response.status_code == 422

    def test_a_ptz_type_without_ptz_support_is_rejected(self, client):
        response = create(client, camera_type="ptz", supports_ptz=False)
        assert response.status_code == 422

    def test_an_unknown_field_is_rejected_not_ignored(self, client):
        assert create(client, lattitude=23.0).status_code == 422

    def test_a_rejected_payload_stores_nothing(self, client):
        create(client, latitude=91.0)
        assert client.get("/api/v1/cameras").json()["total"] == 0


class TestGet:
    def test_a_camera_can_be_fetched_by_id(self, client):
        created = create(client).json()
        response = client.get(f"/api/v1/cameras/{created['id']}")
        assert response.status_code == 200
        assert response.json()["camera_code"] == "AHM-SAT-0142"

    def test_an_unknown_id_is_404_with_a_stable_code(self, client):
        response = client.get(f"/api/v1/cameras/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "camera_not_found"

    def test_a_malformed_uuid_is_422_not_500(self, client):
        assert client.get("/api/v1/cameras/not-a-uuid").status_code == 422


class TestList:
    def _seed(self, client):
        create(client, camera_code="AHM-001", district="Ahmedabad",
               camera_name="Satellite Circle", status="active", vendor="Hikvision")
        create(client, camera_code="AHM-002", district="Ahmedabad",
               camera_name="Ring Road", status="inactive", vendor="Dahua")
        create(client, camera_code="SUR-001", district="Surat",
               camera_name="Gate Two", status="active", vendor="Hikvision")

    def test_an_empty_registry_lists_cleanly(self, client):
        body = client.get("/api/v1/cameras").json()
        assert body == {"items": [], "total": 0, "page": 1, "page_size": 20,
                        "total_pages": 0, "has_more": False}

    def test_every_camera_is_listed(self, client):
        self._seed(client)
        assert client.get("/api/v1/cameras").json()["total"] == 3

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("district=Ahmedabad", {"AHM-001", "AHM-002"}),
            ("district=Surat", {"SUR-001"}),
            ("status=active", {"AHM-001", "SUR-001"}),
            ("vendor=Dahua", {"AHM-002"}),
            ("protocol=rtsp", {"AHM-001", "AHM-002", "SUR-001"}),
            ("district=Ahmedabad&status=active", {"AHM-001"}),
        ],
    )
    def test_filters_narrow_the_result(self, client, query, expected):
        self._seed(client)
        body = client.get(f"/api/v1/cameras?{query}").json()
        assert {c["camera_code"] for c in body["items"]} == expected

    def test_an_unknown_enum_value_in_a_filter_is_422(self, client):
        assert client.get("/api/v1/cameras?status=retired").status_code == 422

    def test_sorting_ascending_and_descending(self, client):
        self._seed(client)
        up = client.get("/api/v1/cameras?sort_by=camera_code").json()
        down = client.get("/api/v1/cameras?sort_by=camera_code&descending=true").json()
        codes = [c["camera_code"] for c in up["items"]]
        assert codes == ["AHM-001", "AHM-002", "SUR-001"]
        assert [c["camera_code"] for c in down["items"]] == list(reversed(codes))

    def test_sort_by_outside_the_allow_list_is_refused(self, client):
        """sort_by reaches an ORDER BY clause."""
        response = client.get("/api/v1/cameras?sort_by=camera_code;DROP TABLE cameras")
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "invalid_query"


class TestSearch:
    def test_search_matches_the_name_case_insensitively(self, client):
        create(client, camera_code="AHM-001", camera_name="Satellite Circle")
        create(client, camera_code="SUR-001", camera_name="Ring Road")
        body = client.get("/api/v1/cameras?search=satellite").json()
        assert [c["camera_code"] for c in body["items"]] == ["AHM-001"]

    def test_search_matches_the_code(self, client):
        create(client, camera_code="AHM-001")
        create(client, camera_code="SUR-001")
        body = client.get("/api/v1/cameras?search=sur").json()
        assert [c["camera_code"] for c in body["items"]] == ["SUR-001"]

    def test_a_wildcard_is_matched_literally(self, client):
        create(client, camera_code="AHM-001")
        assert client.get("/api/v1/cameras?search=%25").json()["total"] == 0


class TestPagination:
    def _seed(self, client, count=5):
        for n in range(count):
            create(client, camera_code=f"AHM-{n:03d}")

    def test_a_page_reports_the_full_total(self, client):
        self._seed(client)
        body = client.get("/api/v1/cameras?page=1&page_size=2").json()
        assert len(body["items"]) == 2
        assert body["total"] == 5 and body["total_pages"] == 3
        assert body["has_more"] is True

    def test_pages_do_not_overlap_or_skip(self, client):
        self._seed(client)
        seen = []
        for page in (1, 2, 3):
            body = client.get(f"/api/v1/cameras?page={page}&page_size=2").json()
            seen += [c["camera_code"] for c in body["items"]]
        assert seen == sorted(seen) and len(set(seen)) == 5

    def test_the_last_page_reports_no_more(self, client):
        self._seed(client)
        body = client.get("/api/v1/cameras?page=3&page_size=2").json()
        assert len(body["items"]) == 1 and body["has_more"] is False

    def test_a_page_past_the_end_is_empty_not_an_error(self, client):
        self._seed(client)
        body = client.get("/api/v1/cameras?page=99&page_size=2").json()
        assert body["items"] == [] and body["total"] == 5

    @pytest.mark.parametrize("query", ["page=0", "page_size=0", "page_size=201"])
    def test_nonsensical_paging_is_refused(self, client, query):
        assert client.get(f"/api/v1/cameras?{query}").status_code == 422


class TestUpdate:
    def test_only_the_given_fields_change(self, client):
        created = create(client).json()
        response = client.patch(
            f"/api/v1/cameras/{created['id']}", json={"firmware_version": "V6.0.0"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["firmware_version"] == "V6.0.0"
        assert body["camera_name"] == created["camera_name"]

    def test_an_explicit_null_clears_a_field(self, client):
        created = create(client).json()
        body = client.patch(
            f"/api/v1/cameras/{created['id']}", json={"owner": None}
        ).json()
        assert body["owner"] is None

    def test_an_empty_patch_is_a_no_op(self, client):
        created = create(client).json()
        body = client.patch(f"/api/v1/cameras/{created['id']}", json={}).json()
        assert body["updated_at"] == created["updated_at"]

    def test_updating_an_unknown_camera_is_404(self, client):
        response = client.patch(
            f"/api/v1/cameras/{uuid.uuid4()}", json={"camera_name": "X"}
        )
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "camera_not_found"

    def test_camera_code_cannot_be_patched(self, client):
        created = create(client).json()
        response = client.patch(
            f"/api/v1/cameras/{created['id']}", json={"camera_code": "AHM-NEW-0001"}
        )
        assert response.status_code == 422

    def test_a_ptz_switch_without_support_is_refused(self, client):
        """Valid on its own; invalid against the stored row."""
        created = create(client).json()
        response = client.patch(
            f"/api/v1/cameras/{created['id']}", json={"camera_type": "ptz"}
        )
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "invalid_camera_configuration"

    def test_switching_type_and_support_together_is_allowed(self, client):
        created = create(client).json()
        response = client.patch(
            f"/api/v1/cameras/{created['id']}",
            json={"camera_type": "ptz", "supports_ptz": True},
        )
        assert response.status_code == 200 and response.json()["camera_type"] == "ptz"

    def test_a_url_contradicting_the_stored_protocol_is_refused(self, client):
        created = create(client).json()
        response = client.patch(
            f"/api/v1/cameras/{created['id']}", json={"stream_url": "https://cam/x"}
        )
        assert response.status_code == 422


class TestDelete:
    def test_delete_retires_rather_than_destroys(self, client):
        created = create(client).json()
        response = client.delete(f"/api/v1/cameras/{created['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "decommissioned"
        # Still retrievable by id: the record and its history survive.
        assert client.get(f"/api/v1/cameras/{created['id']}").status_code == 200

    def test_a_retired_camera_leaves_the_default_listing(self, client):
        created = create(client).json()
        client.delete(f"/api/v1/cameras/{created['id']}")
        assert client.get("/api/v1/cameras").json()["total"] == 0

    def test_a_retired_camera_is_still_findable_on_request(self, client):
        created = create(client).json()
        client.delete(f"/api/v1/cameras/{created['id']}")
        body = client.get("/api/v1/cameras?status=decommissioned").json()
        assert [c["camera_code"] for c in body["items"]] == ["AHM-SAT-0142"]
        assert client.get(
            "/api/v1/cameras?include_decommissioned=true"
        ).json()["total"] == 1

    def test_retiring_twice_is_not_an_error(self, client):
        """A retried request that already succeeded must not fail."""
        created = create(client).json()
        client.delete(f"/api/v1/cameras/{created['id']}")
        assert client.delete(f"/api/v1/cameras/{created['id']}").status_code == 200

    def test_deleting_an_unknown_camera_is_404(self, client):
        response = client.delete(f"/api/v1/cameras/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "camera_not_found"

    def test_hard_delete_is_refused_unless_the_server_opts_in(self, client):
        created = create(client).json()
        response = client.delete(f"/api/v1/cameras/{created['id']}?hard=true")
        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "hard_delete_disabled"
        assert client.get(f"/api/v1/cameras/{created['id']}").status_code == 200

    def test_hard_delete_destroys_the_row_when_enabled(self, client, monkeypatch):
        from sentinel_system.core import config as registry_config

        monkeypatch.setenv("SENTINEL_ALLOW_HARD_DELETE", "true")
        registry_config.reset_settings()
        try:
            created = create(client).json()
            response = client.delete(f"/api/v1/cameras/{created['id']}?hard=true")
            assert response.status_code == 204
            assert client.get(f"/api/v1/cameras/{created['id']}").status_code == 404
        finally:
            registry_config.reset_settings()


class TestOpenAPI:
    def test_every_endpoint_is_documented(self, client):
        spec = client.get("/openapi.json").json()
        paths = spec["paths"]
        assert set(paths["/api/v1/cameras"]) >= {"get", "post"}
        assert set(paths["/api/v1/cameras/{camera_id}"]) >= {"get", "patch", "delete"}

    def test_each_operation_has_a_summary_and_description(self, client):
        spec = client.get("/openapi.json").json()
        for path in ("/api/v1/cameras", "/api/v1/cameras/{camera_id}"):
            for method, operation in spec["paths"][path].items():
                assert operation.get("summary"), f"{method} {path} has no summary"
                assert operation.get("description"), f"{method} {path} has no description"

    def test_error_responses_are_documented(self, client):
        spec = client.get("/openapi.json").json()
        assert "409" in spec["paths"]["/api/v1/cameras"]["post"]["responses"]
        assert "404" in spec["paths"]["/api/v1/cameras/{camera_id}"]["get"]["responses"]

    def test_the_request_and_response_models_are_published(self, client):
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        assert {"CameraCreate", "CameraRead", "CameraUpdate", "CameraPageResponse"} <= set(
            schemas
        )


class TestExistingApiUntouched:
    """M1.2 adds routes; it must not displace any."""

    def test_the_alpr_endpoints_still_respond(self, client):
        assert client.get("/health").status_code == 200
        assert client.get("/logs").status_code == 200

    def test_the_trajectory_routes_are_still_mounted(self, client):
        spec = client.get("/openapi.json").json()
        assert any(p.startswith("/trajectory-api") for p in spec["paths"])

    def test_an_alpr_validation_error_keeps_its_original_shape(self, client):
        """The validation logger delegates to FastAPI's stock handler, so
        existing error bodies must be unchanged: a list, not our dict."""
        response = client.post("/entry", json={"plate_number": 12345})
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)


class TestLogging:
    """The project's loggers set propagate=False, so caplog's root handler
    never sees them. These attach to the named loggers directly."""

    @pytest.fixture()
    def captured(self):
        import logging

        records: list[logging.LogRecord] = []

        class Collector(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Collector()
        loggers = [
            logging.getLogger("api.sentinel_routes"),
            logging.getLogger("api.server"),
        ]
        for logger in loggers:
            logger.addHandler(handler)
        try:
            yield records
        finally:
            for logger in loggers:
                logger.removeHandler(handler)

    def test_a_stream_url_password_never_reaches_the_log(self, client, captured):
        create(
            client,
            camera_code="AHM-SEC-0001",
            stream_url="rtsp://admin:hunter2@10.20.4.11:554/live",
        )
        combined = " ".join(record.getMessage() for record in captured)
        assert "Camera created" in combined
        assert "hunter2" not in combined
        # The username is kept deliberately -- "which account is this using?"
        # is what makes a failed login diagnosable.
        assert "admin:***@10.20.4.11" in combined

    def test_creation_update_and_retirement_are_logged(self, client, captured):
        created = create(client).json()
        client.patch(f"/api/v1/cameras/{created['id']}", json={"fps": 30})
        client.delete(f"/api/v1/cameras/{created['id']}")
        combined = " ".join(record.getMessage() for record in captured)
        assert "Camera created" in combined
        assert "Camera updated" in combined
        assert "Camera decommissioned" in combined

    def test_a_rejected_body_is_logged_with_the_offending_field(self, client, captured):
        create(client, latitude=91.0)
        combined = " ".join(record.getMessage() for record in captured)
        assert "rejected" in combined.lower()
        assert "latitude" in combined

    def test_a_duplicate_is_logged(self, client, captured):
        create(client)
        create(client)
        combined = " ".join(record.getMessage() for record in captured)
        assert "already registered" in combined


# ── M1.3: verification endpoint ──────────────────────────────────────────


@pytest.fixture()
def verifying(client, tmp_path):
    """Point the verify endpoint at a scripted sampler and a temp thumbnail dir.

    No camera and no network: the dependency is overridden so the endpoint
    exercises the real service, model and storage while the only thing faked
    is the stream itself.
    """
    import numpy as np

    from sentinel_system.core.config import Settings
    from sentinel_system.verification.service import CameraVerificationService
    from src.api.server import app
    from src.api.sentinel_routes import get_registry_session, get_verification_service
    from src.cameras.stream_probe import StreamSample

    scripted: dict = {
        "sample": StreamSample(
            ok=True,
            message="Read 5 frame(s) at 1920x1080.",
            host="10.20.4.11", port=554,
            connect_latency_ms=3.1, open_latency_ms=12.0,
            first_frame_latency_ms=26.9,
            frames_read=5, measured_fps=24.9, declared_fps=25.0,
            width=1920, height=1080, fourcc="hevc",
            frame=np.zeros((1080, 1920, 3), np.uint8),
        )
    }
    settings = Settings(thumbnail_directory=str(tmp_path / "cameras"))

    def override(session=Depends(get_registry_session)):
        return CameraVerificationService(
            session, settings=settings, sampler=lambda url, **kw: scripted["sample"]
        )

    app.dependency_overrides[get_verification_service] = override
    try:
        yield scripted
    finally:
        app.dependency_overrides.pop(get_verification_service, None)


class TestVerifyEndpoint:
    def test_a_working_camera_verifies(self, client, verifying):
        created = create(client).json()
        response = client.post(f"/api/v1/cameras/{created['id']}/verify")
        assert response.status_code == 200
        body = response.json()
        assert body["verified"] is True
        assert body["connection_status"] == "reachable"
        assert body["measured_resolution"] == "1920x1080"
        assert body["measured_fps"] == 24.9
        assert body["codec_detected"] == "hevc"
        assert body["errors"] == []
        assert body["thumbnail_path"] == f"/thumbnails/cameras/{created['id']}.jpg"
        assert body["checked_at"]

    def test_a_broken_camera_is_a_200_not_an_error(self, client, verifying):
        """'This camera is unreachable' must stay distinguishable from
        'the verify endpoint is broken'."""
        from src.cameras.stream_probe import StreamSample

        verifying["sample"] = StreamSample(
            ok=False, message="Cannot resolve the camera address 'cam.local'."
        )
        created = create(client).json()
        response = client.post(f"/api/v1/cameras/{created['id']}/verify")
        assert response.status_code == 200
        body = response.json()
        assert body["verified"] is False
        assert body["connection_status"] == "unreachable"
        assert body["errors"] == ["Cannot resolve the camera address 'cam.local'."]
        assert body["measured_fps"] is None

    def test_verifying_an_unknown_camera_is_404(self, client, verifying):
        response = client.post(f"/api/v1/cameras/{uuid.uuid4()}/verify")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "camera_not_found"

    def test_verification_never_alters_the_camera(self, client, verifying):
        """Registered specification and measured reality stay separate."""
        created = create(client).json()
        client.post(f"/api/v1/cameras/{created['id']}/verify")
        after = client.get(f"/api/v1/cameras/{created['id']}").json()
        for field in ("resolution", "fps", "codec", "status", "health"):
            assert after[field] == created[field]

    def test_history_records_every_attempt_newest_first(self, client, verifying):
        from src.cameras.stream_probe import StreamSample

        created = create(client).json()
        client.post(f"/api/v1/cameras/{created['id']}/verify")
        verifying["sample"] = StreamSample(ok=False, message="gone dark")
        client.post(f"/api/v1/cameras/{created['id']}/verify")

        history = client.get(f"/api/v1/cameras/{created['id']}/verifications").json()
        assert len(history) == 2
        assert history[0]["verified"] is False
        assert history[1]["verified"] is True

    def test_history_for_an_unknown_camera_is_404(self, client, verifying):
        """404, not an empty list that reads as 'never verified'."""
        response = client.get(f"/api/v1/cameras/{uuid.uuid4()}/verifications")
        assert response.status_code == 404

    def test_history_is_empty_before_any_verification(self, client, verifying):
        created = create(client).json()
        assert client.get(f"/api/v1/cameras/{created['id']}/verifications").json() == []

    def test_the_endpoints_are_documented(self, client):
        spec = client.get("/openapi.json").json()
        verify = spec["paths"]["/api/v1/cameras/{camera_id}/verify"]["post"]
        assert verify["summary"] and verify["description"]
        assert "404" in verify["responses"]
        assert "VerificationResult" in str(spec["components"]["schemas"].keys())


# ── M1.4: bulk import endpoints ──────────────────────────────────────────

_IMPORT_HEADER = (
    "camera_code,camera_name,department,district,latitude,longitude,"
    "camera_type,protocol,stream_url"
)


def _import_row(code="AHM-001", lat="23.02"):
    return (
        f"{code},Gate {code},Gujarat Police,Ahmedabad,{lat},72.57,"
        f"bullet,rtsp,rtsp://10.0.0.1/{code}"
    )


def _csv(*rows: str) -> bytes:
    return ("\n".join([_IMPORT_HEADER, *rows]) + "\n").encode()


def _upload(client, content: bytes, name="cams.csv", url="/api/v1/cameras/import",
            mime="text/csv", **params):
    return client.post(url, files={"file": (name, content, mime)}, params=params)


class TestImportTemplate:
    def test_the_template_downloads_as_a_csv_attachment(self, client):
        response = client.get("/api/v1/cameras/import/template")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]
        assert "camera_import_template.csv" in response.headers["content-disposition"]

    def test_the_template_has_a_header_and_one_example_row(self, client):
        lines = client.get("/api/v1/cameras/import/template").text.strip().splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("camera_code,camera_name")
        assert "AHM-SAT-0142" in lines[1]

    def test_template_is_not_captured_as_a_job_id(self, client):
        """/import/template must be matched before /import/{job_id}."""
        assert client.get("/api/v1/cameras/import/template").status_code == 200


class TestImportEndpoint:
    def test_a_good_csv_imports(self, client):
        response = _upload(client, _csv(_import_row("AHM-001"), _import_row("AHM-002")))
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert (body["total_rows"], body["imported"], body["failed"]) == (2, 2, 0)
        assert client.get("/api/v1/cameras").json()["total"] == 2

    def test_partial_success_is_reported_row_by_row(self, client):
        response = _upload(
            client,
            _csv(_import_row("AHM-001"), _import_row("AHM-002", lat="91.0")),
        )
        body = response.json()
        assert body["status"] == "partial"
        assert (body["imported"], body["failed"]) == (1, 1)
        assert body["errors"][0] == {
            "row": 3,
            "camera_code": "AHM-002",
            "field": "latitude",
            "message": body["errors"][0]["message"],
        }
        assert client.get("/api/v1/cameras").json()["total"] == 1

    def test_an_excel_upload_imports(self, client):
        from openpyxl import Workbook
        import io as _io

        book = Workbook()
        sheet = book.active
        sheet.append(_IMPORT_HEADER.split(","))
        sheet.append(_import_row("AHM-010").split(","))
        buffer = _io.BytesIO()
        book.save(buffer)
        response = _upload(
            client, buffer.getvalue(), name="cams.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        assert response.json()["imported"] == 1
        assert response.json()["file_format"] == "xlsx"

    def test_a_json_upload_imports(self, client):
        payload = json.dumps([{
            "camera_code": "AHM-020", "camera_name": "Gate",
            "department": "Gujarat Police", "district": "Ahmedabad",
            "latitude": 23.02, "longitude": 72.57,
            "camera_type": "bullet", "protocol": "rtsp",
            "stream_url": "rtsp://10.0.0.1/s",
        }]).encode()
        response = _upload(client, payload, name="cams.json", mime="application/json")
        assert response.json()["imported"] == 1
        assert response.json()["file_format"] == "json"

    def test_an_unsupported_file_is_a_report_not_a_crash(self, client):
        response = _upload(client, b"\x00\x01binary", name="cams.pdf",
                           mime="application/pdf")
        assert response.status_code == 200
        assert response.json()["status"] == "failed"
        assert "Unsupported file type" in response.json()["message"]

    def test_an_empty_upload_is_refused(self, client):
        response = _upload(client, b"")
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "import_file_empty"

    def test_a_missing_required_column_fails_the_file(self, client):
        response = _upload(client, b"camera_code,camera_name\nAHM-001,Gate\n")
        body = response.json()
        assert body["status"] == "failed"
        assert "district" in body["message"]

    def test_a_duplicate_of_an_existing_camera_is_rejected(self, client):
        create(client, camera_code="AHM-001")
        response = _upload(client, _csv(_import_row("AHM-001")))
        body = response.json()
        assert body["imported"] == 0 and body["failed"] == 1
        assert "already registered" in body["errors"][0]["message"]

    def test_imports_go_through_the_same_validation_as_the_single_post(self, client):
        """A row the single-camera POST would refuse must be refused here."""
        bad = _import_row("AHM-030").replace("rtsp://10.0.0.1/AHM-030",
                                             "https://cam/stream")
        assert _upload(client, _csv(bad)).json()["failed"] == 1
        direct = create(client, camera_code="AHM-031",
                        stream_url="https://cam/stream")
        assert direct.status_code == 422


class TestDryRunEndpoints:
    def test_dry_run_creates_nothing(self, client):
        response = _upload(client, _csv(_import_row("AHM-001")), dry_run="true")
        body = response.json()
        assert body["status"] == "validated" and body["dry_run"] is True
        assert body["imported"] == 1
        assert client.get("/api/v1/cameras").json()["total"] == 0

    def test_the_validate_endpoint_creates_nothing(self, client):
        response = _upload(
            client, _csv(_import_row("AHM-001")),
            url="/api/v1/cameras/import/validate",
        )
        assert response.json()["status"] == "validated"
        assert client.get("/api/v1/cameras").json()["total"] == 0

    def test_validate_reports_the_same_errors_as_a_commit_would(self, client):
        response = _upload(
            client, _csv(_import_row("AHM-001", lat="91.0")),
            url="/api/v1/cameras/import/validate",
        )
        assert response.json()["errors"][0]["field"] == "latitude"


class TestImportHistoryEndpoint:
    def test_a_report_is_retrievable_afterwards(self, client):
        job = _upload(client, _csv(_import_row("AHM-001"))).json()
        again = client.get(f"/api/v1/cameras/import/{job['job_id']}")
        assert again.status_code == 200
        assert again.json()["job_id"] == job["job_id"]
        assert again.json()["imported"] == 1

    def test_the_stored_report_keeps_its_errors(self, client):
        job = _upload(
            client, _csv(_import_row("AHM-001"), _import_row("AHM-002", lat="91.0"))
        ).json()
        stored = client.get(f"/api/v1/cameras/import/{job['job_id']}").json()
        assert len(stored["errors"]) == 1 and stored["errors"][0]["row"] == 3

    def test_an_unknown_job_is_404(self, client):
        response = client.get(f"/api/v1/cameras/import/{uuid.uuid4()}")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "import_job_not_found"


class TestImportOpenAPI:
    def test_all_four_endpoints_are_documented(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        assert "post" in paths["/api/v1/cameras/import"]
        assert "post" in paths["/api/v1/cameras/import/validate"]
        assert "get" in paths["/api/v1/cameras/import/template"]
        assert "get" in paths["/api/v1/cameras/import/{job_id}"]

    def test_each_has_a_summary_and_description(self, client):
        paths = client.get("/openapi.json").json()["paths"]
        for path in (
            "/api/v1/cameras/import",
            "/api/v1/cameras/import/validate",
            "/api/v1/cameras/import/template",
            "/api/v1/cameras/import/{job_id}",
        ):
            for operation in paths[path].values():
                assert operation.get("summary") and operation.get("description")

    def test_the_report_model_is_published(self, client):
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        assert "ImportReport" in schemas and "RowError" in schemas
