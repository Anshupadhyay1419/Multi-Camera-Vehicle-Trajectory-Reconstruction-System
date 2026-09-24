"""Sentinel portal integration: the portal-independent layer.

No request is made to any real portal. A scripted FakePortal implements the
same PortalClient contract a real client will, so discovery, stream
selection, auth failures and unavailable cameras are driven directly.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from sentinel_system.portal import (
    AuthExpiredError,
    AuthenticationError,
    CameraUnavailableError,
    CatalogueSync,
    NoSupportedStreamError,
    PermissionDeniedError,
    PortalCamera,
    PortalClient,
    PortalNotConfiguredError,
    PortalTimeoutError,
    REQUIRED_INFORMATION,
    ResilientPortalSession,
    StreamOffer,
    StreamOfflineError,
    StreamType,
    UnconfiguredPortalClient,
    derive_camera_code,
    load_credentials,
    select_stream,
)
from sentinel_system.portal.credentials import PortalCredentials, mask_identity
from sentinel_system.registry.enums import CameraStatus, HealthStatus, Protocol

T = StreamType


def cam(portal_id="CAM-001", *, streams=None, online=True, **overrides) -> PortalCamera:
    data = dict(
        portal_id=portal_id, name=f"Camera {portal_id}",
        department="Gujarat Police", district="Ahmedabad",
        location="Satellite Circle", latitude=23.02, longitude=72.57,
        status="online" if online else "offline", online=online,
        streams=streams if streams is not None else (
            StreamOffer(T.RTSP, f"rtsp://10.0.0.1/{portal_id}"),
        ),
    )
    data.update(overrides)
    return PortalCamera(**data)


class FakePortal:
    """Scripted stand-in for a real PortalClient."""

    def __init__(self, cameras=(), *, auth_error=None, expire_times=0,
                 list_error=None, camera_errors=None):
        self.cameras = {c.portal_id: c for c in cameras}
        self.auth_error = auth_error
        self.expire_times = expire_times
        self.list_error = list_error
        self.camera_errors = camera_errors or {}
        self.auth_calls = 0
        self.list_calls = 0

    def authenticate(self):
        self.auth_calls += 1
        if self.auth_error:
            raise self.auth_error

    def list_cameras(self):
        self.list_calls += 1
        if self.expire_times > 0:
            self.expire_times -= 1
            raise AuthExpiredError("session expired")
        if self.list_error:
            raise self.list_error
        return list(self.cameras.values())

    def get_camera(self, portal_id):
        if portal_id in self.camera_errors:
            raise self.camera_errors[portal_id]
        if portal_id not in self.cameras:
            raise CameraUnavailableError(f"No camera {portal_id!r}")
        return self.cameras[portal_id]


def test_the_fake_satisfies_the_real_contract():
    assert isinstance(FakePortal(), PortalClient)


# ── discovery ─────────────────────────────────────────────────────────────


class TestDiscovery:
    def test_every_camera_on_the_account_is_discovered(self):
        portal = ResilientPortalSession(FakePortal([cam("A"), cam("B"), cam("C")]))
        assert [c.portal_id for c in portal.list_cameras()] == ["A", "B", "C"]

    def test_the_session_authenticates_before_the_first_call(self):
        fake = FakePortal([cam()])
        ResilientPortalSession(fake).list_cameras()
        assert fake.auth_calls == 1

    def test_later_calls_reuse_the_session(self):
        fake = FakePortal([cam()])
        session = ResilientPortalSession(fake)
        session.list_cameras()
        session.list_cameras()
        assert fake.auth_calls == 1

    def test_availability_reports_offered_and_up(self):
        camera = cam(streams=(StreamOffer(T.RTSP, "rtsp://x", available=False),
                              StreamOffer(T.HLS, "https://x/a.m3u8")))
        assert camera.availability() == {"rtsp": False, "whep": False, "hls": True}


# ── authentication ────────────────────────────────────────────────────────


class TestAuthentication:
    def test_a_wrong_password_fails_and_is_not_retried(self):
        fake = FakePortal([cam()], auth_error=AuthenticationError("bad credentials"))
        with pytest.raises(AuthenticationError):
            ResilientPortalSession(fake).list_cameras()
        assert fake.auth_calls == 1

    def test_an_expired_session_is_renewed_once_and_the_call_succeeds(self):
        fake = FakePortal([cam()], expire_times=1)
        result = ResilientPortalSession(fake).list_cameras()
        assert len(result) == 1
        assert fake.auth_calls == 2      # initial + one renewal
        assert fake.list_calls == 2

    def test_a_second_expiry_is_raised_not_looped_on(self):
        """A fresh session rejected immediately is a real problem; retrying
        forever would hide it while hammering the portal."""
        fake = FakePortal([cam()], expire_times=5)
        with pytest.raises(AuthExpiredError):
            ResilientPortalSession(fake).list_cameras()
        assert fake.list_calls == 2


# ── unavailable cameras and permissions ──────────────────────────────────


class TestUnavailable:
    def test_a_camera_not_in_the_catalogue_is_unavailable(self):
        with pytest.raises(CameraUnavailableError):
            ResilientPortalSession(FakePortal([cam("A")])).get_camera("NOPE")

    def test_a_camera_the_account_cannot_see_is_permission_denied(self):
        fake = FakePortal([cam("A")],
                          camera_errors={"A": PermissionDeniedError("not permitted")})
        with pytest.raises(PermissionDeniedError):
            ResilientPortalSession(fake).get_camera("A")

    def test_a_portal_timeout_surfaces_as_such(self):
        fake = FakePortal(list_error=PortalTimeoutError("no answer in 10s"))
        with pytest.raises(PortalTimeoutError):
            ResilientPortalSession(fake).list_cameras()


# ── stream selection ──────────────────────────────────────────────────────


class TestStreamSelection:
    def test_rtsp_wins_when_everything_is_up(self):
        camera = cam(streams=(StreamOffer(T.HLS, "https://h/a.m3u8"),
                              StreamOffer(T.WHEP, "https://w/whep"),
                              StreamOffer(T.RTSP, "rtsp://r/s")))
        assert select_stream(camera).stream_type is T.RTSP

    def test_whep_is_next_when_rtsp_is_down(self):
        camera = cam(streams=(StreamOffer(T.HLS, "https://h/a.m3u8"),
                              StreamOffer(T.WHEP, "https://w/whep"),
                              StreamOffer(T.RTSP, "rtsp://r/s", available=False)))
        assert select_stream(camera).stream_type is T.WHEP

    def test_hls_is_the_last_resort(self):
        camera = cam(streams=(StreamOffer(T.HLS, "https://h/a.m3u8"),))
        assert select_stream(camera).stream_type is T.HLS

    def test_all_offered_but_down_is_offline_not_unsupported(self):
        """Only this case is worth retrying later."""
        camera = cam(streams=(StreamOffer(T.RTSP, "rtsp://r/s", available=False),))
        with pytest.raises(StreamOfflineError):
            select_stream(camera)

    def test_nothing_playable_is_unsupported(self):
        with pytest.raises(NoSupportedStreamError):
            select_stream(cam(streams=()))

    def test_the_registry_path_can_pick_an_offline_stream(self):
        camera = cam(streams=(StreamOffer(T.RTSP, "rtsp://r/s", available=False),))
        assert select_stream(camera, require_available=False).stream_type is T.RTSP


# ── registry sync ─────────────────────────────────────────────────────────


class TestRegistrySync:
    def test_discovered_cameras_are_registered(self, session, service):
        report = CatalogueSync(session, FakePortal([cam("A"), cam("B")])).run()
        assert (report.discovered, report.created) == (2, 2)
        assert service.get_by_code("SEN-A").protocol is Protocol.RTSP

    def test_portal_cameras_are_namespaced(self, session, service):
        """They can never overwrite a camera registered by hand."""
        CatalogueSync(session, FakePortal([cam("AHM-SAT-0142")])).run()
        assert service.get_by_code("SEN-AHM-SAT-0142")

    def test_a_second_sync_changes_nothing(self, session):
        fake = FakePortal([cam("A")])
        CatalogueSync(session, fake).run()
        report = CatalogueSync(session, fake).run()
        assert (report.created, report.unchanged) == (0, 1)

    def test_a_changed_stream_is_refreshed(self, session, service):
        CatalogueSync(session, FakePortal([cam("A")])).run()
        moved = cam("A", streams=(StreamOffer(T.RTSP, "rtsp://10.9.9.9/new"),))
        report = CatalogueSync(session, FakePortal([moved])).run()
        assert report.updated == 1
        assert service.get_by_code("SEN-A").stream_url == "rtsp://10.9.9.9/new"

    def test_nothing_is_invented_for_missing_fields(self, session, service):
        """A placeholder location on a police map is worse than no marker."""
        report = CatalogueSync(
            session, FakePortal([cam("A", district=None, latitude=None)])
        ).run()
        assert report.created == 0
        assert "district" in report.skipped[0]["reason"]
        assert "latitude" in report.skipped[0]["reason"]
        assert service.list().total == 0

    def test_an_offline_camera_is_still_registered_as_unreachable(self, session, service):
        """A temporary outage must not look like a decommissioning."""
        offline = cam("A", online=False,
                      streams=(StreamOffer(T.RTSP, "rtsp://r/s", available=False),))
        CatalogueSync(session, FakePortal([offline])).run()
        assert service.get_by_code("SEN-A").health is HealthStatus.UNREACHABLE

    def test_a_camera_with_no_playable_stream_is_skipped(self, session):
        report = CatalogueSync(session, FakePortal([cam("A", streams=())])).run()
        assert report.created == 0 and report.skipped

    def test_resync_never_undoes_an_operator_decommission(self, session, service):
        CatalogueSync(session, FakePortal([cam("A")])).run()
        service.decommission(service.get_by_code("SEN-A").id)
        CatalogueSync(session, FakePortal([cam("A", name="Renamed")])).run()
        camera = service.get_by_code("SEN-A")
        assert camera.status is CameraStatus.DECOMMISSIONED
        assert camera.camera_name == "Renamed"

    def test_whep_and_hls_map_to_registry_protocols(self, session, service):
        CatalogueSync(session, FakePortal([
            cam("W", streams=(StreamOffer(T.WHEP, "https://w/whep"),)),
            cam("H", streams=(StreamOffer(T.HLS, "https://h/a.m3u8"),)),
        ])).run()
        assert service.get_by_code("SEN-W").protocol is Protocol.WEBRTC
        assert service.get_by_code("SEN-H").protocol is Protocol.HLS

    def test_a_row_the_registry_rejects_is_reported_not_fatal(self, session):
        bad = cam("A", latitude=95.0)   # the registry's own range check
        report = CatalogueSync(session, FakePortal([bad, cam("B")])).run()
        assert report.created == 1 and len(report.errors) == 1

    @pytest.mark.parametrize("portal_id,code", [
        ("CAM-0042", "SEN-CAM-0042"),
        ("gj/ahd/101", "SEN-GJ-AHD-101"),
        ("  ", None),
    ])
    def test_camera_codes_are_derived_safely(self, portal_id, code):
        assert derive_camera_code(portal_id) == code


# ── credentials ───────────────────────────────────────────────────────────


class TestCredentials:
    def test_missing_settings_are_named_not_guessed(self, monkeypatch, tmp_path):
        for key in ("SENTINEL_PORTAL_URL", "SENTINEL_USERNAME", "SENTINEL_PASSWORD"):
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(PortalNotConfiguredError) as info:
            load_credentials(tmp_path / "absent.env")
        assert len(info.value.missing) == 3

    def test_environment_variables_win_over_the_local_file(self, monkeypatch, tmp_path):
        local = tmp_path / "portal.env"
        local.write_text("SENTINEL_PORTAL_URL=https://file\nSENTINEL_USERNAME=f@x.in\n"
                         "SENTINEL_PASSWORD=from-file\n")
        monkeypatch.setenv("SENTINEL_PASSWORD", "from-env")
        monkeypatch.delenv("SENTINEL_PORTAL_URL", raising=False)
        monkeypatch.delenv("SENTINEL_USERNAME", raising=False)
        creds = load_credentials(local)
        assert creds.password == "from-env" and creds.base_url == "https://file"

    def test_the_password_never_renders(self):
        creds = PortalCredentials("https://p", "someone@example.com", "hunter2")
        assert "hunter2" not in repr(creds) and "hunter2" not in str(creds)
        assert "so***@example.com" in repr(creds)

    def test_identities_are_masked(self):
        assert mask_identity("someone@example.com") == "so***@example.com"
        assert mask_identity(None) == "<unset>"

    def test_the_real_credentials_file_is_git_ignored(self):
        """A property of the repository, so only checkable where one exists --
        the runtime container ships neither git nor .git."""
        import shutil
        import subprocess
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        if shutil.which("git") is None or not (root / ".git").exists():
            pytest.skip("not running inside a git checkout")
        result = subprocess.run(
            ["git", "check-ignore", "-q", "config/sentinel_portal.env"],
            cwd=root, capture_output=True,
        )
        assert result.returncode == 0


class TestUnconfigured:
    def test_every_call_answers_with_the_checklist(self):
        client = UnconfiguredPortalClient(["SENTINEL_PASSWORD"])
        for call in (client.authenticate, client.list_cameras,
                     lambda: client.get_camera("A")):
            with pytest.raises(PortalNotConfiguredError) as info:
                call()
            assert "SENTINEL_PASSWORD" in info.value.missing
            assert len(info.value.missing) == 1 + len(REQUIRED_INFORMATION)


# ── API ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path}/api.db")
    from sentinel_system.core import database as registry_db
    from src.api.server import app

    with TestClient(app) as client:
        yield client, app
    app.dependency_overrides.clear()
    registry_db.reset_engine()


def _with_portal(app, fake):
    from src.api.sentinel_routes import get_portal_session

    app.dependency_overrides[get_portal_session] = lambda: ResilientPortalSession(fake)


class TestPortalApi:
    def test_unconfigured_is_a_503_carrying_the_checklist(self, api):
        client, _ = api
        response = client.get("/api/v1/portal/cameras")
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "portal_not_configured"
        assert any("documentation" in item for item in detail["missing"])

    def test_status_reports_the_gap_without_values(self, api, monkeypatch):
        client, _ = api
        monkeypatch.setenv("SENTINEL_PASSWORD", "do-not-echo-me")
        body = client.get("/api/v1/portal/status").json()
        assert body["client_available"] is False
        assert body["required_documentation"] == list(REQUIRED_INFORMATION)
        assert "do-not-echo-me" not in str(body)

    def test_discovery_lists_cameras_with_availability(self, api):
        client, app = api
        _with_portal(app, FakePortal([cam("A"), cam("B")]))
        body = client.get("/api/v1/portal/cameras").json()
        assert [c["portal_id"] for c in body] == ["A", "B"]
        assert body[0]["availability"]["rtsp"] is True

    def test_stream_opening_picks_by_priority(self, api):
        client, app = api
        _with_portal(app, FakePortal([cam("A", streams=(
            StreamOffer(T.HLS, "https://h/a.m3u8"),
            StreamOffer(T.WHEP, "https://w/whep")))]))
        body = client.get("/api/v1/portal/cameras/A/stream").json()
        assert body["stream_type"] == "whep"
        assert body["priority"] == ["rtsp", "whep", "hls"]

    @pytest.mark.parametrize("error,http,code", [
        (AuthenticationError("bad"), 502, "portal_auth_failed"),
        (PermissionDeniedError("no"), 403, "portal_permission_denied"),
        (PortalTimeoutError("slow"), 504, "portal_timeout"),
    ])
    def test_failures_map_to_distinct_statuses(self, api, error, http, code):
        client, app = api
        _with_portal(app, FakePortal([cam("A")], camera_errors={"A": error}))
        response = client.get("/api/v1/portal/cameras/A/stream")
        assert response.status_code == http
        assert response.json()["detail"]["code"] == code

    def test_an_unavailable_camera_is_404(self, api):
        client, app = api
        _with_portal(app, FakePortal([cam("A")]))
        response = client.get("/api/v1/portal/cameras/NOPE/stream")
        assert response.status_code == 404
        assert response.json()["detail"]["code"] == "portal_camera_not_found"

    def test_an_offline_stream_is_503_and_unplayable_is_422(self, api):
        client, app = api
        _with_portal(app, FakePortal([
            cam("OFF", streams=(StreamOffer(T.RTSP, "rtsp://r", available=False),)),
            cam("NONE", streams=()),
        ]))
        assert client.get("/api/v1/portal/cameras/OFF/stream").status_code == 503
        assert client.get("/api/v1/portal/cameras/NONE/stream").status_code == 422

    def test_sync_populates_the_registry(self, api):
        client, app = api
        _with_portal(app, FakePortal([cam("A"), cam("B")]))
        report = client.post("/api/v1/portal/sync").json()
        assert report["created"] == 2
        codes = {c["camera_code"] for c in client.get("/api/v1/cameras").json()["items"]}
        assert codes == {"SEN-A", "SEN-B"}

    def test_a_stream_password_never_reaches_the_log(self, api):
        client, app = api
        records: list[logging.LogRecord] = []

        class Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Collect()
        for name in ("api.sentinel_routes", "sentinel.portal.sync"):
            logging.getLogger(name).addHandler(handler)
        try:
            _with_portal(app, FakePortal([cam("A", streams=(
                StreamOffer(T.RTSP, "rtsp://admin:hunter2@10.0.0.1/live"),))]))
            client.get("/api/v1/portal/cameras/A/stream")
            client.post("/api/v1/portal/sync")
        finally:
            for name in ("api.sentinel_routes", "sentinel.portal.sync"):
                logging.getLogger(name).removeHandler(handler)
        text = " ".join(r.getMessage() for r in records)
        assert "hunter2" not in text and "admin:***@" in text
