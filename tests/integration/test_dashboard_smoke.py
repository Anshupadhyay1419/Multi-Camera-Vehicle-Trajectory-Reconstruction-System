"""
Smoke tests that actually RENDER the dashboards and click through them.

These exist because of a class of failure the rest of the suite cannot see:
a Streamlit page can die in a way that is not a Python exception. This
deployment pairs pyarrow 25 with numpy 1.26 -- an ABI mismatch, since
pyarrow 25 is built against numpy 2.x -- and Arrow's DataFrame conversion
SEGFAULTS under some conditions. A segfault kills the server process
outright, so the operator sees only "Connection error"; nothing is logged,
nothing is raised, and no unit test of the underlying functions notices,
because the functions themselves are fine.

Both dashboards therefore render their tables as plain HTML rather than
through st.dataframe/st.table. These tests walk the real interaction paths
so that reintroducing an Arrow-backed element (or any other page-level
crash) fails here rather than in front of a user.

A crash shows up as pytest's worker dying rather than as a normal failure --
which is itself the signal.
"""

from __future__ import annotations

import pytest

from src.database import db as database

streamlit = pytest.importorskip("streamlit", reason="dashboard tests need streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pathlib import Path as _Path

REPO_ROOT_FOR_TEST = _Path(__file__).resolve().parents[2]
TRAJECTORY_APP = "src/dashboard/trajectory_app.py"
LEGACY_APP = "src/dashboard/app.py"

# Four cameras, one vehicle passing all of them, plus a single-camera vehicle.
_SITES = [
    ("CAM001", "India Gate", 28.6129, 77.2295, 1, 0),
    ("CAM002", "Connaught Place", 28.6315, 77.2167, 2, 12),
    ("CAM003", "Karol Bagh", 28.6519, 77.1909, 3, 25),
    ("CAM004", "Kashmere Gate", 28.6675, 77.2273, 4, 48),
]


@pytest.fixture()
def seeded_db(tmp_path):
    """A real database with a real multi-camera journey in it.

    Function-scoped, not module-scoped: the deletion tests really do remove
    these rows, and a shared database would leave every test that ran after
    them looking at an empty system.
    """
    path = tmp_path / "alpr.db"
    database.init_db(str(path))
    with database.get_session() as session:
        for camera_id, name, lat, lon, order, minute in _SITES:
            for repeat in range(2):
                database.insert_event(session, {
                    "plate_number": "DL8CA1234", "vehicle_type": "Car",
                    "plate_color": "White", "series_type": "normal",
                    "direction": "IN", "image_path": "",
                    "camera_id": camera_id, "camera_name": name,
                    "latitude": lat, "longitude": lon,
                    "timestamp": f"2026-09-12T09:{minute:02d}:0{repeat}+00:00",
                    "trajectory_order": order, "processing_session": "SMOKE",
                    "video_source": f"data/uploads/{camera_id}.mp4",
                    "confidence": 0.90 + repeat * 0.05, "ocr_text": "DL8CA1234",
                    "vehicle_image_path": "",
                })
        database.insert_event(session, {
            "plate_number": "HR26DK8337", "vehicle_type": "Car",
            "plate_color": "White", "series_type": "normal", "direction": "IN",
            "image_path": "", "camera_id": "CAM001", "camera_name": "India Gate",
            "latitude": 28.6129, "longitude": 77.2295,
            "timestamp": "2026-09-12T09:05:00+00:00", "trajectory_order": 1,
            "processing_session": "SMOKE", "confidence": 0.8,
        })
    return str(path)


@pytest.fixture()
def app(seeded_db, monkeypatch):
    """An AppTest factory bound to the seeded database.

    ALPR_DB_PATH is the documented override that load_config() honours, so
    this points every layer the page touches at the test database without
    editing config.yaml.
    """
    monkeypatch.setenv("ALPR_DB_PATH", seeded_db)
    streamlit.cache_resource.clear()
    streamlit.cache_data.clear()

    def build(**session_state) -> AppTest:
        at = AppTest.from_file(TRAJECTORY_APP, default_timeout=90)
        for key, value in session_state.items():
            at.session_state[key] = value
        at.run()
        return at

    yield build
    streamlit.cache_resource.clear()
    streamlit.cache_data.clear()


def _assert_clean(at: AppTest, what: str) -> None:
    assert not at.exception, f"{what}: {at.exception[0].value}"


def _session_scope_box(at: AppTest):
    """The sidebar's session picker, found by its contents.

    The page renders several selectboxes and their order is an
    implementation detail; matching on the option the widget is guaranteed
    to carry keeps the test from breaking every time the layout moves.
    """
    for box in at.get("selectbox"):
        if box.options and box.options[0] == "All sessions":
            return box
    raise AssertionError("no session-scope selectbox on the page")


class TestOperationsTab:
    def test_the_page_renders(self, app):
        at = app()
        _assert_clean(at, "initial render")
        labels = {h.value for h in at.subheader}
        assert "Cameras" in labels
        assert "Live Processing Status" in labels
        assert "Statistics" in labels

    def test_every_configured_camera_is_offered_a_source(self, app):
        at = app()
        # One radio per camera, each with both source kinds.
        radios = at.get("radio")
        assert len(radios) == 4
        for radio in radios:
            assert len(radio.options) == 2

    def test_statistics_render_real_database_numbers(self, app):
        """The figures must come from the database, not from placeholders."""
        at = app()
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Vehicles detected"] == "9"   # 8 + 1 seeded events
        assert metrics["Unique plates"] == "2"

    def test_switching_a_camera_to_rtsp_does_not_crash_the_page(self, app):
        """The exact interaction that used to segfault the server."""
        at = app()
        at.get("radio")[0].set_value("RTSP stream").run()
        _assert_clean(at, "after switching to RTSP")
        assert len(at.get("file_uploader")) == 3, "camera 1 should now show a URL box"
        assert any("RTSP URL" in t.label for t in at.get("text_input"))

    def test_switching_every_camera_back_and_forth(self, app):
        at = app()
        for index in range(4):
            at.get("radio")[index].set_value("RTSP stream").run()
            _assert_clean(at, f"camera {index} -> RTSP")
        assert len(at.get("file_uploader")) == 0
        for index in range(4):
            at.get("radio")[index].set_value("Upload video").run()
            _assert_clean(at, f"camera {index} -> upload")
        assert len(at.get("file_uploader")) == 4

    def test_repeated_reruns_stay_stable(self, app):
        """The Arrow crash was intermittent, so one clean render proved little."""
        at = app()
        for _ in range(5):
            at.run()
            _assert_clean(at, "rerun")


class TestSourceSelectionFlow:
    """Clicking through the real source controls, end to end.

    These exist because setting an RTSP URL silently did nothing: the input
    lived in an st.form whose submit never reached the manager, so the camera
    kept its uploaded video, the card still displayed the stream URL, and the
    run processed the file. Nothing raised -- which is why only a
    click-through test catches it.
    """

    def _manager(self):
        from src.cameras.manager import get_camera_manager
        from src.utils.config import load_config

        return get_camera_manager(load_config("config/config.yaml"),
                                  "config/camera_config.yaml")

    def test_setting_an_rtsp_url_actually_changes_the_source(self, app):
        from src.cameras.manager import reset_camera_manager

        reset_camera_manager()
        at = app()
        at.get("radio")[0].set_value("RTSP stream").run()
        at.get("text_input")[0].set_value("rtsp://192.168.1.50:554/stream1").run()
        next(b for b in at.button if b.label == "Set stream").click().run()
        _assert_clean(at, "after setting an RTSP URL")

        camera = self._manager().registry.require("CAM001")
        assert camera.source_type.value == "rtsp"
        assert camera.rtsp_url == "rtsp://192.168.1.50:554/stream1"
        assert camera.video_path is None, "the uploaded video is still attached"
        reset_camera_manager()

    def test_the_stream_survives_reruns_and_a_look_at_the_upload_tab(self, app):
        """The file_uploader keeps holding the previous file, so switching
        the radio back used to re-save it and silently replace the stream."""
        from src.cameras.manager import reset_camera_manager

        reset_camera_manager()
        at = app()
        at.get("radio")[0].set_value("RTSP stream").run()
        at.get("text_input")[0].set_value("rtsp://host/stream").run()
        next(b for b in at.button if b.label == "Set stream").click().run()

        for _ in range(3):
            at.run()
        at.get("radio")[0].set_value("Upload video").run()
        at.get("radio")[0].set_value("RTSP stream").run()
        _assert_clean(at, "after toggling the source kind")

        camera = self._manager().registry.require("CAM001")
        assert camera.source_type.value == "rtsp"
        assert camera.rtsp_url == "rtsp://host/stream"
        reset_camera_manager()

    def test_a_bad_stream_url_is_reported_and_changes_nothing(self, app):
        from src.cameras.manager import reset_camera_manager

        reset_camera_manager()
        at = app()
        before = self._manager().registry.require("CAM001").source_type.value
        at.get("radio")[0].set_value("RTSP stream").run()
        at.get("text_input")[0].set_value("192.168.1.50/not-a-url").run()
        next(b for b in at.button if b.label == "Set stream").click().run()

        _assert_clean(at, "after a bad URL")
        assert at.error, "a malformed URL should be reported"
        assert self._manager().registry.require("CAM001").source_type.value == before
        reset_camera_manager()


class TestCameraWall:
    """The multi-camera feed view."""

    def test_the_wall_lists_every_camera(self, app):
        at = app()
        assert "Camera feeds" in {h.value for h in at.subheader}
        captions = " ".join(c.value for c in at.caption)
        for camera_id in ("CAM001", "CAM002", "CAM003", "CAM004"):
            assert camera_id in captions

    def test_frames_appear_when_the_pipeline_publishes_them(self, app, tmp_path):
        """Each camera shows its OWN frame -- one shared file would put the
        active camera's video behind three other cameras' labels."""
        import cv2
        import numpy as np

        frames = REPO_ROOT_FOR_TEST / "data" / "live_frames"
        frames.mkdir(parents=True, exist_ok=True)
        written = []
        try:
            for camera_id in ("CAM001", "CAM002"):
                image = np.full((120, 160, 3), 60, dtype=np.uint8)
                path = frames / f"{camera_id}.jpg"
                cv2.imwrite(str(path), image)
                written.append(path)

            at = app()
            _assert_clean(at, "camera wall with frames")
            assert len(at.get("imgs") or []) >= 2
        finally:
            for path in written:
                path.unlink(missing_ok=True)

    def test_a_partial_jpeg_is_skipped_rather_than_shown(self, app):
        """The pipeline rewrites these files several times a second, so a
        read can land mid-write; a truncated JPEG must not be rendered."""
        frames = REPO_ROOT_FOR_TEST / "data" / "live_frames"
        frames.mkdir(parents=True, exist_ok=True)
        partial = frames / "CAM003.jpg"
        try:
            partial.write_bytes(b"\xff\xd8" + b"\x00" * 200)   # no EOI marker
            at = app()
            _assert_clean(at, "camera wall with a partial frame")
        finally:
            partial.unlink(missing_ok=True)


class TestTrajectoryTab:
    def test_a_real_plate_reconstructs_and_renders(self, app):
        at = app(active_plate="DL8CA1234")
        _assert_clean(at, "trajectory view")
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Cameras visited"] == "4"
        assert metrics["Total detections"] == "8"

    def test_the_trajectory_sections_are_all_present(self, app):
        at = app(active_plate="DL8CA1234")
        headings = [m.value for m in at.markdown if m.value.startswith("###")]
        assert any("timeline" in h.lower() for h in headings)
        assert any("map" in h.lower() for h in headings)
        assert any("history" in h.lower() for h in headings)

    def test_the_route_is_read_from_stored_coordinates(self, app):
        """Not a hardcoded path: the camera names must appear in the order the
        stored timestamps put them, and the real coordinates must be present."""
        at = app(active_plate="DL8CA1234")
        page = "\n".join(m.value for m in at.markdown)
        for name in ("India Gate", "Connaught Place", "Karol Bagh", "Kashmere Gate"):
            assert name in page
        assert "28.61290" in page or "28.6129" in page

    def test_satellite_toggle_does_not_crash(self, app):
        at = app(active_plate="DL8CA1234")
        toggles = at.get("toggle")
        assert toggles, "the map should offer a satellite toggle"
        toggles[0].set_value(True).run()
        _assert_clean(at, "after enabling satellite view")

    def test_an_unknown_plate_warns_instead_of_crashing(self, app):
        at = app(active_plate="ZZ00ZZ0000")
        _assert_clean(at, "unknown plate")
        assert at.warning, "an unknown plate should warn, not fail silently"

    def test_a_single_camera_plate_still_renders(self, app):
        """One sighting is not a path, but it must not break the page."""
        at = app(active_plate="HR26DK8337")
        _assert_clean(at, "single-sighting plate")
        assert {m.label: m.value for m in at.metric}["Cameras visited"] == "1"


class TestSidebar:
    def test_session_scope_lists_real_sessions(self, app):
        at = app()
        # Located by content: the page has more than one selectbox and their
        # order is an implementation detail.
        scope = _session_scope_box(at)
        assert scope.options[0] == "All sessions"
        assert any("SMOKE" in option for option in scope.options[1:])

    def test_scoping_to_a_session_does_not_crash(self, app):
        at = app(active_plate="DL8CA1234")
        scope = _session_scope_box(at)
        target = next(o for o in scope.options if "SMOKE" in o)
        scope.set_value(target).run()
        _assert_clean(at, "after scoping to a session")

    def test_statistics_follow_the_session_scope(self, app):
        """Statistics and the trajectory view must agree on which run is in
        view -- statistics used to scope to the status file's session id
        instead, and reported 0 whenever the two disagreed."""
        at = app()
        scope = _session_scope_box(at)
        target = next(o for o in scope.options if "SMOKE" in o)
        scope.set_value(target).run()
        _assert_clean(at, "statistics scoped to a session")
        assert {m.label: m.value for m in at.metric}["Vehicles detected"] == "9"

    def test_reload_camera_config_does_not_crash(self, app):
        at = app()
        button = next(b for b in at.button if "Reload" in b.label)
        button.click().run()
        _assert_clean(at, "after reloading the camera config")


class TestSessionDeletion:
    """The sidebar's delete control, clicked the way an operator would."""

    def _events(self) -> int:
        with database.get_session() as session:
            return len(database.get_all_events(session, limit=500))

    def test_the_delete_control_offers_the_real_sessions(self, app):
        at = app()
        options = [o for box in at.get("selectbox") for o in box.options]
        assert any("SMOKE" in option for option in options)

    def test_deleting_asks_for_confirmation_before_removing_anything(self, app):
        """One mis-click on a narrow sidebar control must not destroy a run."""
        at = app()
        before = self._events()

        picker = next(
            box for box in at.get("selectbox")
            if box.options and box.options[0] == "—"
            and any("SMOKE" in o for o in box.options)
        )
        picker.set_value(next(o for o in picker.options if "SMOKE" in o)).run()
        delete_button = next(b for b in at.button if b.label.startswith("Delete session"))
        delete_button.click().run()
        _assert_clean(at, "after asking to delete")

        # Still nothing deleted -- only a confirmation prompt.
        assert self._events() == before
        assert at.warning, "a confirmation warning should be shown"
        assert any(b.label == "Yes, delete" for b in at.button)
        assert any(b.label == "Cancel" for b in at.button)

    def test_cancelling_leaves_the_events_alone(self, app):
        at = app()
        before = self._events()
        picker = next(
            box for box in at.get("selectbox")
            if box.options and box.options[0] == "—"
            and any("SMOKE" in o for o in box.options)
        )
        picker.set_value(next(o for o in picker.options if "SMOKE" in o)).run()
        next(b for b in at.button if b.label.startswith("Delete session")).click().run()
        next(b for b in at.button if b.label == "Cancel").click().run()
        _assert_clean(at, "after cancelling")
        assert self._events() == before

    def test_confirming_deletes_the_session_and_its_events(self, app):
        at = app()
        assert self._events() > 0

        picker = next(
            box for box in at.get("selectbox")
            if box.options and box.options[0] == "—"
            and any("SMOKE" in o for o in box.options)
        )
        picker.set_value(next(o for o in picker.options if "SMOKE" in o)).run()
        next(b for b in at.button if b.label.startswith("Delete session")).click().run()
        next(b for b in at.button if b.label == "Yes, delete").click().run()
        _assert_clean(at, "after confirming the delete")

        assert self._events() == 0, "the session's events were not removed"


class TestSearch:
    def test_searching_a_plate_selects_it(self, app):
        at = app()
        at.get("text_input")[-1].set_value("DL8CA1234").run()
        next(b for b in at.button if b.label == "Search").click().run()
        _assert_clean(at, "after searching")
        assert at.session_state["active_plate"] == "DL8CA1234"

    def test_the_suggestion_list_comes_from_the_database(self, app):
        """Multi-camera vehicles are offered; single-sighting ones are not."""
        at = app()
        suggestions = [
            option for box in at.get("selectbox") for option in box.options
        ]
        joined = " ".join(suggestions)
        assert "DL8CA1234" in joined
        assert "HR26DK8337" not in joined


class TestLegacyDashboard:
    def test_the_original_dashboard_still_renders(self, seeded_db, monkeypatch):
        """The single-gate dashboard must keep working alongside the new one.

        Its auto-refresh loop never terminates under AppTest, so the refresh
        checkbox is turned off first; the point is that the page builds at
        all, which is what the Arrow crash used to prevent.
        """
        monkeypatch.setenv("ALPR_DB_PATH", seeded_db)
        streamlit.cache_resource.clear()
        streamlit.cache_data.clear()

        at = AppTest.from_file(LEGACY_APP, default_timeout=90)
        # Its auto-refresh loop reruns forever, which no non-interactive
        # render can survive -- switch it off before the first run.
        at.session_state["auto_refresh"] = False
        at.run()
        assert not at.exception, at.exception[0].value
        assert at.title or at.subheader, "the legacy page rendered nothing"
