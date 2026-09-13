"""Natural-language vehicle search: parser, profile search, dashboard flow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.database import db as database
from src.database.vehicle_profiles import search_profiles
from src.search.nl_query import parse_query

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 9, 13, 14, 0, tzinfo=IST)
CAMERAS = [("CAM001", "India Gate"), ("CAM002", "Connaught Place")]


def q(text):
    return parse_query(text, NOW, CAMERAS)


class TestParser:
    def test_colour_and_type(self):
        query = q("show all white car image")
        assert query.colors == ["White"] and query.classes == ["car"]

    def test_synonyms(self):
        assert q("grey lorry").colors == ["Gray"] and q("grey lorry").classes == ["truck"]

    def test_time_after_today(self):
        query = q("red trucks after 9am today")
        assert query.start == datetime(2026, 9, 13, 9, 0, tzinfo=IST)

    def test_relative_window(self):
        query = q("blue car last 2 hours")
        assert query.start == NOW - timedelta(hours=2) and query.end == NOW

    def test_between_shares_the_meridiem(self):
        query = q("bikes between 9 and 11am")
        assert (query.start.hour, query.end.hour) == (9, 11)

    def test_yesterday_after_is_bounded_to_that_day(self):
        query = q("white car yesterday after 9am")
        assert query.start.day == 12 and query.end.day == 12

    def test_camera_by_name(self):
        assert q("cars at India Gate").camera_ids == ["CAM001"]

    def test_full_plate_and_explicit_fragment(self):
        assert q("DL8CA1234").plate_fragment == "DL8CA1234"
        assert q("plate KA02").plate_fragment == "KA02"

    def test_a_date_is_not_mistaken_for_a_plate(self):
        """Regression: "on 2026-09-12" was read as plate ON2026."""
        assert q("cars on 2026-09-12").plate_fragment is None

    def test_nothing_recognisable_is_empty(self):
        assert q("hello there").is_empty


@pytest.fixture()
def db_with_vehicles(tmp_path):
    database.init_db(str(tmp_path / "search.db"))

    def add(plate, colour, cls, camera, name, when):
        with database.get_session() as session:
            database.insert_event(session, {
                "plate_number": plate, "vehicle_type": "Private", "plate_color": "White",
                "series_type": "normal", "direction": "IN", "image_path": "",
                "camera_id": camera, "camera_name": name,
                "timestamp": when.astimezone(timezone.utc).isoformat(),
                "processing_session": "S1", "confidence": 0.9,
                "vehicle_class": cls, "vehicle_color": colour,
            })

    add("DL1WHITE01", "White", "car", "CAM001", "India Gate", NOW - timedelta(hours=5))    # 09:00
    add("DL2WHITE02", "White", "car", "CAM002", "Connaught Place", NOW - timedelta(days=1))
    add("DL3RED0003", "Red", "truck", "CAM001", "India Gate", NOW - timedelta(hours=1))
    yield
    database._engine = database._SessionFactory = database._db_type = None


def plates(text):
    with database.get_session() as session:
        return sorted(p["plate_number"] for p in search_profiles(session, q(text)))


class TestSearch:
    def test_colour_and_type_regardless_of_time(self, db_with_vehicles):
        assert plates("show all white car image") == ["DL1WHITE01", "DL2WHITE02"]

    def test_time_window(self, db_with_vehicles):
        assert plates("white cars today") == ["DL1WHITE01"]
        assert plates("white cars yesterday") == ["DL2WHITE02"]

    def test_camera_and_time_must_hold_for_the_same_visit(self, db_with_vehicles):
        assert plates("white car at India Gate") == ["DL1WHITE01"]
        assert plates("white car at Connaught Place today") == []

    def test_other_attributes(self, db_with_vehicles):
        assert plates("red truck last 2 hours") == ["DL3RED0003"]
        assert plates("plate RED") == ["DL3RED0003"]


class TestDashboard:
    def test_search_shows_vehicles_and_opens_a_trajectory(self, seeded_db_for_search, monkeypatch):
        from streamlit.testing.v1 import AppTest
        import streamlit

        from src.cameras import preview_server

        class NoServer:
            running, port, error = False, 0, "disabled in tests"

        monkeypatch.setattr(preview_server, "get_preview_server", lambda *a, **k: NoServer())
        monkeypatch.setenv("ALPR_DB_PATH", seeded_db_for_search)
        streamlit.cache_resource.clear()

        at = AppTest.from_file("src/dashboard/trajectory_app.py", default_timeout=90)
        at.run()
        next(t for t in at.text_input if t.key == "vehicle_search_input").set_value("white cars")
        next(b for b in at.button if b.key == "vehicle_search_go").click().run()
        assert not at.exception, at.exception[0].value
        assert any("Understood: colour White, type car" in c.value for c in at.caption)

        next(b for b in at.button if b.key == "nl_pick_DL1WHITE01").click().run()
        assert not at.exception, at.exception[0].value
        assert at.session_state["active_plate"] == "DL1WHITE01"

    @pytest.fixture()
    def seeded_db_for_search(self, tmp_path):
        path = tmp_path / "dash.db"
        database.init_db(str(path))
        with database.get_session() as session:
            database.insert_event(session, {
                "plate_number": "DL1WHITE01", "vehicle_type": "Private", "plate_color": "White",
                "series_type": "normal", "direction": "IN", "image_path": "",
                "camera_id": "CAM001", "camera_name": "India Gate",
                "timestamp": datetime.now(timezone.utc).isoformat(), "processing_session": "S1",
                "confidence": 0.9, "vehicle_class": "car", "vehicle_color": "White",
            })
        yield str(path)
        database._engine = database._SessionFactory = database._db_type = None
