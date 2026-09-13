"""
Integration tests for the Vehicle Profile layer.

Covers the whole life of a profile against a real SQLite database:
creation from detections, attribute voting, image selection, camera-visit
history, out-of-order arrival, session deletion, backfill of an existing
database, the pipeline's store step (real thumbnails on disk), the APIs, and
backward compatibility with callers that know nothing about profiles.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.database import db as database
from src.database.models import VehicleProfile
from src.database.vehicle_profiles import (
    get_profile,
    list_profiles,
    rebuild_all_profiles,
    rebuild_profile,
)

SITES = {
    "CAM001": ("India Gate", 28.6129, 77.2295),
    "CAM002": ("Connaught Place", 28.6315, 77.2167),
    "CAM003": ("Karol Bagh", 28.6519, 77.1909),
    "CAM004": ("Kashmere Gate", 28.6675, 77.2273),
}


@pytest.fixture(autouse=True)
def _reset_engine():
    yield
    database._engine = None
    database._SessionFactory = None
    database._db_type = None


@pytest.fixture()
def fresh(tmp_path):
    database.init_db(str(tmp_path / "profiles.db"))
    return tmp_path


def detection(plate="DL8CA1234", camera="CAM001", minute=0, second=0, **overrides):
    name, lat, lon = SITES[camera]
    event = {
        "plate_number": plate, "vehicle_type": "Private", "plate_color": "White",
        "series_type": "normal", "direction": "IN", "image_path": f"plate_{camera}.jpg",
        "camera_id": camera, "camera_name": name, "latitude": lat, "longitude": lon,
        "timestamp": f"2026-09-13T09:{minute:02d}:{second:02d}+00:00",
        "processing_session": "S1", "confidence": 0.9,
        "vehicle_class": "car", "vehicle_color": "Blue",
        "vehicle_thumbnail_path": f"vt_{camera}_{minute}.jpg",
        "plate_thumbnail_path": f"pt_{camera}_{minute}.jpg",
    }
    event.update(overrides)
    return event


def store(*events):
    with database.get_session() as session:
        for event in events:
            assert database.insert_event(session, event) is not None


def profile(plate="DL8CA1234"):
    with database.get_session() as session:
        return get_profile(session, plate)


def journey():
    return [
        detection(camera="CAM001", minute=0, second=0, confidence=0.88),
        detection(camera="CAM001", minute=0, second=4, confidence=0.97),
        detection(camera="CAM002", minute=12, confidence=0.93, vehicle_color="Silver"),
        detection(camera="CAM003", minute=25, confidence=0.95),
        detection(camera="CAM004", minute=48, confidence=0.91,
                  vehicle_class="truck", vehicle_color="Unknown"),
    ]


class TestProfileContents:
    def test_every_required_field_is_populated(self, fresh):
        store(*journey())
        p = profile()
        required = {
            "plate_number", "vehicle_thumbnail_path", "plate_thumbnail_path",
            "vehicle_class", "vehicle_color", "camera_id", "camera_name",
            "latitude", "longitude", "ocr_confidence", "first_seen", "last_seen",
            "processing_session", "total_camera_visits", "trajectory_history",
        }
        missing = {field for field in required if p.get(field) in (None, "", [])}
        assert not missing, f"empty profile fields: {missing}"

    def test_times_span_first_to_last_detection(self, fresh):
        store(*journey())
        p = profile()
        assert p["first_seen"].startswith("2026-09-13T09:00:00")
        assert p["last_seen"].startswith("2026-09-13T09:48:00")

    def test_camera_visit_history_is_in_travel_order(self, fresh):
        store(*journey())
        history = profile()["trajectory_history"]
        assert [v["camera_name"] for v in history] == [
            "India Gate", "Connaught Place", "Karol Bagh", "Kashmere Gate"
        ]

    def test_repeat_reads_at_one_camera_are_one_visit(self, fresh):
        store(*journey())
        p = profile()
        assert p["total_detections"] == 5
        assert p["total_camera_visits"] == 4
        assert p["unique_cameras"] == 4
        assert p["trajectory_history"][0]["detections"] == 2

    def test_a_genuine_return_to_a_camera_is_a_new_visit(self, fresh):
        store(detection(camera="CAM001", minute=0),
              detection(camera="CAM002", minute=10),
              detection(camera="CAM001", minute=30))
        p = profile()
        assert [v["camera_id"] for v in p["trajectory_history"]] == ["CAM001", "CAM002", "CAM001"]
        assert p["total_camera_visits"] == 3
        assert p["unique_cameras"] == 2

    def test_the_same_camera_after_a_long_gap_is_a_new_visit(self, fresh):
        store(detection(camera="CAM001", minute=0), detection(camera="CAM001", minute=30))
        assert profile()["total_camera_visits"] == 2

    def test_the_same_camera_in_another_session_is_a_new_visit(self, fresh):
        store(detection(camera="CAM001", minute=0, processing_session="RUN1"),
              detection(camera="CAM001", minute=1, processing_session="RUN2"))
        assert profile()["total_camera_visits"] == 2

    def test_location_and_session_are_from_the_latest_detection(self, fresh):
        store(*journey())
        p = profile()
        assert p["camera_id"] == "CAM004"
        assert p["latitude"] == pytest.approx(28.6675)
        assert p["ocr_confidence"] == pytest.approx(0.91)


class TestAttributeVoting:
    def test_class_and_colour_are_the_majority_not_the_latest(self, fresh):
        """One misread camera (truck, silver) must not flip the profile."""
        store(*journey())
        p = profile()
        assert p["vehicle_class"] == "car"
        assert p["vehicle_color"] == "Blue"

    def test_unknown_colour_never_wins(self, fresh):
        store(detection(minute=0, vehicle_color="Unknown"),
              detection(camera="CAM002", minute=10, vehicle_color="Unknown"),
              detection(camera="CAM003", minute=20, vehicle_color="Red"))
        assert profile()["vehicle_color"] == "Red"

    def test_registration_category_is_kept_separately(self, fresh):
        store(*journey())
        p = profile()
        assert p["vehicle_type"] == "Private"
        assert p["plate_color"] == "White"


class TestImages:
    def test_images_come_from_the_highest_confidence_detection(self, fresh):
        store(*journey())
        p = profile()
        assert p["best_confidence"] == pytest.approx(0.97)
        assert p["vehicle_thumbnail_path"] == "vt_CAM001_0.jpg"

    def test_images_are_filled_even_without_a_confidence(self, fresh):
        store(detection(confidence=None))
        assert profile()["vehicle_thumbnail_path"]


class TestConsistency:
    def test_incremental_and_rebuilt_profiles_are_identical(self, fresh):
        store(*journey())
        before = {k: v for k, v in profile().items() if k != "updated_at"}
        with database.get_session() as session:
            rebuild_profile(session, "DL8CA1234")
        after = {k: v for k, v in profile().items() if k != "updated_at"}
        assert before == after

    def test_an_out_of_order_detection_is_placed_in_time_order(self, fresh):
        store(detection(camera="CAM002", minute=20), detection(camera="CAM003", minute=30))
        store(detection(camera="CAM001", minute=5))           # arrives late
        p = profile()
        assert [v["camera_id"] for v in p["trajectory_history"]] == ["CAM001", "CAM002", "CAM003"]
        assert p["first_seen"].startswith("2026-09-13T09:05")

    def test_plates_are_matched_case_insensitively(self, fresh):
        """Regression: a lowercase plate from /entry used to make a rebuild
        find no detections and delete the profile."""
        store(detection(plate="dl8ca1234"))
        with database.get_session() as session:
            rebuild_profile(session, "DL8CA1234")
        assert profile("DL8CA1234") is not None
        assert profile("dl8ca1234")["plate_number"] == "DL8CA1234"

    def test_one_profile_per_plate(self, fresh):
        store(*journey(), detection(plate="HR26DK8337"))
        with database.get_session() as session:
            assert session.query(VehicleProfile).count() == 2
            assert [p["plate_number"] for p in list_profiles(session)][0] == "DL8CA1234"


class TestIsolation:
    def test_a_profile_failure_never_loses_the_detection(self, fresh, monkeypatch):
        from src.database import vehicle_profiles

        def explode(*args, **kwargs):
            raise RuntimeError("profile store is broken")

        monkeypatch.setattr(vehicle_profiles, "_apply_event", explode)
        store(detection())
        with database.get_session() as session:
            assert len(database.get_all_events(session)) == 1   # detection kept
            assert get_profile(session, "DL8CA1234") is None     # profile skipped


class TestSessionDeletion:
    def test_deleting_a_session_rebuilds_from_what_remains(self, fresh):
        store(detection(camera="CAM001", minute=0, processing_session="OLD"),
              detection(camera="CAM002", minute=10, processing_session="NEW"),
              detection(camera="CAM003", minute=20, processing_session="NEW"))
        with database.get_session() as session:
            result = database.delete_session(session, "NEW")
        p = profile()
        assert p["total_detections"] == 1
        assert [v["camera_id"] for v in p["trajectory_history"]] == ["CAM001"]
        assert p["processing_session"] == "OLD"
        # thumbnails of the deleted detections are reported for removal
        assert "vt_CAM002_10.jpg" in result["image_paths"]

    def test_deleting_the_only_session_removes_the_profile(self, fresh):
        store(*journey())
        with database.get_session() as session:
            database.delete_session(session, "S1")
        assert profile() is None

    def test_clear_all_data_removes_profiles(self, fresh):
        from src.utils.data_reset import clear_all_data

        store(*journey())
        clear_all_data({"database": {"path": str(fresh / "profiles.db")}, "api": {}})
        with database.get_session() as session:
            assert session.query(VehicleProfile).count() == 0


class TestBackfill:
    LEGACY_V2 = """
    CREATE TABLE vehicle_events (
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, plate_number VARCHAR NOT NULL,
        vehicle_type VARCHAR NOT NULL, plate_color VARCHAR NOT NULL,
        series_type VARCHAR NOT NULL, timestamp VARCHAR NOT NULL,
        direction VARCHAR NOT NULL, image_path VARCHAR NOT NULL,
        camera_id VARCHAR, camera_name VARCHAR, latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION
    );
    INSERT INTO vehicle_events (plate_number, vehicle_type, plate_color, series_type,
        timestamp, direction, image_path, camera_id, camera_name)
    VALUES ('KA02MH7256','Private','White','normal','2026-09-01T10:00:00+00:00','IN','a.jpg','GATE-01','Main Gate'),
           ('KA02MH7256','Private','White','normal','2026-09-02T10:00:00+00:00','OUT','b.jpg','GATE-01','Main Gate'),
           ('DL7CD5017','Commercial','Yellow','normal','2026-09-03T10:00:00+00:00','IN','c.jpg',NULL,NULL);
    """

    def test_an_existing_database_gains_profiles_for_its_history(self, tmp_path):
        path = tmp_path / "legacy.db"
        connection = sqlite3.connect(str(path))
        connection.executescript(self.LEGACY_V2)
        connection.commit(); connection.close()

        database.init_db(str(path))

        with database.get_session() as session:
            assert session.query(VehicleProfile).count() == 2
            main_gate = get_profile(session, "KA02MH7256")
        assert main_gate["total_detections"] == 2
        assert main_gate["vehicle_type"] == "Private"
        assert main_gate["vehicle_class"] is None        # never recorded for old rows
        # old rows keep reading back fine
        with database.get_session() as session:
            assert len(database.get_all_events(session)) == 3

    def test_backfill_runs_once_not_on_every_start(self, tmp_path, monkeypatch):
        database.init_db(str(tmp_path / "once.db"))
        database._engine = None

        from src.database import vehicle_profiles

        calls = []
        monkeypatch.setattr(vehicle_profiles, "rebuild_all_profiles",
                            lambda session, *a, **k: calls.append(1) or 0)
        database.init_db(str(tmp_path / "once.db"))
        assert calls == []


class TestPipelineStoreStep:
    """_store_event produces the thumbnails, colour and class for real."""

    def test_a_stored_vehicle_gets_a_complete_profile(self, fresh):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from scripts.run_pipeline import _store_event
        from src.classification.vehicle_color import VehicleColorDetector
        from src.utils.logger import get_logger

        class Classifier:
            def classify(self, value): return "White"

        class VehicleTypes:
            def classify(self, value): return "Private"

        class Dedup:
            def is_duplicate(self, *a): return False
            def record(self, *a): pass

        class Direction:
            def update(self, *a): return "IN"

        vehicle = np.zeros((480, 640, 3), np.uint8); vehicle[:] = (90, 110, 95)
        cv2.rectangle(vehicle, (40, 80), (600, 430), (170, 70, 20), -1)        # blue car
        plate = np.full((60, 220, 3), 230, np.uint8)
        thumbs = fresh / "thumbs"

        stored = _store_event(
            plate_number="DL8CA1234", series_type="normal", plate_crop=plate,
            color_classifier=Classifier(), vehicle_classifier=VehicleTypes(),
            dup_filter=Dedup(), direction_detector=Direction(), track_id=7,
            centroid=(10.0, 10.0), image_save_path=str(fresh / "plates"),
            camera_meta={"camera_id": "CAM001", "camera_name": "India Gate",
                         "latitude": 28.6129, "longitude": 77.2295},
            database=database, log=get_logger("test"),
            vehicle_crop=vehicle, confidence=0.94, ocr_text="DL8CA1234",
            session_context={"processing_session": "S1", "trajectory_order": 1,
                             "video_source": "clip.mp4"},
            vehicle_image_save_path=str(fresh / "vehicles"),
            vehicle_class="car", vehicle_color_detector=VehicleColorDetector(),
            plate_thumbnail_source=plate, thumbnail_dir=str(thumbs),
        )
        assert stored

        p = profile()
        assert p["vehicle_class"] == "car"
        assert p["vehicle_color"] == "Blue"
        vehicle_thumb = cv2.imread(p["vehicle_thumbnail_path"])
        plate_thumb = cv2.imread(p["plate_thumbnail_path"])
        assert vehicle_thumb is not None and max(vehicle_thumb.shape[:2]) <= 320
        assert plate_thumb is not None and max(plate_thumb.shape[:2]) <= 240
        assert Path(p["vehicle_thumbnail_path"]).parent == thumbs / "vehicles"


class TestThumbnails:
    def test_large_images_are_shrunk_to_max_side(self, tmp_path):
        path = database.save_thumbnail(np.zeros((1080, 1920, 3), np.uint8), "P", str(tmp_path), max_side=320)
        assert cv2.imread(path).shape[:2] == (180, 320)

    def test_a_tiny_plate_is_enlarged_to_a_readable_width(self, tmp_path):
        """Real plate crops can be 38x11 px at source -- a speck on a card."""
        path = database.save_thumbnail(np.zeros((11, 38, 3), np.uint8), "P", str(tmp_path),
                                       max_side=240, min_width=176)
        height, width = cv2.imread(path).shape[:2]
        assert width == 176 and 45 <= height <= 56          # aspect ratio kept

    def test_without_min_width_small_images_are_left_alone(self, tmp_path):
        path = database.save_thumbnail(np.zeros((11, 38, 3), np.uint8), "P", str(tmp_path))
        assert cv2.imread(path).shape[:2] == (11, 38)

    def test_an_unusable_image_returns_an_empty_path_not_an_error(self, tmp_path):
        assert database.save_thumbnail(None, "P", str(tmp_path)) == ""
        assert database.save_thumbnail(np.zeros((0, 0, 3), np.uint8), "P", str(tmp_path)) == ""


class TestAPIs:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path}/api.db")
        from src.api.server import app

        with TestClient(app) as client:
            store(*journey())
            yield client

    def test_event_listings_include_profile_fields(self, client):
        event = client.get("/logs").json()[0]
        for field in ("vehicle_class", "vehicle_color",
                      "vehicle_thumbnail_path", "plate_thumbnail_path"):
            assert field in event

    def test_vehicle_history_includes_the_profile(self, client):
        body = client.get("/vehicles/DL8CA1234").json()
        assert body["total_events"] == 5                       # unchanged key
        assert body["profile"]["total_camera_visits"] == 4

    def test_trajectory_includes_the_profile(self, client):
        body = client.get("/trajectory-api/trajectory/DL8CA1234").json()
        assert body["path_labels"][0] == "India Gate"          # reconstruction unchanged
        assert body["profile"]["vehicle_color"] == "Blue"

    def test_profile_endpoints(self, client):
        assert client.get("/trajectory-api/vehicles").json()[0]["plate_number"] == "DL8CA1234"
        one = client.get("/trajectory-api/vehicles/dl8ca1234").json()
        assert len(one["trajectory_history"]) == 4
        assert client.get("/trajectory-api/vehicles/NOPE").status_code == 404

    def test_a_legacy_entry_without_profile_fields_still_works(self, client):
        response = client.post("/entry", json={
            "plate_number": "UP32AB1234", "vehicle_type": "Car",
            "plate_color": "White", "series_type": "normal", "direction": "IN",
        })
        assert response.status_code == 201
        assert response.json()["vehicle_color"] is None
        assert client.get("/trajectory-api/vehicles/UP32AB1234").status_code == 200

    def test_thumbnails_are_served(self, client):
        from src.api import server

        target = server._thumbnail_dir / "vehicles" / "api_test_thumb.jpg"
        cv2.imwrite(str(target), np.zeros((10, 10, 3), np.uint8))
        try:
            response = client.get("/thumbnails/vehicles/api_test_thumb.jpg")
            assert response.status_code == 200
            assert response.headers["content-type"] == "image/jpeg"
        finally:
            target.unlink(missing_ok=True)
