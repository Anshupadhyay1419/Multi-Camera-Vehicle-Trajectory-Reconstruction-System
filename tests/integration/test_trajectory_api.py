"""
Integration tests for the multi-camera database columns and API routes.

Two risks are covered here that the unit tests cannot reach:

1. Schema migration. The trajectory columns were added to a table that is
   already deployed and already holding rows. create_all() never alters an
   existing table, so db._migrate_schema() has to -- from EITHER prior
   version (the original eight columns, or the four-camera-column one).

2. Route isolation. The new /trajectory-api routes are mounted alongside the
   existing single-gate ones and a catch-all static mount at "/". They must
   add without displacing.
"""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect

from src.database import db as database

TRAJECTORY_COLUMNS = [
    "vehicle_image_path", "trajectory_order", "processing_session",
    "video_source", "confidence", "ocr_text",
]
CAMERA_COLUMNS = ["camera_id", "camera_name", "latitude", "longitude"]

# vehicle_events as it shipped ORIGINALLY, before any camera attribution.
_V1_SCHEMA = """
CREATE TABLE vehicle_events (
    id           INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    plate_number VARCHAR NOT NULL,
    vehicle_type VARCHAR NOT NULL,
    plate_color  VARCHAR NOT NULL,
    series_type  VARCHAR NOT NULL,
    timestamp    VARCHAR NOT NULL,
    direction    VARCHAR NOT NULL,
    image_path   VARCHAR NOT NULL
);
"""

# ... and as it was after camera attribution but before trajectories.
_V2_SCHEMA = _V1_SCHEMA.replace(
    "    image_path   VARCHAR NOT NULL\n);",
    """    image_path   VARCHAR NOT NULL,
    camera_id    VARCHAR,
    camera_name  VARCHAR,
    latitude     DOUBLE PRECISION,
    longitude    DOUBLE PRECISION
);""",
)

_LEGACY_ROW = (
    "INSERT INTO vehicle_events (plate_number, vehicle_type, plate_color, "
    "series_type, timestamp, direction, image_path) VALUES "
    "('DL1CA0001', 'Car', 'White', 'normal', '2026-01-01T08:00:00+00:00', 'IN', 'old.jpg')"
)


def _legacy_db(tmp_path, schema: str, name: str) -> str:
    path = tmp_path / name
    connection = sqlite3.connect(str(path))
    connection.executescript(schema)
    connection.execute(_LEGACY_ROW)
    connection.commit()
    connection.close()
    return str(path)


@pytest.fixture(autouse=True)
def _isolated_db():
    """Reset the module-global engine between tests."""
    yield
    database._engine = None
    database._SessionFactory = None
    database._db_type = None


def _event(**overrides) -> dict:
    base = {
        "plate_number": "DL8CA1234",
        "vehicle_type": "Car",
        "plate_color": "White",
        "series_type": "normal",
        "direction": "IN",
        "image_path": "plate.jpg",
        "timestamp": "2026-09-12T09:00:00+00:00",
        "camera_id": "CAM001",
        "camera_name": "India Gate",
        "latitude": 28.6129,
        "longitude": 77.2295,
        "vehicle_image_path": "vehicle.jpg",
        "trajectory_order": 1,
        "processing_session": "SESS1",
        "video_source": "data/uploads/CAM001.mp4",
        "confidence": 0.93,
        "ocr_text": "DL8CA1234",
    }
    base.update(overrides)
    return base


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path}/api.db")
    from src.api import trajectory_routes
    from src.api.server import app
    from src.cameras.manager import CameraManager
    from src.cameras.registry import load_camera_registry
    from src.cameras.status_store import StatusStore

    # Give the routes a manager whose uploads and status land in tmp_path.
    # Without this the module-level manager uses the real registry, and
    # an upload test would write junk .mp4 files into the project's own
    # data/uploads -- leaving cameras looking "ready" with 256 bytes of
    # test data in them.
    registry = load_camera_registry("config/camera_config.yaml")
    registry._settings["processing"] = {
        "upload_dir": str(tmp_path / "uploads"),
        "status_file": str(tmp_path / "status.json"),
    }
    monkeypatch.setattr(
        trajectory_routes, "_manager",
        CameraManager(
            {"video": {}}, registry,
            runner=lambda **kwargs: {},
            status_store=StatusStore(str(tmp_path / "status.json")),
        ),
    )
    monkeypatch.setattr(trajectory_routes, "_registry", registry)

    with TestClient(app) as client:
        with database.get_session() as session:
            sites = [
                ("CAM001", "India Gate", 28.6129, 77.2295, 1, 0),
                ("CAM002", "Connaught Place", 28.6315, 77.2167, 2, 12),
                ("CAM003", "Karol Bagh", 28.6519, 77.1909, 3, 25),
                ("CAM004", "Kashmere Gate", 28.6675, 77.2273, 4, 48),
            ]
            for camera_id, name, lat, lon, order, minute in sites:
                database.insert_event(session, _event(
                    camera_id=camera_id, camera_name=name,
                    latitude=lat, longitude=lon, trajectory_order=order,
                    timestamp=f"2026-09-12T09:{minute:02d}:00+00:00",
                ))
        yield client


class TestSchemaMigration:
    @pytest.mark.parametrize(
        "schema,name",
        [(_V1_SCHEMA, "v1.db"), (_V2_SCHEMA, "v2.db")],
        ids=["from-original-8-columns", "from-camera-columns"],
    )
    def test_an_existing_database_is_migrated_in_place(self, tmp_path, schema, name):
        path = _legacy_db(tmp_path, schema, name)
        database.init_db(path)

        columns = {c["name"] for c in inspect(database.get_engine()).get_columns("vehicle_events")}
        for column in CAMERA_COLUMNS + TRAJECTORY_COLUMNS:
            assert column in columns, f"{column} was not added"

    def test_existing_rows_survive_and_read_back_as_null(self, tmp_path):
        path = _legacy_db(tmp_path, _V1_SCHEMA, "keep.db")
        database.init_db(path)

        with database.get_session() as session:
            rows = database.get_all_events(session)
        assert len(rows) == 1
        assert rows[0]["plate_number"] == "DL1CA0001"
        # New columns are NULL, not zero or empty -- the row predates them.
        for column in TRAJECTORY_COLUMNS:
            assert rows[0][column] is None

    def test_migration_is_idempotent(self, tmp_path):
        path = _legacy_db(tmp_path, _V1_SCHEMA, "twice.db")
        database.init_db(path)
        database.init_db(path)
        database.init_db(path)
        with database.get_session() as session:
            assert len(database.get_all_events(session)) == 1

    def test_the_supporting_indexes_exist_after_migration(self, tmp_path):
        path = _legacy_db(tmp_path, _V1_SCHEMA, "idx.db")
        database.init_db(path)
        names = {i["name"] for i in inspect(database.get_engine()).get_indexes("vehicle_events")}
        assert "idx_camera_id" in names
        assert "idx_processing_session" in names


class TestInsertAndRead:
    @pytest.fixture()
    def fresh(self, tmp_path):
        database.init_db(str(tmp_path / "fresh.db"))
        return str(tmp_path / "fresh.db")

    def test_a_full_multi_camera_event_round_trips(self, fresh):
        with database.get_session() as session:
            database.insert_event(session, _event())
        with database.get_session() as session:
            row = database.get_all_events(session)[0]
        assert row["processing_session"] == "SESS1"
        assert row["trajectory_order"] == 1
        assert row["confidence"] == pytest.approx(0.93)
        assert row["vehicle_image_path"] == "vehicle.jpg"
        assert row["video_source"].endswith("CAM001.mp4")

    def test_the_single_gate_pipeline_can_still_insert_without_them(self, fresh):
        """The old caller must keep working untouched -- this is the
        no-regression guarantee for the existing deployment."""
        with database.get_session() as session:
            event = database.insert_event(session, {
                "plate_number": "DL9XY0001", "vehicle_type": "Car",
                "plate_color": "White", "series_type": "normal",
                "direction": "OUT", "image_path": "p.jpg",
            })
            assert event is not None
        with database.get_session() as session:
            row = database.get_all_events(session)[0]
        for column in TRAJECTORY_COLUMNS:
            assert row[column] is None

    def test_a_bad_trajectory_order_degrades_instead_of_losing_the_event(self, fresh):
        """A hand-edited config should never cost the gate a vehicle."""
        with database.get_session() as session:
            database.insert_event(session, _event(trajectory_order="not-a-number"))
        with database.get_session() as session:
            assert database.get_all_events(session)[0]["trajectory_order"] is None

    def test_a_string_order_is_coerced(self, fresh):
        with database.get_session() as session:
            database.insert_event(session, _event(trajectory_order="3"))
        with database.get_session() as session:
            assert database.get_all_events(session)[0]["trajectory_order"] == 3

    def test_detections_come_back_oldest_first(self, fresh):
        """A trajectory is read forwards, unlike the newest-first log view."""
        with database.get_session() as session:
            for order, minute in enumerate([48, 0, 25, 12], start=1):
                database.insert_event(session, _event(
                    camera_id=f"CAM00{order}",
                    timestamp=f"2026-09-12T09:{minute:02d}:00+00:00",
                ))
        with database.get_session() as session:
            rows = database.get_plate_detections(session, "DL8CA1234")
        assert [r["timestamp"] for r in rows] == sorted(r["timestamp"] for r in rows)

    def test_plate_lookup_is_case_insensitive(self, fresh):
        with database.get_session() as session:
            database.insert_event(session, _event())
        with database.get_session() as session:
            assert len(database.get_plate_detections(session, "dl8ca1234")) == 1
            assert len(database.get_plate_detections(session, "  DL8CA1234 ")) == 1

    def test_session_filter_scopes_a_query_to_one_run(self, fresh):
        """Repeated demos of the same video must not blend together."""
        with database.get_session() as session:
            database.insert_event(session, _event(processing_session="RUN1"))
            database.insert_event(session, _event(processing_session="RUN2"))
        with database.get_session() as session:
            assert len(database.get_plate_detections(session, "DL8CA1234")) == 2
            assert len(database.get_plate_detections(
                session, "DL8CA1234", processing_session="RUN1")) == 1

    def test_multi_camera_plates_excludes_single_sightings(self, fresh):
        """A plate seen at one camera is a sighting, not a path."""
        with database.get_session() as session:
            for order in (1, 2, 3):
                database.insert_event(session, _event(camera_id=f"CAM00{order}"))
            database.insert_event(session, _event(plate_number="HR26DK8337"))
        with database.get_session() as session:
            plates = database.get_multi_camera_plates(session, min_cameras=2)
        assert [p["plate_number"] for p in plates] == ["DL8CA1234"]
        assert plates[0]["camera_count"] == 3

    def test_unique_plate_count_is_not_the_sum_of_per_camera_counts(self, fresh):
        """One vehicle at four cameras is 1 unique plate, not 4."""
        with database.get_session() as session:
            for order in range(1, 5):
                database.insert_event(session, _event(camera_id=f"CAM00{order}"))
        with database.get_session() as session:
            stats = database.get_session_stats(session, "SESS1")
        assert stats["total_detections"] == 4
        assert stats["unique_plates"] == 1
        assert stats["cameras_reporting"] == 4
        assert sum(c["unique_plates"] for c in stats["per_camera"]) == 4  # the wrong number


class TestTrajectoryRoutes:
    def test_camera_registry_is_served_in_processing_order(self, client):
        response = client.get("/trajectory-api/cameras")
        assert response.status_code == 200
        assert [c["camera_id"] for c in response.json()] == [
            "CAM001", "CAM002", "CAM003", "CAM004"
        ]

    def test_trajectory_reconstructs_the_full_path(self, client):
        body = client.get("/trajectory-api/trajectory/DL8CA1234").json()
        assert body["path_labels"] == [
            "India Gate", "Connaught Place", "Karol Bagh", "Kashmere Gate"
        ]
        assert body["cameras_visited"] == 4
        assert body["total_distance_km"] > 0
        assert len(body["legs"]) == 3

    def test_trajectory_is_case_insensitive(self, client):
        assert client.get("/trajectory-api/trajectory/dl8ca1234").status_code == 200

    def test_unknown_plate_is_a_404(self, client):
        assert client.get("/trajectory-api/trajectory/ZZ00ZZ0000").status_code == 404

    def test_map_endpoint_returns_geojson_with_correct_axis_order(self, client):
        body = client.get("/trajectory-api/map/DL8CA1234").json()
        assert body["type"] == "FeatureCollection"
        line = body["features"][0]
        assert line["geometry"]["type"] == "LineString"
        longitude, latitude = line["geometry"]["coordinates"][0]
        assert 76 < longitude < 78 and 28 < latitude < 29

    def test_map_html_renders_a_leaflet_page(self, client):
        response = client.get("/trajectory-api/map/DL8CA1234/html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "leaflet" in response.text.lower()
        assert "World_Imagery" in response.text        # satellite layer
        assert "DL8CA1234" in response.text

    def test_map_of_an_unknown_plate_is_still_valid_geojson(self, client):
        """Clients must never have to special-case the response shape."""
        body = client.get("/trajectory-api/map/ZZ00ZZ0000").json()
        assert body["type"] == "FeatureCollection"
        assert body["features"] == []

    def test_statistics_aggregate_the_session(self, client):
        body = client.get("/trajectory-api/statistics?session=SESS1").json()
        assert body["total_detections"] == 4
        assert body["unique_plates"] == 1
        assert body["cameras_reporting"] == 4

    def test_plates_endpoint_lists_trackable_vehicles(self, client):
        body = client.get("/trajectory-api/plates?min_cameras=2").json()
        assert body[0]["plate_number"] == "DL8CA1234"
        assert body[0]["camera_count"] == 4

    def test_sessions_endpoint_lists_runs(self, client):
        body = client.get("/trajectory-api/sessions").json()
        assert body[0]["processing_session"] == "SESS1"
        assert body[0]["detections"] == 4

    def test_processing_status_is_available_before_any_run(self, client):
        assert client.get("/trajectory-api/processing/status").json()["state"] == "idle"

    def test_bulk_upload_broadcasts_one_video_to_every_camera(self, client, tmp_path):
        """The dashboard's single uploader: one file in, every camera sourced,
        each with its own copy."""
        response = client.post(
            "/trajectory-api/cameras/upload",
            files=[("files", ("demo.mp4", b"\x00" * 1024, "video/mp4"))],
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["files_received"] == 1
        assert body["cameras_assigned"] == 4
        assert set(body["assigned"]) == {"CAM001", "CAM002", "CAM003", "CAM004"}
        # No two cameras share a path, even though the bytes are identical.
        assert len(set(body["assigned"].values())) == 4

    def test_bulk_upload_assigns_one_file_per_camera_in_order(self, client):
        response = client.post(
            "/trajectory-api/cameras/upload",
            files=[
                ("files", (f"clip{i}.mp4", bytes([i]) * 256, "video/mp4"))
                for i in range(4)
            ],
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["files_received"] == 4
        assert list(body["assigned"]) == ["CAM001", "CAM002", "CAM003", "CAM004"]

    def test_bulk_upload_with_too_many_files_is_a_400(self, client):
        response = client.post(
            "/trajectory-api/cameras/upload",
            files=[
                ("files", (f"clip{i}.mp4", b"x" * 64, "video/mp4"))
                for i in range(5)
            ],
        )
        assert response.status_code == 400
        assert "only 4 camera" in response.json()["detail"]

    def test_upload_to_an_unknown_camera_is_a_404(self, client):
        response = client.post(
            "/trajectory-api/cameras/NOPE/upload",
            files={"file": ("clip.mp4", b"data", "video/mp4")},
        )
        assert response.status_code == 404


class TestSessionDeletion:
    """Deleting one run, its events, and its images."""

    @pytest.fixture()
    def seeded(self, tmp_path):
        database.init_db(str(tmp_path / "del.db"))
        images = []
        with database.get_session() as session:
            for index, run in enumerate(("RUN_A", "RUN_A", "RUN_B")):
                plate_image = tmp_path / f"plate_{index}.jpg"
                vehicle_image = tmp_path / f"vehicle_{index}.jpg"
                plate_image.write_bytes(b"x")
                vehicle_image.write_bytes(b"x")
                images.append((run, plate_image, vehicle_image))
                database.insert_event(session, _event(
                    processing_session=run,
                    image_path=str(plate_image),
                    vehicle_image_path=str(vehicle_image),
                ))
        return images

    def test_deleting_a_session_removes_only_its_events(self, seeded):
        with database.get_session() as session:
            result = database.delete_session(session, "RUN_A")
        assert result["events"] == 2

        with database.get_session() as session:
            remaining = database.get_all_events(session)
        assert len(remaining) == 1
        assert remaining[0]["processing_session"] == "RUN_B"

    def test_the_images_of_the_deleted_events_are_reported(self, seeded):
        with database.get_session() as session:
            result = database.delete_session(session, "RUN_A")
        # Both the plate crop and the vehicle crop of each deleted event.
        assert len(result["image_paths"]) == 4

    def test_deleting_without_a_session_id_is_refused(self, seeded):
        """An empty id would match every single-gate event ever recorded --
        never what a 'delete this run' button means."""
        for empty in ("", "   ", None):
            with database.get_session() as session:
                with pytest.raises(ValueError, match="needs a session id"):
                    database.delete_session(session, empty)
        with database.get_session() as session:
            assert len(database.get_all_events(session)) == 3

    def test_deleting_an_unknown_session_removes_nothing(self, seeded):
        with database.get_session() as session:
            assert database.delete_session(session, "NOPE")["events"] == 0
            assert len(database.get_all_events(session)) == 3

    def test_the_helper_also_deletes_the_image_files(self, seeded, tmp_path):
        from src.utils.data_reset import delete_processing_session

        config = {"database": {"path": str(tmp_path / "del.db")}}
        counts = delete_processing_session(config, "RUN_A")

        assert counts["events"] == 2
        assert counts["images"] == 4
        for run, plate_image, vehicle_image in seeded:
            if run == "RUN_A":
                assert not plate_image.exists()
                assert not vehicle_image.exists()
            else:
                assert plate_image.exists(), "another run's images were deleted"

    def test_a_missing_image_file_does_not_fail_the_delete(self, seeded, tmp_path):
        """The rows are already gone; a stuck file must not hide that."""
        from src.utils.data_reset import delete_processing_session

        for run, plate_image, _ in seeded:
            if run == "RUN_A":
                plate_image.unlink()

        counts = delete_processing_session(
            {"database": {"path": str(tmp_path / "del.db")}}, "RUN_A"
        )
        assert counts["events"] == 2


class TestSessionDeleteRoute:
    def test_deleting_a_session_over_the_api(self, client):
        listed = client.get("/trajectory-api/sessions").json()
        assert listed and listed[0]["processing_session"] == "SESS1"

        response = client.delete("/trajectory-api/sessions/SESS1")
        assert response.status_code == 200, response.text
        assert response.json()["deleted_events"] == 4

        assert client.get("/trajectory-api/sessions").json() == []
        assert client.get("/trajectory-api/trajectory/DL8CA1234").status_code == 404

    def test_deleting_an_unknown_session_is_a_404(self, client):
        assert client.delete("/trajectory-api/sessions/NOPE").status_code == 404

    def test_deleting_is_refused_while_a_run_is_in_progress(self, client, monkeypatch):
        from src.api import trajectory_routes

        monkeypatch.setattr(
            trajectory_routes._manager, "is_running", lambda: True
        )
        response = client.delete("/trajectory-api/sessions/SESS1")
        assert response.status_code == 409
        # ...and nothing was deleted.
        assert client.get("/trajectory-api/sessions").json()


class TestDatabaseBinding:
    """The API must serve the SAME database the pipeline writes to."""

    def test_the_api_opens_the_database_named_by_the_config(self, tmp_path, monkeypatch):
        """Regression: the server used to call init_db() with no argument, so
        it always opened sqlite:///data/alpr.db no matter what
        `database.path` said. With the stock config the two coincide, which
        hid it -- but point the pipeline elsewhere and the API silently
        served a different database and reported "no detections" for plates
        that had definitely been recorded.
        """
        pipeline_db = tmp_path / "pipeline.db"
        monkeypatch.delenv("DB_URL", raising=False)
        monkeypatch.setenv("ALPR_DB_PATH", str(pipeline_db))

        # Write an event the way the pipeline does, into the configured file.
        database.init_db(str(pipeline_db))
        with database.get_session() as session:
            database.insert_event(session, _event(plate_number="DL1AB2345"))

        import importlib

        from src.api import server as server_module
        importlib.reload(server_module)

        with TestClient(server_module.app) as client:
            assert str(pipeline_db) in str(database.get_engine().url)
            # The API can see what the pipeline wrote.
            assert len(client.get("/search?plate=DL1AB2345").json()) == 1

    def test_db_url_still_wins_when_set(self, tmp_path, monkeypatch):
        """$DB_URL stays the explicit override -- it is the only way to point
        this system at PostgreSQL, so config.yaml must not displace it."""
        explicit = tmp_path / "explicit.db"
        monkeypatch.setenv("DB_URL", f"sqlite:///{explicit}")
        monkeypatch.setenv("ALPR_DB_PATH", str(tmp_path / "ignored.db"))

        import importlib

        from src.api import server as server_module
        importlib.reload(server_module)

        with TestClient(server_module.app):
            assert str(explicit) in str(database.get_engine().url)


class TestExistingRoutesUnaffected:
    """The new routes are additive. Nothing that worked may stop working."""

    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DB_URL", f"sqlite:///{tmp_path}/api.db")
        from src.api.server import app

        with TestClient(app) as client:
            yield client

    def test_health_still_responds(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_the_single_gate_entry_endpoint_still_works(self, client):
        response = client.post("/entry", json={
            "plate_number": "DL5CX9999", "vehicle_type": "Car",
            "plate_color": "White", "series_type": "normal", "direction": "IN",
        })
        assert response.status_code == 201
        # The new columns come back as nulls rather than breaking the schema.
        assert response.json()["processing_session"] is None

    def test_logs_search_and_stats_still_respond(self, client):
        client.post("/entry", json={
            "plate_number": "DL5CX9999", "vehicle_type": "Car",
            "plate_color": "White", "series_type": "normal", "direction": "IN",
        })
        assert client.get("/logs").status_code == 200
        assert len(client.get("/search?plate=DL5CX9999").json()) == 1
        assert client.get("/stats").status_code == 200
        assert client.get("/vehicles/DL5CX9999").status_code == 200
