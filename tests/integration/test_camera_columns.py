"""
Integration tests for camera attribution on stored events.

The four camera columns (camera_id, camera_name, latitude, longitude) were
added to vehicle_events after the system was already deployed, so the risky
part is not writing them -- it's that an existing database file keeps its old
shape. SQLAlchemy's create_all() only creates missing *tables*, never alters
an existing one, so db._migrate_schema() has to add the columns in place.

Tests cover:
- A legacy eight-column table is migrated in place, keeping its rows
- Migration is idempotent across repeated init_db() calls
- insert_event stores, and to_dict round-trips, the camera fields
- Callers that omit the camera fields still work (columns stay NULL)
- The API inherits this device's camera config, but never another camera's
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect

from src.database import db as database
from src.database.db import get_all_events, insert_event
from src.database.models import VehicleEvent

CAMERA_COLUMNS = ["camera_id", "camera_name", "latitude", "longitude"]

# vehicle_events exactly as it existed before camera attribution -- used to
# build a "legacy" database file for the migration tests.
_LEGACY_SCHEMA = """
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
CREATE INDEX idx_plate_number ON vehicle_events (plate_number);
CREATE INDEX idx_timestamp ON vehicle_events (timestamp);
"""


@pytest.fixture()
def reset_db_module():
    """Release the module-level engine after a test touches init_db()."""
    yield
    database._engine = None
    database._SessionFactory = None
    database._db_type = None


def _legacy_db(tmp_path, rows: int = 3) -> str:
    """Create a pre-migration database file with *rows* events in it."""
    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(_LEGACY_SCHEMA)
    conn.executemany(
        "INSERT INTO vehicle_events (plate_number, vehicle_type, plate_color, "
        "series_type, timestamp, direction, image_path) VALUES (?,?,?,?,?,?,?)",
        [(f"KA19TR{i:04d}", "Private", "White", "normal",
          datetime.now(timezone.utc).isoformat(), "IN", "") for i in range(rows)],
    )
    conn.commit()
    conn.close()
    return db_path


def _sample_event(**overrides) -> dict:
    base = {
        "plate_number": "KA19TR0234",
        "vehicle_type": "Private",
        "plate_color":  "White",
        "series_type":  "normal",
        "direction":    "IN",
        "image_path":   "data/plate_crops/test.jpg",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    def test_fresh_database_has_camera_columns(self, tmp_path, reset_db_module):
        database.init_db(str(tmp_path / "fresh.db"))
        cols = {c["name"] for c in inspect(database.get_engine()).get_columns("vehicle_events")}
        assert set(CAMERA_COLUMNS).issubset(cols)

    def test_legacy_table_gains_camera_columns(self, tmp_path, reset_db_module):
        db_path = _legacy_db(tmp_path)
        legacy_engine = create_engine(f"sqlite:///{db_path}")
        before = {c["name"] for c in
                  inspect(legacy_engine).get_columns("vehicle_events")}
        legacy_engine.dispose()
        assert not set(CAMERA_COLUMNS) & before, "fixture should start pre-migration"

        database.init_db(db_path)
        after = {c["name"] for c in
                 inspect(database.get_engine()).get_columns("vehicle_events")}
        assert set(CAMERA_COLUMNS).issubset(after)

    def test_migration_preserves_existing_rows(self, tmp_path, reset_db_module):
        db_path = _legacy_db(tmp_path, rows=5)
        database.init_db(db_path)
        with database.get_session() as session:
            events = get_all_events(session)
        assert len(events) == 5
        # Pre-existing rows have no camera to attribute them to.
        assert all(e["camera_id"] is None for e in events)
        assert all(e["plate_number"].startswith("KA19TR") for e in events)

    def test_migration_creates_the_camera_index(self, tmp_path, reset_db_module):
        db_path = _legacy_db(tmp_path)
        database.init_db(db_path)
        indexes = {i["name"] for i in
                   inspect(database.get_engine()).get_indexes("vehicle_events")}
        assert "idx_camera_id" in indexes

    def test_migration_is_idempotent(self, tmp_path, reset_db_module):
        """init_db() runs on every pipeline/API start, so a second pass over
        an already-migrated database must be a silent no-op."""
        db_path = _legacy_db(tmp_path)
        database.init_db(db_path)
        database.init_db(db_path)
        database.init_db(db_path)
        cols = [c["name"] for c in
                inspect(database.get_engine()).get_columns("vehicle_events")]
        # No column added twice, and the migrated table matches the model
        # exactly. Derived from VehicleEvent rather than a hardcoded count so
        # this keeps testing idempotency as the schema grows, instead of
        # needing an edit every time a column is added.
        expected = [c.name for c in VehicleEvent.__table__.columns]
        assert len(cols) == len(set(cols))
        assert set(cols) == set(expected)

    def test_partially_migrated_table_is_completed(self, tmp_path, reset_db_module):
        """Another process (or an interrupted start) may have added only some
        of the columns; the rest must still be added."""
        db_path = _legacy_db(tmp_path)
        conn = sqlite3.connect(db_path)
        conn.execute("ALTER TABLE vehicle_events ADD COLUMN camera_id VARCHAR")
        conn.commit()
        conn.close()

        database.init_db(db_path)
        cols = [c["name"] for c in
                inspect(database.get_engine()).get_columns("vehicle_events")]
        assert set(CAMERA_COLUMNS).issubset(cols)
        assert cols.count("camera_id") == 1

    def test_column_added_concurrently_does_not_fail_startup(
        self, tmp_path, reset_db_module, monkeypatch
    ):
        """The pipeline and the API both run init_db() at startup and are
        usually started together. If the other one adds a column between this
        one's inspect and its ALTER, "duplicate column" means the work is
        done -- not that startup should fail."""
        db_path = _legacy_db(tmp_path)
        real_ddl = database._apply_additive_ddl

        def racing_ddl(conn, statement, description):
            if "ADD COLUMN camera_name" in statement:
                other = sqlite3.connect(db_path)
                other.execute("ALTER TABLE vehicle_events ADD COLUMN camera_name VARCHAR")
                other.commit()
                other.close()
            return real_ddl(conn, statement, description)

        monkeypatch.setattr(database, "_apply_additive_ddl", racing_ddl)

        database.init_db(db_path)  # must not raise

        cols = [c["name"] for c in
                inspect(database.get_engine()).get_columns("vehicle_events")]
        assert set(CAMERA_COLUMNS).issubset(cols)
        assert cols.count("camera_name") == 1

    def test_unexpected_ddl_failure_still_propagates(self, tmp_path, reset_db_module):
        """Only "already exists" is tolerated -- init_db() must not report a
        healthy database it could not actually migrate."""
        from sqlalchemy import create_engine as _create_engine
        from sqlalchemy.exc import OperationalError

        db_path = _legacy_db(tmp_path)
        engine = _create_engine(f"sqlite:///{db_path}")
        try:
            with engine.begin() as conn:
                with pytest.raises(OperationalError):
                    database._apply_additive_ddl(
                        conn, "ALTER TABLE no_such_table ADD COLUMN x VARCHAR",
                        "nonsense",
                    )
        finally:
            engine.dispose()

    def test_migrated_database_accepts_camera_writes(self, tmp_path, reset_db_module):
        """The real point of the migration: new events can be stored."""
        db_path = _legacy_db(tmp_path)
        database.init_db(db_path)
        with database.get_session() as session:
            insert_event(session, _sample_event(
                plate_number="DL3CBJ1384", camera_id="GATE-01",
                camera_name="Main Gate", latitude=28.613939, longitude=77.209023,
            ))
        with database.get_session() as session:
            stored = next(e for e in get_all_events(session)
                          if e["plate_number"] == "DL3CBJ1384")
        assert stored["camera_id"] == "GATE-01"
        assert stored["latitude"] == pytest.approx(28.613939)


# ---------------------------------------------------------------------------
# insert_event
# ---------------------------------------------------------------------------

class TestInsertEventCameraFields:
    def test_round_trips_all_four_fields(self, db_session):
        event = insert_event(db_session, _sample_event(
            camera_id="GATE-02", camera_name="North Gate",
            latitude=-12.04318, longitude=-77.02824,
        ))
        assert event.camera_id == "GATE-02"
        assert event.camera_name == "North Gate"
        assert event.latitude == pytest.approx(-12.04318)
        assert event.longitude == pytest.approx(-77.02824)

    def test_omitted_camera_fields_store_as_null(self, db_session):
        """Callers written before camera attribution still work."""
        event = insert_event(db_session, _sample_event())
        assert event.camera_id is None
        assert event.camera_name is None
        assert event.latitude is None
        assert event.longitude is None

    def test_string_coordinates_are_stored_as_floats(self, db_session):
        event = insert_event(db_session, _sample_event(
            latitude="28.613939", longitude="77.209023",
        ))
        assert isinstance(event.latitude, float)
        assert event.latitude == pytest.approx(28.613939)

    def test_unparseable_coordinate_stores_null_rather_than_failing(self, db_session):
        """A bad coordinate costs the event its map position, not the event."""
        event = insert_event(db_session, _sample_event(
            plate_number="MH12AB1234", latitude="n/a", longitude=None,
        ))
        assert event is not None
        assert event.plate_number == "MH12AB1234"
        assert event.latitude is None

    def test_to_dict_exposes_camera_fields(self, db_session):
        insert_event(db_session, _sample_event(
            camera_id="GATE-01", camera_name="Main Gate",
            latitude=28.613939, longitude=77.209023,
        ))
        db_session.flush()
        event = get_all_events(db_session)[0]
        assert set(CAMERA_COLUMNS).issubset(event.keys())
        assert event["camera_name"] == "Main Gate"


# ---------------------------------------------------------------------------
# POST /entry camera inheritance
# ---------------------------------------------------------------------------

class TestApiCameraInheritance:
    """The API stamps its own configured camera onto events posted without
    one -- but must not lend this gate's name or coordinates to an event
    posted on behalf of a different camera."""

    @pytest.fixture(autouse=True)
    def api_db(self, tmp_path):
        database.init_db(str(tmp_path / "test_api_camera.db"))
        yield
        database._engine = None
        database._SessionFactory = None

    @pytest.fixture()
    def client(self):
        from src.api.server import app
        return TestClient(app)

    def _payload(self, **overrides) -> dict:
        base = {
            "plate_number": "KA19TR0234",
            "vehicle_type": "Private",
            "plate_color":  "White",
            "series_type":  "normal",
            "direction":    "IN",
        }
        base.update(overrides)
        return base

    def test_omitted_camera_inherits_local_config(self, client):
        from src.api.server import _camera_meta

        data = client.post("/entry", json=self._payload()).json()
        assert data["camera_id"] == _camera_meta["camera_id"]
        assert data["camera_name"] == _camera_meta["camera_name"]

    def test_explicit_fields_are_stored_as_sent(self, client):
        data = client.post("/entry", json=self._payload(
            camera_id="GATE-02", camera_name="North Gate",
            latitude=28.613939, longitude=77.209023,
        )).json()
        assert data["camera_id"] == "GATE-02"
        assert data["camera_name"] == "North Gate"
        assert data["latitude"] == pytest.approx(28.613939)

    def test_other_camera_does_not_inherit_this_gates_name(self, client):
        """Inheriting "Main Gate" for GATE-99 would label the event with a
        gate the vehicle never passed."""
        data = client.post("/entry", json=self._payload(camera_id="GATE-99")).json()
        assert data["camera_id"] == "GATE-99"
        assert data["camera_name"] is None
        assert data["latitude"] is None

    def test_same_camera_id_still_inherits_remaining_fields(self, client):
        from src.api.server import _camera_meta

        data = client.post("/entry", json=self._payload(
            camera_id=_camera_meta["camera_id"],
        )).json()
        assert data["camera_name"] == _camera_meta["camera_name"]

    def test_settings_endpoint_reports_the_camera(self, client):
        camera = client.get("/settings").json()["camera"]
        assert set(camera.keys()) == {"camera_id", "camera_name", "latitude", "longitude"}

    def test_logs_include_camera_fields(self, client):
        client.post("/entry", json=self._payload())
        events = client.get("/logs").json()
        assert events
        assert set(CAMERA_COLUMNS).issubset(events[0].keys())
