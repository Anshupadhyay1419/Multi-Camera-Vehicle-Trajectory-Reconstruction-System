"""
Shared pytest fixtures and Hypothesis configuration for the ALPR test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base

# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------

settings.register_profile(
    "fast",
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
settings.load_profile("fast")


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------

@pytest.fixture()
def random_bgr_image():
    """Return a factory that creates random BGR images of given size."""
    def _make(height: int = 64, width: int = 128) -> np.ndarray:
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return _make


@pytest.fixture()
def random_gray_image():
    """Return a factory that creates random grayscale images of given size."""
    def _make(height: int = 32, width: int = 80) -> np.ndarray:
        return np.random.randint(0, 256, (height, width), dtype=np.uint8)
    return _make


@pytest.fixture()
def small_plate_crop():
    """A small (30×80) BGR plate crop — below SR threshold."""
    return np.random.randint(0, 256, (30, 80, 3), dtype=np.uint8)


@pytest.fixture()
def large_plate_crop():
    """A large (60×200) BGR plate crop — above SR threshold."""
    return np.random.randint(0, 256, (60, 200, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# In-memory SQLite session
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_session():
    """Provide an in-memory SQLite session for database tests."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def initialized_db(tmp_path):
    """Initialize the database module with a temp SQLite file."""
    from src.database import db as database
    db_path = str(tmp_path / "test_alpr.db")
    database.init_db(db_path)
    yield database
    # Cleanup: reset module-level engine
    database._engine = None
    database._SessionFactory = None


# ---------------------------------------------------------------------------
# Camera registry
# ---------------------------------------------------------------------------

# The four camera sites the tests are written against.
#
# A FIXED registry, not a copy of config/camera_config.yaml: cameras are now
# added, renamed and removed from the dashboard, so the deployment's own file
# is operator state that changes between runs. Tests that read it broke the
# moment somebody renamed a camera -- and worse, tests that add or remove
# cameras would write to the live deployment.
TEST_CAMERA_SITES = (
    ("CAM001", "India Gate",      28.6129, 77.2295, 1),
    ("CAM002", "Connaught Place", 28.6315, 77.2167, 2),
    ("CAM003", "Karol Bagh",      28.6519, 77.1909, 3),
    ("CAM004", "Kashmere Gate",   28.6675, 77.2273, 4),
)

def _test_camera_config(directory) -> str:
    """The YAML for TEST_CAMERA_SITES, with every path inside *directory*.

    upload_dir and status_file are pointed at the temp directory too: left at
    their defaults they resolve to the deployment's own data/, so a test
    would read the status of the operator's last real run (and write junk
    uploads into it).
    """
    return "\n".join(
        ["defaults:",
         '  source_type: "upload"',
         "  enabled: true",
         "",
         "cameras:"]
        + [f'  - camera_id: "{camera_id}"\n'
           f'    camera_name: "{name}"\n'
           f"    latitude: {latitude}\n"
           f"    longitude: {longitude}\n"
           f'    video_path: "{directory / (camera_id + ".mp4")}"\n'
           f"    order: {order}\n"
           for camera_id, name, latitude, longitude, order in TEST_CAMERA_SITES]
        + ["processing:",
           f'  upload_dir: "{directory / "uploads"}"',
           f'  status_file: "{directory / "processing_status.json"}"',
           "  continue_on_error: true",
           "",
           "trajectory:",
           '  order_by: "auto"',
           "  collapse_per_camera: true",
           "  revisit_gap_seconds: 300",
           ""]
    )


@pytest.fixture()
def camera_config_file(tmp_path):
    """A four-camera registry of this suite's own, in a temp directory.

    Written per test, so a test may add, rename or remove cameras freely:
    those changes persist beside this file (cameras_runtime.yaml), never in
    the deployment's config/.
    """
    path = tmp_path / "camera_config.yaml"
    path.write_text(_test_camera_config(tmp_path), encoding="utf-8")
    return path
