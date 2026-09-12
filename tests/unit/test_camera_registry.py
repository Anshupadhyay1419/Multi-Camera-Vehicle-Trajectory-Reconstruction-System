"""
Unit tests for the camera registry (cameras/registry.py).

The registry decides which cameras exist, what order they run in, and where
their video comes from -- so the tests here pin down the split the loader
deliberately makes between fatal and degradable problems:

  * Structural damage (no cameras list, duplicate ids) raises, because
    silently running a subset of the deployment corrupts trajectories.
  * Per-field damage (a swapped coordinate, an unknown source_type) is
    logged and degraded, because losing a site's whole feed to a typo is
    worse than losing its map pin.
"""

from __future__ import annotations

import textwrap

import pytest

from src.cameras.models import SourceType
from src.cameras.registry import (
    CameraConfigError,
    CameraRegistry,
    load_camera_registry,
)


def _write(tmp_path, body: str):
    path = tmp_path / "camera_config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


VALID = """
    cameras:
      - camera_id: "CAM001"
        camera_name: "India Gate"
        latitude: 28.6129
        longitude: 77.2295
        video_path: "data/uploads/CAM001.mp4"
        order: 1
      - camera_id: "CAM002"
        camera_name: "Connaught Place"
        latitude: 28.6315
        longitude: 77.2167
        video_path: "data/uploads/CAM002.mp4"
        order: 2
    """


class TestLoading:
    def test_loads_every_camera_in_order(self, tmp_path):
        registry = load_camera_registry(_write(tmp_path, VALID))
        assert [c.camera_id for c in registry] == ["CAM001", "CAM002"]
        assert registry.require("CAM001").camera_name == "India Gate"
        assert registry.require("CAM001").latitude == pytest.approx(28.6129)

    def test_orders_by_order_field_not_file_position(self, tmp_path):
        """`order` is the queue, so a file listing them backwards still runs
        them forwards."""
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "B", camera_name: "Second", order: 2}
              - {camera_id: "A", camera_name: "First",  order: 1}
            """)
        assert [c.camera_id for c in load_camera_registry(path)] == ["A", "B"]

    def test_equal_orders_break_ties_deterministically(self, tmp_path):
        """Two cameras claiming the same slot must still produce a stable
        sequence -- a run that reorders itself between invocations would make
        trajectory_order meaningless."""
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "Z", camera_name: "Z", order: 1}
              - {camera_id: "A", camera_name: "A", order: 1}
            """)
        first = [c.camera_id for c in load_camera_registry(path)]
        second = [c.camera_id for c in load_camera_registry(path)]
        assert first == second == ["A", "Z"]

    def test_defaults_block_applies_to_cameras_that_omit_the_key(self, tmp_path):
        path = _write(tmp_path, """
            defaults:
              enabled: false
              source_type: "rtsp"
            cameras:
              - {camera_id: "A", camera_name: "A", order: 1}
              - {camera_id: "B", camera_name: "B", order: 2, enabled: true}
            """)
        registry = load_camera_registry(path)
        assert registry.require("A").enabled is False
        assert registry.require("A").source_type is SourceType.RTSP
        # An explicit per-camera value beats the default.
        assert registry.require("B").enabled is True

    def test_missing_name_falls_back_to_the_id(self, tmp_path):
        path = _write(tmp_path, 'cameras:\n  - {camera_id: "CAM009", order: 1}\n')
        assert load_camera_registry(path).require("CAM009").camera_name == "CAM009"


class TestFatalProblems:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(CameraConfigError, match="not found"):
            load_camera_registry(str(tmp_path / "nope.yaml"))

    def test_unparseable_yaml_raises(self, tmp_path):
        with pytest.raises(CameraConfigError, match="Failed to parse"):
            load_camera_registry(_write(tmp_path, "cameras: [unclosed\n"))

    def test_no_cameras_list_raises(self, tmp_path):
        with pytest.raises(CameraConfigError, match="non-empty 'cameras' list"):
            load_camera_registry(_write(tmp_path, "processing: {}\n"))

    def test_empty_cameras_list_raises(self, tmp_path):
        with pytest.raises(CameraConfigError, match="non-empty 'cameras' list"):
            load_camera_registry(_write(tmp_path, "cameras: []\n"))

    def test_duplicate_camera_id_raises(self, tmp_path):
        """Two sites sharing an id merge every trajectory that passes either,
        irrecoverably -- so this must never be tolerated."""
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "CAM001", camera_name: "A", order: 1}
              - {camera_id: "CAM001", camera_name: "B", order: 2}
            """)
        with pytest.raises(CameraConfigError, match="Duplicate camera_id"):
            load_camera_registry(path)

    def test_missing_camera_id_raises(self, tmp_path):
        with pytest.raises(CameraConfigError, match="missing a camera_id"):
            load_camera_registry(_write(tmp_path, 'cameras:\n  - {camera_name: "X"}\n'))


class TestDegradedFields:
    """Bad per-field values cost that field, never the camera."""

    def test_unsurveyed_camera_loads_without_coordinates(self, tmp_path):
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "A", camera_name: "A", order: 1,
                 latitude: null, longitude: null}
            """)
        camera = load_camera_registry(path).require("A")
        assert camera.latitude is None and camera.longitude is None
        assert camera.has_location is False

    def test_swapped_coordinates_are_rejected_not_stored(self, tmp_path):
        """A latitude of 100 is impossible; storing it would plot the camera
        somewhere it isn't, which is worse than having no pin."""
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "A", camera_name: "A", order: 1,
                 latitude: 100.0, longitude: 77.2}
            """)
        camera = load_camera_registry(path).require("A")
        assert camera.latitude is None
        assert camera.longitude == pytest.approx(77.2)

    def test_string_coordinates_are_coerced(self, tmp_path):
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "A", camera_name: "A", order: 1,
                 latitude: "28.6129", longitude: "77.2295"}
            """)
        assert load_camera_registry(path).require("A").latitude == pytest.approx(28.6129)

    def test_unknown_source_type_falls_back_to_upload(self, tmp_path):
        """UPLOAD fails loudly at open time; guessing RTSP would make a typo
        look like a network outage."""
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "A", camera_name: "A", order: 1, source_type: "carrier-pigeon"}
            """)
        assert load_camera_registry(path).require("A").source_type is SourceType.UPLOAD

    def test_non_integer_order_falls_back_to_file_position(self, tmp_path):
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "A", camera_name: "A", order: "first"}
            """)
        assert load_camera_registry(path).require("A").order == 1


class TestRegistryBehaviour:
    def test_enabled_excludes_disabled_cameras(self, tmp_path):
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "A", camera_name: "A", order: 1, enabled: true}
              - {camera_id: "B", camera_name: "B", order: 2, enabled: false}
            """)
        registry = load_camera_registry(path)
        assert [c.camera_id for c in registry.enabled] == ["A"]
        assert len(registry.all) == 2       # still configured, just not queued

    def test_require_raises_for_unknown_camera(self, tmp_path):
        registry = load_camera_registry(_write(tmp_path, VALID))
        assert registry.get("NOPE") is None
        with pytest.raises(KeyError, match="NOPE"):
            registry.require("NOPE")

    def test_replace_swaps_a_camera_and_keeps_the_order(self, tmp_path):
        """The dashboard uses replace() after an upload; it must not disturb
        the queue or lose the other cameras."""
        import dataclasses

        registry = load_camera_registry(_write(tmp_path, VALID))
        updated = dataclasses.replace(
            registry.require("CAM002"), video_path="/tmp/new.mp4"
        )
        registry.replace(updated)
        assert registry.require("CAM002").video_path == "/tmp/new.mp4"
        assert [c.camera_id for c in registry] == ["CAM001", "CAM002"]

    def test_replace_rejects_an_unregistered_camera(self, tmp_path):
        import dataclasses

        registry = load_camera_registry(_write(tmp_path, VALID))
        stranger = dataclasses.replace(registry.require("CAM001"), camera_id="GHOST")
        with pytest.raises(KeyError, match="GHOST"):
            registry.replace(stranger)

    def test_settings_blocks_fall_back_to_defaults(self, tmp_path):
        """A registry written before `processing:`/`trajectory:` existed must
        still load with usable behaviour."""
        registry = load_camera_registry(_write(tmp_path, VALID))
        assert registry.processing["status_file"].endswith(".json")
        assert registry.trajectory["order_by"] == "auto"

    def test_settings_blocks_are_overridable(self, tmp_path):
        path = _write(tmp_path, """
            cameras:
              - {camera_id: "A", camera_name: "A", order: 1}
            trajectory:
              order_by: "trajectory_order"
              collapse_per_camera: false
            """)
        registry = load_camera_registry(path)
        assert registry.trajectory["order_by"] == "trajectory_order"
        assert registry.trajectory["collapse_per_camera"] is False
        # Unspecified keys still come from the defaults.
        assert "revisit_gap_seconds" in registry.trajectory


class TestCameraConfigHelpers:
    def test_event_metadata_matches_the_event_column_names(self, tmp_path):
        """The result is splatted into event_data, so the keys must be the
        VehicleEvent column names exactly."""
        camera = load_camera_registry(_write(tmp_path, VALID)).require("CAM001")
        assert set(camera.event_metadata()) == {
            "camera_id", "camera_name", "latitude", "longitude"
        }

    def test_source_uri_follows_source_type(self, tmp_path):
        import dataclasses

        camera = load_camera_registry(_write(tmp_path, VALID)).require("CAM001")
        assert camera.source_uri == "data/uploads/CAM001.mp4"

        live = dataclasses.replace(
            camera, source_type=SourceType.RTSP, rtsp_url="rtsp://host/stream"
        )
        assert live.source_uri == "rtsp://host/stream"

    def test_has_location_requires_both_coordinates(self, tmp_path):
        import dataclasses

        camera = load_camera_registry(_write(tmp_path, VALID)).require("CAM001")
        assert camera.has_location is True
        assert dataclasses.replace(camera, latitude=None).has_location is False
        assert dataclasses.replace(camera, longitude=None).has_location is False


def test_shipped_camera_config_is_valid():
    """The registry that actually ships must load and describe four sites."""
    registry = load_camera_registry("config/camera_config.yaml")
    assert len(registry) == 4
    assert [c.camera_id for c in registry] == ["CAM001", "CAM002", "CAM003", "CAM004"]
    assert all(c.has_location for c in registry), "every demo camera is surveyed"
    assert [c.order for c in registry] == [1, 2, 3, 4]
